#!/usr/bin/env python3
"""
Cosmos3-Edge Generator — 显存占比 + 多模式测试（VRAM & Capability Test）
=======================================================================
用 diffusers 的 Cosmos3OmniPipeline 加载 Cosmos3-Edge，测试 Edge 支持的三种
生成模式，并分段测量 GPU 显存占比（占比 = 占用 / 显卡总显存）：

    --mode text2video   文生视频（默认，121帧/480p）
    --mode text2image   文生图（1帧/480p）
    --mode image2video  图生视频（121帧/480p，需 --image）

前置条件：diffusers 必须是 git HEAD 版本（PyPI 0.39.0 不支持 Edge）。
Edge 官方参数（cookbook）：480p/832×480、50步、guidance=5.0、flow_shift=3.0、无音频。

分段记录：
    [阶段0] 加载前基线
    [阶段1] 模型加载完成（纯权重）     —— Edge ≈ 8GB
    [阶段2] 生成中峰值（重置后 max）   —— 480p×121帧×50步
    [阶段3] 生成完成后回落

用法（远程 env_edge，Python 3.13 + git HEAD diffusers）：
    export HF_TOKEN=...
    python generator_vram.py                              # 文生视频（默认）
    python generator_vram.py --mode text2image --prompt "..."
    python generator_vram.py --mode image2video --image /path/to/car.jpg
    python generator_vram.py --mode text2video --guidance 6.0 --seed 42 --num-steps 60

想只测权重不跑生成，加 --skip-generate。
"""

import argparse
import os
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/mnt/disk8/fengyun/huggingface")
# 从环境变量读取，避免硬编码泄露；需要登录时 export HF_TOKEN=...
HF_TOKEN = os.environ.get("HF_TOKEN", "")

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

model_id = "nvidia/Cosmos3-Edge"

# 每个模式的默认参数（分辨率统一 480p/832×480，Edge 上限）
DEFAULTS = {
    "text2video": {
        "num_frames": 121,
        "fps": 24,
        "ext": ".mp4",
        "prompt": (
            "A white mobile robot with a rotating sensor turret rolls slowly down the "
            "center aisle of a modern warehouse, tall shelves stacked with cardboard boxes "
            "on both sides. The camera follows smoothly behind it at a steady walking pace, "
            "softly dollying forward. Warm overhead lighting, shallow depth of field, "
            "gentle reflections on the polished concrete floor, photorealistic, crisp "
            "details, smooth consistent motion."
        ),
        "negative_prompt": "blurry, distorted, low quality, jittery, flickering, deformed robot, wrong anatomy",
    },
    "text2image": {
        "num_frames": 1,
        "fps": 24,
        "ext": ".png",
        "prompt": (
            "A sleek white autonomous delivery robot on a sunlit city sidewalk in front "
            "of a modern glass building, golden hour light, shallow depth of field, "
            "photorealistic, high detail, cinematic composition."
        ),
        "negative_prompt": "blurry, distorted, low quality, watermark, text",
    },
    "image2video": {
        "num_frames": 121,
        "fps": 24,
        "ext": ".mp4",
        "prompt": (
            "The car drives steadily forward along the road, green trees passing by on "
            "both sides, the camera tracks the car smoothly, natural consistent motion, "
            "photorealistic."
        ),
        "negative_prompt": "blurry, distorted, low quality, jittery, flickering",
    },
}

DEFAULT_IMAGE = "/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/generator/audiovisual/assets/images/image2video/car_driving.jpg"


# ── 显存探针（占比 = 占用 / 显卡总显存）──
def _total_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def log_vram(tag: str = "", *, reset_peak: bool = False) -> tuple[float, float, float]:
    """打印显存占用 GB 与占显卡总显存的百分比。"""
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    total = _total_gb()
    alloc = torch.cuda.memory_allocated() / 1024**3
    resv = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"  [{tag:30s}] alloc={alloc:7.2f}G ({alloc/total*100:5.1f}%)  "
        f"reserved={resv:7.2f}G ({resv/total*100:5.1f}%)  "
        f"peak={peak:7.2f}G ({peak/total*100:5.1f}%)  [total={total:.1f}G]"
    )
    return alloc, resv, peak


def main() -> None:
    parser = argparse.ArgumentParser(description="Cosmos3-Edge 显存占比 + 多模式测试")
    parser.add_argument("--mode", choices=list(DEFAULTS), default="text2video",
                        help="生成模式（默认 text2video）")
    parser.add_argument("--prompt", default=None, help="覆盖默认 prompt")
    parser.add_argument("--negative-prompt", default=None, help="覆盖默认 negative prompt")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="image2video 的输入图")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--output-dir", default="edge_vram_outputs")
    parser.add_argument("--skip-generate", action="store_true", help="只测权重不跑生成")
    parser.add_argument("--offline", action="store_true",
                        help="只用本地缓存加载（不联网），权重需已下载过")
    args = parser.parse_args()

    cfg = DEFAULTS[args.mode]
    prompt = args.prompt or cfg["prompt"]
    negative_prompt = args.negative_prompt or cfg["negative_prompt"]

    # ── 阶段0：加载前基线 ──
    total_gb = _total_gb()
    print("=" * 78)
    print(f"GPU: {torch.cuda.get_device_name(0)}  (总显存 {total_gb:.1f} GB)")
    print(f"模式: {args.mode}  | {cfg['num_frames']}帧 832x480 | {args.num_steps}步 | "
          f"guidance={args.guidance} | flow_shift={args.flow_shift} | seed={args.seed}")
    print(f"prompt: {prompt[:90]}{'...' if len(prompt) > 90 else ''}")
    print("=" * 78)
    log_vram("0. 加载前 (baseline)", reset_peak=True)

    # ── 阶段1：加载模型（权重上 GPU）──
    print("\n--- 加载 Cosmos3-Edge（权重移动到 GPU） ---")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        safety_checker=None,
        enable_safety_checker=False,   # 关闭 guardrail，避免额外显存干扰测量
        token=HF_TOKEN or None,
        local_files_only=args.offline,   # --offline 时纯离线加载缓存权重
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)
    pipe.to("cuda")
    log_vram("1. 加载完成（纯权重）", reset_peak=True)
    weight_gb = torch.cuda.memory_allocated() / 1024**3

    # 各子模块参数量（大致，bf16 估算）
    print("\n各子模块参数量:")
    for name in ['language_model', 'transformer', 'vae']:
        mod = getattr(pipe, name, None)
        if mod is not None:
            params = sum(p.numel() for p in mod.parameters()
                         if p.dtype in [torch.float32, torch.bfloat16, torch.float16])
            print(f"  {name:28s}: {params/1e6:8.1f}M params ≈ {params*2/1024**3:6.2f} GB (bf16)")

    # ── 阶段2：生成（重置峰值，只看生成段）──
    peak_gen_gb = 0.0
    if args.skip_generate:
        print("\n[--skip-generate] 仅测量权重占用，跳过生成。")
    else:
        if args.mode == "image2video":
            print(f"\n--- 生成（{cfg['num_frames']}帧 832x480 {args.num_steps}步，输入图 {args.image}） ---")
        else:
            print(f"\n--- 生成（{cfg['num_frames']}帧 832x480 {args.num_steps}步） ---")
        torch.cuda.reset_peak_memory_stats()

        if args.mode == "image2video":
            if not os.path.exists(args.image):
                print(f"  错误: 输入图不存在: {args.image}")
                print("  请用 --image 指定一张真实存在的图片，例如:")
                print("    cookbooks/cosmos3/generator/audiovisual/assets/images/image2video/car_driving.jpg")
                return
            image = load_image(args.image)
        else:
            image = None
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_frames=cfg["num_frames"],
            height=480,
            width=832,
            fps=cfg["fps"],
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance,
            enable_sound=False,
            add_resolution_template=False,
            add_duration_template=False,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        )
        torch.cuda.synchronize()
        peak_gen_gb = torch.cuda.max_memory_allocated() / 1024**3
        log_vram("2. 生成完成（峰值已含 VAE 解码）", reset_peak=True)

        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"edge_{args.mode}_seed{args.seed}{cfg['ext']}")
        if cfg["ext"] == ".png":
            result.video[0].save(out_path)
        else:
            export_to_video(result.video, out_path, fps=cfg["fps"], macro_block_size=1)
        print(f"  已保存: {out_path}")

    # ── 结论 ──
    print("\n" + "=" * 78)
    print("最终结论（显存占比 = 占用 / 显卡总显存）")
    print("=" * 78)
    print(f"  纯权重（模型加载后）     : {weight_gb:6.2f} GB ({weight_gb/total_gb*100:5.1f}%)")
    if peak_gen_gb > 0:
        print(f"  生成峰值                 : {peak_gen_gb:6.2f} GB ({peak_gen_gb/total_gb*100:5.1f}%)")
        print(f"  生成增量（激活值+缓存）  : {peak_gen_gb - weight_gb:6.2f} GB")
    print(f"  显卡总显存               : {total_gb:6.1f} GB")
    print("=" * 78)


if __name__ == "__main__":
    main()
