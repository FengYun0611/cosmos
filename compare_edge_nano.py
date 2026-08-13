#!/usr/bin/env python3
"""
Cosmos3 Edge vs Nano — image2video 性能对比（控制变量）
=======================================================
同一张输入图 + 同一生成参数，分别用 Cosmos3-Edge 和 Cosmos3-Nano 跑图生视频，
对比：权重显存 / 生成峰值显存 / 生成耗时 / 参数量 / 显存占比。

控制变量（保证两边一致，只让模型不同）：
    image=car_driving.jpg, 480p/832×480, 121帧, 24fps, 50步, guidance=5.0, 相同 seed
    分辨率取 480p 是因为 Edge 上限 480p，Nano 也能跑，两边都满足。
    每模型用各自的 480p shift：nano=5.0, edge=3.0（shift 只影响画质不影响速度/显存）。

前置：在 env_edge（Python 3.13 + git HEAD diffusers）里跑，Edge 和 Nano 都能加载。
权重从同一 HF_HOME 缓存读取。

用法：
    python compare_edge_nano.py                 # Edge + Nano 都跑（时间较长）
    python compare_edge_nano.py --models nano   # 只跑 Nano
    python compare_edge_nano.py --models edge nano
    python compare_edge_nano.py --offline       # 纯离线（权重需已缓存）
"""

import argparse
import gc
import os
import time
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/mnt/disk8/fengyun/huggingface")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

# 每模型的 480p 配置（shift 各用各的推荐值）
MODELS = {
    "edge": {
        "id": "nvidia/Cosmos3-Edge",
        "shift": 3.0,   # cookbook: Edge 480p flow_shift=3.0
    },
    "nano": {
        "id": "nvidia/Cosmos3-Nano",
        "shift": 5.0,   # framework: Nano(8B) 480p shift=5.0
    },
}

DEFAULT_IMAGE = "/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/generator/audiovisual/assets/images/image2video/car_driving.jpg"
PROMPT = "The car drives forward on a road with trees on both sides."
NEGATIVE = "blurry, distorted, low quality, jittery, flickering"


def _total_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def log_vram(tag: str = "", *, reset_peak: bool = False) -> None:
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


def run_one(name: str, args: argparse.Namespace, total_gb: float) -> dict:
    cfg = MODELS[name]
    print("\n" + "=" * 78)
    print(f">>> 模型: {name.upper()} ({cfg['id']})")
    print("=" * 78)

    log_vram(f"{name} 加载前", reset_peak=True)
    t0 = time.perf_counter()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        cfg["id"],
        dtype=torch.bfloat16,
        safety_checker=None,
        enable_safety_checker=False,
        token=HF_TOKEN or None,
        local_files_only=args.offline,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=cfg["shift"])
    pipe.to("cuda")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    log_vram(f"{name} 加载完成（纯权重）", reset_peak=True)
    weight_gb = torch.cuda.memory_allocated() / 1024**3

    total_params = 0
    for mod_name in ["language_model", "transformer", "vae"]:
        mod = getattr(pipe, mod_name, None)
        if mod is not None:
            n = sum(p.numel() for p in mod.parameters()
                    if p.dtype in (torch.float32, torch.bfloat16, torch.float16))
            total_params += n
    print(f"  统计参数量: {total_params/1e6:.0f}M ≈ {total_params*2/1024**3:.1f} GB (bf16)")

    torch.cuda.reset_peak_memory_stats()
    print(f"  生成中...（{args.num_frames}帧 {args.width}x{args.height} {args.num_steps}步）")
    t0 = time.perf_counter()
    peak_gb = 0.0
    oom = False
    try:
        image = load_image(args.image)
        result = pipe(
            prompt=args.prompt,
            negative_prompt=NEGATIVE,
            image=image,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            fps=args.fps,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance,
            enable_sound=False,
            add_resolution_template=False,
            add_duration_template=False,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        )
        torch.cuda.synchronize()
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        print(f"  ⚠ 显存不足 (OOM): {e}")
        oom = True
    gen_s = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    log_vram(f"{name} 生成完成", reset_peak=True)

    if not oom:
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, f"i2v_{name}_seed{args.seed}.mp4")
        export_to_video(result.video, out, fps=args.fps, macro_block_size=1)
        print(f"  已保存: {out}")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "model": name, "weight_gb": weight_gb, "params_m": total_params / 1e6,
        "load_s": load_s, "gen_s": gen_s, "peak_gb": peak_gb,
        "peak_pct": peak_gb / total_gb * 100, "oom": oom,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cosmos3 Edge vs Nano image2video 性能对比")
    parser.add_argument("--models", nargs="+", choices=["edge", "nano"], default=["edge", "nano"])
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="edge_nano_compare")
    parser.add_argument("--offline", action="store_true", help="只用本地缓存加载（不联网）")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"错误: 输入图不存在: {args.image}")
        return

    total_gb = _total_gb()
    print("=" * 78)
    print(f"GPU: {torch.cuda.get_device_name(0)}  (总显存 {total_gb:.1f} GB)")
    print(f"控制变量: image={args.image}")
    print(f"          分辨率={args.height}x{args.width} 帧数={args.num_frames} fps={args.fps} "
          f"步数={args.num_steps} guidance={args.guidance} seed={args.seed}")
    print(f"对比模型: {', '.join(m.upper() for m in args.models)}")
    print("=" * 78)

    results = [run_one(m, args, total_gb) for m in args.models]

    print("\n" + "=" * 78)
    print("对比结果")
    print("=" * 78)
    print(f"{'模型':<8}{'权重GB':>9}{'参数M':>9}{'加载s':>8}{'生成s':>9}{'峰值GB':>9}{'峰值%':>8}  OOM")
    print("-" * 78)
    for r in results:
        print(
            f"{r['model']:<8}{r['weight_gb']:>9.2f}{r['params_m']:>9.0f}"
            f"{r['load_s']:>8.1f}{r['gen_s']:>9.1f}{r['peak_gb']:>9.2f}"
            f"{r['peak_pct']:>8.1f}  {'⚠' if r['oom'] else '—'}"
        )
    print("=" * 78)
    if len(results) == 2 and not any(r["oom"] for r in results):
        e, n = {r["model"]: r for r in results}["edge"], {r["model"]: r for r in results}["nano"]
        print(f"Nano 比 Edge 峰值高 {n['peak_gb'] - e['peak_gb']:.1f}G，"
              f"生成慢 {n['gen_s'] / e['gen_s']:.1f}×")
    print("=" * 78)


if __name__ == "__main__":
    main()
