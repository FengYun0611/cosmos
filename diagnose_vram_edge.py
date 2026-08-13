#!/usr/bin/env python3
"""
Cosmos3-Edge Generator 显存占比诊断脚本
========================================
在 Cosmos Framework 推理路径上测量 Edge（Nemotron-2B-Dense-VL + DiT Tokenizer）
加载与生成时的 GPU 显存占用 / 占比。

与 generator_t2v.py 完全同路径：
    python -m cosmos_framework.scripts.inference --checkpoint-path Cosmos3-Edge

分段记录：
    [阶段0] 空载基线
    [阶段1] 模型加载完成（纯权重）      —— Edge ≈ 8GB（2B × 2 towers × 2B/tower）
    [阶段2] 生成过程中峰值（后台轮询）   —— 480p×121帧×24fps×35步 真实峰值
    [阶段3] 生成完成后回落

用法（远程服务器，与 generator_t2v.py 相同环境）：
    export HF_HOME=/mnt/disk8/fengyun/huggingface
    export HF_TOKEN=...
    cd /path/to/cosmos/packages/cosmos3
    source .venv/bin/activate
    python diagnose_vram_edge.py --num-steps 35 --num-frames 121 --resolution 480

可选参数（用于对比显存增量）：
    --num-steps 5/15/35    步数越少峰值越低（但权重不变）
    --num-frames 61/121    帧长影响 latent token 数与激活值
    --resolution 256/480   Edge 支持 256p / 480p
"""

import argparse
import os
import threading
import time
from pathlib import Path

# ============ 可配置（远程服务器，与 generator_t2v.py 一致） ============
_OS_ENV = {
    "HF_HOME": os.environ.get("HF_HOME", "/mnt/disk8/fengyun/huggingface"),
}
if os.environ.get("HF_TOKEN"):
    _OS_ENV["HF_TOKEN"] = os.environ["HF_TOKEN"]

# 先初始化框架（配置 log / 分布式 / CUDA device），再导入 torch 与框架 API
from cosmos_framework.inference.common.init import init_script

init_script(env=_OS_ENV)

import torch
from cosmos_framework.inference.args import OmniSampleOverrides, OmniSetupOverrides
from cosmos_framework.inference.inference import OmniInference

_PROMPT = "A mobile robot navigates a warehouse aisle and stops at a shelf."


# ============ 显存探针 ============
def _total_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def print_vram(tag: str, *, reset_peak: bool = False) -> None:
    """打印显存占用 GB 与占显卡总显存的百分比。"""
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    total = _total_gb()
    alloc = torch.cuda.memory_allocated() / 1024**3
    resv = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"  [{tag:34s}] alloc={alloc:7.2f}G ({alloc/total*100:5.1f}%)  "
        f"reserved={resv:7.2f}G ({resv/total*100:5.1f}%)  "
        f"peak={peak:7.2f}G ({peak/total*100:5.1f}%)  [total={total:.1f}G]"
    )


class VramMeter:
    """后台线程轮询峰值显存；stop() 返回生成期间的最大 alloc（GB）。"""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._peak_gb = 0.0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        torch.cuda.reset_peak_memory_stats()
        self._peak_gb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak_gb = max(self._peak_gb, torch.cuda.max_memory_allocated() / 1024**3)
            time.sleep(0.05)

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._peak_gb = max(self._peak_gb, torch.cuda.max_memory_allocated() / 1024**3)
        return self._peak_gb


# ============ 主流程 ============
def main() -> None:
    parser = argparse.ArgumentParser(description="Cosmos3-Edge 生成显存占比诊断")
    parser.add_argument("--num-steps", type=int, default=35, help="扩散采样步数（默认35）")
    parser.add_argument("--num-frames", type=int, default=121, help="视频帧数（默认121）")
    parser.add_argument("--resolution", type=str, default="480", choices=["256", "480"])
    parser.add_argument("--fps", type=int, default=24, help="帧率（默认24）")
    parser.add_argument("--guidance", type=float, default=6.0, help="CFG guidance（默认6.0）")
    parser.add_argument("--shift", type=float, default=5.0, help="UniPC shift（480p 训练 shift=5）")
    parser.add_argument("--output-root", type=str, default="edge_vram_outputs")
    parser.add_argument("--skip-generate", action="store_true", help="只测权重占用，不跑生成")
    args = parser.parse_args()

    total_gb = _total_gb()
    dev_name = torch.cuda.get_device_name(0)
    print("=" * 78)
    print(f"GPU: {dev_name}  (总显存 {total_gb:.1f} GB)")
    print(
        f"Edge 参数: resolution={args.resolution} num_frames={args.num_frames} fps={args.fps} "
        f"num_steps={args.num_steps} guidance={args.guidance}"
    )
    print("=" * 78)

    # ---------- 阶段0：加载前基线 ----------
    torch.cuda.reset_peak_memory_stats()
    print_vram("0. 模型加载前 (baseline)")

    # ---------- 构建 setup（与 generator_t2v 相同的 Cosmos3-Edge 路径） ----------
    output_dir = Path(args.output_root).absolute()
    setup = OmniSetupOverrides(
        checkpoint_path="Cosmos3-Edge",
        output_dir=str(output_dir),
        parallelism_preset="latency",
        guardrails=False,
        benchmark=False,
        warmup=0,
        num_iterations=1,
        sample_overrides=OmniSampleOverrides(
            model_mode="text2video",
            name="vram_test",
            prompt=_PROMPT,
            num_frames=args.num_frames,
            fps=args.fps,
            resolution=args.resolution,
            aspect_ratio="16,9",
            num_steps=args.num_steps,
            guidance=args.guidance,
            shift=args.shift,
            enable_sound=False,
        ),
    ).build_setup()

    # ---------- 阶段1：模型加载（权重上 GPU） ----------
    print("\n--- 加载 Cosmos3-Edge 模型（权重移动到 GPU） ---")
    pipe = OmniInference.create(setup)
    print_vram("1. 模型加载完成（纯权重）", reset_peak=True)

    # ---------- 权重子模块参数量统计 ----------
    print("\n各子模块参数量（bf16 估算）:")
    model = pipe.model
    root = getattr(model, "net", None) or model
    for name in ["language_model", "tokenizer_vision_gen"]:
        mod = getattr(root, name, None)
        if mod is None:
            print(f"  {name:30s}: 未找到")
            continue
        params = sum(
            p.numel() for p in mod.parameters() if p.dtype in (torch.float32, torch.bfloat16, torch.float16)
        )
        print(f"  {name:30s}: {params/1e6:8.1f}M params ≈ {params*2/1024**3:6.2f} GB (bf16)")

    # ---------- 阶段2：生成（可选） ----------
    peak_gen_gb = 0.0
    if args.skip_generate:
        print("\n[跳过生成] 仅测量权重占用。")
    else:
        sample_args = setup.sample_overrides.build_sample(model_config=pipe.model_config)
        sample_args.output_dir = output_dir / "vram_test"

        print("\n--- 执行一次生成（后台线程监测峰值显存） ---")
        meter = VramMeter()
        meter.start()
        pipe.generate([sample_args])
        peak_gen_gb = meter.stop()
        print_vram("2. 生成完成后", reset_peak=True)

    # ---------- 结论 ----------
    print("\n" + "=" * 78)
    print("最终结论（显存占比 = 占用 / 显卡总显存）")
    print("=" * 78)
    weight_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  纯权重（模型加载后）        : {weight_gb:6.2f} GB ({weight_gb/total_gb*100:5.1f}%)")
    if peak_gen_gb > 0:
        print(f"  生成峰值                  : {peak_gen_gb:6.2f} GB ({peak_gen_gb/total_gb*100:5.1f}%)")
        print(f"  生成增量（激活值+缓存）    : {peak_gen_gb - weight_gb:6.2f} GB")
    print(f"  显卡总显存                : {total_gb:6.1f} GB")
    print("=" * 78)


if __name__ == "__main__":
    main()
