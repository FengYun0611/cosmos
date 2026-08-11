#!/usr/bin/env python3
"""
Cosmos3-Edge Generator - Text-to-Video（文生视频）

背景：diffusers 的 Cosmos3OmniPipeline 仅支持 Qwen3VL 架构（Nano/Super），
不支持 Nemotron-2B-Dense-VL 架构的 Cosmos3-Edge。因此 Edge 改走官方推荐的
Cosmos Framework 推理路径：
    python -m cosmos_framework.scripts.inference --checkpoint-path Cosmos3-Edge

Edge 约束：无音频、仅 256p/480p、12-30fps、50-150 帧。
官方推荐：resolution=480, num_frames=121, fps=24。

本脚本只负责生成输入 JSON + 打印远程命令，不直接调模型/GUI。
请同步到远程后按实际路径修改下方常量再运行。
"""

import json
import os
from pathlib import Path

# ============ 可配置项（远程服务器） ============
INPUT_DIR = Path("/home/shenyanyuan/fengyun/cosmos/edge_t2v_inputs")
FRAMEWORK_DIR = "/home/shenyanyuan/fengyun/cosmos/packages/cosmos3"
OUTPUT_ROOT = "/home/shenyanyuan/fengyun/cosmos/edge_t2v_outputs"
HF_HOME = "/mnt/disk8/fengyun/huggingface"
# 从环境变量读取，避免硬编码泄露；需要登录时在执行前 export HF_TOKEN=...
HF_TOKEN = os.environ.get("HF_TOKEN", "")

PROMPTS = [
    "A mobile robot navigates a warehouse aisle and stops at a shelf.",
    "A humanoid robot walking on a sunny street, people watching in the background.",
    "A drone flying over a construction site, capturing aerial footage.",
]

# Edge 生成参数
NUM_FRAMES = 121
FPS = 24
RESOLUTION = "480"
ASPECT_RATIO = "16,9"
NUM_STEPS = 35
GUIDANCE = 6.0
SHIFT = 10.0


def build_sample(name: str, prompt: str) -> dict:
    return {
        "model_mode": "text2video",
        "name": name,
        "prompt": prompt,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "resolution": RESOLUTION,
        "aspect_ratio": ASPECT_RATIO,
        "num_steps": NUM_STEPS,
        "guidance": GUIDANCE,
        "shift": SHIFT,
        "enable_sound": False,
    }


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_files = []
    for i, prompt in enumerate(PROMPTS):
        name = f"t2v_edge_{i + 1}"
        path = INPUT_DIR / f"{name}.json"
        path.write_text(json.dumps(build_sample(name, prompt), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        input_files.append(path)
        print(f"[ok] 已生成输入文件: {path}")

    files_arg = " ".join(str(p) for p in input_files)
    env_block = f"export HF_HOME={HF_HOME}"
    if HF_TOKEN:
        env_block += f"\nexport HF_TOKEN={HF_TOKEN}"

    cmd = f"""{env_block}
cd {FRAMEWORK_DIR}
source .venv/bin/activate
python -m cosmos_framework.scripts.inference \\
    --parallelism-preset=latency \\
    -i {files_arg} \\
    -o {OUTPUT_ROOT} \\
    --checkpoint-path Cosmos3-Edge \\
    --no-guardrails \\
    --seed=0
"""
    print("\n" + "=" * 72)
    print("在远程服务器执行以下命令(Cosmos Framework 加载 Cosmos3-Edge):")
    print("=" * 72)
    print(cmd)
    print("=" * 72)
    print(f"  * 输出: {OUTPUT_ROOT}/t2v_edge_<N>/vision.mp4  (N=1,2,3)")
    print("  * 首次运行自动下载 nvidia/Cosmos3-Edge 权重到 HF_HOME")
    print("  * 如需安全检测（需装 guardrail 依赖+apt 系统包），去掉 --no-guardrails")


if __name__ == "__main__":
    main()