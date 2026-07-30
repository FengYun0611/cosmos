"""
Cosmos3-Nano Generator — Image-to-Video（图生视频）
给定一张图片，生成后续视频
优化：添加显存监控 + CPU offload 降低显存峰值
"""
import os
os.environ["HF_HOME"] = "/mnt/disk8/fengyun/huggingface"
os.environ["HF_TOKEN"] = "hf_"

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

model_id = "nvidia/Cosmos3-Nano"
assets_dir = "/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/generator/audiovisual/assets"

# ── 显存监控 ──
def log_vram(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_alloc = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{tag:20s}] allocated={allocated:.1f}G | reserved={reserved:.1f}G | max_alloc={max_alloc:.1f}G")
    return allocated, reserved, max_alloc

# ── 加载模型 ──
log_vram("before_load")
pipe = Cosmos3OmniPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    safety_checker=None,
    enable_safety_checker=False,
    token=os.environ["HF_TOKEN"],
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)
pipe.to("cuda")
log_vram("after_move_to_cuda")

# ── 关键：显存优化（真正降低峰值） ──
# 1) 顺序 CPU offload：不用的模块暂时挪到 CPU，大幅降低显存（但会慢一点）
pipe.enable_sequential_cpu_offload()
# 2) Attention slicing：切分 attention 计算图，减少中间激活值显存
pipe.enable_attention_slicing()
# 3) VAE slicing：切分 VAE 解码，防止 VAE 时显存暴涨
pipe.enable_vae_slicing()
log_vram("after_optimizations")

# 重置峰值计数器，以便只看推理阶段的增量
torch.cuda.reset_peak_memory_stats()

# 测试图片：车、机器人、海岸公路
examples = [
    {
        "image": f"{assets_dir}/images/image2video/car_driving.jpg",
        "prompt": "The car drives forward on a road with trees on both sides.",
        "output": "/home/shenyanyuan/fengyun/cosmos/i2v_car.mp4",
    },
    {
        "image": f"{assets_dir}/images/image2video/humanoid_robot.jpg",
        "prompt": "The humanoid robot walks forward slowly.",
        "output": "/home/shenyanyuan/fengyun/cosmos/i2v_robot.mp4",
    },
    {
        "image": f"{assets_dir}/images/image2video/coastal_road_audio.jpg",
        "prompt": "A car drives along a scenic coastal road with ocean view.",
        "output": "/home/shenyanyuan/fengyun/cosmos/i2v_coastal.mp4",
    },
]

for i, ex in enumerate(examples, 1):
    print(f"\n{'='*60}")
    print(f"样例 {i}: {ex['output']}")
    print(f"{'='*60}")
    image = load_image(ex["image"])
    
    # 推理前的显存状态
    before_alloc, _, _ = log_vram("before_inference")
    
    result = pipe(
        prompt=ex["prompt"],
        negative_prompt="blurry, distorted, low quality",
        image=image,
        num_frames=25,
        height=320,
        width=512,
        fps=10,
        num_inference_steps=20,
        guidance_scale=6.0,
        enable_sound=False,
        add_resolution_template=False,
        add_duration_template=False,
        generator=torch.Generator(device="cuda").manual_seed(1234),
    )
    
    # 推理后的显存（包括峰值）
    after_alloc, _, infer_peak = log_vram("after_inference")
    print(f"推理增量显存: {infer_peak - before_alloc:.1f}G")
    
    export_to_video(result.video, ex["output"], fps=10, macro_block_size=1)
    print(f"  已保存: {ex['output']}")

print("\n全部完成！")
