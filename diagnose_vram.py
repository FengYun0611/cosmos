"""
诊断脚本：分段测量 Cosmos3 各组件加载到 GPU 各占多少显存
不开启任何 offload 优化，以便看清真实占用
"""
import os
os.environ["HF_HOME"] = "/mnt/disk8/fengyun/huggingface"
os.environ["HF_TOKEN"] = "hf_imUvc"

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import load_image

model_id = "nvidia/Cosmos3-Nano"

def print_vram(tag=""):
    alloc = torch.cuda.memory_allocated() / 1024**3
    resv = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  [{tag:35s}] alloc={alloc:.2f}G  reserved={resv:.2f}G  peak={peak:.2f}G")
    return alloc

# 重置统计
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

print("=" * 70)
print("第1步：加载模型（HF 下载/缓存，还在 CPU）")
print("=" * 70)
pipe = Cosmos3OmniPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    safety_checker=None,
    enable_safety_checker=False,
    token=os.environ["HF_TOKEN"],
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)
print_vram("after load (still on CPU)")

# 逐个查看可获取参数量的子模块
print("\n各子模块的参数量（大致）:")
for name in ['transformer', 'vae']:
    mod = getattr(pipe, name, None)
    if mod is not None:
        params = sum(p.numel() for p in mod.parameters() if p.dtype in [torch.float32, torch.bfloat16, torch.float16])
        param_gb = params * 2 / 1024**3  # bf16 = 2 bytes per param
        print(f"  {name:30s}: {params/1e6:.1f}M params ≈ {param_gb:.2f}GB (bf16)")

input("\n按 Enter 继续，将逐个移动组件到 CUDA 观察...")

print("\n--- 1) 移动 VAE 到 GPU ---")
if hasattr(pipe, 'vae') and pipe.vae is not None:
    pipe.vae.to("cuda")
    print_vram("after vae.to(cuda)")

print("\n--- 2) 移动 transformer (DiT 核心) 到 GPU ---")
if hasattr(pipe, 'transformer') and pipe.transformer is not None:
    pipe.transformer.to("cuda")
    print_vram("after transformer.to(cuda)")

total_weights = torch.cuda.memory_allocated() / 1024**3
print(f"\n{'='*60}")
print(f">>> 纯权重占用（transformer + VAE在GPU上）: {total_weights:.2f} GB")
print(f"{'='*60}")

# ========== 执行一次推理看峰值 ==========
input("\n按 Enter 继续，将执行一次推理（5步，无任何 offload 优化）...")

torch.cuda.reset_peak_memory_stats()
assets_dir = "/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/generator/audiovisual/assets"
image = load_image(f"{assets_dir}/images/image2video/car_driving.jpg")

print("\n--- 3) 推理前 ---")
before = print_vram("before inference")

result = pipe(
    prompt="The car drives forward on a road with trees on both sides.",
    negative_prompt="blurry, distorted, low quality",
    image=image,
    num_frames=25,
    height=320,
    width=512,
    fps=10,
    num_inference_steps=5,
    guidance_scale=6.0,
    enable_sound=False,
    add_resolution_template=False,
    add_duration_template=False,
    generator=torch.Generator(device="cuda").manual_seed(1234),
)

print("\n--- 4) 推理后 ---")
print_vram("after inference")

peak = torch.cuda.max_memory_allocated() / 1024**3

print("\n" + "=" * 70)
print("最终结论")
print("=" * 70)
print(f"  纯权重（transformer + VAE）: {total_weights:.2f} GB")
print(f"  推理峰值                  : {peak:.2f} GB")
print(f"  推理增量（激活值+中间缓存）  : {peak - total_weights:.2f} GB")
print(f"\n  结论：权重本身已占 {total_weights:.2f}GB，峰值自然就是 ~30GB")
print(f"  降低分辨率只影响那 {peak - total_weights:.2f}GB 的增量，不影响主体")
print("=" * 70)

