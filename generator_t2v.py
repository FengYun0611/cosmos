"""
Cosmos3-Edge Generator — Text-to-Video（文生视频）
参考 cookbooks/cosmos3/generator/audiovisual/run_with_diffusers.ipynb 编写
"""
import os

os.environ["HF_HOME"] = "/mnt/disk8/fengyun/huggingface"
os.environ["HF_TOKEN"] = "hf_imUvVUUMjc"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import gc
import random
import time
import torch
import torch.nn as nn
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video


def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(
            f"\n[{tag}] GPU Memory:"
            f"\n  Allocated: {allocated:.2f} GB"
            f"\n  Reserved : {reserved:.2f} GB"
            f"\n  Peak     : {max_allocated:.2f} GB\n"
        )


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model_id = "nvidia/Cosmos3-Edge"

print_gpu_memory("Before loading model")

t0 = time.time()
pipe = Cosmos3OmniPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    safety_checker=None,
    enable_safety_checker=False,
    token=os.environ["HF_TOKEN"],
)
print(f"Loaded pipeline in {time.time() - t0:.1f}s")

print_gpu_memory("After loading model (CPU)")

# =============================
# 修复：Edge 模型的部分参数（norm_q, norm_k, mlp_moe_gen.gate_proj 等）
# 不在 checkpoint 中，被留在 meta device 上。直接 .to("cuda") 会报错：
#   "Cannot copy out of meta tensor; no data!"
# 需要先递归物化所有 meta tensor，再移到 GPU。
#
# 注意：Cosmos3OmniPipeline 本身不是 nn.Module（diffusers Pipeline 容器），
# 没有 .modules()/.parameters() 方法。必须通过 pipe.components 拿到
# 内部真正的 nn.Module 组件（transformer、vae、text_encoder 等）再处理。
# =============================
def materialize_meta_tensors(module: nn.Module, dtype=torch.bfloat16):
    """递归物化单个 nn.Module 上的 meta 参数和 buffer"""
    # 物化 parameters
    for name, param in module.named_parameters(recurse=False):
        if param.device.type == "meta":
            new_param = nn.Parameter(torch.empty(param.size(), dtype=dtype))
            nn.init.kaiming_uniform_(new_param)
            del module._parameters[name]
            module.register_parameter(name, new_param)
    # 物化 buffers
    for name, buf in module.named_buffers(recurse=False):
        if buf.device.type == "meta":
            new_buf = torch.empty(buf.size(), dtype=dtype)
            del module._buffers[name]
            module.register_buffer(name, new_buf)
    # 递归子模块
    for child in module.children():
        materialize_meta_tensors(child, dtype)


def materialize_pipeline_meta_tensors(pipeline):
    """
    遍历 pipeline 的所有组件（components dict），
    对每个 nn.Module 组件递归物化 meta tensor。
    """
    for component_name, component in pipeline.components.items():
        if isinstance(component, nn.Module):
            meta_names = [n for n, p in component.named_parameters() if p.device.type == "meta"]
            if meta_names:
                print(f"  Materializing meta tensors in component '{component_name}' ({len(meta_names)} params)")
            materialize_meta_tensors(component)


print("Materializing meta tensors ...")
materialize_pipeline_meta_tensors(pipe)
print("Done materializing meta tensors.")

# 设置 scheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=10.0
)

# 移到 GPU
pipe.to("cuda")

print_gpu_memory("After moving model to CUDA")

# =============================
# 推理参数（参考 notebook 的 FIXED_SAMPLING）
# =============================
FIXED_SAMPLING = {
    "num_steps": 35,
    "guidance": 6.0,
    "shift": 10.0,
    "fps": 24,
    "num_frames": 121,     # Edge 标准 profile
    "resolution": "480",   # 480p
    "aspect_ratio": "16,9",
}

prompts = [
    "A mobile robot navigates a warehouse aisle and stops at a shelf.",
    "A humanoid robot walking on a sunny street, people watching in the background.",
    "A drone flying over a construction site, capturing aerial footage.",
]

height, width = 480, 832  # 480p 16:9

for i, prompt in enumerate(prompts):

    print(f"\n[{i+1}/{len(prompts)}] 生成: {prompt[:60]}...")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print_gpu_memory("Before inference")

    seed = random.randint(0, 2**31)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    t0 = time.time()

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality",

            num_frames=FIXED_SAMPLING["num_frames"],
            height=height,
            width=width,
            fps=FIXED_SAMPLING["fps"],

            num_inference_steps=FIXED_SAMPLING["num_steps"],
            guidance_scale=FIXED_SAMPLING["guidance"],

            enable_sound=False,
            add_resolution_template=False,
            add_duration_template=False,

            generator=generator,
        )

    elapsed = time.time() - t0
    print(f"Generated in {elapsed:.1f}s")

    print_gpu_memory("After inference")

    output_path = f"/home/shenyanyuan/fengyun/cosmos/t2v_edge_output_{i+1}.mp4"

    export_to_video(
        result.video,
        output_path,
        fps=FIXED_SAMPLING["fps"],
        macro_block_size=1
    )
    print(f"  已保存: {output_path}")

    # 清理
    del result
    gc.collect()
    torch.cuda.empty_cache()

    print_gpu_memory("After cleanup")

print("\n全部完成！")
