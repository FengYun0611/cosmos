import os

os.environ["HF_HOME"] = "/mnt/disk8/fengyun/huggingface"
os.environ["HF_TOKEN"] = "hf_im"

import random
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video


# =========================
# 显存监控函数
# =========================
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


# 清空缓存，记录峰值
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


model_id = "nvidia/Cosmos3-Nano"


print_gpu_memory("Before loading model")


pipe = Cosmos3OmniPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    safety_checker=None,
    enable_safety_checker=False,
    token=os.environ["HF_TOKEN"],
)


print_gpu_memory("After loading model (CPU)")


pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=10.0
)


pipe.enable_model_cpu_offload()


print_gpu_memory("After moving model to CUDA")


prompts = [
    "A mobile robot navigates a warehouse aisle and stops at a shelf.",
    "A humanoid robot walking on a sunny street, people watching in the background.",
    "A drone flying over a construction site, capturing aerial footage.",
]


for i, prompt in enumerate(prompts):

    print(f"\n[{i+1}/{len(prompts)}] 生成: {prompt[:50]}...")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


    print_gpu_memory("Before inference")


    seed = random.randint(0, 2**31)


    with torch.no_grad():

        result = pipe(
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality",

            num_frames=25,
            height=320,
            width=512,
            fps=10,

            num_inference_steps=20,
            guidance_scale=6.0,

            enable_sound=False,
            add_resolution_template=False,
            add_duration_template=False,

            generator=torch.Generator(
                device="cpu"
            ).manual_seed(seed),
        )


    print_gpu_memory("After inference")


    output_path = f"/home/shenyanyuan/fengyun/cosmos/t2v_output_{i+1}.mp4"

    export_to_video(
        result.video,
        output_path,
        fps=10,
        macro_block_size=1
    )


    print(f"  已保存: {output_path}")


    # 释放当前视频结果
    del result
    torch.cuda.empty_cache()


    print_gpu_memory("After cleanup")


print("\n全部完成！")