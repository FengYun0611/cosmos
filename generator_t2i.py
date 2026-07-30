"""
Cosmos3-Nano Generator — Text-to-Image（文生图）
多组 prompt 示例
"""
import os
os.environ["HF_HOME"] = "/mnt/disk8/fengyun/huggingface"
os.environ["HF_TOKEN"] = "hf_ic"

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

model_id = "nvidia/Cosmos3-Nano"

pipe = Cosmos3OmniPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    safety_checker=None,
    enable_safety_checker=False,
    token=os.environ["HF_TOKEN"],
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)
pipe.to("cuda")

prompts = [
    "A self-driving car navigating a busy city intersection at night with streetlights and pedestrians.",
    "Bird's eye view of an autonomous vehicle driving on a highway with multiple lanes during sunset.",
    "Dashboard camera view of a car approaching a crosswalk with a pedestrian waiting to cross.",
    "An autonomous truck driving through a tunnel with bright overhead lights and reflective lane markings.",
    "Aerial view of a traffic jam on a highway with autonomous vehicles in dedicated lanes.",
]

for i, prompt in enumerate(prompts):
    print(f"[{i+1}/{len(prompts)}] 生成: {prompt[:50]}...")
    result = pipe(
        prompt=prompt,
        negative_prompt="blurry, distorted, low quality",
        num_frames=1,
        height=320,
        width=512,
        num_inference_steps=20,
        guidance_scale=6.0,
        add_resolution_template=False,
        add_duration_template=False,
        generator=torch.Generator(device="cuda").manual_seed(1234 + i),
    )
    output_path = f"/home/shenyanyuan/fengyun/cosmos/t2i_drive_{i+1}.png"
    result.video[0].save(output_path)
    print(f"  已保存: {output_path}")

print("全部完成！")
