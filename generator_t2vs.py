"""
Cosmos3-Nano Generator — Text-to-Video with Sound（文生视频+声音）
"""
import os
os.environ["HF_HOME"] = "/mnt/disk8/fengyun/huggingface"
os.environ["HF_TOKEN"] = "hf_imU"

import random
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, encode_video

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
    "A robot arm picks up a metal object on a factory assembly line, with mechanical humming sounds.",
    "A car drives on a rainy city street at night, windshield wipers moving and tires splashing.",
    "A drone flies over a beach with waves crashing, wind sound and seagulls in the background.",
]

for i, prompt in enumerate(prompts):
    print(f"\n[{i+1}/{len(prompts)}] 生成: {prompt[:50]}...")
    seed = random.randint(0, 2**31)
    result = pipe(
        prompt=prompt,
        negative_prompt="blurry, distorted, low quality",
        num_frames=25,
        height=320,
        width=512,
        fps=10,
        num_inference_steps=20,
        guidance_scale=6.0,
        enable_sound=True,          # 开启音频生成
        add_resolution_template=False,
        add_duration_template=False,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    )

    output_path = f"/home/shenyanyuan/fengyun/cosmos/t2vs_output_{i+1}.mp4"
    if result.sound is not None:
        encode_video(
            result.video,
            fps=10,
            output_path=output_path,
            audio=result.sound,
            audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
        )
    else:
        export_to_video(result.video, output_path, fps=10, macro_block_size=1)
    print(f"  已保存: {output_path} (含音频: {result.sound is not None})")

print("\n全部完成！")
