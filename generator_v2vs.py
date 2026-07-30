"""
Cosmos3-Nano Generator — Video-to-Video with Sound（视频转换+声音）
"""
import os
os.environ["HF_HOME"] = "/mnt/disk8/fengyun/huggingface"
os.environ["HF_TOKEN"] = "hf_i"

import random
import cv2
import numpy as np
from PIL import Image
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


def load_video_frames(video_path, max_frames=25, target_height=320):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            scale = target_height / h
            frame = cv2.resize(frame, (int(w * scale), target_height))
            frames.append(Image.fromarray(frame))
    cap.release()
    print(f"  源视频: {len(frames)} 帧")
    return frames


source_video = "/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/reasoner/assets/drive_scene_next_action.mp4"

tasks = [
    {
        "prompt": "The car honks and turns right at the intersection, engine revving as it accelerates away.",
        "output": "/home/shenyanyuan/fengyun/cosmos/v2vs_turn_right.mp4",
    },
    {
        "prompt": "The car slows down and pulls over to the side of the road, engine idling with turn signal clicking.",
        "output": "/home/shenyanyuan/fengyun/cosmos/v2vs_pull_over.mp4",
    },
]

frames = load_video_frames(source_video, max_frames=25)

for i, task in enumerate(tasks):
    print(f"\n[{i+1}/{len(tasks)}] {task['prompt'][:50]}...")
    seed = random.randint(0, 2**31)
    result = pipe(
        prompt=task["prompt"],
        negative_prompt="blurry, distorted, low quality",
        video=frames,
        num_frames=25,
        height=320,
        width=512,
        fps=10,
        num_inference_steps=20,
        guidance_scale=6.0,
        enable_sound=True,
        condition_frame_indexes_vision=[0],
        add_resolution_template=False,
        add_duration_template=False,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    )
    if result.sound is not None:
        encode_video(
            result.video, fps=10, output_path=task["output"],
            audio=result.sound,
            audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
        )
    else:
        export_to_video(result.video, task["output"], fps=10, macro_block_size=1)
    print(f"  已保存: {task['output']} (含音频: {result.sound is not None})")

print("全部完成！")
