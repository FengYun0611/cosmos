"""
Cosmos3-Nano Reasoner — Embodied Reasoning（具身推理）
机器人/自动驾驶/辅助任务的下一步动作推理
"""
import os
os.environ["TORCHVISION_USE_AV"] = "0"

import cv2
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

model_id = "nvidia/Cosmos3-Nano"
assets_dir = Path("/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/reasoner/assets")

processor = AutoProcessor.from_pretrained(model_id)
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)


def read_frames_cpu(video_path, max_frames=32):
    """OpenCV CPU 解码，返回 PIL Image 列表"""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    print(f"  采样 {len(frames)} 帧 (共 {total} 帧)")
    return frames


def run_video(video_name, prompt, max_frames=8, max_tokens=1024):
    video_path = (assets_dir / video_name).resolve()
    frames = read_frames_cpu(video_path, max_frames=max_frames)

    # 把每帧作为图片传入
    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, torch.bfloat16)
    # 用采样代替贪心解码，避免重复
    generated_ids = model.generate(
        **inputs, do_sample=True, temperature=0.7, top_p=0.9,
        max_new_tokens=max_tokens
    )
    trimmed = [out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids, inputs.input_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# ===== 任务 1: 机器人下一步 =====
print("=" * 60)
print("任务 1: 机器人下一步动作")
print("视频: robotics_next_action.mp4")
print("=" * 60)
prompt1 = (
    "What can be the next immediate action?\n\n"
    "Answer the question using the following format:\n"
    "<think>\nYour reasoning.\n</think>\n"
    "Write your final answer immediately after the </think> tag."
)
print(run_video("robotics_next_action.mp4", prompt1))
print()

# ===== 任务 2: 自动驾驶 =====
print("=" * 60)
print("任务 2: 自动驾驶轨迹规划")
print("视频: drive_scene_next_action.mp4")
print("=" * 60)
prompt2 = (
    "You are an autonomous vehicle planning system. "
    "The video depicts the observation from the vehicle's camera. "
    "You need to observe the critical objects in the environment "
    "and reason your next action and the driving trajectory ahead."
)
print(run_video("drive_scene_next_action.mp4", prompt2, max_frames=8))
print()

# ===== 任务 3: 辅助任务 =====
print("=" * 60)
print("任务 3: 辅助任务 — 打印机换墨盒")
print("视频: assisted_task_next_action.mp4")
print("=" * 60)
prompt3 = (
    'This is the overall task: "The student exchanges the black ink cartridge of the printer."\n'
    'In the video, the agent is following the instruction: "place old ink_cartridge."\n'
    "What should be the next action of the agent?\n\n"
    "Answer using the format:\n"
    "<think>\nYour reasoning.\n</think>\n"
    "Write your final answer after the </think> tag."
)
print(run_video("assisted_task_next_action.mp4", prompt3))
