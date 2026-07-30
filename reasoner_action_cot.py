"""
Cosmos3-Nano Reasoner — Action CoT（动作链推理）
预测机械臂轨迹坐标 + 自动驾驶场景推理
"""
import os
os.environ["TORCHVISION_USE_AV"] = "0"

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

model_id = "nvidia/Cosmos3-Nano"
assets_dir = Path("/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/reasoner/assets")

processor = AutoProcessor.from_pretrained(model_id)
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)


def run_image(image_name, task_desc, max_tokens=1024):
    image_path = (assets_dir / image_name).resolve()
    prompt = (
        f'You are given the task "{task_desc}". '
        "Specify the 2D trajectory your end effector should follow in pixel space. "
        "Return the trajectory coordinates in JSON format like this: "
        '{"point_2d": [x, y], "label": "gripper trajectory"}.\n\n'
        "Answer the question using the following format:\n\n"
        "<think>\n"
        "Your reasoning.\n"
        "</think>\n\n"
        "Write your final answer immediately after the </think> tag."
    )
    messages = [
        {"role": "user", "content": [
            {"type": "image", "path": str(image_path)},
            {"type": "text", "text": prompt},
        ]}
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, torch.bfloat16)
    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens)
    trimmed = [out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids, inputs.input_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# ===== 任务 1: 轨迹坐标 — Move pink bowl =====
print("=" * 60)
print("任务 1: 轨迹预测 — 'Move the pink bowl to the right'")
print("图片: action_cot_trajectory.png")
print("=" * 60)
print(run_image("action_cot_trajectory.png", "Move the pink bowl to the right"))
print()

# ===== 任务 2: 轨迹坐标 — Put flower into bottle =====
print("=" * 60)
print("任务 2: 轨迹预测 — 'Put flower into the red bottle'")
print("图片: robot_planning.png")
print("=" * 60)
print(run_image("robot_planning.png", "Put flower into the red bottle"))
print()

# ===== 任务 3: 自动驾驶场景推理 =====
print("=" * 60)
print("任务 3: 自动驾驶场景推理（Driving Scene）")
print("视频: action_cot_driving_scene.mp4")
print("=" * 60)

video_path = (assets_dir / "action_cot_driving_scene.mp4").resolve()
cap = cv2.VideoCapture(str(video_path))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
indices = np.linspace(0, total - 1, min(16, total), dtype=int)
frames = []
for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        scale = 320 / h
        frame = cv2.resize(frame, (int(w * scale), 320))
        frames.append(Image.fromarray(frame))
cap.release()
print(f"  采样 {len(frames)} 帧")

prompt3 = (
    "The video depicts the observation from the vehicle's camera. "
    "You need to think step by step and identify the objects in the scene "
    "that are critical for safe navigation.\n\n"
    "Answer the question using the following format:\n"
    "<think>\nYour reasoning.\n</think>\n"
    "Write your final answer immediately after the </think> tag."
)
content = [{"type": "image", "image": f} for f in frames]
content.append({"type": "text", "text": prompt3})
messages3 = [{"role": "user", "content": content}]
inputs3 = processor.apply_chat_template(
    messages3, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).to(model.device, torch.bfloat16)
generated_ids3 = model.generate(**inputs3, do_sample=False, max_new_tokens=512)
trimmed3 = [out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids3, inputs3.input_ids)]
print(processor.batch_decode(trimmed3, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
