"""
Cosmos3-Nano Reasoner — 时序定位（Temporal Localization）
OpenCV CPU 解码 + 采样模式，避免 av 解码和重复问题
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


def read_frames_cpu(video_path, max_frames=8, target_height=336):
    """OpenCV CPU 解码，缩放后返回 PIL Image 列表"""
    cap = cv2.VideoCapture(str(video_path))
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
            new_w = int(w * scale)
            frame = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(frame))
    cap.release()
    print(f"  采样 {len(frames)} 帧 (共 {total} 帧), 缩放至 {target_height}px 高")
    return frames


def run(video_name, prompt, max_frames=8, max_tokens=512):
    video_path = (assets_dir / video_name).resolve()
    frames = read_frames_cpu(video_path, max_frames=max_frames)

    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, torch.bfloat16)

    generated_ids = model.generate(
        **inputs, do_sample=True, temperature=0.7, top_p=0.9,
        max_new_tokens=max_tokens
    )
    trimmed = [out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids, inputs.input_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# ===== 任务 1: 动作段时序定位 =====
print("=" * 60)
print("任务 1: 时序定位 — 列出所有动作段")
print("视频: temporal_localization_1.mp4")
print("=" * 60)
prompt1 = (
    'List all action segments in the video.\n\n'
    "Provide the result in json format with 'seconds' for time depiction. "
    "Use 'start', 'end' and 'caption' in the json output. "
    "List multiple events if applicable.\n\n"
    "```json\n"
    '[{"start": t_start, "end": t_end, "caption": EVENT1}]\n'
    "```"
)
print(run("temporal_localization_1.mp4", prompt1))
print()

# ===== 任务 2: 事件时间线 =====
print("=" * 60)
print("任务 2: 事件时间线（mm:ss.ff 格式）")
print("视频: temporal_localization_2.mp4")
print("=" * 60)
prompt2 = (
    "Describe the notable events in the provided video. "
    "Provide the result in json format with 'mm:ss.ff' format. "
    "Use 'start', 'end' and 'caption' in the json output."
)
print(run("temporal_localization_2.mp4", prompt2))
print()

# ===== 任务 3: 时间戳查询 =====
print("=" * 60)
print("任务 3: 时间戳查询")
print("视频: temporal_localization_2.mp4")
print("=" * 60)
prompt3 = (
    'When is "A man in a white sweater walks out of a room carrying a box" '
    "depicted? Provide json with 'start', 'end' in mm:ss.ff format."
)
print(run("temporal_localization_2.mp4", prompt3))
