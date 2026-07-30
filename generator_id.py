"""
Cosmos3-Nano Generator — Inverse Dynamics（逆向动力学）
给定一段视频，反推出产生该运动所需的动作序列
"""
import os
os.environ["HF_HOME"] = "/mnt/disk8/fengyun/huggingface"
os.environ["HF_TOKEN"] = "hf_i"

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition
from diffusers.utils import export_to_video

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

# 用一段行车视频反推动作
video_path = "/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/reasoner/assets/drive_scene_next_action.mp4"

# OpenCV 读取视频帧
cap = cv2.VideoCapture(video_path)
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
print(f"源视频: 采样 {len(frames)} 帧")

action = CosmosActionCondition(
    mode="inverse_dynamics",
    chunk_size=len(frames) - 1,
    domain_name="av",
    video=frames,
    resolution_tier=256,
)

prompt = "A car driving on a city road approaching an intersection."
print(f"运行 Inverse Dynamics")
print(f"任务: {prompt}")

result = pipe(
    prompt=prompt,
    action=action,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
)

video_out = "/home/shenyanyuan/fengyun/cosmos/id_output.mp4"
export_to_video(result.video, video_out, fps=10, macro_block_size=1)
print(f"视频已保存: {video_out} ({len(result.video)} 帧)")

if result.action is not None:
    action_data = result.action[0]
    print(f"反推动作: {action_data.shape[0]} 步 x {action_data.shape[1]} 维")
    print(f"前 5 步:")
    print(f"{'步数':>4} | {'tx':>8} {'ty':>8} {'tz':>8} {'r1':>8} {'r2':>8} {'r3':>8} {'r4':>8} {'r5':>8} {'r6':>8}")
    print("-" * 90)
    for t in range(min(5, action_data.shape[0])):
        vals = action_data[t].float()
        row = " | ".join(f"{v:>8.4f}" for v in vals)
        print(f"{t:>4} | {row}")
    np.savetxt("/home/shenyanyuan/fengyun/cosmos/id_actions.csv",
               action_data.float().cpu().numpy(), delimiter=",")
    print(f"完整动作已保存: id_actions.csv")
else:
    print("无反推动作输出")
