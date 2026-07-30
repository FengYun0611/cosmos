"""
Cosmos3-Nano Generator — Forward Dynamics（正向动力学）
给定起始画面 + 动作序列，预测未来视频
"""
import os
os.environ["HF_HOME"] = "/mnt/disk8/fengyun/huggingface"
os.environ["HF_TOKEN"] = "hf_xxx"

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition
from diffusers.utils import export_to_video, load_image

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

image_path = "/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/generator/audiovisual/assets/images/image2video/car_driving.jpg"
image = load_image(image_path)

# 模拟 AV 动作序列: [steer, accel, brake, ...] (9维)
# 直行→右转→直行，共 60 步 ≈ 6 秒
chunk_size = 60
raw_actions = torch.zeros(chunk_size, 9)
raw_actions[:, 1] = 0.2      # 持续轻加速
raw_actions[15:35, 0] = 0.4  # 第15-35帧: 右转
raw_actions[35:, 0] = -0.1   # 回正+小幅左修正

action = CosmosActionCondition(
    mode="forward_dynamics",
    chunk_size=chunk_size,
    domain_name="av",
    image=image,
    raw_actions=raw_actions,
    resolution_tier=256,
)

print(f"运行 Forward Dynamics: {image_path}")
result = pipe(
    prompt="",
    action=action,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
)

output_path = "/home/shenyanyuan/fengyun/cosmos/fd_output.mp4"
export_to_video(result.video, output_path, fps=10, macro_block_size=1)
print(f"已保存: {output_path} ({len(result.video)} 帧)")
