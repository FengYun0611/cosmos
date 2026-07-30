"""
Cosmos3-Nano Generator — Action Policy（动作策略）
给定起始画面 + 任务指令，预测动作序列和未来视频
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

# 用机器人操作台的图片
image_path = "/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/generator/audiovisual/assets/images/image2video/car_driving.jpg"
image = load_image(image_path)

action = CosmosActionCondition(
    mode="policy",
    chunk_size=30,
    domain_name="av",
    image=image,
    resolution_tier=256,
)

prompt = "The car turns right at the intersection and then accelerates on the straight road."

print(f"运行 Action Policy")
print(f"任务: {prompt}")
result = pipe(
    prompt=prompt,
    action=action,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
)

video_path = "/home/shenyanyuan/fengyun/cosmos/policy_output.mp4"
export_to_video(result.video, video_path, fps=10, macro_block_size=1)
print(f"视频已保存: {video_path} ({len(result.video)} 帧)")

if result.action is not None:
    print(f"预测动作: {len(result.action)} 组, 每组形状: {result.action[0].shape}")
    # 打印前 10 步的动作值
    action_data = result.action[0]  # [30, 9]
    print(f"前 10 步动作值 (9维: translation xyz + 6D rotation):")
    print(f"{'步数':>4} | {'tx':>8} {'ty':>8} {'tz':>8} {'r1':>8} {'r2':>8} {'r3':>8} {'r4':>8} {'r5':>8} {'r6':>8}")
    print("-" * 90)
    for t in range(min(10, action_data.shape[0])):
        vals = action_data[t].float()
        row = " | ".join(f"{v:>8.4f}" for v in vals)
        print(f"{t:>4} | {row}")
    # 保存到文件（bf16 → float32 → numpy）
    import numpy as np
    np.savetxt("/home/shenyanyuan/fengyun/cosmos/policy_actions.csv",
               action_data.float().cpu().numpy(), delimiter=",",
               header="tx,ty,tz,r1,r2,r3,r4,r5,r6")
    print(f"\n完整动作已保存: policy_actions.csv")
else:
    print("无动作输出")
