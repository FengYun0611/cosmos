"""
Cosmos3-Nano Reasoner — 常识推理（Common Sense Reasoning）
判断视频中的物理场景是否合理
"""
from pathlib import Path
import torch
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

model_id = "nvidia/Cosmos3-Nano"
assets_dir = Path("/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/reasoner/assets")
video_path = (assets_dir / "common_sense_reasoning.mp4").resolve()

processor = AutoProcessor.from_pretrained(model_id)
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

# 使用提示词中的 <think> 格式启用推理过程
prompt = (
    "Can the countertop support the weight of the juicers?\n\n"
    "Answer the question using the following format:\n\n"
    "<think>\n"
    "Your reasoning.\n"
    "</think>\n\n"
    "Write your final answer immediately after the </think> tag."
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": str(video_path)},
            {"type": "text", "text": prompt},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, fps=2, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).to(model.device, torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=1024)
trimmed = [out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids, inputs.input_ids)]
output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)
