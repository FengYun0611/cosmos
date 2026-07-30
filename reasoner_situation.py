"""
Cosmos3-Nano Reasoner — Situation Understanding（场景理解）
理解视频中的人物行为，预测下一步动作
"""
from pathlib import Path
import torch
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

model_id = "nvidia/Cosmos3-Nano"
assets_dir = Path("/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/reasoner/assets")
video_path = (assets_dir / "situation_understanding.mp4").resolve()

processor = AutoProcessor.from_pretrained(model_id)
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

prompt = (
    "What is the person doing with the skillet? "
    "What will the person likely do next in this situation?"
)

messages = [
    {"role": "user", "content": [
        {"type": "video", "path": str(video_path)},
        {"type": "text", "text": prompt},
    ]}
]

inputs = processor.apply_chat_template(
    messages, fps=2, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).to(model.device, torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=512)
trimmed = [out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids, inputs.input_ids)]
output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)
