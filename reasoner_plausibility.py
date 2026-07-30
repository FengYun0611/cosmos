"""
Cosmos3-Nano Reasoner — 物理合理性分析（Physical Plausibility）
判断视频是否符合物理规律
"""
from pathlib import Path
import torch
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

model_id = "nvidia/Cosmos3-Nano"
assets_dir = Path("/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/reasoner/assets")
video_path = (assets_dir / "physical_plausibility.mp4").resolve()

processor = AutoProcessor.from_pretrained(model_id)
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

prompt = (
    "Is this video physically plausible/possible according to your understanding "
    "of e.g. object permanence, shape constancy (objects maintain shape over time), "
    "continuous trajectories of objects? Assume it is the normal laws of physics.\n\n"
    "Your answer should be based on the events in the video and ignore the quality "
    "of the simulation engine. The rising wall is part of the experiment setup and "
    "should not be judged for plausibility.\n\n"
    "(A) Possible\n"
    "(B) Impossible"
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

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
trimmed = [out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids, inputs.input_ids)]
output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)
