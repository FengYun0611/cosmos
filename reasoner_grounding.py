"""
Cosmos3-Nano Reasoner — 2D Grounding（目标定位）
用图片中的物体，输出 bounding box 坐标
"""
from pathlib import Path
import torch
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

model_id = "nvidia/Cosmos3-Nano"
assets_dir = Path("/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/reasoner/assets")
image_path = (assets_dir / "grounding_2d.png").resolve()

processor = AutoProcessor.from_pretrained(model_id)
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

# ========== 任务 1: 定位 "load as a whole" ==========
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": str(image_path)},
            {"type": "text", "text": "Locate the accurate bounding box of the load as a whole. Return a json."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
print("=" * 60)
print("任务 1: Locate bounding box of 'load as a whole'")
print("=" * 60)
print(processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0])
print()

# ========== 任务 2: 定位 fork 叉车 ==========
messages2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": str(image_path)},
            {"type": "text", "text": "Locate the accurate bounding box of the fork. Return a json."},
        ],
    }
]

inputs2 = processor.apply_chat_template(
    messages2,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, torch.bfloat16)

generated_ids2 = model.generate(**inputs2, do_sample=False, max_new_tokens=512)
generated_ids2_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs2.input_ids, generated_ids2)
]
print("=" * 60)
print("任务 2: Locate bounding box of 'fork'")
print("=" * 60)
print(processor.batch_decode(
    generated_ids2_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0])
print()

# ========== 任务 3: 定位 person ==========
messages3 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": str(image_path)},
            {"type": "text", "text": "Locate the accurate bounding box of the person. Return a json."},
        ],
    }
]

inputs3 = processor.apply_chat_template(
    messages3,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, torch.bfloat16)

generated_ids3 = model.generate(**inputs3, do_sample=False, max_new_tokens=512)
generated_ids3_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs3.input_ids, generated_ids3)
]
print("=" * 60)
print("任务 3: Locate bounding box of 'person'")
print("=" * 60)
print(processor.batch_decode(
    generated_ids3_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0])
