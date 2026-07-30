"""
Cosmos3-Nano Reasoner — Describe Anything（属性描述）
描述图片中标记物体的属性和类别
"""
from pathlib import Path
import torch
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

model_id = "nvidia/Cosmos3-Nano"
assets_dir = Path("/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/reasoner/assets")
image_path = (assets_dir / "describe_anything.png").resolve()

processor = AutoProcessor.from_pretrained(model_id)
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": str(image_path)},
            {
                "type": "text",
                "text": "Please caption the notable attributes in the provided image. "
                        "List and describe all marked subjects in the image with their "
                        'categories and detailed captions using a json with keyword '
                        '"subject_id", "category" and "caption".'
            },
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
output = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]
print(output)
