"""
Cosmos3-Nano Reasoner 图像理解 Demo
"""
from pathlib import Path
import torch
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

model_id = "nvidia/Cosmos3-Nano"
image_path = Path("/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/reasoner/assets/grounding_2d.png").resolve()

# 加载处理器和模型
processor = AutoProcessor.from_pretrained(model_id)
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

# 定义消息
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": str(image_path)},
            {"type": "text", "text": "Caption the image in detail."},
        ],
    }
]

# 处理输入
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, torch.bfloat16)

# 推理
generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(output[0])
