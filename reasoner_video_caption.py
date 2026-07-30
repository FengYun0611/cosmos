"""
Cosmos3-Nano Reasoner — Video Caption（视频描述）
给一段视频，输出详细的文字描述
"""
from pathlib import Path
import torch
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

model_id = "nvidia/Cosmos3-Nano"
assets_dir = Path("/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/reasoner/assets")
video_path = (assets_dir / "video_caption.mp4").resolve()

processor = AutoProcessor.from_pretrained(model_id)
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

# Video caption — 用 fps=2 控制采样帧率
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": str(video_path)},
            {"type": "text", "text": "Describe the video in detail."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    fps=2,
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
