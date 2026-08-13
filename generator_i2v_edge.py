"""
Cosmos3-Edge Generator — Image-to-Video（图生视频）
==================================================
模仿 generator_i2v.py 的 car_driving 示例，改用 Cosmos3-Edge 权重生成一遍，
用于和已有的 Nano 输出做对比。

与 generator_i2v.py 完全一致的参数：
    prompt、negative_prompt、num_frames=25、fps=10、num_inference_steps=20、
    guidance_scale=6.0、seed=1234、enable_sound=False、输入图 car_driving.jpg

仅因 Edge 硬限制做的差异（Edge 官方只支持 256p/480p，且 320x512 不在其训练分布内）：
    分辨率:   Nano 用 320x512；这里默认 480p(832x480)。
              想更贴近 Nano 的低分辨率 → 把 RESOLUTION 改成 "256"（192x320）。
    flow_shift: Edge 480p 用 3.0（Nano 用 10.0，各模型不同，不可照搬）。

注意：Edge 官方推荐 480p/121帧/50步/12-30fps。这里 25帧/fps=10 是为了和
Nano 对齐，画质会比 Edge 的最佳配置差；若对比后想看重置为推荐参数，
把 NUM_FRAMES 调 121、FPS 调 24 即可。

前置：diffusers git HEAD（PyPI 0.39.0 不支持 Edge），在 env_edge venv 跑。
"""
import os
os.environ["HF_HOME"] = "/mnt/disk8/fengyun/huggingface"
# 从环境变量读取，避免硬编码泄露；需要登录时 export HF_TOKEN=...
HF_TOKEN = os.environ.get("HF_TOKEN", "")

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

model_id = "nvidia/Cosmos3-Edge"
assets_dir = "/home/shenyanyuan/fengyun/cosmos/cookbooks/cosmos3/generator/audiovisual/assets"

# ── 可配置 ──
RESOLUTION = "480"            # "480" → 832x480（Edge 推荐）；"256" → 192x320
RES_MAP = {"480": (832, 480), "256": (320, 192)}
WIDTH, HEIGHT = RES_MAP[RESOLUTION]
FLOW_SHIFT = 3.0              # Edge 480p 的推荐 shift
NUM_FRAMES = 121               # 与 generator_i2v 一致（Edge 推荐 121）
FPS = 24                      # 与 generator_i2v 一致（Edge 推荐 24）
NUM_STEPS = 50                # 与 generator_i2v 一致（Edge 推荐 50）
GUIDANCE = 6.0
SEED = 1234
OFFLINE = True                # 权重已缓存时 True（不联网）；需下载时改 False 并 export HF_ENDPOINT=https://hf-mirror.com
OUTPUT_PATH = "/home/shenyanyuan/fengyun/cosmos/edge_i2v_car.mp4"

# ── 显存监控（沿用 generator_i2v 风格）──
def log_vram(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_alloc = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(
        f"[{tag:20s}] allocated={allocated:.1f}G ({allocated/total*100:.0f}%) | "
        f"reserved={reserved:.1f}G | max_alloc={max_alloc:.1f}G"
    )
    return allocated, reserved, max_alloc

# ── 加载模型 ──
log_vram("before_load")
pipe = Cosmos3OmniPipeline.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    safety_checker=None,
    enable_safety_checker=False,
    token=HF_TOKEN or None,
    local_files_only=OFFLINE,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=FLOW_SHIFT)
pipe.to("cuda")
log_vram("after_move_to_cuda")

# 重置峰值计数器，只看推理阶段
torch.cuda.reset_peak_memory_stats()

# ── car_driving 示例（与 generator_i2v.py 相同输入）──
image_path = f"{assets_dir}/images/image2video/car_driving.jpg"
image = load_image(image_path)
prompt = "The car drives forward on a road with trees on both sides."

print(f"\n[Edge] 图生视频: {image_path}")
print(f"参数: {WIDTH}x{HEIGHT}, {NUM_FRAMES}帧, fps={FPS}, {NUM_STEPS}步, guidance={GUIDANCE}, seed={SEED}")

before_alloc, _, _ = log_vram("before_inference")

result = pipe(
    prompt=prompt,
    negative_prompt="blurry, distorted, low quality",
    image=image,
    num_frames=NUM_FRAMES,
    height=HEIGHT,
    width=WIDTH,
    fps=FPS,
    num_inference_steps=NUM_STEPS,
    guidance_scale=GUIDANCE,
    enable_sound=False,
    add_resolution_template=False,
    add_duration_template=False,
    generator=torch.Generator(device="cuda").manual_seed(SEED),
)

torch.cuda.synchronize()
after_alloc, _, infer_peak = log_vram("after_inference")
print(f"生成增量显存: {infer_peak - before_alloc:.1f}G | 推理耗时见上")

export_to_video(result.video, OUTPUT_PATH, fps=FPS, macro_block_size=1)
print(f"已保存: {OUTPUT_PATH}")

print("\n全部完成！")
