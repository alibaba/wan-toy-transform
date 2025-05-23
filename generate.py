import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    parser.add_argument("--lora_path", type=str, default="Alibaba-Research-Intelligence-Computing/wan-toy-transform")
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--image", type=str)
    parser.add_argument(
        "--save_file",
        type=str,
        default=f"samples/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4",
    )
    parser.add_argument(
        "--offload_type",
        type=str,
        default="none",
        choices=["leaf_level", "block_level", "none", "model"],
    )
    parser.add_argument("--num_blocks_per_group", type=int, default=1)
    return parser.parse_args()


args = parse_args()

model_id = args.model_id
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(
    model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
)
if args.lora_path is not None:
    pipe.load_lora_weights(args.lora_path)

onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
if args.offload_type == "none":
    pipe.to(onload_device)
elif args.offload_type == "model":
    pipe.enable_model_cpu_offload(device=onload_device)
elif args.offload_type == "block_level":
    pipe.transformer.enable_group_offload(
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type=args.offload_type,
        use_stream=True,
        num_blocks_per_group=args.num_blocks_per_group,
    )
    apply_group_offloading(
        pipe.text_encoder,
        onload_device=onload_device,
        offload_type=args.offload_type,
        num_blocks_per_group=args.num_blocks_per_group,
    )
    apply_group_offloading(
        pipe.image_encoder,
        onload_device=onload_device,
        offload_type=args.offload_type,
        num_blocks_per_group=args.num_blocks_per_group,
    )
    apply_group_offloading(pipe.vae, onload_device=onload_device, offload_type="leaf_level")
elif args.offload_type == "leaf_level":
    pipe.transformer.enable_group_offload(
        onload_device=onload_device, offload_device=offload_device, offload_type=args.offload_type, use_stream=True
    )
    apply_group_offloading(pipe.text_encoder, onload_device=onload_device, offload_type=args.offload_type)
    apply_group_offloading(pipe.image_encoder, onload_device=onload_device, offload_type=args.offload_type)
    apply_group_offloading(pipe.vae, onload_device=onload_device, offload_type="leaf_level")

image = load_image(args.image)
max_area = 480 * 832
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))
prompt = args.prompt
negative_prompt = ""

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=81,
    guidance_scale=5.0,
).frames[0]

Path(args.save_file).parent.mkdir(parents=True, exist_ok=True)
export_to_video(output, args.save_file, fps=16)
