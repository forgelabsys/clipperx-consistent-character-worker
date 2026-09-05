import base64
import io
import os

import runpod
import torch
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPVisionModelWithProjection
from PIL import Image

BASE_MODEL = os.environ.get("BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"

# The "_vit-h" checkpoint name is literal: it was trained against OpenCLIP
# ViT-H-14 (1280-dim) embeddings — that encoder lives under models/image_encoder
# in the h94/IP-Adapter repo, NOT sdxl_models/image_encoder (which is the
# bigger ViT-bigG, 1664-dim, used by the *other* non-"vit-h" SDXL checkpoints).
# Mixing these up is what caused the "514x1664 vs 1280x1280" shape error.
print("Loading IP-Adapter image encoder (ViT-H)...")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    IP_ADAPTER_REPO, subfolder="models/image_encoder", torch_dtype=torch.float16
)

print("Loading SDXL base pipeline...")
pipeline = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL, image_encoder=image_encoder, torch_dtype=torch.float16, variant="fp16"
).to("cuda")

print("Loading IP-Adapter weights...")
pipeline.load_ip_adapter(
    IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHT
)
pipeline.set_ip_adapter_scale(0.6)
print("Ready.")


def decode_image(b64_data):
    raw = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def encode_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(event):
    inp = event["input"]

    prompt = inp["prompt"]
    negative_prompt = inp.get(
        "negative_prompt",
        "photo, photorealistic, 3d render, realistic shading, blurry, low quality, extra limbs, deformed",
    )
    ref_image = decode_image(inp["reference_image_base64"])
    ip_adapter_scale = inp.get("ip_adapter_scale", 0.6)
    steps = inp.get("num_inference_steps", 28)
    guidance_scale = inp.get("guidance_scale", 6.0)
    width = inp.get("width", 1024)
    height = inp.get("height", 1024)
    seed = inp.get("seed")

    pipeline.set_ip_adapter_scale(ip_adapter_scale)

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_image=ref_image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )
    image = result.images[0]

    return {"image": encode_image(image)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
