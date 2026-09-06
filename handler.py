import base64
import io
import os

import runpod
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection
from PIL import Image

# SD 1.5 (not SDXL) + a face-specific IP-Adapter variant, which only exists
# for 1.5 — trained to lock facial identity specifically, rather than the
# general "plus" adapter's whole-image style/content, which is what SDXL was
# fighting us on (face vs. pose vs. style all competing for the same scale knob).
BASE_MODEL = os.environ.get("BASE_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "models"
IP_ADAPTER_WEIGHT = "ip-adapter-plus-face_sd15.bin"

print("Loading IP-Adapter image encoder (ViT-H)...")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    IP_ADAPTER_REPO, subfolder="models/image_encoder", torch_dtype=torch.float16
)

print("Loading SD 1.5 base pipeline...")
pipeline = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL, image_encoder=image_encoder, torch_dtype=torch.float16
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
    ref_image_b64 = inp.get("reference_image_base64")
    ref_image = decode_image(ref_image_b64) if ref_image_b64 else None
    ip_adapter_scale = inp.get("ip_adapter_scale", 0.6)
    steps = inp.get("num_inference_steps", 28)
    guidance_scale = inp.get("guidance_scale", 6.0)
    width = inp.get("width", 1024)
    height = inp.get("height", 1024)
    seed = inp.get("seed")

    pipeline.set_ip_adapter_scale(ip_adapter_scale if ref_image else 0.0)

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    # Once an IP-Adapter is loaded onto the pipeline, diffusers' UNet forward
    # pass structurally expects *some* ip_adapter_image on every call — it
    # can't just be omitted. With no reference supplied, pass a blank neutral
    # image and rely on scale=0.0 (set above) to make it a no-op.
    ip_adapter_input = ref_image if ref_image is not None else Image.new("RGB", (224, 224), (255, 255, 255))

    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_image=ip_adapter_input,
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
