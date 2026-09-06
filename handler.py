import base64
import io

import runpod
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from PIL import Image

# FLUX.2 [klein] 4B — Apache 2.0, distilled for speed (as low as 4 steps),
# with native multi-reference image editing support. This replaces the
# SDXL+IP-Adapter attempt: instead of bolting a CLIP-based adapter onto a
# model that was never trained for it, this model was trained end-to-end to
# do reference-image-conditioned editing, which should be far more stable.
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

print("Loading FLUX.2-klein-4B pipeline...")
pipeline = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
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
    steps = inp.get("num_inference_steps", 4)
    guidance_scale = inp.get("guidance_scale", 1.0)
    width = inp.get("width", 1024)
    height = inp.get("height", 1024)
    seed = inp.get("seed")

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    # Accept either a single reference (reference_image_base64) or several
    # (reference_images_base64, a list) for multi-reference editing.
    ref_images = None
    if inp.get("reference_images_base64"):
        ref_images = [decode_image(b) for b in inp["reference_images_base64"]]
    elif inp.get("reference_image_base64"):
        ref_images = decode_image(inp["reference_image_base64"])

    gen_kwargs = dict(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    if ref_images is not None:
        # Editing mode — model infers output size from the reference by
        # default; only force width/height if explicitly provided.
        gen_kwargs["image"] = ref_images
        if inp.get("width") and inp.get("height"):
            gen_kwargs["height"] = height
            gen_kwargs["width"] = width
    else:
        gen_kwargs["width"] = width
        gen_kwargs["height"] = height

    result = pipeline(**gen_kwargs)
    image = result.images[0]

    return {"image": encode_image(image)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
