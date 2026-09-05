FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download the SDXL base weights + IP-Adapter weights at build time so
# workers boot with everything already on disk (no multi-GB download on
# first request / cold start).
RUN python3 -c "\
import torch; \
from diffusers import AutoPipelineForText2Image; \
from transformers import CLIPVisionModelWithProjection; \
image_encoder = CLIPVisionModelWithProjection.from_pretrained('h94/IP-Adapter', subfolder='sdxl_models/image_encoder', torch_dtype=torch.float16); \
pipe = AutoPipelineForText2Image.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', image_encoder=image_encoder, torch_dtype=torch.float16, variant='fp16'); \
pipe.load_ip_adapter('h94/IP-Adapter', subfolder='sdxl_models', weight_name='ip-adapter-plus_sdxl_vit-h.safetensors')"

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
