FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download the FLUX.2-klein-4B weights at build time so workers boot
# with everything already on disk (no multi-GB download on first request).
RUN python3 -c "\
import torch; \
from diffusers import DiffusionPipeline; \
DiffusionPipeline.from_pretrained('black-forest-labs/FLUX.2-klein-4B', torch_dtype=torch.bfloat16)"

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
