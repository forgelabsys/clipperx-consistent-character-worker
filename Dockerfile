FROM nvidia/cuda:12.8.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# RunPod's GPU pool includes Blackwell-generation cards (RTX 5090, RTX PRO
# 6000 Blackwell) even when we request Ada-class types — its scheduler
# apparently treats them as an equivalent tier. torch==2.6.0/cu124 has no
# compiled kernels for Blackwell's sm_120 ("CUDA error: no kernel image is
# available"), so we need a current torch build with cu128+ Blackwell
# support instead of trying to dodge which physical GPU gets assigned.
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

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
