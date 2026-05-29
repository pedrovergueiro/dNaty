# dNATY — Evolutionary AI Model Compression
# Multi-stage build for efficient images

FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY setup.py requirements.txt* ./

# Install PyTorch + dNATY
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e .

# Copy source code
COPY . .

# Production stage
FROM base as production

ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""

ENTRYPOINT ["python"]
CMD ["-c", "from dnaty import compress; print('dNATY ready!')"]

---

# GPU stage (optional)
FROM base as gpu

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

ENV CUDA_VISIBLE_DEVICES=0

ENTRYPOINT ["python"]
