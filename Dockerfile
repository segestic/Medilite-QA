FROM nvidia/cuda:13.1.0-devel-ubuntu22.04

# Core Environment Variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH="/usr/local/bin:$PATH"

# Set to your actual GPU compute capability (8.6 = A6000/3090) to speed up flash-attn build
ENV TORCH_CUDA_ARCH_LIST="8.6"

# 1. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    ninja-build \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

WORKDIR /workspace

# 2. Upgrade build tools & enforce setuptools pin FIRST
RUN pip3 install --no-cache-dir --upgrade pip wheel packaging ninja "setuptools<70.0.0"

# 3. Install core Python dependencies (Torch + HF Stack)
# Leveraging the --extra-index-url already defined in requirements.txt
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 4. Compile Flash Attention
RUN pip3 install --no-cache-dir flash-attn==2.8.3 --no-build-isolation

# 5. Build llama.cpp for CPU (edge/deployment simulation)
RUN git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp \
    && cd /workspace/llama.cpp \
    && cmake -B build -DGGML_CUDA=OFF \
    && cmake --build build --config Release -j "$(nproc)"

ENV LLAMA_CPP_DIR=/workspace/llama.cpp

# 6. Pipeline Code + Config
COPY src/ /workspace/src/

WORKDIR /workspace

# Fix: Run python as a module from the root directory
ENTRYPOINT ["python", "-m", "src.train"]
