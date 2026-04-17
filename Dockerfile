FROM node:20-slim

RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

RUN pip3 install torch torchvision torchaudio \
    transformers datasets numpy --break-system-packages

WORKDIR /workspace
