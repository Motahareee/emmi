# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This repository defines a Docker environment (`Dockerfile`) based on `node:20-slim` with:

- **Node.js 20** with `@anthropic-ai/claude-code` installed globally
- **Python 3** with PyTorch, torchvision, torchaudio, Hugging Face `transformers`, `datasets`, and `numpy`
- **Git** and **curl**

The working directory inside the container is `/workspace`.

## Building the Docker Image

```bash
docker build -t <image-name> .
```

## Running the Container

```bash
docker run -it --rm -v $(pwd):/workspace <image-name>
```
