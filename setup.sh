#!/usr/bin/env bash
# Create the virtual environment on a Linux machine with an NVIDIA GPU.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c 'import torch, transformers, bitsandbytes; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__)'
echo SETUP_DONE
