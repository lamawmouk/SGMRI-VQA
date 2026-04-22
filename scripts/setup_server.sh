#!/bin/bash
# Setup script for PACE ICE cluster
# Run this on the login node (or GPU node) ONCE

set -e

REMOTE_DIR="/storage/ice1/8/4/zzhao465/VLM_grounding"
cd "$REMOTE_DIR"

echo "=== Creating conda environment ==="
conda create -n vlm_eval python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate vlm_eval

echo "=== Installing PyTorch with CUDA 12.1 ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing core dependencies ==="
pip install Pillow tqdm python-dotenv numpy scipy pyyaml

echo "=== Installing HuggingFace stack ==="
pip install "transformers>=4.52.0" "accelerate>=0.25.0" einops timm

echo "=== Installing vLLM ==="
pip install "vllm>=0.6.0"

echo "=== Installing model-specific utils ==="
pip install qwen-vl-utils
pip install qwen-omni-utils
pip install keye-vl-utils
pip install molmo_utils

echo "=== Installing flash-attn (optional) ==="
pip install flash-attn --no-build-isolation || echo "flash-attn build failed, models will use sdpa fallback"

echo "=== Installing LLaVA-NeXT ==="
cd "$REMOTE_DIR/LLaVA-NeXT" && pip install -e . && cd "$REMOTE_DIR"

echo "=== Installing perception_models (PLM) ==="
cd "$REMOTE_DIR/perception_models" && pip install -e . && cd "$REMOTE_DIR"

echo "=== Verifying installation ==="
cd "$REMOTE_DIR/Grounding"

python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python3 -c "from qwen_vl_utils import process_vision_info; print('qwen_vl_utils OK')"
python3 -c "from qwen_omni_utils import process_mm_info; print('qwen_omni_utils OK')"
python3 -c "from keye_vl_utils import process_vision_info; print('keye_vl_utils OK')"
python3 -c "import sys; sys.path.insert(0,'../InternVL/internvl_chat'); from internvl.model.internvl_chat import InternVLChatModel; print('InternVL OK')"
python3 -c "from llava.model.builder import load_pretrained_model; print('LLaVA OK')"
python3 -c "import sys; sys.path.insert(0,'../perception_models'); from apps.plm.generate import load_consolidated_model_and_tokenizer; print('PLM OK')"

python3 -c "
from evaluation.data.loader import load_all_samples
print(f'Image samples: {len(load_all_samples(eval_mode=\"image\"))}')
print(f'Video samples: {len(load_all_samples(eval_mode=\"video\"))}')
"

echo "=== Setup complete! ==="
