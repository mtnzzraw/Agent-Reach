#!/usr/bin/env bash
# Instala ComfyUI en tu PC con GPU NVIDIA y deja el pipeline listo. Uso: bash setup.sh
set -e
git clone https://github.com/comfyanonymous/ComfyUI ../ComfyUI 2>/dev/null || true
python3 -m venv ../ComfyUI/venv && source ../ComfyUI/venv/bin/activate
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -q -r ../ComfyUI/requirements.txt pyyaml
echo "Descarga un checkpoint SDXL (p.ej. sd_xl_base_1.0.safetensors) a ../ComfyUI/models/checkpoints"
echo "Arranca ComfyUI:  cd ../ComfyUI && python main.py"
echo "Luego:  python plan.py 30 && python run.py 7"
