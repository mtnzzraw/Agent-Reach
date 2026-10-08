#!/usr/bin/env bash
# Instala ComfyUI y deja el pipeline listo. Mac Apple Silicon (MPS) o PC con GPU NVIDIA. Uso: bash setup.sh
set -e
git clone https://github.com/comfyanonymous/ComfyUI ../ComfyUI 2>/dev/null || true
python3 -m venv ../ComfyUI/venv && source ../ComfyUI/venv/bin/activate
if [ "$(uname -s)" = "Darwin" ]; then
  pip install -q torch torchvision            # build de PyTorch con soporte MPS (Apple Silicon)
  RUN="python main.py --force-fp16 --use-split-cross-attention --lowvram"
  CKPT_HINT="SD 1.5 (v1-5-pruned-emaonly.safetensors), con 8 GB de RAM no uses SDXL"
else
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
  RUN="python main.py"
  CKPT_HINT="SD 1.5 o SDXL (sd_xl_base_1.0.safetensors)"
fi
pip install -q -r ../ComfyUI/requirements.txt pyyaml
echo "Descarga un checkpoint: $CKPT_HINT -> ../ComfyUI/models/checkpoints"
echo "Arranca ComfyUI:  cd ../ComfyUI && source venv/bin/activate && $RUN"
echo "Luego:  python plan.py 30 && python run.py 7        (PROFILE=sdxl para GPUs grandes)"
