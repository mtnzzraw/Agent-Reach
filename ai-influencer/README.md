# Lía Marín — creadora virtual (IA) de viajes
Todo etiquetado como IA (bio + caption + etiqueta "AI info" de Instagram al publicar).

## Pasos
1. `bash setup.sh` y arranca ComfyUI como indica. Mac 8 GB: perfil `sd15` por defecto (512x768, SD 1.5, ~1-2 min/imagen).
   GPU grande: `PROFILE=sdxl`.
2. **Personaje consistente**: genera ~30 imágenes de la misma cara, entrena un LoRA (kohya/ai-toolkit),
   cópialo a `ComfyUI/models/loras` y usa la palabra `lia_character`.
3. `pip install pyyaml` · `python plan.py 30` (plan de 30 días) · `CKPT=... LORA=lia_character.safetensors python run.py 7`
4. Revisa las imágenes a mano antes de publicar (manos, texto raro).
5. Publicación: `IG_USER_ID, IG_TOKEN, PUBLIC_BASE_URL` y cron cada hora → `python publish.py`.
   Cron diario → `python track.py`.

## Mac 8 GB
- `PROFILE=sd15` (por defecto) · `PROFILE=sdxl-lowvram` experimental (arranca ComfyUI con `--lowvram`; si da error de memoria vuelve a sd15).
- Upscale opcional: pon un modelo en `ComfyUI/models/upscale_models` y `UPSCALE=RealESRGAN_x4plus.pth python run.py 1`.
- `training_prompts.txt`: 30 prompts con distintos ángulos, luces y looks para sacar las imágenes del LoRA.
  Elige una cara (misma `seed`/imagen de referencia) y usa solo las que mantengan los mismos rasgos.

## Fotos realistas (recomendado): Google Colab + SDXL
Sigue **EMPIEZA_AQUI.md**. Tres cuadernos (`1_candidatas`, `2_entrenar_lora`, `3_generar`) con RealVisXL + LoRA de la cara,
todo guardado en tu Google Drive. Se regeneran con `python build_notebooks.py` si cambias `persona.yaml`.
