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
