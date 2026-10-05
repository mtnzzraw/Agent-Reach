# Lía Marín — creadora virtual (IA) de viajes
Todo etiquetado como IA (bio + caption + etiqueta "AI info" de Instagram al publicar).

## Pasos
1. ComfyUI en marcha (`python main.py`) con un checkpoint SDXL.
2. **Personaje consistente**: genera ~30 imágenes de la misma cara, entrena un LoRA (kohya/ai-toolkit),
   cópialo a `ComfyUI/models/loras` y usa la palabra `lia_character`.
3. `pip install pyyaml` (y `ffmpeg` instalado para los Reels).
   - `calendar.yaml`: un post por entrada (carrusel o Reel), con gancho, texto, pregunta y escenas.
     Los pies de foto se montan en español con el aviso de IA; las escenas (en inglés) solo van al prompt.
   - `python plan.py` → `plan.json` (valida aviso de IA, ≤2.200 caracteres, ≤30 hashtags, 2–10 fotos por carrusel).
   - `CKPT=... LORA=lia_character.safetensors python run.py 7` → carruseles en 4:5 y Reels en 9:16
     (imagen + vídeo de 6 s con zoom lento y el gancho encima, `reel.py`).
4. Revisa las imágenes a mano antes de publicar (manos, texto raro). Los posts con `verify: true`
   nombran sitios, precios u horarios: compruébalos y cambia su `status` a `"verified"` en `plan.json`.
5. Publicación: `IG_USER_ID, IG_TOKEN, PUBLIC_BASE_URL` (sube `output/` ahí) y cron cada hora → `python publish.py`.
   - Carruseles: se publican solos.
   - Reels: por defecto quedan en `output/reel_FECHA.mp4` + `.txt` con el pie, para subirlos desde el móvil
     con un audio en tendencia (la API no lo permite). `python publish.py --reels api` los publica sin música.
   - Etiqueta "AI info": ponla al subir; el aviso en el pie y la bio va siempre.
   Cron diario → `python track.py`.
