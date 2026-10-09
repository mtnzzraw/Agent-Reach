# Empieza aquí — Lía Marín (influencer IA, gratis)

Todo lo que se podía automatizar ya está hecho. Estos son los pasos que **solo puedes hacer tú**, en orden.
Cada paso dice cuánto tarda y qué comprobar antes de seguir.

## 1. Probar que tu Mac genera imágenes (30–60 min la primera vez)
1. Abre la app **Terminal** y pega:
   ```
   git clone https://github.com/mtnzzraw/Agent-Reach
   cd Agent-Reach && git checkout claude/beautiful-goodall-w8susi
   cd ai-influencer && bash setup.sh
   ```
2. Descarga el modelo **SD 1.5** (`v1-5-pruned-emaonly.safetensors`, unos 4 GB, desde Hugging Face:
   `stable-diffusion-v1-5/stable-diffusion-v1-5`) y ponlo en `ComfyUI/models/checkpoints/`.
3. Arranca ComfyUI con el comando que te imprime `setup.sh` y déjalo abierto.
4. En otra ventana de Terminal: `cd Agent-Reach/ai-influencer && python plan.py 30 && python run.py 1`
5. ✅ Comprobación: aparece una imagen en `ai-influencer/output/`. Apunta cuánto tardó.
   (En la nube, con CPU, tardó ~9–12 min. Tu Mac debería ir más rápido; no está medido.)

## 2. Crear la cara de Lía (1–2 horas)
1. Genera las 30 imágenes: `python gen_dataset.py` (unos 45 min en tu Mac; se puede parar y reanudar). Salen en `dataset/
2. Quédate con **15–30 imágenes donde se vea la misma cara**. Borra las que no se parezcan.
3. ✅ Comprobación: si pones todas juntas, parece la misma persona.

## 3. Entrenar el LoRA gratis (1–2 horas)
1. Entra en **kaggle.com** (cuenta gratis, verifica el teléfono para activar la GPU) o en **colab.research.google.com**.
2. Sube `train_lora.ipynb`. Activa GPU (T4/P100) e Internet.
3. Sube tu carpeta de imágenes como `lia_dataset` y ejecuta las celdas en orden.
4. Descarga `lia_character.safetensors` y cópialo a `ComfyUI/models/loras/`.
5. ✅ Comprobación: `LORA=lia_character.safetensors python run.py 1` genera una imagen con esa cara.

## 4. Producir contenido
- `LORA=lia_character.safetensors python run.py 7` genera la semana (déjalo de noche).
- Revisa cada imagen a mano (manos, cara, texto raro). Descarta las malas.
- Mejora de calidad opcional: `UPSCALE=RealESRGAN_x4plus.pth` (más lento).

## 5. Instagram
- Crea la cuenta como **Creator/Business**, con "IA" en la bio, y marca cada publicación con la etiqueta "AI info".
- Al principio, publica **a mano** desde el móvil con los captions de `plan.json`. La API (`publish.py`) solo merece
  la pena cuando todo lo demás funcione: pide cuenta vinculada a una página de Facebook y una app de Meta.
- Tono: atrevido pero sin desnudos ni contenido explícito. Si lo cruzas, Instagram limita o cierra la cuenta.

## Si algo falla
Copia el mensaje de error completo y pídele ayuda a Claude con el archivo implicado.
