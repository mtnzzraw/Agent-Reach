# Empieza aquí — Lía Marín (influencer IA, gratis, fotos realistas)

Todo se hace en **Google Colab** (GPU gratis) y se guarda en tu **Google Drive**. El Mac ya no hace falta para generar:
con SDXL realista salen caras mucho mejores que con SD 1.5. Necesitas solo una cuenta de Google.

> Tiempos y límites de Colab gratis **sin medir**: la GPU no está garantizada (a veces dice "no hay GPU disponible", repite más tarde)
> y las sesiones se cortan tras unas horas. Todos los cuadernos retoman donde se quedaron.

## Cómo abrir un cuaderno
1. Enlace directo (repo público):
   `https://colab.research.google.com/github/mtnzzraw/Agent-Reach/blob/claude/beautiful-goodall-w8susi/ai-influencer/1_candidatas.ipynb`
   (cambia `1_candidatas` por `2_entrenar_lora` o `3_generar` en los otros pasos).
   Si falla: Colab → Archivo → Abrir cuaderno → pestaña **GitHub** → `mtnzzraw/Agent-Reach` → elige la rama → el archivo.
2. **Entorno de ejecución → Cambiar tipo de entorno → GPU T4 → Guardar.**
3. **Entorno de ejecución → Ejecutar todo.** La primera vez pide permiso para tu Drive: acepta.

## Paso 1 — Candidatas: cara y cuerpo (`1_candidatas.ipynb`)
- Genera 100 imágenes estilo foto casera de móvil de Lía (mujer normal, pelo cobrizo, sin pecas, figura definida pero natural):
  25 selfies, 35 de espejo y 40 de cuerpo. Se guardan en `Mi unidad/lia2/candidatas/` (más `hoja_XX.jpg`). Unos 1–1,5 h (estimado).
- Todo va en la carpeta nueva `lia2`: la carpeta anterior `lia` (otra cara, con pecas) no se toca ni hace falta borrarla.

## Paso 2 — Elegir (cara y cuerpo coherentes)
- Mira las `hoja_XX.jpg` (o súbelas al chat de Claude) y elige **20–30 imágenes donde cara y cuerpo se parezcan**,
  mezclando retratos, cintura para arriba y cuerpo entero. Apunta los números de `img_XX.png`.
- Ponlos en la lista `ELEGIDAS` de `2_entrenar_lora.ipynb`: se copian solos a `lia2/dataset`.

## Paso 3 — Entrenar el LoRA (`2_entrenar_lora.ipynb`)
- Aprende cara **y cuerpo** de `lia2/dataset`. Resultado en `lia2/lora/pytorch_lora_weights.safetensors`.
- ✅ La última celda dice "OK, LoRA guardado".

## Paso 4 — Generar las fotos (`3_generar.ipynb`)
- Genera las publicaciones del plan de 30 días con estilo de foto de móvil (selfies, fotos casuales). Cambia `DESDE, HASTA` para hacerlo por tandas.
- Salen en `lia2/salida_iphone/` con un `captions.csv` (texto con el aviso de IA). Los selfies de espejo piden el iPhone con su logo.
- Última celda (opcional): crea `salida_iphone/movil/*.jpg` con grano y compresión de móvil. Solo estética, sin metadatos falsos.
- **Revisa cada imagen a mano** (manos, cara, proporciones, texto raro) y descarta las malas.

## Paso 5 — Instagram
- Cuenta **Creator/Business**, con "IA" en la bio y la etiqueta "AI info" en cada publicación.
- Al principio publica **a mano** desde el móvil con los captions de `captions.csv`.
  La API (`publish.py`) solo merece la pena cuando todo lo demás funcione (pide página de Facebook y app de Meta).
- Tono: atrevido pero sin desnudos ni contenido explícito. Si lo cruzas, Instagram limita o cierra la cuenta.

## Opcional: generar en tu Mac (peor calidad)
`setup.sh`, `run.py`, `gen_dataset.py`, `contact_sheet.py` siguen funcionando con SD 1.5 en tu Mac de 8 GB,
pero las caras salen peores. Útil solo para pruebas rápidas.

## Si algo falla
- Error `cannot import name 'FLAX_WEIGHTS_NAME'` o similar: es de versiones. Vuelve a abrir el cuaderno desde el enlace (versión nueva),
  y en Colab: **Entorno de ejecución → Reiniciar sesión y ejecutar todo**.
Copia el mensaje de error completo y pídele ayuda a Claude.
