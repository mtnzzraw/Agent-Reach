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

## Paso 1 — Caras candidatas (`1_candidatas.ipynb`)
- Genera 60 retratos y los guarda en `Mi unidad/lia/candidatas/` (más unas `hoja_XX.jpg` para verlas de golpe).
- ✅ Abre esa carpeta en Drive y mira las fotos.

## Paso 2 — Elegir la cara
- Elige **20–30 fotos donde se vea la misma persona** y **muévelas** a `Mi unidad/lia/dataset/`.
- Si quieres ayuda para elegir, sube las `hoja_XX.jpg` al chat de Claude.
- Si ninguna se parece, repite el paso 1 (o dime y cambio la semilla).

## Paso 3 — Entrenar el LoRA (`2_entrenar_lora.ipynb`)
- Aprende la cara de tu carpeta `lia/dataset`. Resultado en `lia/lora/pytorch_lora_weights.safetensors`.
- ✅ La última celda dice "OK, LoRA guardado".

## Paso 4 — Generar las fotos (`3_generar.ipynb`)
- Genera las publicaciones del plan de 30 días con la cara de Lía. Cambia `DESDE, HASTA` para hacerlo por tandas.
- Salen en `lia/salida/` con un `captions.csv` (texto con el aviso de IA).
- **Revisa cada imagen a mano** (manos, cara, texto raro) y descarta las malas.

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
