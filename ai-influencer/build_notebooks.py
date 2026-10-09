"""Genera los 3 cuadernos de Google Colab (SDXL realista + LoRA) a partir de persona.yaml.
Uso: python build_notebooks.py   ->  1_candidatas.ipynb, 2_entrenar_lora.ipynb, 3_generar.ipynb"""
import json, re, yaml
from plan import make_plan

P = yaml.safe_load(open("persona.yaml"))
# Los prompts de SDXL se truncan a 77 tokens: lo importante va primero y sin sintaxis de pesos "(x:1.4)".
STYLE = "RAW photo, detailed skin texture, detailed eyes, 85mm, film grain"
NEGATIVE = ("nsfw, nudity, deformed, bad anatomy, deformed iris, deformed pupils, uncanny, ugly, bad teeth, "
            "extra fingers, blurry, watermark, text, logo, brand logo, nike, adidas, swoosh, plastic skin, lowres, low quality")
SUFFIX = re.compile(r",\s*RAW photo, natural skin texture.*$")
faces = [SUFFIX.sub("", l.strip()).replace("lia_character, ", "") for l in open("training_prompts_faces.txt") if l.strip()][:60]
# El LoRA ya lleva la identidad: el prompt solo necesita el disparador + la escena (si no, SDXL trunca a 77 tokens).
plan = [{"pillar": x["pillar"], "prompt": f"photo of lia_character woman, {x['scene']}, natural light, candid",
         "caption": x["caption"]} for x in make_plan(30)]

def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}
def nb(cells): return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}

SETUP = code('''
from google.colab import drive
drive.mount('/content/drive')
import os
BASE = '/content/drive/MyDrive/lia'      # todo se guarda aquí, en tu Google Drive
for d in ('candidatas', 'dataset', 'lora', 'salida'):
    os.makedirs(f'{BASE}/{d}', exist_ok=True)
import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA -> Entorno de ejecución > Cambiar tipo > GPU T4')
''')
# diffusers 0.41.0 es la versión probada con el transformers 5.x de Colab (0.31 daba "cannot import FLAX_WEIGHTS_NAME").
# Colab trae una torchao antigua que rompe el import de diffusers 0.41 (cannot import name 'FqnToConfig'); no la usamos.
INSTALL = code("!pip uninstall -y -q torchao\n!pip install -q diffusers==0.41.0 transformers accelerate peft safetensors")
LOAD = '''
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
pipe = StableDiffusionXLPipeline.from_pretrained('SG161222/RealVisXL_V4.0', torch_dtype=torch.float16,
                                                 variant='fp16', use_safetensors=True).to('cuda')
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
'''
CONFIG = f"STYLE = {STYLE!r}\nNEGATIVE = {NEGATIVE!r}\n"

# ---------- 1: candidatas ----------
n1 = nb([
 md('''
# 1 · Caras candidatas de Lía (SDXL realista)
**Colab → Entorno de ejecución → Cambiar tipo de entorno → GPU T4.** Ejecuta las celdas en orden.

Genera 60 retratos con el modelo RealVisXL (mucho más realista que SD 1.5). Se guardan en tu Google Drive,
carpeta `lia/candidatas`. Luego **tú eliges** las 20–30 donde se vea la misma cara y las **mueves** a `lia/dataset`.
Se puede parar y repetir: salta las que ya existen. Tiempo aproximado: 30–45 min (estimado, no medido).
La cuenta final se publica siempre **etiquetada como IA**.
'''),
 SETUP, INSTALL,
 code(CONFIG + f"SEED = 777\nPROMPTS = {json.dumps(faces, ensure_ascii=False, indent=1)}\n"),
 code(LOAD + '''
import os
for i, core in enumerate(PROMPTS, 1):
    f = f'{BASE}/candidatas/cara_{i:02d}.png'
    if os.path.exists(f):
        continue
    g = torch.Generator('cuda').manual_seed(SEED)       # misma semilla = pelo y rasgos más parecidos
    img = pipe(prompt=f'{STYLE}, {core}', negative_prompt=NEGATIVE, width=896, height=1152,
               num_inference_steps=30, guidance_scale=5.0, generator=g).images[0]
    img.save(f)
    print(i, '/', len(PROMPTS), flush=True)
'''),
 md("## Ver todas de golpe\nCrea hojas numeradas en `lia/candidatas/hoja_XX.jpg`. Ábrelas en Drive o súbelas al chat de Claude para elegir."),
 code('''
import glob
from PIL import Image, ImageDraw
files = sorted(glob.glob(f'{BASE}/candidatas/cara_*.png'))
COLS, ROWS, T = 4, 3, 360
for n in range(0, len(files), COLS * ROWS):
    sheet = Image.new('RGB', (COLS * T, ROWS * T), 'white'); d = ImageDraw.Draw(sheet)
    for k, f in enumerate(files[n:n + COLS * ROWS]):
        im = Image.open(f).convert('RGB'); im.thumbnail((T, T))
        x, y = (k % COLS) * T, (k // COLS) * T
        sheet.paste(im, (x, y)); d.rectangle([x, y, x + 44, y + 24], fill='black')
        d.text((x + 6, y + 6), os.path.basename(f)[5:7], fill='white')
    sheet.save(f'{BASE}/candidatas/hoja_{n // (COLS * ROWS) + 1:02d}.jpg', quality=88)
print('Hojas creadas en', f'{BASE}/candidatas')
'''),
])

# ---------- 2: entrenar ----------
n2 = nb([
 md('''
# 2 · Entrenar el LoRA de Lía (SDXL)
La primera celda copia a `lia/dataset` las fotos elegidas (lista `ELEGIDAS`); edítala si quieres otra selección.
**GPU T4.** Tiempo aproximado: 1 h (estimado, no medido). Si Colab corta la sesión, vuelve a ejecutar:
el resultado final queda en `lia/lora`.
'''),
 SETUP, INSTALL,
 code('''
# Fotos elegidas de lia/candidatas (números de cara_XX.png). Edita la lista si quieres cambiar la selección.
# Se copian solas a lia/dataset: no hace falta moverlas a mano.
ELEGIDAS = [1, 2, 4, 8, 9, 11, 14, 16, 17, 20, 21, 24, 25, 27, 30, 31, 32, 38, 40, 42, 45, 46, 47, 48, 50, 53, 54, 58, 59]
import shutil
for n in ELEGIDAS:
    shutil.copy(f'{BASE}/candidatas/cara_{n:02d}.png', f'{BASE}/dataset/')
print(len(ELEGIDAS), 'fotos copiadas a lia/dataset')
'''),
 code('''
!pip install -q bitsandbytes
!wget -q -O train_dreambooth_lora_sdxl.py https://raw.githubusercontent.com/huggingface/diffusers/v0.41.0/examples/dreambooth/train_dreambooth_lora_sdxl.py
import glob
n = len(glob.glob(f'{BASE}/dataset/*.png') + glob.glob(f'{BASE}/dataset/*.jpg'))
print(n, 'imágenes en el dataset')
assert n >= 10, 'Mueve primero tus mejores imágenes a lia/dataset (mínimo 10, ideal 20-30)'
'''),
 code('''
!accelerate launch --mixed_precision=fp16 train_dreambooth_lora_sdxl.py \\
  --pretrained_model_name_or_path=SG161222/RealVisXL_V4.0 \\
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \\
  --instance_data_dir={BASE}/dataset \\
  --instance_prompt="photo of lia_character woman" \\
  --output_dir={BASE}/lora \\
  --resolution=768 --center_crop \\
  --train_batch_size=1 --gradient_accumulation_steps=2 \\
  --gradient_checkpointing --use_8bit_adam --mixed_precision=fp16 \\
  --learning_rate=1e-4 --lr_scheduler=constant --lr_warmup_steps=0 \\
  --rank=16 --max_train_steps=1000 --checkpointing_steps=500 --seed=42
'''),
 code('''
import os
f = f'{BASE}/lora/pytorch_lora_weights.safetensors'
print('OK, LoRA guardado:' if os.path.exists(f) else 'NO se creó el LoRA, revisa los errores de arriba:', f)
'''),
])

# ---------- 3: generar ----------
n3 = nb([
 md('''
# 3 · Generar las fotos de Lía (SDXL + su LoRA)
Usa el LoRA de `lia/lora`. Genera las publicaciones del plan de 30 días y las guarda en `lia/salida`
junto a `captions.csv` (texto con el aviso de IA). Cambia `DESDE` y `HASTA` para hacerlo por tandas.
**GPU T4.** Aproximadamente 1 min por imagen (estimado). **Revisa cada imagen a mano** (manos, cara, texto raro) antes de publicar,
y publica siempre con la etiqueta de IA.
'''),
 SETUP, INSTALL,
 code(CONFIG + f"PLAN = {json.dumps(plan, ensure_ascii=False, indent=1)}\nDESDE, HASTA = 1, 10     # días del plan a generar\n"),
 code(LOAD + '''
pipe.load_lora_weights(f'{BASE}/lora', weight_name='pytorch_lora_weights.safetensors')
import csv, os, random
cap = f'{BASE}/salida/captions.csv'
if not os.path.exists(cap):
    open(cap, 'w', encoding='utf-8').write('dia,pilar,archivo,caption\\n')
for n in range(DESDE, HASTA + 1):
    it = PLAN[n - 1]
    f = f'{BASE}/salida/{n:02d}_{it["pillar"]}.png'
    if os.path.exists(f):
        continue
    seed = random.randint(0, 2**31)
    g = torch.Generator('cuda').manual_seed(seed)
    img = pipe(prompt=f'{STYLE}, {it["prompt"]}', negative_prompt=NEGATIVE, width=832, height=1216,
               num_inference_steps=30, guidance_scale=5.0, generator=g,
               cross_attention_kwargs={'scale': 0.9}).images[0]
    img.save(f)
    with open(cap, 'a', newline='', encoding='utf-8') as fh:
        csv.writer(fh).writerow([n, it['pillar'], os.path.basename(f), it['caption']])
    print(n, it['pillar'], 'seed', seed, flush=True)
'''),
])

for name, n in (("1_candidatas", n1), ("2_entrenar_lora", n2), ("3_generar", n3)):
    json.dump(n, open(f"{name}.ipynb", "w"), ensure_ascii=False, indent=1)
    print("escrito", f"{name}.ipynb")
