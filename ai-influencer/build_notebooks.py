"""Genera los 3 cuadernos de Google Colab (SDXL realista + LoRA) a partir de persona.yaml.
Uso: python build_notebooks.py   ->  1_candidatas.ipynb, 2_entrenar_lora.ipynb, 3_generar.ipynb"""
import itertools, json, random, yaml
from plan import make_plan

P = yaml.safe_load(open("persona.yaml"))
DRIVE_DIR = "lia2"   # carpeta nueva en Drive (la anterior, "lia", tenía otra cara con pecas)

# Los prompts de SDXL se truncan a 77 tokens: lo importante va primero y sin sintaxis de pesos "(x:1.4)".
CHAR = ("26 year old ordinary woman, long wavy copper auburn hair, hazel eyes, clear natural skin, curvy hourglass figure")
# Estilo "foto casera de móvil" (referencias: selfies de espejo en baños y dormitorios, flash, fondos cotidianos).
STYLE_DS = "amateur phone photo, candid, natural skin pores, imperfect framing"
NEGATIVE_DS = ("nsfw, nudity, freckles, moles, deformed, bad anatomy, deformed iris, uncanny, bad teeth, extra fingers, blurry, "
               "see-through clothing, nipples, topless, watermark, text, plastic skin, airbrushed, professional photography, studio lighting, exaggerated proportions, "
               "huge breasts, disproportionate body, cartoon, 3d render")

# Estilo para las publicaciones. Solo estética: no se añaden metadatos falsos de cámara; las fotos siguen siendo IA
# y se publican etiquetadas como tal. Se permiten logos reales (p. ej. la manzana del iPhone).
STYLE3 = "amateur iPhone photo, candid, unedited, slight grain"
NEGATIVE3 = ("nsfw, nudity, see-through clothing, nipples, topless, freckles, deformed, bad anatomy, deformed iris, uncanny, bad teeth, "
             "extra fingers, blurry, watermark, plastic skin, exaggerated proportions, huge breasts, disproportionate body, studio lighting, professional photography, "
             "bokeh, retouched, airbrushed, HDR, cinematic, perfect")

# --- 100 candidatas 1024x1024, TODAS con la cara visible (el LoRA aprende cara y cuerpo): 25 selfies, 35 de espejo, 40 de cuerpo ---
selfie = ["front camera selfie, close-up face, looking at camera", "front camera selfie, slight smile, head and shoulders",
          "selfie from slightly above, looking up at camera", "front camera selfie, laughing, messy hair",
          "selfie, head tilted, relaxed expression, scoop-neck top"]
selfie_bg = ["bedroom", "small tiled bathroom", "car interior", "cafe", "street", "plain wall"]
mirror = ["mirror selfie from the waist up, holding phone at chest height, face visible, fitted black scoop-neck top, modest cleavage",
          "mirror selfie from the waist up, face visible, white fitted tank top and light blue jeans",
          "mirror selfie from the waist up, face visible, open blue windbreaker over a low-cut black top, modest cleavage",
          "mirror selfie from the waist up, face visible, hoodie unzipped over a fitted black top",
          "gym mirror selfie from the waist up, fitted sports bra and leggings, face visible",
          "mirror selfie from the waist up, fitted v-neck top, modest cleavage, smiling"]
mirror_bg = ["small tiled bathroom", "bedroom with a round vintage mirror", "plain wall", "bedroom"]
body = ["full-length mirror selfie, face visible, fitted top and light blue jeans, bedroom",
        "full-length mirror selfie, face visible, black scoop-neck crop top and floral skirt, tiled bathroom",
        "full body, sitting on an outdoor cafe sofa with an iced coffee, varsity jacket, ripped jeans, white sneakers",
        "full body, standing against a plain wall, fitted tank top and jeans, relaxed pose",
        "full body, walking on a sunny street, fitted top and jeans",
        "full body, sports bra and leggings in a gym, relaxed pose",
        "full body, one-piece swimsuit on a beach, relaxed pose",
        "full body, oversized blazer and bike shorts on a street, relaxed pose"]
lights = ["harsh ceiling lights", "camera flash at night", "soft window light", "overcast daylight", "warm indoor light"]
rnd = random.Random(11)
def combos(shots, bgs, n):
    c = list(itertools.product(shots, bgs, lights)); rnd.shuffle(c); return c[:n]
CANDIDATES = ([f"{CHAR}, {s}, {b}, {l}" for s, b, l in combos(selfie, selfie_bg, 25) + combos(mirror, mirror_bg, 35)]
              + [f"{CHAR}, {s}, {l}" for s, _, l in combos(body, [""], 40)])

plan = [{"pillar": x["pillar"], "prompt": f"photo of lia_character woman, {x['scene']}", "caption": x["caption"]} for x in make_plan(30)]

def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}
def nb(cells): return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}

SETUP = code(f'''
from google.colab import drive
drive.mount('/content/drive')
import os
BASE = '/content/drive/MyDrive/{DRIVE_DIR}'      # todo se guarda aquí, en tu Google Drive
for d in ('candidatas', 'dataset', 'lora'):
    os.makedirs(f'{{BASE}}/{{d}}', exist_ok=True)
import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA -> Entorno de ejecución > Cambiar tipo > GPU T4')
''')
# Colab trae una torchao antigua que rompe el import de diffusers 0.41 (cannot import name 'FqnToConfig'); no la usamos.
# diffusers 0.41.0 es la versión probada con el transformers 5.x de Colab.
INSTALL = code("!pip uninstall -y -q torchao\n!pip install -q diffusers==0.41.0 transformers accelerate peft safetensors")
LOAD = '''
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
pipe = StableDiffusionXLPipeline.from_pretrained('SG161222/RealVisXL_V4.0', torch_dtype=torch.float16,
                                                 variant='fp16', use_safetensors=True).to('cuda')
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
'''

# ---------- 1: candidatas ----------
n1 = nb([
 md('''
# 1 · Candidatas de Lía: cara y cuerpo (SDXL, estilo foto de móvil)
**Colab → Entorno de ejecución → Cambiar tipo de entorno → GPU T4.** Ejecuta las celdas en orden.

Genera 100 imágenes estilo foto casera de móvil (selfies, espejos, cuerpo entero) de una mujer normal, con pelo cobrizo,
sin pecas y con figura definida pero natural: 25 selfies, 35 de espejo y 40 de cuerpo (así el LoRA aprende **cara y cuerpo** juntos).
Se guardan en tu Google Drive, carpeta `lia2/candidatas`. Luego eliges las 20–30 donde **cara y cuerpo se parezcan**.
Tiempo aproximado: 1 h–1 h 20 min (estimado, no medido). Se puede parar y repetir: salta las que ya existen.
La cuenta final se publica siempre **etiquetada como IA**.
'''),
 SETUP, INSTALL,
 code(f"STYLE = {STYLE_DS!r}\nNEGATIVE = {NEGATIVE_DS!r}\nSEED = 777\nPROMPTS = {json.dumps(CANDIDATES, ensure_ascii=False, indent=1)}\n"),
 code(LOAD + '''
import os
for i, core in enumerate(PROMPTS, 1):
    f = f'{BASE}/candidatas/img_{i:02d}.png'
    if os.path.exists(f):
        continue
    g = torch.Generator('cuda').manual_seed(SEED)       # misma semilla = rasgos más parecidos
    img = pipe(prompt=f'{STYLE}, {core}', negative_prompt=NEGATIVE, width=1024, height=1024,
               num_inference_steps=30, guidance_scale=5.0, generator=g).images[0]
    img.save(f)
    print(i, '/', len(PROMPTS), flush=True)
'''),
 md("## Ver todas de golpe\nCrea hojas numeradas en `lia2/candidatas/hoja_XX.jpg`. Ábrelas en Drive o súbelas al chat de Claude para elegir."),
 code('''
import glob
from PIL import Image, ImageDraw
files = sorted(glob.glob(f'{BASE}/candidatas/img_*.png'), key=lambda f: int(os.path.basename(f)[4:-4]))
COLS, ROWS, T = 4, 3, 360
for n in range(0, len(files), COLS * ROWS):
    sheet = Image.new('RGB', (COLS * T, ROWS * T), 'white'); d = ImageDraw.Draw(sheet)
    for k, f in enumerate(files[n:n + COLS * ROWS]):
        im = Image.open(f).convert('RGB'); im.thumbnail((T, T))
        x, y = (k % COLS) * T, (k // COLS) * T
        sheet.paste(im, (x, y)); d.rectangle([x, y, x + 52, y + 24], fill='black')
        d.text((x + 6, y + 6), os.path.basename(f)[4:-4], fill='white')
    sheet.save(f'{BASE}/candidatas/hoja_{n // (COLS * ROWS) + 1:02d}.jpg', quality=88)
print('Hojas creadas en', f'{BASE}/candidatas')
'''),
])

# ---------- 2: entrenar ----------
n2 = nb([
 md(f'''
# 2 · Entrenar el LoRA de Lía (SDXL): cara y cuerpo
Antes: elige las **20–30 imágenes** de `lia2/candidatas` donde cara y cuerpo se parezcan (mezcla retratos, cintura para arriba y cuerpo
entero) y pon sus números en la lista `ELEGIDAS` de la primera celda: se copian solas a `lia2/dataset`.
**GPU T4.** Tiempo aproximado: 1 h (estimado, no medido). El resultado queda en `lia2/lora`.
'''),
 SETUP, INSTALL,
 code('''
# Números de img_XX.png elegidos de lia2/candidatas. Se copian solos a lia2/dataset. Edita la lista para cambiar la selección.
ELEGIDAS = [28, 29, 30, 31, 33, 38, 39, 42, 43, 44, 52, 54, 56, 58, 61, 62, 63, 64, 75, 79, 80, 81, 84, 85, 86, 88, 91, 97]
import shutil
for n in ELEGIDAS:
    shutil.copy(f'{BASE}/candidatas/img_{n:02d}.png', f'{BASE}/dataset/')
print(len(ELEGIDAS), 'fotos copiadas a lia2/dataset')
'''),
 code('''
!pip install -q bitsandbytes
!wget -q -O train_dreambooth_lora_sdxl.py https://raw.githubusercontent.com/huggingface/diffusers/v0.41.0/examples/dreambooth/train_dreambooth_lora_sdxl.py
import glob
n = len(glob.glob(f'{BASE}/dataset/*.png') + glob.glob(f'{BASE}/dataset/*.jpg'))
print(n, 'imágenes en el dataset')
assert n >= 10, 'Pon en ELEGIDAS los números de tus mejores imágenes (mínimo 10, ideal 20-30)'
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
 md(f'''
# 3 · Generar las fotos de Lía (SDXL + su LoRA), estilo foto de móvil
Usa el LoRA de `lia2/lora`. Genera las publicaciones del plan de 30 días con estilo de selfie / foto casual y las guarda en
`lia2/salida_iphone` junto a `captions.csv` (texto con el aviso de IA). Cambia `DESDE` y `HASTA` para hacerlo por tandas.
Los selfies de espejo piden el iPhone con su logo visible y el móvil tapando la cara (la IA no siempre lo dibuja bien: se descartan las que salgan mal).
**GPU T4.** Aproximadamente 1 min por imagen (estimado). **Revisa cada imagen a mano** (manos, cara, proporciones, texto raro)
y publica siempre con la etiqueta de IA.
'''),
 SETUP, INSTALL,
 code(f"STYLE = {STYLE3!r}\nNEGATIVE = {NEGATIVE3!r}\nOUT = 'salida_iphone'\nPLAN = {json.dumps(plan, ensure_ascii=False, indent=1)}\nDESDE, HASTA = 1, 10     # días del plan a generar\n"),
 code(LOAD + '''
pipe.load_lora_weights(f'{BASE}/lora', weight_name='pytorch_lora_weights.safetensors')
import csv, os, random
os.makedirs(f'{BASE}/{OUT}', exist_ok=True)
cap = f'{BASE}/{OUT}/captions.csv'
if not os.path.exists(cap):
    open(cap, 'w', encoding='utf-8').write('dia,pilar,archivo,caption\\n')
for n in range(DESDE, HASTA + 1):
    it = PLAN[n - 1]
    f = f'{BASE}/{OUT}/{n:02d}_{it["pillar"]}.png'
    if os.path.exists(f):
        continue
    seed = random.randint(0, 2**31)
    g = torch.Generator('cuda').manual_seed(seed)
    if 'mirror selfie' in it['prompt']:
        cam = 'mirror selfie, iPhone 16 Pro with Apple logo visible on its back, flash glare'
    elif 'selfie' in it['prompt']:
        cam = 'front camera selfie at arm length, wide angle, slightly imperfect framing'
    else:
        cam = 'shot on a phone, casual framing'
    img = pipe(prompt=f'{STYLE}, {cam}, {it["prompt"]}', negative_prompt=NEGATIVE, width=832, height=1216,
               num_inference_steps=30, guidance_scale=4.5, generator=g,
               cross_attention_kwargs={'scale': 0.9}).images[0]
    img.save(f)
    with open(cap, 'a', newline='', encoding='utf-8') as fh:
        csv.writer(fh).writerow([n, it['pillar'], os.path.basename(f), it['caption']])
    print(n, it['pillar'], 'seed', seed, flush=True)
'''),
 md("## Efecto «foto de móvil» (opcional)\nCopia cada imagen de `salida_iphone` a `salida_iphone/movil/*.jpg` con grano de sensor, un poco menos de nitidez, viñeteo suave y compresión JPEG, como una foto subida desde el móvil. "
    "Solo cambia la estética: **no** añade metadatos falsos de cámara. Las originales se conservan."),
 code('''
import glob, numpy as np
from PIL import Image, ImageFilter, ImageEnhance
os.makedirs(f'{BASE}/{OUT}/movil', exist_ok=True)
def efecto_movil(img, seed):
    rng = np.random.default_rng(seed)
    im = img.convert('RGB').filter(ImageFilter.GaussianBlur(0.6))
    a = np.asarray(im).astype(np.float32) + rng.normal(0, 3.5, (im.size[1], im.size[0], 3))
    h, w = a.shape[:2]; yy, xx = np.mgrid[0:h, 0:w]
    a *= (1 - 0.10 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2))[..., None]
    return ImageEnhance.Contrast(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))).enhance(1.05)
for f in sorted(glob.glob(f'{BASE}/{OUT}/*.png')):
    out = f'{BASE}/{OUT}/movil/' + os.path.basename(f)[:-4] + '.jpg'
    if not os.path.exists(out):
        efecto_movil(Image.open(f), abs(hash(f)) % 2**32).save(out, quality=82)
print('Listo: lia2/' + OUT + '/movil')
'''),
])

for name, n in (("1_candidatas", n1), ("2_entrenar_lora", n2), ("3_generar", n3)):
    json.dump(n, open(f"{name}.ipynb", "w"), ensure_ascii=False, indent=1)
    print("escrito", f"{name}.ipynb")