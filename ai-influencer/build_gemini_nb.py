"""Genera 4_gemini_lote.ipynb: genera las fotos de Lía en lote con la API de Gemini (a partir de prompts_gemini.md) y las escala."""
import json, re
txt = open("prompts_gemini.md", encoding="utf-8").read()
dias = []
for m in re.finditer(r"### Día (\d+)( 🌙)?\n\*\*Escena:\*\* (.*?)\n\*\*Caption sugerido:\*\* (.*?)\n", txt):
    dias.append({"dia": int(m.group(1)), "tatuaje": bool(m.group(2)), "escena": m.group(3), "caption": m.group(4)})
assert len(dias) == 30, len(dias)

def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}

cells = [
md("""# 4 · Fotos de Lía en lote con Gemini (sin pegar cada texto a mano)
Genera las fotos de los días que elijas con la **API de Gemini** usando tus fotos de referencia, las guarda en tu Drive y las escala a mejor resolución.
**No necesita GPU** (Entorno de ejecución → CPU vale).
**Antes de empezar:**
1. En Drive crea la carpeta `Mi unidad/lia2/referencias` y mete 3 fotos buenas de Lía. **Para cambiar la cara basta con cambiar estas fotos** y, si quieres, la variable `PELO`.
2. Consigue una clave gratis en https://aistudio.google.com/apikey (botón *Create API key*). **No la compartas con nadie.**
Ojo: la generación de imágenes por API puede tener **cupo gratis muy limitado o no estar incluida**. Si el cuaderno dice "cuota" o "billing", mira los límites en tu cuenta de AI Studio antes de seguir."""),
code("""!pip install -q google-genai pillow
import os, glob, time, csv, io
from PIL import Image
from google.colab import drive
drive.mount('/content/drive')
BASE = '/content/drive/MyDrive/lia2'
for d in ('referencias', 'gemini_out', 'gemini_hd'):
    os.makedirs(f'{BASE}/{d}', exist_ok=True)
refs = sorted(glob.glob(f'{BASE}/referencias/*.[jp][pn]*g'))
print(len(refs), 'fotos de referencia:', [os.path.basename(r) for r in refs])
assert refs, 'Pon 3 fotos de Lía en Mi unidad/lia2/referencias'"""),
code("""from getpass import getpass
API_KEY = getpass('Pega tu clave de AI Studio y pulsa Enter: ')
from google import genai
from google.genai import types
client = genai.Client(api_key=API_KEY)"""),
code("""PELO = 'pelo largo ondulado cobrizo'   # cámbialo si quieres otro color o estilo (por ejemplo: 'pelo largo liso castaño oscuro')
FRASE_BASE = ('Crea una foto nueva de la misma mujer de las imágenes de referencia: mismo rostro, ' + PELO + ', figura natural, sin pecas, mujer adulta de unos 26 años. '
              'Foto casera hecha con un iPhone, luz natural, ligero grano, aspecto 100 % real y sin retoque. Sin collares ni colgantes. '
              'Si aparece un móvil, es un iPhone con la manzana y tres cámaras bien dibujadas. Formato vertical 4:5.')
TATUAJE = ' Lleva un pequeño tatuaje de línea fina de luna creciente en el pecho, entre las clavículas, sin collar ni colgante.'
PLAN = __PLAN__
MODELO = 'gemini-2.5-flash-image'    # si te da error de modelo, mira el nombre actual en AI Studio (modelos de imagen)
DESDE, HASTA = 1, 3     # días a generar
VARIANTES = 2           # versiones por día (así eliges la mejor)
PAUSA = 8               # segundos entre peticiones (para no pasar el límite por minuto)""".replace('__PLAN__', json.dumps(dias, ensure_ascii=False, indent=1))),
code("""imgs_ref = [Image.open(r).convert('RGB') for r in refs]
cap = f'{BASE}/gemini_out/captions.csv'
if not os.path.exists(cap):
    open(cap, 'w', encoding='utf-8').write('dia,archivo,caption\\n')
parar = False
for it in PLAN:
    if parar: break
    if not (DESDE <= it['dia'] <= HASTA): continue
    for v in range(1, VARIANTES + 1):
        f = f"{BASE}/gemini_out/dia{it['dia']:02d}_v{v}.png"
        if os.path.exists(f): continue
        prompt = FRASE_BASE + ' ' + it['escena'] + (TATUAJE if it['tatuaje'] else '')
        for intento in range(3):
            try:
                r = client.models.generate_content(model=MODELO, contents=[prompt] + imgs_ref,
                        config=types.GenerateContentConfig(response_modalities=['IMAGE']))
                partes = (r.candidates[0].content.parts if r.candidates and r.candidates[0].content else []) or []
                dat = [p.inline_data.data for p in partes if getattr(p, 'inline_data', None) and p.inline_data.data]
                if not dat:
                    print('día', it['dia'], 'v', v, ': sin imagen (posible rechazo). Respuesta:', str(r.candidates[0].finish_reason if r.candidates else r.prompt_feedback)[:150]); break
                Image.open(io.BytesIO(dat[0])).save(f)
                with open(cap, 'a', newline='', encoding='utf-8') as fh: csv.writer(fh).writerow([it['dia'], os.path.basename(f), it['caption'] + '\\n\\n✨ Imagen generada con IA · Virtual creator'])
                print('OK', os.path.basename(f), Image.open(f).size, flush=True); break
            except Exception as e:
                m = str(e)
                print('error:', m[:200])
                if '429' in m or 'RESOURCE_EXHAUSTED' in m or 'quota' in m.lower():
                    if 'per day' in m.lower() or 'billing' in m.lower() or 'limit: 0' in m.lower():
                        print('Cupo agotado o sin cupo gratis para imágenes: para aquí.'); parar = True; break
                    time.sleep(40)
                else:
                    time.sleep(5)
        if parar: break
        time.sleep(PAUSA)
print('Listo: lia2/gemini_out')"""),
md("""## Escalar a mejor resolución
Las fotos de Gemini salen pequeñas. Este paso las amplía x2 (con Real-ESRGAN si se puede instalar, y si no con un método normal de suavizado) y guarda una versión lista para Instagram (1080 px de ancho) en `lia2/gemini_hd`."""),
code("""import numpy as np, torch, urllib.request
from PIL import ImageFilter
modelo_esr = None
try:
    !pip install -q spandrel
    from spandrel import ModelLoader
    pesos = '/content/RealESRGAN_x4plus.pth'
    if not os.path.exists(pesos):
        urllib.request.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', pesos)
    modelo_esr = ModelLoader().load_from_file(pesos).cuda().eval() if torch.cuda.is_available() else ModelLoader().load_from_file(pesos).eval()
    print('Real-ESRGAN cargado')
except Exception as e:
    print('Sin Real-ESRGAN (se usa el método normal):', str(e)[:120])

def esr4(img, tile=256, pad=16):
    dev = next(modelo_esr.model.parameters()).device
    a = torch.from_numpy(np.asarray(img).astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
    _, _, H, W = a.shape
    out = torch.zeros(1, 3, H * 4, W * 4)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            y0, x0, y1, x1 = max(y - pad, 0), max(x - pad, 0), min(y + tile + pad, H), min(x + tile + pad, W)
            with torch.no_grad():
                o = modelo_esr(a[:, :, y0:y1, x0:x1].to(dev)).cpu()
            oy, ox = (y - y0) * 4, (x - x0) * 4
            h, w = min(tile, H - y), min(tile, W - x)
            out[:, :, y * 4:(y + h) * 4, x * 4:(x + w) * 4] = o[:, :, oy:oy + h * 4, ox:ox + w * 4]
    return Image.fromarray((out.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))

for f in sorted(glob.glob(f'{BASE}/gemini_out/*.png')):
    sal = f'{BASE}/gemini_hd/' + os.path.basename(f)[:-4] + '.jpg'
    if os.path.exists(sal): continue
    im = Image.open(f).convert('RGB')
    if modelo_esr is not None:
        try: big = esr4(im).resize((im.width * 2, im.height * 2), Image.LANCZOS)
        except Exception as e: print('ESRGAN falló, método normal:', str(e)[:80]); big = im.resize((im.width * 2, im.height * 2), Image.LANCZOS).filter(ImageFilter.UnsharpMask(1.2, 60, 2))
    else:
        big = im.resize((im.width * 2, im.height * 2), Image.LANCZOS).filter(ImageFilter.UnsharpMask(1.2, 60, 2))
    big = big.resize((1080, round(big.height * 1080 / big.width)), Image.LANCZOS)
    big.save(sal, quality=95)
    print('HD', os.path.basename(sal), big.size, flush=True)
print('Listo: lia2/gemini_hd')"""),
]
nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("4_gemini_lote.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("escrito 4_gemini_lote.ipynb")
