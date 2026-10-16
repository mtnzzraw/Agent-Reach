"""Genera las fotos de Lía en lote con la API de Gemini (REST, sin dependencias salvo Pillow y numpy).
Lee la clave de la variable de entorno GEMINI_API_KEY (NUNCA la escribas en el código ni en el chat).
Uso: python gen_gemini.py --refs carpeta_refs --out carpeta_salida --desde 1 --hasta 3 --variantes 2 [--pelo "pelo largo liso castaño oscuro"]"""
import argparse, base64, csv, glob, io, json, os, re, sys, time, urllib.request, urllib.error
from PIL import Image
import numpy as np

MODELO = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
URL = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent"
TATUAJE = " Lleva un pequeño tatuaje de línea fina de luna creciente en el pecho, entre las clavículas, sin collar ni colgante."

def plan():
    txt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts_gemini.md"), encoding="utf-8").read()
    out = []
    for m in re.finditer(r"### Día (\d+)( 🌙)?\n\*\*Escena:\*\* (.*?)\n\*\*Caption sugerido:\*\* (.*?)\n", txt):
        out.append({"dia": int(m.group(1)), "tatuaje": bool(m.group(2)), "escena": m.group(3), "caption": m.group(4)})
    return out

def llamar(clave, prompt, refs):
    partes = [{"text": prompt}] + [{"inline_data": {"mime_type": "image/jpeg", "data": r}} for r in refs]
    for modalidades in (["IMAGE"], ["TEXT", "IMAGE"]):
        body = json.dumps({"contents": [{"parts": partes}], "generationConfig": {"responseModalities": modalidades}}).encode()
        req = urllib.request.Request(URL % MODELO, data=body, headers={"Content-Type": "application/json", "x-goog-api-key": clave})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                j = json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "ignore")[:400]
            if e.code == 400 and "modalit" in msg.lower() and modalidades == ["IMAGE"]:
                continue
            raise RuntimeError("HTTP %d: %s" % (e.code, msg))
        for c in j.get("candidates", []):
            for p in c.get("content", {}).get("parts", []):
                d = (p.get("inlineData") or p.get("inline_data") or {}).get("data")
                if d: return base64.b64decode(d)
        return None
    return None

def hd(img, semilla):
    from PIL import ImageFilter
    big = img.convert("RGB").resize((img.width * 2, img.height * 2), Image.LANCZOS).filter(ImageFilter.UnsharpMask(1.2, 60, 2))
    big = big.resize((1080, round(big.height * 1080 / big.width)), Image.LANCZOS)
    a = np.asarray(big).astype(np.float32) + np.random.default_rng(semilla).normal(0, 2.2, (big.height, big.width, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--desde", type=int, default=1); ap.add_argument("--hasta", type=int, default=3)
    ap.add_argument("--variantes", type=int, default=2); ap.add_argument("--pausa", type=float, default=8)
    ap.add_argument("--pelo", default="pelo largo ondulado cobrizo")
    a = ap.parse_args()
    clave = os.environ.get("GEMINI_API_KEY")
    if not clave: sys.exit("Falta la variable de entorno GEMINI_API_KEY")
    rutas = sorted(glob.glob(os.path.join(a.refs, "*.[jp][pn]*g")))[:5]
    if not rutas: sys.exit("No hay fotos de referencia en " + a.refs)
    refs = []
    for r in rutas:
        im = Image.open(r).convert("RGB"); im.thumbnail((1024, 1024)); b = io.BytesIO(); im.save(b, "JPEG", quality=92); refs.append(base64.b64encode(b.getvalue()).decode())
    base = ("Crea una foto nueva de la misma mujer de las imágenes de referencia: mismo rostro, %s, figura natural, sin pecas, mujer adulta. "
            "Foto casera hecha con un iPhone, luz natural, ligero grano, aspecto 100 %% real y sin retoque: piel con poros y pequeñas imperfecciones, pelo con algo de frizz, enfoque de móvil sin perfección de estudio. "
            "Sin collares ni colgantes. Si aparece un móvil, es un iPhone con la manzana y tres cámaras bien dibujadas. Formato vertical 4:5.") % a.pelo
    os.makedirs(a.out, exist_ok=True); os.makedirs(os.path.join(a.out, "hd"), exist_ok=True)
    cap = os.path.join(a.out, "captions.csv")
    if not os.path.exists(cap): open(cap, "w", encoding="utf-8").write("dia,archivo,caption\n")
    for it in plan():
        if not (a.desde <= it["dia"] <= a.hasta): continue
        for v in range(1, a.variantes + 1):
            f = os.path.join(a.out, "dia%02d_v%d.png" % (it["dia"], v))
            if os.path.exists(f): continue
            prompt = base + " " + it["escena"] + (TATUAJE if it["tatuaje"] else "")
            for intento in range(3):
                try:
                    dat = llamar(clave, prompt, refs)
                except Exception as e:
                    m = str(e); print("error:", m[:300], flush=True)
                    if "429" in m or "RESOURCE_EXHAUSTED" in m:
                        if any(k in m.lower() for k in ("per day", "billing", "limit: 0", "perday")): sys.exit("Cupo diario agotado o sin cupo gratis para imágenes.")
                        time.sleep(40); continue
                    if "403" in m or "401" in m or "API key" in m: sys.exit("Clave rechazada o sin permiso.")
                    time.sleep(5); continue
                if not dat: print("día", it["dia"], "v", v, ": sin imagen (posible rechazo)", flush=True); break
                im = Image.open(io.BytesIO(dat)); im.save(f)
                hd(im, it["dia"] * 10 + v).save(os.path.join(a.out, "hd", "dia%02d_v%d.jpg" % (it["dia"], v)), quality=95)
                with open(cap, "a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow([it["dia"], os.path.basename(f), it["caption"] + "\n\n✨ Imagen generada con IA · Virtual creator"])
                print("OK", os.path.basename(f), im.size, flush=True); break
            time.sleep(a.pausa)

if __name__ == "__main__":
    main()
