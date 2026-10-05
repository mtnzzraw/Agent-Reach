"""Convierte calendar.yaml en plan.json: prompts por imagen + pie de foto en español (con aviso de IA) + fecha.
Uso: python plan.py   (conserva el estado de los posts ya generados o publicados)"""
import json, datetime as dt, re
from pathlib import Path
import yaml

MAX_CAPTION, MAX_HASHTAGS = 2200, 30          # límites de Instagram
MIN_IMAGES, MAX_IMAGES = 2, 10                # carrusel en la Graph API

def caption(e, p):
    lines = [e["hook"], "", e["body"].strip(), "", f"💬 {e['question']}", p["cta"][e["format"]], "",
             p["disclosure_caption"], " ".join(dict.fromkeys(e.get("tags", []) + p["base_hashtags"]))]
    return "\n".join(lines)

def validate(it, p):
    errs, c, n = [], it["caption"], len(it["prompts"])
    if p["disclosure_caption"] not in c: errs.append("falta el aviso de IA")
    if len(c) > MAX_CAPTION: errs.append(f"pie de {len(c)} caracteres (máx. {MAX_CAPTION})")
    if len(re.findall(r"#\w+", c)) > MAX_HASHTAGS: errs.append(f"más de {MAX_HASHTAGS} hashtags")
    if it["format"] == "reel" and n != 1: errs.append("un Reel lleva exactamente 1 escena")
    if it["format"] == "carousel" and not MIN_IMAGES <= n <= MAX_IMAGES:
        errs.append(f"un carrusel lleva {MIN_IMAGES}-{MAX_IMAGES} escenas, tiene {n}")
    if it["format"] not in ("carousel", "reel"): errs.append(f"formato desconocido: {it['format']}")
    if errs: raise ValueError(f"{it['date']}: " + "; ".join(errs))

def make_plan(calendar="calendar.yaml"):
    p = yaml.safe_load(open("persona.yaml"))
    items = []
    for e in yaml.safe_load(open(calendar)):
        when = dt.datetime.combine(dt.date.fromisoformat(str(e["date"])),
                                   dt.time.fromisoformat(e.get("time", p["post_time"])))
        it = {
            "date": when.isoformat(), "format": e["format"], "pillar": e["pillar"], "hook": e["hook"],
            "prompts": [f"photo of {p['character']}, {s}, natural light, 35mm, candid, detailed skin texture, "
                        "instagram travel photography" for s in e["scenes"]],
            "caption": caption(e, p), "verify": bool(e.get("verify")), "status": "planned",
        }
        validate(it, p)
        items.append(it)
    return items

def merge(new, old):
    """Mantiene los posts que ya no están en 'planned' (generados, publicados...) tal cual estaban."""
    done = {it["date"]: it for it in old if it.get("status") != "planned"}
    return [done.get(it["date"], it) for it in new]

if __name__ == "__main__":
    old = json.load(open("plan.json")) if Path("plan.json").exists() else []
    plan = merge(make_plan(), old)
    json.dump(plan, open("plan.json", "w"), ensure_ascii=False, indent=2)
    r = sum(it["format"] == "reel" for it in plan)
    print(f"plan.json con {len(plan)} publicaciones ({len(plan) - r} carruseles, {r} Reels)")
