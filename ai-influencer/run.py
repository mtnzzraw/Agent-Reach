"""Genera las imágenes pendientes del plan.json con ComfyUI.  Uso: python run.py [N]"""
import json, sys, yaml, os
from pathlib import Path
import comfy

PROFILE = os.environ.get("PROFILE", "sd15")                         # sd15 (Mac 8 GB) | sdxl-lowvram (experimental) | sdxl
CHECKPOINT = comfy.default_checkpoint(PROFILE)   # CKPT=... para forzar otro; usa Realistic Vision si lo descargaste
HIRES = os.environ.get("HIRES", "0") == "1"                          # 2ª pasada que mejora las caras, SIN PROBAR (HIRES=1 para activarla)
UPSCALE = os.environ.get("UPSCALE")                                 # p.ej. RealESRGAN_x4plus.pth (opcional, más lento)
LORA = os.environ.get("LORA")                                       # p.ej. lia_character.safetensors

def main(limit=7):
    p = yaml.safe_load(open("persona.yaml"))
    plan = json.load(open("plan.json"))
    Path("output").mkdir(exist_ok=True)
    done = 0
    for it in plan:
        if it["status"] != "planned" or done >= limit:
            continue
        wf = comfy.build_workflow(it["prompt"], p["negative"], CHECKPOINT, LORA, upscale_model=UPSCALE, vae=comfy.default_vae(), hires=HIRES, **comfy.PROFILES["sd15-tall" if PROFILE == "sd15" and "full body" in it["prompt"] else PROFILE])
        it["images"] = comfy.generate(wf, "output")
        it["status"] = "generated"
        done += 1
        json.dump(plan, open("plan.json", "w"), ensure_ascii=False, indent=2)
        print("OK", it["date"], it["images"])

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
