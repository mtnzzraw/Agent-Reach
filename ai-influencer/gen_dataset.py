"""Genera imágenes para el dataset de la cara (LoRA). Uso: python gen_dataset.py [N] [SEED]   (opcional: PROMPTS=archivo.txt PREFIX=cara)
Usa la misma semilla en todos los prompts: en SD 1.5 eso mantiene la cara bastante parecida entre imágenes.
Las imágenes quedan en dataset/ ; quédate con las 15-30 donde se vea la misma persona."""
import os, sys, time
from pathlib import Path
import yaml
import comfy

PROFILE = os.environ.get("PROFILE", "sd15")
CHECKPOINT = os.environ.get("CKPT", "v1-5-pruned-emaonly.safetensors" if PROFILE == "sd15" else "sd_xl_base_1.0.safetensors")

def main(n=30, seed=1234):
    negative = yaml.safe_load(open("persona.yaml"))["negative"]
    prompts = [l.strip() for l in open(os.environ.get("PROMPTS", "training_prompts.txt")) if l.strip()][:n]
    prefix = os.environ.get("PREFIX", "lia")
    out = Path("dataset"); out.mkdir(exist_ok=True)
    for i, p in enumerate(prompts, 1):
        if (out / f"{prefix}_{i:02d}.png").exists():
            continue   # permite reanudar si lo paras
        prof = "sd15-tall" if PROFILE == "sd15" and "full body" in p else PROFILE
        wf = comfy.build_workflow(p, negative, CHECKPOINT, seed=seed, **comfy.PROFILES[prof])
        for attempt in range(1, 4):   # sin supervisión: si una imagen falla, reintenta y sigue con la siguiente
            try:
                saved = comfy.generate(wf, str(out))
                Path(saved[0]).rename(out / f"{prefix}_{i:02d}.png")
                print(f"{i}/{len(prompts)} OK", flush=True)
                break
            except Exception as e:
                print(f"{i}/{len(prompts)} fallo (intento {attempt}/3): {e}", flush=True)
                time.sleep(20)

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30, int(sys.argv[2]) if len(sys.argv) > 2 else 1234)
