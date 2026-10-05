"""Genera las imágenes pendientes del plan.json con ComfyUI.  Uso: python run.py [N]"""
import json, sys, yaml, os
from pathlib import Path
import comfy

CHECKPOINT = os.environ.get("CKPT", "sd_xl_base_1.0.safetensors")   # nombre en ComfyUI/models/checkpoints
LORA = os.environ.get("LORA")                                       # p.ej. lia_character.safetensors

def main(limit=7):
    p = yaml.safe_load(open("persona.yaml"))
    plan = json.load(open("plan.json"))
    Path("output").mkdir(exist_ok=True)
    done = 0
    for it in plan:
        if it["status"] != "planned" or done >= limit:
            continue
        wf = comfy.build_workflow(it["prompt"], p["negative"], CHECKPOINT, LORA)
        it["images"] = comfy.generate(wf, "output")
        it["status"] = "generated"
        done += 1
        json.dump(plan, open("plan.json", "w"), ensure_ascii=False, indent=2)
        print("OK", it["date"], it["images"])

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
