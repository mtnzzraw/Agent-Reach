"""Genera los posts pendientes del plan.json con ComfyUI: carrusel = 1 imagen por escena, Reel = imagen + vídeo.
Uso: python run.py [N]"""
import json, sys, yaml, os, random
from pathlib import Path
import comfy, reel

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
        w, h = p["sizes"][it["format"]]
        seed = random.randint(0, 2**32 - 11)   # semillas consecutivas: luz parecida entre fotos del carrusel
        it["images"] = []
        for i, prompt in enumerate(it["prompts"]):
            wf = comfy.build_workflow(prompt, p["negative"], CHECKPOINT, LORA, w=w, h=h, seed=seed + i)
            it["images"] += comfy.generate(wf, "output")
        if it["format"] == "reel":
            it["video"] = reel.make_reel(it["images"][0], it["hook"], f"output/reel_{it['date'][:10]}.mp4")
        it["status"] = "generated"
        done += 1
        json.dump(plan, open("plan.json", "w"), ensure_ascii=False, indent=2)
        print("OK", it["date"], it["format"], it.get("video") or it["images"])

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
