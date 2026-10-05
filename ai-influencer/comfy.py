"""Cliente mínimo de ComfyUI: construye un workflow SDXL+LoRA, lo envía y descarga las imágenes."""
import json, time, random, urllib.request, urllib.parse
from pathlib import Path

HOST = "http://127.0.0.1:8188"

def build_workflow(prompt, negative, checkpoint, lora=None, lora_strength=0.9,
                   w=832, h=1216, steps=30, cfg=5.5, seed=None):
    seed = random.randint(0, 2**32 - 1) if seed is None else seed
    g = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}}}
    model, clip = ["1", 0], ["1", 1]
    if lora:
        g["2"] = {"class_type": "LoraLoader", "inputs": {"model": model, "clip": clip,
                  "lora_name": lora, "strength_model": lora_strength, "strength_clip": lora_strength}}
        model, clip = ["2", 0], ["2", 1]
    g["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip}}
    g["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": clip}}
    g["5"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
    g["6"] = {"class_type": "KSampler", "inputs": {"model": model, "positive": ["3", 0], "negative": ["4", 0],
              "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": cfg,
              "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}}
    g["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}}
    g["8"] = {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "lia"}}
    return g

def _req(path, data=None):
    r = urllib.request.Request(HOST + path, data=json.dumps(data).encode() if data else None,
                               headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=60).read()

def generate(workflow, out_dir, timeout=600):
    pid = json.loads(_req("/prompt", {"prompt": workflow}))["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = json.loads(_req(f"/history/{pid}"))
        if pid in hist:
            saved = []
            for node in hist[pid]["outputs"].values():
                for im in node.get("images", []):
                    q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]})
                    p = Path(out_dir) / im["filename"]
                    p.write_bytes(_req("/view?" + q))
                    saved.append(str(p))
            return saved
        time.sleep(2)
    raise TimeoutError("ComfyUI no terminó a tiempo")
