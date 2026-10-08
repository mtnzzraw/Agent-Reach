"""Cliente mínimo de ComfyUI: construye un workflow SDXL+LoRA, lo envía y descarga las imágenes."""
import json, time, random, urllib.request, urllib.parse
from pathlib import Path

HOST = "http://127.0.0.1:8188"

# Perfiles: sd15 para Mac con 8 GB (512x768 se generan en ~1-2 min); sdxl para GPUs con >=12 GB.
PROFILES = {"sd15": dict(w=512, h=768, steps=25, cfg=7.0),
            "sdxl": dict(w=832, h=1216, steps=30, cfg=5.5),
            # SDXL en 8 GB: resolución reducida; arrancar ComfyUI con --lowvram. Experimental, puede fallar por memoria.
            "sdxl-lowvram": dict(w=704, h=1024, steps=28, cfg=5.5)}

def build_workflow(prompt, negative, checkpoint, lora=None, lora_strength=0.9,
                   w=512, h=768, steps=25, cfg=7.0, seed=None, upscale_model=None):
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
    img = ["7", 0]
    if upscale_model:   # modelo de ../ComfyUI/models/upscale_models (p.ej. RealESRGAN_x4plus.pth); ComfyUI lo aplica por mosaicos
        g["9"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": upscale_model}}
        g["10"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["9", 0], "image": img}}
        g["11"] = {"class_type": "ImageScaleBy", "inputs": {"image": ["10", 0], "upscale_method": "lanczos", "scale_by": 0.5}}
        img = ["11", 0]   # x4 del modelo reducido a x2 final
    g["8"] = {"class_type": "SaveImage", "inputs": {"images": img, "filename_prefix": "lia"}}
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
