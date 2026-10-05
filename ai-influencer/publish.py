"""Publica en Instagram vía Graph API oficial lo que toque según la fecha. Programa con cron cada hora.
Requiere cuenta Business/Creator + IG_USER_ID + IG_TOKEN. La API necesita URLs públicas: sube output/*
(imágenes y .mp4) a tu hosting (S3, R2...) y pon la base en PUBLIC_BASE_URL.

Carruseles: se publican solos. Reels: por defecto se dejan listos en output/ (vídeo + pie .txt) para subirlos
desde el móvil con un audio en tendencia, que la API no permite. Con --reels api se publican por API sin música.
Los posts con verify: true no se publican hasta que pongas su status en "verified" tras revisarlos.
Uso: python publish.py [--reels manual|api]"""
import argparse, json, os, time, datetime as dt, urllib.error, urllib.request, urllib.parse
from pathlib import Path

G = "https://graph.facebook.com/v21.0"
READY = ("generated", "verified")

def api(path, method="POST", **data):
    q = urllib.parse.urlencode({**data, "access_token": os.environ["IG_TOKEN"]})
    req = urllib.request.Request(G + path, q.encode()) if method == "POST" else urllib.request.Request(f"{G}{path}?{q}")
    try:
        return json.loads(urllib.request.urlopen(req, timeout=60).read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode(errors="replace")) from None

def url(file):
    return f"{os.environ['PUBLIC_BASE_URL'].rstrip('/')}/{Path(file).name}"

def wait_ready(cid, timeout=300):
    """Instagram procesa el vídeo (y a veces el carrusel) antes de dejar publicarlo."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = api(f"/{cid}", "GET", fields="status_code")["status_code"]
        if s == "FINISHED": return
        if s in ("ERROR", "EXPIRED"): raise RuntimeError(f"contenedor {cid}: {s}")
        time.sleep(5)
    raise TimeoutError(f"contenedor {cid} sin procesar tras {timeout} s")

def publish_carousel(uid, it):
    kids = [api(f"/{uid}/media", image_url=url(f), is_carousel_item="true")["id"] for f in it["images"]]
    c = api(f"/{uid}/media", media_type="CAROUSEL", children=",".join(kids), caption=it["caption"])
    wait_ready(c["id"])
    return api(f"/{uid}/media_publish", creation_id=c["id"])["id"]

def publish_reel(uid, it):
    c = api(f"/{uid}/media", media_type="REELS", video_url=url(it["video"]), caption=it["caption"], share_to_feed="true")
    wait_ready(c["id"])
    return api(f"/{uid}/media_publish", creation_id=c["id"])["id"]

def publish_photo(uid, it):   # posts antiguos del plan, sin formato
    c = api(f"/{uid}/media", image_url=url(it["images"][0]), caption=it["caption"])
    return api(f"/{uid}/media_publish", creation_id=c["id"])["id"]

def leave_for_manual(it):
    txt = Path(it["video"]).with_suffix(".txt")
    txt.write_text(it["caption"], encoding="utf-8")
    print(f"Reel listo para subir a mano: {it['video']} (pie en {txt})")

def main(reels="manual", plan_file="plan.json"):
    uid = os.environ.get("IG_USER_ID")
    plan = json.load(open(plan_file))
    for it in plan:
        if it["status"] not in READY or dt.datetime.fromisoformat(it["date"]) > dt.datetime.now():
            continue
        if it.get("verify") and it["status"] != "verified":
            print(f"Pendiente de revisar (verify): {it['date']} · {it.get('hook', '')}")
            continue
        fmt = it.get("format", "photo")
        try:
            if fmt == "reel" and reels == "manual":
                leave_for_manual(it)
                it["status"] = "manual"
            else:
                fn = {"carousel": publish_carousel, "reel": publish_reel, "photo": publish_photo}[fmt]
                it["media_id"] = fn(uid, it)
                it["status"] = "published"
                print("Publicado", it["date"], fmt)
        except Exception as e:
            it["status"], it["error"] = "error", str(e)[:500]
            print("Error", it["date"], fmt, it["error"])
        json.dump(plan, open(plan_file, "w"), ensure_ascii=False, indent=2)   # guarda tras cada post

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--reels", choices=["manual", "api"], default="manual")
    main(a.parse_args().reels)
