"""Publica en Instagram vía Graph API oficial lo que toque según la fecha. Programa con cron cada hora.
Requiere cuenta Business/Creator + IG_USER_ID + IG_TOKEN. La API necesita una URL pública de la imagen:
sube output/*.png a tu hosting (S3, R2...) y pon la base en PUBLIC_BASE_URL."""
import json, os, datetime as dt, urllib.request, urllib.parse
from pathlib import Path

G = "https://graph.facebook.com/v21.0"
UID, TOKEN, BASE = os.environ["IG_USER_ID"], os.environ["IG_TOKEN"], os.environ["PUBLIC_BASE_URL"]

def post(path, **data):
    r = urllib.request.urlopen(G + path, urllib.parse.urlencode({**data, "access_token": TOKEN}).encode(), timeout=60)
    return json.loads(r.read())

plan = json.load(open("plan.json"))
for it in plan:
    if it["status"] == "generated" and dt.datetime.fromisoformat(it["date"]) <= dt.datetime.now():
        url = f"{BASE.rstrip('/')}/{Path(it['images'][0]).name}"
        c = post(f"/{UID}/media", image_url=url, caption=it["caption"])
        post(f"/{UID}/media_publish", creation_id=c["id"])
        it["status"] = "published"
        print("Publicado", it["date"])
json.dump(plan, open("plan.json", "w"), ensure_ascii=False, indent=2)
