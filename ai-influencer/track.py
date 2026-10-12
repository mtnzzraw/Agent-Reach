"""Registra seguidores cada día en followers.csv y muestra progreso hacia la meta. Cron diario."""
import csv, json, os, datetime as dt, urllib.request, urllib.parse
GOAL = 10_000
q = urllib.parse.urlencode({"fields": "followers_count", "access_token": os.environ["IG_TOKEN"]})
n = json.loads(urllib.request.urlopen(f"https://graph.facebook.com/v21.0/{os.environ['IG_USER_ID']}?{q}").read())["followers_count"]
new = not os.path.exists("followers.csv")
with open("followers.csv", "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(["date", "followers"])
    w.writerow([dt.date.today().isoformat(), n])
print(f"{n} seguidores — {n / GOAL:.1%} de la meta ({GOAL})")
