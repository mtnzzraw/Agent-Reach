"""Genera un plan de N días: escena + prompt + caption (con aviso de IA) + fecha de publicación."""
import json, random, sys, datetime as dt, yaml

def make_plan(days=30, per_day=1, start=None, seed=1):
    p = yaml.safe_load(open("persona.yaml"))
    rnd = random.Random(seed)
    pillars = [x["name"] for x in p["pillars"] for _ in range(x["weight"])]
    start = start or dt.date.today() + dt.timedelta(days=1)
    items = []
    for d in range(days):
        for k in range(per_day):
            pil = rnd.choice(pillars)
            scene = rnd.choice(p["scenes"][pil])
            when = dt.datetime.combine(start + dt.timedelta(days=d), dt.time(18 + k * 2, 0))
            items.append({
                "date": when.isoformat(), "pillar": pil, "scene": scene,
                "prompt": f"photo of {p['character']}, {scene}, natural light, candid, detailed skin texture, instagram lifestyle photography",
                "caption": f"{scene.capitalize()} ✨\n\n{p['disclosure_caption']}\n\n" + " ".join(p["base_hashtags"]),
                "status": "planned",
            })
    return items

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    json.dump(make_plan(n), open("plan.json", "w"), ensure_ascii=False, indent=2)
    print(f"plan.json con {n} publicaciones")
