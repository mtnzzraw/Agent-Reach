"""Junta las imágenes de dataset/ en una hoja numerada: dataset/hoja_01.jpg (y hoja_02... si hay muchas).
Uso: python contact_sheet.py   -> sube esas hojas al chat para decidir cuáles se parecen."""
from pathlib import Path
from PIL import Image, ImageDraw

COLS, ROWS, THUMB = 5, 3, 320   # 15 fotos por hoja

def main():
    files = sorted(p for p in Path("dataset").glob("lia_*.png"))
    if not files:
        raise SystemExit("No hay imágenes en dataset/")
    per = COLS * ROWS
    for n in range(0, len(files), per):
        chunk = files[n:n + per]
        sheet = Image.new("RGB", (COLS * THUMB, ROWS * THUMB), "white")
        d = ImageDraw.Draw(sheet)
        for i, f in enumerate(chunk):
            im = Image.open(f).convert("RGB")
            im.thumbnail((THUMB, THUMB))
            x, y = (i % COLS) * THUMB, (i // COLS) * THUMB
            sheet.paste(im, (x, y))
            d.rectangle([x, y, x + 46, y + 26], fill="black")
            d.text((x + 6, y + 6), f.stem.split("_")[1], fill="white")   # número de la foto
        out = Path("dataset") / f"hoja_{n // per + 1:02d}.jpg"
        sheet.save(out, quality=85)
        print("creada", out)

if __name__ == "__main__":
    main()
