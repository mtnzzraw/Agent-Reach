"""Convierte una imagen 9:16 en un Reel de 6 s con ffmpeg: zoom lento + gancho escrito los primeros segundos.
Lleva pista de audio en silencio; la música se pone al subirlo (la API no permite audios en tendencia)."""
import os, re, subprocess, tempfile, textwrap

W, H, FPS, SECONDS, HOOK_SECONDS = 1080, 1920, 30, 6, 2.5
FONT = os.environ.get("REEL_FONT")   # ruta a un .ttf/.otf; sin ella ffmpeg usa la fuente por defecto del sistema

def _overlay_text(hook):
    # drawtext no pinta emojis ni parte líneas: se quitan y se envuelve a mano
    plain = re.sub(r"[^\w\s¿?¡!.,:;'’“”\"()€%/+-]", "", hook).strip()
    return "\n".join(textwrap.wrap(plain, 20))

def make_reel(image, hook, out):
    frames = FPS * SECONDS
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(_overlay_text(hook))
    font = f":fontfile='{FONT}'" if FONT else ""
    vf = (f"scale={W * 2}:{H * 2}:force_original_aspect_ratio=increase,crop={W * 2}:{H * 2},"
          f"zoompan=z='min(zoom+0.0007,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}:fps={FPS},"
          f"drawtext=textfile='{f.name}'{font}:fontsize=72:fontcolor=white:borderw=5:bordercolor=black@0.55:"
          f"line_spacing=12:x=(w-text_w)/2:y=h*0.16:enable='lt(t,{HOOK_SECONDS})'")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", image,
                        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-vf", vf, "-t", str(SECONDS),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-shortest",
                        "-movflags", "+faststart", out], check=True)
    finally:
        os.unlink(f.name)
    return out
