"""Draws the DVD Restore icon (the rainbow disc of the sidebar logo) -> app/icon.png (512) and app/icon.ico."""
import math
import os

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOPS = ["#7c93ff", "#c084fc", "#f472b6", "#fbbf24", "#34d399", "#38bdf8", "#7c93ff"]   # styles.css .logo
HOLE = (18, 21, 27)                                                                   # --rail


def rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def colour(t):
    t = t % 1.0
    n = len(STOPS) - 1
    i = min(int(t * n), n - 1)
    f = t * n - i
    a, b = rgb(STOPS[i]), rgb(STOPS[i + 1])
    return tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))


def draw(size):
    s = size * 4                                                  # supersampled, then reduced (smooth edges)
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    px = img.load()
    c = (s - 1) / 2
    r_out, r_hole = s * 0.47, s * 0.47 * 0.36
    for y in range(s):
        for x in range(s):
            dx, dy = x - c, y - c
            d = math.hypot(dx, dy)
            if d > r_out:
                continue
            if d < r_hole:
                px[x, y] = HOLE + (255,)
                continue
            ang = (math.degrees(math.atan2(dx, -dy)) - 200) % 360 / 360       # conic-gradient(from 200deg, ...)
            px[x, y] = colour(ang) + (255,)
    ImageDraw.Draw(img).ellipse([c - r_hole * 0.42, c - r_hole * 0.42, c + r_hole * 0.42, c + r_hole * 0.42],
                                fill=(0, 0, 0, 0))                # the disc's centre hole
    return img.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    big = draw(256)
    big.resize((512, 512), Image.LANCZOS).save(os.path.join(ROOT, "app", "icon.png"))
    big.save(os.path.join(ROOT, "app", "icon.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("app/icon.png, app/icon.ico")
