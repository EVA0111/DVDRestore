"""Video scopes for the Colour step, drawn from the previewed (After) frame: waveform, RGB parade,
vectorscope and histogram. Input: HxWx3 uint8 RGB (the preview frame). Output: PNG bytes."""
import io
import math

import numpy as np
from PIL import Image, ImageDraw

BG = (14, 16, 20)


def _png(img):
    b = io.BytesIO()
    img.save(b, "PNG", compress_level=1)
    return b.getvalue()


def _wave(ch, cols=256, rows=256):
    """2-D histogram: column position x value (0 top = 255)."""
    h, w = ch.shape
    xi = (np.arange(w) * cols // w)[None, :].repeat(h, 0).ravel()
    v = ch.ravel().astype(np.int64)
    hist = np.zeros((rows, cols), np.float64)
    np.add.at(hist, (255 - v, xi), 1)
    hist = np.log1p(hist) / max(np.log1p(hist).max(), 1e-6)
    return hist


def _grid(d, w, h, labels=True):
    for v in (0, 64, 128, 191, 255):
        y = int((255 - v) / 255 * (h - 1))
        d.line([(0, y), (w, y)], fill=(45, 50, 62))
    if labels:
        for v, t in ((235, "100"), (16, "0")):
            y = int((255 - v) / 255 * (h - 1))
            d.line([(0, y), (w, y)], fill=(90, 70, 40))
            d.text((2, y - 11 if v > 128 else y + 1), t, fill=(150, 120, 70))


def waveform(rgb):
    y = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).clip(0, 255)
    h = _wave(y, 320, 256)
    img = np.zeros((256, 320, 3), np.float32) + BG
    img += (h[..., None] * np.array([90, 230, 140]))
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    _grid(ImageDraw.Draw(im), 320, 256)
    return _png(im)


def parade(rgb):
    cols = []
    for c, col in enumerate(((255, 80, 80), (80, 230, 110), (90, 140, 255))):
        h = _wave(rgb[..., c].astype(np.float32), 160, 256)
        img = np.zeros((256, 160, 3), np.float32) + BG + h[..., None] * np.array(col)
        cols.append(img)
        cols.append(np.zeros((256, 4, 3), np.float32) + (30, 33, 40))
    im = Image.fromarray(np.hstack(cols[:-1]).clip(0, 255).astype(np.uint8))
    _grid(ImageDraw.Draw(im), im.width, 256)
    return _png(im)


def vectorscope(rgb, size=256):
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    half = size / 2
    sc = half / 128 * 0.95
    x = np.clip(half + cb * sc, 0, size - 1).astype(np.int64).ravel()
    y = np.clip(half - cr * sc, 0, size - 1).astype(np.int64).ravel()
    hist = np.zeros((size, size), np.float64)
    np.add.at(hist, (y, x), 1)
    hist = np.log1p(hist) / max(np.log1p(hist).max(), 1e-6)
    # colour each dot by where it is (hue), like a real vectorscope
    yy, xx = np.mgrid[0:size, 0:size]
    u, v = (xx - half) / (half * 0.95) * 128, (half - yy) / (half * 0.95) * 128
    tint = np.dstack([128 + 1.402 * v, 128 - 0.344 * u - 0.714 * v, 128 + 1.772 * u]).clip(40, 255)
    img = np.zeros((size, size, 3), np.float32) + BG + hist[..., None] * tint
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for rr in (0.25, 0.5, 0.75, 1.0):
        q = half * 0.95 * rr
        d.ellipse([half - q, half - q, half + q, half + q], outline=(45, 50, 62))
    d.line([(half, 0), (half, size)], fill=(40, 44, 54)); d.line([(0, half), (size, half)], fill=(40, 44, 54))
    # 75 % colour bar targets and the skin tone line
    for name, (R, G, B) in (("R", (191, 0, 0)), ("Mg", (191, 0, 191)), ("B", (0, 0, 191)), ("Cy", (0, 191, 191)),
                            ("G", (0, 191, 0)), ("Yl", (191, 191, 0))):
        tcb = -0.168736 * R - 0.331264 * G + 0.5 * B
        tcr = 0.5 * R - 0.418688 * G - 0.081312 * B
        tx, ty = half + tcb * sc, half - tcr * sc
        d.rectangle([tx - 5, ty - 5, tx + 5, ty + 5], outline=(120, 120, 130))
        d.text((tx + 7, ty - 6), name, fill=(150, 150, 160))
    ang = math.radians(123)
    d.line([(half, half), (half + math.cos(ang) * half * 0.95, half - math.sin(ang) * half * 0.95)], fill=(170, 120, 90))
    d.text((6, size - 14), "skin line", fill=(170, 120, 90))
    return _png(im)


def histogram(rgb, w=320, h=200):
    img = np.zeros((h, w, 3), np.float32) + BG
    for c, col in enumerate(((255, 70, 70), (70, 220, 100), (80, 130, 255))):
        cnt = np.bincount(rgb[..., c].ravel(), minlength=256).astype(np.float64)
        cnt = np.sqrt(cnt) / max(np.sqrt(cnt).max(), 1e-6)
        xs = (np.arange(256) * w / 256).astype(int)
        for i in range(256):
            top = int(h - cnt[i] * (h - 4))
            img[top:, xs[i]:xs[i] + 2] += np.array(col) * 0.45
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for v in (16, 235):
        x = int(v * w / 256); d.line([(x, 0), (x, h)], fill=(90, 70, 40))
    return _png(im)


SCOPES = {"waveform": waveform, "parade": parade, "vectorscope": vectorscope, "histogram": histogram}


def draw(kind, rgb):
    return SCOPES.get(kind, waveform)(rgb)
