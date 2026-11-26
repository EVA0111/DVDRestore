"""Auto colour (Colour step): measures a source and solves the grade that corrects its transfer.

What a transfer gets wrong, and what is corrected (everything else is the art and is left alone):
  levels      washed-out black (lifted), detail under legal black, whites too dim or above legal white
  white cast  painted white (paper, teeth, eyes, white clothes) tinted - per-channel gain of the transfer
  black cast  line art / black tinted - per-channel offset of the transfer (shadows go red, blue...)
  fade        colour much weaker than any well-transferred source (faded film / weak chroma gain)
  gamut       colour stored beyond what a screen can show (squeezed in instead of cut off)

Every correction needs evidence that holds over the source, not one shot: a cast must be the same in a good
share of the frames that show whites (or blacks) - a cast that changes from shot to shot is the art (sunset,
night) and is left alone. A source that is already right comes back with the neutral grade.

The grade is solved with grade_np(), a numpy copy of vsgraph._grade_one (levels, gamma, saturation,
temperature / tint, lift wheel): the measured anchors are pushed through it and the settings adjusted until
they land where they should, so what was measured is what the export does.
"""
import numpy as np

# BT.601 (both DVD matrices use these coefficients)
KR, KB = 0.299, 0.114
M_RGB = np.array([[1, 0, 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]])     # (Y, Cb, Cr) -> RGB
M_YUV = np.linalg.inv(M_RGB)

# ---------------------------------------------------------------- thresholds (bench/bench_colour.py)
WHITE_BAND = 0.15              # white candidates: within this of the frame's top tones ...
WHITE_CHROMA = 0.12            # ... and |Cb, Cr| below this (12 %: a strong transfer cast still passes)
BLACK_MAX = 0.16               # black anchor: the frame's darkest tones, at most this bright
CAST_MIN = 1.5 / 224           # smaller casts are left alone (1.5 8-bit levels)
SHARE_MIN = 0.25               # anchors in at least this share of the frames
CONSIST_MIN = 0.25             # the same cast in at least this (brightness-weighted) share of the frames with anchors
SAT_REF = 0.105                # p90 colour strength of well-transferred sources: lowest of the reference set (Golden Boy)
SAT_FADED = 0.75               # a source below SAT_FADED * SAT_REF is faded ...
SAT_MAX = 1.6                  # ... and brought up to SAT_REF, at most this much
SAT_PEAK_FADED = 0.16          # strongest colours (p99.5 |Cb, Cr|, 90th pct of frames) below this = faded: every good
                               # disc measured reaches 0.17-0.43; a 0.6 fade of a pastel show lands at 0.10-0.15
SAT_PEAK_REF = 0.20            # ... brought up to this
# colour prior (no reference) - gates / gains / bias from the held-out evaluation (models/colourprior_train.py eval)
PRIOR_SHOTS = 4                # judged over at least this many different shots
PRIOR_SAT_BIAS = -0.023        # log colour scale the prior says about good transfers on average (held-out median)
PRIOR_SAT_GATE = 0.2           # |log| beyond this is corrected: 12 % of good discs, fades 0.6 / 0.7 / 0.8 caught 94 / 72 / 44 %
PRIOR_SAT_GATE_DOWN = 0.3       # less colour: a stricter bar (held out: 0 % of good discs desaturated; 25 % of 1.3x oversaturation caught)
PRIOR_SAT_GAIN = 1.0           # ... the part beyond the gate times this
PRIOR_GAMMA_GATE = 0.08        # 12 % of good discs; gamma 0.8 / 1.25 caught ~100 %, 0.9 / 1.12 72 %
PRIOR_GAMMA_GAIN = 1.0
PRIOR_HUE_GATE = 3.0           # 9 % of good discs; +-15 deg caught 97 %, +-8 deg 56-75 %
PRIOR_HUE_GAIN = 1.0
REF_PX = 3000                  # reference match: pixels sampled per frame
REF_CMIN = 0.03                # ... colourless below this |Cb, Cr| (reference side; the source's follows the fade)
REF_TONE_Q = list(range(10, 91, 10))   # ... brightness percentiles per hue family for the tone fit
REF_TONE_STAT = "mean"          # ... "mean" or "median" of the per-family errors
REF_TONE_GAIN = 0.7            # ... midtones / shadow lift only set when they cut the brightness mismatch by 30 %+
REF_HUE_MAX = 30               # ... hue rotation searched within +-30 degrees
REF_HUE_MIN = 3                # ... and set from 3 degrees
REF_HUE_GAIN = 1.05            # ... when it lines the colours up clearly better than no rotation
REF_DARK_MIN = 1.1             # ... shadow colour set when the darks lost 10 %+ more colour than the mid tones (same scenes only)
REF_METHOD = "families"        # reference match: "register" (3-D histogram overlap, joint fit) or "families"
REG_PX = 40000                 # ... pixels per side for the registration
REG_ROUNDS = 12                # ... coordinate-descent rounds
REG_GAIN = 1.005               # ... settings only set when they raise the overlap by 0.5 %+
REF_NMIN = 300                 # ... pixels a hue family needs on both sides
REF_SAME_COR = 0.9             # ... thumbnails this alike show the same picture
REF_SAME_SHARE = 0.5           # ... "same scenes" when this share of frames has its twin on the other side
REF_MIN_OTHER = 1.2            # ... other scenes: saturation only matched when 20 %+ off (scene noise ~15 %)
REF_MIN = 1.05                 # ... ratios closer to 1 than this leave saturation alone
REF_MAX = 2.0                  # ... at most this much stronger (or 1 / this weaker)
SAT_SHOTS = 8                  # fade is judged against other sources, so only over several different shots
MONO_SPREAD_CLIP = 0.05        # ... in a source of fewer than SAT_SHOTS shots
MONO_SPREAD = 0.035            # colour spread (p90 around the median) below this = a monochrome / tinted look


# ---------------------------------------------------------------- grade, numpy copy
def grade_np(y, u, v, g):
    """Full-range Y 0..1, Cb / Cr -0.5..0.5 (arrays of any shape) -> the same after vsgraph._grade_one's levels,
    contrast, gamma, shadow lift, saturation, shadow colour, temperature / tint, hue, and the lift / gain / gamma wheels.
    (gamut compression, the gamma / gain wheels, curves and LUTs are not copied - auto colour does not set them)"""
    bp, wp = (float(g.get("black", 16)) - 16) / 219, (float(g.get("white", 235)) - 16) / 219
    wp = max(wp, bp + 0.2)
    con, gam, sh = float(g.get("contrast", 1)), max(0.2, float(g.get("gamma", 1))), float(g.get("shadows", 0))
    lin = ((y - bp) / (wp - bp) - 0.5) * con + 0.5
    T = np.clip(lin, 1e-6, 1)
    yn = T ** (1 / gam) + sh * 0.12 * T * (1 - T) ** 5 * 14.93 + (lin - T)
    lvl = con / (wp - bp)
    gain = np.clip(np.maximum(yn, 0.03) / np.maximum(lin, 0.03), 0.5, 3) * lvl * float(g.get("sat", 1)) * \
        (1 + float(g.get("dark_sat", 0)) * np.maximum(0, 1 - yn / 0.4) ** 2)
    t, ti = float(g.get("temp", 0)) * 0.05, float(g.get("tint", 0)) * 0.035
    un = u * gain + np.maximum(yn, 0) * (ti - t)
    vn = v * gain + np.maximum(yn, 0) * (ti + t)
    hue, vib = float(g.get("hue", 0)), float(g.get("vibrance", 0))
    if abs(hue) > 1e-6 or abs(vib) > 1e-6:
        cs, sn = np.cos(np.radians(hue)), np.sin(np.radians(hue))
        fac = (1 - np.minimum(1, np.hypot(un, vn) / 0.25)) * vib + 1
        un, vn = (un * cs - vn * sn) * fac, (un * sn + vn * cs) * fac
    wh = [np.array([float(a) for a in (g.get(k) or [0, 0, 0])]) for k in ("lift", "gain_w", "gamma_w")]
    if any(np.abs(x).max() > 1e-9 for x in wh):
        rgb = np.clip(np.stack([yn, un, vn], -1) @ M_RGB.T, 0, 1)
        L, G, Mg = (M_RGB @ wh[0]) * 0.25, (M_RGB @ wh[1]) * 0.5, (M_RGB @ wh[2]) * 0.5
        rgb = np.clip((rgb + L * (1 - rgb)) * (1 + G), 0, 1) ** (1 / np.maximum(0.1, 1 + Mg))
        yn, un, vn = np.moveaxis(rgb @ M_YUV.T, -1, 0)
    return yn, un, vn


# ---------------------------------------------------------------- per frame
def _full(Y, U, V):
    """8-bit 4:2:0 (or 4:4:4) limited -> full-range Y, Cb, Cr at chroma resolution (luma averaged down)."""
    Y = Y.astype(np.float32)
    ry, rx = Y.shape[0] // U.shape[0], Y.shape[1] // U.shape[1]
    if ry > 1 or rx > 1:
        h, w = U.shape[0] * ry, U.shape[1] * rx
        Y = Y[:h, :w].reshape(U.shape[0], ry, U.shape[1], rx).mean((1, 3))
    return (Y - 16) / 219, (U.astype(np.float32) - 128) / 224, (V.astype(np.float32) - 128) / 224


def _peak(u, v, rng):
    """Densest Cb / Cr of a set of pixels (0.25-level bins, lightly smoothed): a mean would be pulled by pale skin."""
    from scipy.ndimage import gaussian_filter
    r = rng / 224
    h, ue, ve = np.histogram2d(u, v, bins=8 * rng, range=[[-r, r], [-r, r]])
    i, j = np.unravel_index(gaussian_filter(h, 2.0).argmax(), h.shape)
    return float(ue[i] + ue[i + 1]) / 2, float(ve[j] + ve[j + 1]) / 2


def _small(y, u, v):
    """The whole frame at 240 x 320 (what the colour prior looks at), float16 Y, Cb, Cr."""
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(np.asarray(a, np.float32), "F").resize((320, 240), Image.BILINEAR))
                     for a in (y, u, v)], -1).astype(np.float16)


def prior_check(stats, g):
    """Ask the colour prior (colourprior.py) about the source as auto colour has graded it so far (levels, casts).
    Returns (colour scale, gamma, hue it sees as the fault, per-frame spread) or None."""
    try:
        import colourprior
        if not colourprior.available():
            return None
        sm = np.stack([s["small"] for s in stats if "small" in s]).astype(np.float32)
        yn, un, vn = grade_np(sm[..., 0], sm[..., 1], sm[..., 2], dict(g, sat=1.0, hue=0.0, gamma=1.0, contrast=1.0,
                                                                       shadows=0.0, dark_sat=0.0, vibrance=0.0))
        rgb = np.clip(np.stack([yn, un, vn], -1) @ M_RGB.T, 0, 1)
        return colourprior.predict((rgb * 255 + 0.5).astype(np.uint8))
    except Exception as e:                                # noqa: BLE001
        import sys
        print(f"colour prior not usable: {e!r}", file=sys.stderr)
        return None


def frame_stats(Y, U, V):
    """Everything auto colour needs from one frame (middle of the frame, so borders and logos count less)."""
    y, u, v = _full(Y, U, V)
    H, W = y.shape
    small = _small(y, u, v)
    sl = (slice(int(H * 0.1), int(H * 0.9)), slice(int(W * 0.1), int(W * 0.9)))
    y, u, v = y[sl], u[sl], v[sl]
    c = np.hypot(u, v)
    inner = (slice(int(H * 0.125), int(H * 0.625)), slice(int(W * 0.125), int(W * 0.625)))   # middle 60 % of the full frame
    yc = y[inner]
    s = {"mean": float(yc.mean()), "lo": float(np.percentile(yc, 0.1)), "hi": float(np.percentile(yc, 99.9)),
         "dark": float(np.mean(yc < 6 / 219))}
    # whites: the frame's top tones that are weakly coloured (|Cb, Cr| < 12 %) and not clipped (a clipped
    # channel hides the cast). Their colour per unit of brightness (a transfer's gain cast grows with brightness)
    # as a histogram weighted by brightness^8 - painted white is the brightest paint, pale hair / skin / cream
    # sit lower. solve() adds the frames' histograms up.
    top = float(np.percentile(y, 99.5))
    rgb = np.stack([y, u, v], -1) @ M_RGB.T
    mx, mn = rgb.max(-1), rgb.min(-1)
    m = (y > max(0.45, top - WHITE_BAND)) & (c < WHITE_CHROMA) & (mx < 0.985)
    if m.sum() >= 60:
        yy = y[m]
        nu, nv = u[m] / yy * 224, v[m] / yy * 224
        wt = (yy / top) ** 8
        h = np.histogram2d(nu, nv, bins=HB, range=[[-HR, HR], [-HR, HR]], weights=wt)[0]
        s["white_h"] = (h / max(1e-9, h.sum())).astype(np.float32)
        s["white_y"] = float(np.average(yy, weights=wt))
    # blacks: the darkest tones (line art, deepest shadow), weakly coloured
    lo = float(np.percentile(y, 0.5))
    m = (y <= max(lo + 0.03, 0.02)) & (y < BLACK_MAX) & (c < 0.08)
    if m.sum() >= 60:
        pu, pv = _peak(u[m], v[m], 15)
        s["black"] = (float(np.median(y[m])), pu, pv)
    # colour strength: p90 of |Cb, Cr| over the mid tones, relative to brightness spread (a dim fade counts too)
    # tint: how much of the colour is one shared offset (a tinted monochrome look - a TV screen, a flashback in
    # sepia) rather than different colours: |median| / p90 distance from it
    mid = (y > 0.12) & (y < 0.92)
    if mid.sum() > 500:
        s["sat"] = float(np.percentile(c[mid], 90))
        s["sat_hi"] = float(np.percentile(c[mid], 99.5))
        # pixel sample (reference match: graded later with the solved levels and wheels, compared per hue family)
        idx = np.arange(y.size)                     # every brightness: a tone fault moves pixels across any band
        idx = idx[np.linspace(0, len(idx) - 1, min(len(idx), REF_PX)).astype(int)]
        s["px"] = np.stack([y.ravel()[idx], u.ravel()[idx], v.ravel()[idx]], -1).astype(np.float32)
        mu, mv = float(np.median(u[mid])), float(np.median(v[mid]))
        s["spread"] = float(np.percentile(np.hypot(u[mid] - mu, v[mid] - mv), 90))
        s["tint"] = float(np.hypot(mu, mv))
    # thumbnail (to tell shots apart)
    h8, w8 = y.shape[0] // 12, y.shape[1] // 16
    s["thumb"] = np.stack([a[:12 * h8, :16 * w8].reshape(12, h8, 16, w8).mean((1, 3)) for a in (y, u, v)], -1)
    s["small"] = small
    # colour beyond the screen range, and colour sitting flat at the limit
    s["gamut"] = 100 * float(np.mean((mx > 1.04) | (mn < -0.04)))
    s["flat"] = 100 * float(np.mean(((mx >= 0.985) | ((mn <= 0.015) & (mx > 0.25))) & (mx - mn > 0.15)))
    return s


# ---------------------------------------------------------------- over the source
def _consistent_cast(pts, n_frames):
    """pts: [(Y, Cb, Cr)] of one anchor over the frames. The cast when it is the transfer's (enough frames, the
    same cast in enough of them), else None. Returns (Y, Cb, Cr, share consistent)."""
    if not pts or len(pts) < max(1, min(4, n_frames), SHARE_MIN * n_frames):
        return None, 0.0
    a = np.array(pts)
    # the densest cluster, not the median: frames without real whites add pale paint that scatters around and
    # drags a median off. Chroma per unit of brightness (a transfer's gain cast grows with brightness); ties go
    # to the brighter anchors.
    k = np.maximum(a[:, 0], 0.05)
    cu, cv = a[:, 1] / k, a[:, 2] / k
    best, bi = None, 0
    for i in range(len(a)):
        tol = max(2.5 / 224 / k[i], 0.25 * np.hypot(cu[i], cv[i]))
        near = np.hypot(cu - cu[i], cv - cv[i]) < tol
        key = (int(near.sum()), float(a[near, 0].mean()))
        if best is None or key > best:
            best, bi, sel = key, i, near
    cons = float(np.mean(sel))
    return (float(np.median(a[sel, 0])), float(np.median(a[sel, 1])), float(np.median(a[sel, 2]))), cons


WB_RAD, WB_FRAME, WB_ZERO = 5.0, 0.15, 0.6     # bench/wb_eval.py sweep 2026-11-24
HB, HR = 80, 40.0               # white histogram: bins, +-range (8-bit levels of Cb / Cr per unit of brightness)


def _white_cast(stats, n_frames):
    """The brightest consistent white over the frames: each frame's histogram counts with weight
    (its white brightness / the brightest frames')^8, so frames whose top tones are real whites outvote frames
    whose top tones are pale paint (Lodoss: 3 frames of grey whites at Y 0.93 vs 40 of blonde hair at 0.63 -
    0.05 each). Returns ((Y, Cb, Cr) or None, weighted share of frames that agree, frames with whites)."""
    from scipy.ndimage import gaussian_filter
    F = [s for s in stats if "white_h" in s]
    if not F or len(F) < max(1, min(4, n_frames), SHARE_MIN * n_frames):
        return None, 0.0, len(F)
    ys = np.array([s["white_y"] for s in F])
    wt = np.clip(ys / max(0.05, float(ys.max())), 0, 1) ** 8
    H = sum(w * s["white_h"] for w, s in zip(wt, F))
    i, j = np.unravel_index(gaussian_filter(H, 1.5).argmax(), H.shape)
    step = 2 * HR / HB
    cen = (np.arange(HB) + 0.5) * step - HR
    pu, pv = float(cen[i]), float(cen[j])
    # refine: centroid within WB_RAD levels; a frame agrees when WB_FRAME of its white mass lies within
    # max(WB_RAD, 30 %) of the peak (painted "white" itself varies by +-5 levels from cel to cel)
    du, dv = np.meshgrid(cen - pu, cen - pv, indexing="ij")
    r = np.hypot(du, dv) < WB_RAD
    if H[r].sum() > 0:
        pu, pv = pu + float((H * du)[r].sum() / H[r].sum()), pv + float((H * dv)[r].sum() / H[r].sum())
    du, dv = np.meshgrid(cen - pu, cen - pv, indexing="ij")
    near = np.hypot(du, dv) < max(WB_RAD, 0.3 * np.hypot(pu, pv))
    agree = np.array([float(s["white_h"][near].sum()) >= WB_FRAME for s in F])
    cons = float(wt[agree].sum() / max(1e-9, wt.sum()))
    # grey whites that carry as much weight as the cast cluster: the source has real whites, the cluster is paint
    zero = np.hypot(*np.meshgrid(cen, cen, indexing="ij")) < 3.0
    if np.hypot(pu, pv) > 2 * WB_RAD and H[zero].sum() >= WB_ZERO * H[near].sum():
        cons = 0.0
    y = float(np.median(ys[agree])) if agree.any() else float(np.median(ys))
    return (y, pu * y / 224, pv * y / 224), cons, len(F)


def corrected_cast(r):
    """The white cast (Cb, Cr per unit of brightness, 8-bit levels) that a solve() result corrects."""
    gw = np.array((r.get("grade") or {}).get("gain_w") or [0, 0, 0], dtype=float)
    P = 1 + (M_RGB @ gw) * 0.5
    ycc = (1 / P) @ M_YUV.T
    return float(ycc[1] / ycc[0] * 224), float(ycc[2] / ycc[0] * 224)


def count_shots(stats, tol=0.05):
    """Number of different shots among the sampled frames (thumbnails that differ by more than tol on average)."""
    reps = []
    for s in stats:
        t = s.get("thumb")
        if t is not None and all(float(np.abs(t - r).mean()) > tol for r in reps):
            reps.append(t)
    return len(reps)


def _graded_px(stats, g):
    px = [s["px"] for s in stats if "px" in s]
    if not px:
        return None
    a = np.concatenate(px).astype(np.float64)
    yn, un, vn = grade_np(a[:, 0], a[:, 1], a[:, 2], dict(g, sat=1.0, vibrance=0.0, dark_sat=0.0))
    return yn, np.hypot(un, vn), np.degrees(np.arctan2(vn, un)) % 360


def ref_ratio(stats, g, ref_stats, gr, ylo=0.15, yhi=0.92):
    """How much stronger the reference's colour is than the source's, both after their own grade (levels, cast
    wheels). Per 15-degree hue family with enough pixels on both sides: the ratio of the 50 / 75 / 90th percentile
    of |Cb, Cr|; the weighted median over families. Returns (ratio or None, families used)."""
    A, R = _graded_px(stats, g), _graded_px(ref_stats, gr)
    if A is None or R is None:
        return None, 0
    ratio, logs, wts = 1.0, [], []
    for _ in range(2):                                   # colourless threshold follows the fade found
        logs, wts = [], []
        ma = (A[0] > ylo) & (A[0] < yhi) & (A[1] > REF_CMIN / ratio)
        mr = (R[0] > ylo) & (R[0] < yhi) & (R[1] > REF_CMIN)
        for b in range(24):
            lo, hi = b * 15, (b + 1) * 15
            sa = ma & (A[2] >= lo) & (A[2] < hi)
            sr = mr & (R[2] >= lo) & (R[2] < hi)
            n = min(int(sa.sum()), int(sr.sum()))
            if n < REF_NMIN:
                continue
            qa, qr = np.percentile(A[1][sa], [50, 75, 90]), np.percentile(R[1][sr], [50, 75, 90])
            logs += list(np.log(qr / np.maximum(qa, 1e-4)))
            wts += [np.sqrt(n)] * 3
        if not logs:
            return None, 0
        o = np.argsort(logs)
        cw = np.cumsum(np.array(wts)[o])
        ratio = float(np.exp(np.array(logs)[o][np.searchsorted(cw, cw[-1] / 2)]))
    return ratio, len(logs) // 3


def thumb_z(y):
    """Brightness structure of a picture for finding the same frame in another release: middle 80 % (crops and
    borders differ between releases), 12 x 16 block means, z-scored (a tone, colour or fade fault does not change
    it). y: 2-D array of any size."""
    y = np.asarray(y, dtype=np.float32)
    H, W = y.shape
    y = y[int(H * 0.1):int(H * 0.9), int(W * 0.1):int(W * 0.9)]
    t = np.array([[b.mean() for b in np.array_split(r, 16, axis=1)] for r in np.array_split(y, 12, axis=0)]).ravel()
    return (t - t.mean()) / max(1e-6, t.std())


def find_twins(index, ref_thumbs):
    """index: (N, 192) thumb_z of the source every few frames; ref_thumbs: (M, 192). For every reference frame the
    best index position and its correlation; positions under REF_SAME_COR are -1."""
    cor = np.asarray(ref_thumbs) @ np.asarray(index).T / index.shape[1]
    best = cor.argmax(1)
    return np.where(cor.max(1) >= REF_SAME_COR, best, -1), cor.max(1)


def content_match(stats, ref_stats):
    """Share of frames (of the smaller side) that show the same picture as some frame of the other side: thumbnail
    brightness structure (z-scored, so a tone, colour or fade fault does not hide it), correlation >= REF_SAME_COR.
    A better release of the same episode, or screenshots of the same moments: ~1. Another episode: low (the
    opening, eye-catches and recaps match, the rest does not)."""
    def z(st):
        out = []
        for s_ in st:
            t = s_.get("thumb")
            if t is None:
                continue
            y = t[..., 0].ravel()
            out.append((y - y.mean()) / max(1e-6, y.std()))
        return np.array(out)
    a, b = z(stats), z(ref_stats)
    if not len(a) or not len(b):
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    cor = a @ b.T / a.shape[1]
    return float(np.mean(cor.max(1) >= REF_SAME_COR))


def tone_curve(x, gam, sh, con=1.0):
    """The Colour step's tone curve on brightness already through levels (vsgraph._grade_one: contrast around mid
    grey, gamma, shadow lift)."""
    lin = (x - 0.5) * con + 0.5
    T = np.clip(lin, 1e-6, 1)
    return T ** (1 / gam) + sh * 0.12 * T * (1 - T) ** 5 * 14.93 + (lin - T)


def ref_tone(stats, g, ref_stats, gr):
    """Midtones (gamma) and Shadow lift that make the source's brightness match the reference's: per hue family
    (and the colourless pixels as one more family) the 10 / 25 / 50 / 75 / 90th percentile of brightness, both
    graded with their own levels and cast wheels; the curve that maps the source's onto the reference's, by grid
    search (mean absolute error, families weighted by sqrt(pixels)). Returns (gamma, shadows, error before, after)
    or None."""
    A, R = _graded_px(stats, g), _graded_px(ref_stats, gr)
    if A is None or R is None:
        return None
    xs, xr, w = [], [], []
    fam_a = np.where(A[1] > REF_CMIN, (A[2] // 15).astype(int), -1)
    fam_r = np.where(R[1] > REF_CMIN, (R[2] // 15).astype(int), -1)
    for b in range(-1, 24):
        sa, sr = A[0][fam_a == b], R[0][fam_r == b]
        n = min(len(sa), len(sr))
        if n < REF_NMIN:
            continue
        q = REF_TONE_Q
        xs += list(np.percentile(sa, q)); xr += list(np.percentile(sr, q)); w += [np.sqrt(n)] * len(q)
    if len(xs) < 15:
        return None
    xs, xr, w = np.array(xs), np.array(xr), np.array(w)
    o = np.argsort(xs)
    xs, xr, w = xs[o], xr[o], w[o]

    def wmed(e):
        """weighted mean (REF_TONE_STAT "mean") or weighted median (robust) of |error| along the last axis"""
        if REF_TONE_STAT == "mean":
            return np.sum(w * e, -1) / w.sum()
        i = np.argsort(e, -1)
        cw = np.cumsum(w[i], -1)
        k = (cw < cw[..., -1:] / 2).sum(-1, keepdims=True)
        return np.take_along_axis(np.take_along_axis(e, i, -1), k, -1)[..., 0]
    e0 = float(wmed(np.abs(tone_curve(xs, 1.0, 0.0) - xr)))
    G = np.arange(0.6, 1.801, 0.02)[:, None, None]
    H = np.arange(-0.5, 1.501, 0.05)[None, :, None]
    best = (e0, 1.0, 0.0, 1.0)
    for con in np.arange(0.8, 1.251, 0.025):
        E = wmed(np.abs(tone_curve(xs[None, None, :], G, H, con) - xr))
        i, j = np.unravel_index(E.argmin(), E.shape)
        if E[i, j] < best[0] - 1e-9:
            best = (float(E[i, j]), float(G[i, 0, 0]), float(H[0, j, 0]), float(con))
    return best[1], best[2], best[3], e0, best[0]


def ref_hue(stats, g, ref_stats, gr):
    """Rotation (degrees) that lines the source's colours up with the reference's - an NTSC phase error turns
    every hue by the same angle. Colour-weighted hue histograms (1 degree), circular cross-correlation within
    +-REF_HUE_MAX. Returns (rotation for the Hue control, correlation gain over no rotation) or None."""
    A, R = _graded_px(stats, g), _graded_px(ref_stats, gr)
    if A is None or R is None:
        return None
    def hist(P):
        m = (P[0] > 0.15) & (P[0] < 0.92) & (P[1] > REF_CMIN)
        h = np.bincount((P[2][m].astype(int)) % 360, weights=P[1][m], minlength=360)
        h = np.convolve(np.concatenate([h[-3:], h, h[:3]]), np.ones(7) / 7, "valid")
        return h / max(1e-9, np.linalg.norm(h))
    ha, hr = hist(A), hist(R)
    shifts = np.arange(-REF_HUE_MAX, REF_HUE_MAX + 1)
    cor = np.array([float(np.dot(np.roll(ha, d), hr)) for d in shifts])
    d = int(shifts[cor.argmax()])
    return d, float(cor.max() / max(1e-9, cor[REF_HUE_MAX]))


REG_KEYS = ("sat", "hue", "contrast", "gamma", "shadows", "dark_sat")
REG_LO = np.array([0.5, -30.0, 0.8, 0.6, -0.5, 0.0])
REG_HI = np.array([2.0, 30.0, 1.25, 1.8, 1.5, 2.0])
REG_STEP = np.array([0.15, 6.0, 0.05, 0.1, 0.15, 0.3])


def _hist3(yn, un, vn):
    """Smoothed, normalised 3-D histogram of (brightness, Cb, Cr): every paint of a show is one peak."""
    from scipy.ndimage import gaussian_filter
    h, _ = np.histogramdd(np.stack([yn, un, vn], -1), bins=(32, 48, 48), range=((0, 1), (-0.3, 0.3), (-0.3, 0.3)))
    h = gaussian_filter(h, 1.0)
    return h / max(1e-12, h.sum())


def ref_register(stats, g, ref_stats, gr):
    """Saturation, hue, contrast, midtones, shadow lift and shadow colour that line the source's colours up with the
    reference's: both graded with their own levels and cast wheels, then the settings that maximise the overlap
    (Bhattacharyya coefficient) of their (brightness, Cb, Cr) histograms. The same show reuses the same paints, so
    their peaks line up when the settings are right; scenes only one side shows add mass that matches nothing and
    hardly moves the best fit. Coarse grid on saturation x hue, then coordinate descent with shrinking steps.
    Returns ({setting: value}, overlap before, after) or None."""
    pa = [s["px"] for s in stats if "px" in s]
    pr = [s["px"] for s in ref_stats if "px" in s]
    if not pa or not pr:
        return None
    a = np.concatenate(pa).astype(np.float64)
    r = np.concatenate(pr).astype(np.float64)
    if len(a) > REG_PX:
        a = a[np.linspace(0, len(a) - 1, REG_PX).astype(int)]
    if len(r) > REG_PX:
        r = r[np.linspace(0, len(r) - 1, REG_PX).astype(int)]
    hr = np.sqrt(_hist3(*grade_np(r[:, 0], r[:, 1], r[:, 2], dict(gr, vibrance=0.0))))

    def score(x):
        gg = dict(g, vibrance=0.0, **{k: float(v) for k, v in zip(REG_KEYS, x)})
        return float(np.sum(np.sqrt(_hist3(*grade_np(a[:, 0], a[:, 1], a[:, 2], gg))) * hr))
    x = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    s0 = best = score(x)
    for sat in np.arange(0.6, 1.81, 0.1):                # coarse: the two that move peaks furthest
        for hue in np.arange(-24, 25, 4):
            y = x.copy(); y[0], y[1] = sat, hue
            v = score(y)
            if v > best:
                best, bx = v, y
    if best > s0:
        x = bx
    step = REG_STEP.copy()
    for _ in range(REG_ROUNDS):
        moved = False
        for i in range(len(x)):
            for d in (step[i], -step[i]):
                y = x.copy(); y[i] = np.clip(y[i] + d, REG_LO[i], REG_HI[i])
                v = score(y)
                if v > best + 1e-6:
                    best, x, moved = v, y, True
        if not moved:
            step /= 2
    return dict(zip(REG_KEYS, (float(v) for v in x))), s0, best


def _round(v, step):
    return float(round(v / step) * step)


def solve(stats, ref_stats=None, pair_stats=None):
    """stats: frame_stats() of the sampled frames (frames that are nearly black are skipped by the caller).
    Returns {"report": {...}, "grade": {...}} - the grade only holds what auto colour sets."""
    n = len(stats)
    lo = np.array([s["lo"] for s in stats]) * 219 + 16
    hi = np.array([s["hi"] for s in stats]) * 219 + 16
    b, w = float(np.percentile(lo, 5)), float(np.percentile(hi, 95))
    dk = 100 * float(np.median([s["dark"] for s in stats]))
    rep = {"frames": n, "level_black": round(b, 1), "level_white": round(w, 1), "level_dark": round(dk, 1)}
    g = {}

    # --- black point: washed-out (nothing darker than 20 anywhere) or real picture under legal black
    bp = int(round(min(b, 40))) if b >= 19.5 else (int(round(b)) if (b <= 12 and dk >= 2.0) else 16)

    # --- whites: the brightest consistent cluster over the frames. Each frame's anchor counts with weight
    # (Y / the source's bright anchors)^8: frames whose top tones are real whites outvote frames whose top tones
    # are pale paint (Lodoss: 3 frames of grey whites at Y 1.0 vs ~15 of blonde hair at 0.7 - 0.06 each).
    white, wcons, nw = _white_cast(stats, n)
    rep.update(wb_frames=nw, wb_consistent=round(100 * wcons), wb_y=round(white[0], 3) if white else None)
    # a tinted monochrome look (one colour offset, hardly any other colour) is the art: no cast, no fade
    sp = [s["spread"] for s in stats if "spread" in s]
    shots = count_shots(stats)
    # (a whole episode is never one tinted look; a short clip of a few shots can be: TMNT's green-grey TV screen,
    # spread 0.048, every good disc 0.065+)
    mono = bool(sp) and float(np.median(sp)) < (MONO_SPREAD if shots >= SAT_SHOTS else MONO_SPREAD_CLIP)
    rep.update(monochrome=mono, shots=shots)
    rep["level_whites"] = round(16 + 219 * white[0], 1) if white is not None else None
    # --- white point: above legal white -> brought down; a source whose brightest tones never reach white
    # (dim / faded transfer: over the brightest sampled frames the top tone stays under 228) -> its brightest
    # tone becomes white, at most a gain of 1.25. Nothing new is clipped beyond the brightest 5 % of frames' top
    # 0.1 %. (Every episode has white somewhere: credits, highlights, paper - if none reaches 228 the transfer is dim.)
    if w >= 240:
        wp = int(round(min(w, 255)))
    elif w < 228:
        wp = int(round(max(w, bp + 219 / 1.25)))
    else:
        wp = 235
    g["black"], g["white"] = bp, wp

    # --- fade: colour much weaker than any well-transferred source (judged over several shots only)
    S = [s["sat"] for s in stats if "sat" in s]
    sat_p = float(np.median(S)) if S else None
    rep["sat_level"] = round(sat_p, 4) if sat_p is not None else None
    lvl = 219 / max(1, wp - bp)
    sat = 1.0
    SH = [s["sat_hi"] for s in stats if "sat_hi" in s]
    hi_p = float(np.percentile(SH, 90)) if SH else None
    rep["sat_peak"] = round(hi_p, 4) if hi_p is not None else None
    if sat_p is not None and hi_p is not None and not mono and shots >= SAT_SHOTS:
        f = 1.0
        if hi_p * lvl < SAT_PEAK_FADED:                   # not one strongly coloured thing in the source: faded
            f = SAT_PEAK_REF / max(1e-4, hi_p * lvl)
        if sat_p * lvl < SAT_FADED * SAT_REF:
            f = max(f, SAT_REF / max(1e-4, sat_p * lvl))
        if f > 1.0:
            sat = _round(min(SAT_MAX, f), 0.05)
    g["sat"] = sat

    # --- casts: the Highlights (gain) and Shadows (lift) wheels, per RGB channel - the exact inverse of a
    # transfer's per-channel gain + offset. Solved so the white anchor and the black anchor (after levels) land on
    # grey at their own brightness: (x + L (1 - x)) * P = target for x = black and x = white of each channel.
    rep.update(wb_u=round(white[1] * 224, 2) if white else None, wb_v=round(white[2] * 224, 2) if white else None)
    B = [s["black"] for s in stats if "black" in s]
    black, bcons = _consistent_cast(B, n)
    rep.update(bk_frames=len(B), bk_consistent=round(100 * bcons),
               bk_u=round(black[1] * 224, 2) if black else None, bk_v=round(black[2] * 224, 2) if black else None)
    use_w = white is not None and not mono and np.hypot(white[1], white[2]) >= CAST_MIN and wcons >= CONSIST_MIN
    use_b = black is not None and not mono and np.hypot(black[1], black[2]) >= CAST_MIN and bcons >= CONSIST_MIN
    g["temp"] = g["tint"] = 0.0
    boost = 1.0
    if use_w and g["white"] < 235:
        # dim whites lifted by the white point AND tinted: the cast's strongest channel would pass 1.0 before the
        # Highlights wheel pulls it back (it works on the clipped picture) - leave that channel room, and let the
        # wheel bring the whites back up by the same amount once they are neutral
        m = float((np.array([1.0, white[1] / white[0], white[2] / white[0]]) @ M_RGB.T).max())
        wp0 = g["white"]
        g["white"] = min(235, int(round(16 + (wp0 - 16) * max(1.0, m))))
        boost = (g["white"] - 16) / max(1, wp0 - 16)
    if use_w or use_b:
        def rgb_after(a):
            yn, un, vn = grade_np(np.array(a[0]), np.array(a[1]), np.array(a[2]), g)
            return np.array([float(yn), float(un), float(vn)]) @ M_RGB.T
        lum = np.array([KR, 1 - KR - KB, KB])
        wr = np.clip(rgb_after(white), 0.02, 1.0) if use_w else None
        br = np.clip(rgb_after(black), 0.0, 0.9) if use_b else None
        if use_w and use_b and float(wr @ lum - br @ lum) > 0.25:
            tw, tb = min(1.0, float(wr @ lum) * boost), float(br @ lum)
            P = (tw * (1 - br) - tb * (1 - wr)) / np.maximum(wr - br, 0.05)
            L = (tb / P - br) / (1 - br)
        elif use_w:
            P, L = min(1.0, float(wr @ lum) * boost) / wr, np.zeros(3)
        else:
            P, L = np.ones(3), (float(br @ lum) - br) / (1 - br)
        P = np.clip(P, 0.6, 1.6)
        gain = M_YUV @ ((P - 1) / 0.5)                      # wheel_luts: G = rgb(gain_w) * 0.5, L = rgb(lift) * 0.25
        lift = M_YUV @ (L / 0.25)
        if use_w:
            g["gain_w"] = [round(float(x), 3) for x in gain]
        if np.abs(lift).max() > 1e-3:
            g["lift"] = [round(float(x), 3) for x in lift]

    # --- reference (a well-transferred release / episode of the same show): its colour strength is the target.
    # A moderate fade cannot be told from a pastel palette inside one source (good shows differ by 2x, every hue
    # family by 25-90 %: bench/fade_probe*.py) - the same show's paints can: both sides are graded with their own
    # levels and cast wheels, then compared per hue family (the same hair, skin, clothes recur), so it does not
    # matter which scenes each side shows (bench/bench_refmatch.py).
    if ref_stats:
        gr = solve(ref_stats)["grade"]
        if REF_METHOD == "register":
            fit = ref_register(stats, g, ref_stats, gr)
            if fit:
                v, s0, s1 = fit
                rep.update(ref_frames=len(ref_stats), ref_overlap=round(s0, 4), ref_overlap_fit=round(s1, 4))
                if s1 > s0 * REG_GAIN:
                    g["sat"] = 1.0 if abs(np.log(v["sat"])) < np.log(REF_MIN) else _round(v["sat"], 0.05)
                    g["hue"] = float(round(v["hue"])) if abs(v["hue"]) >= REF_HUE_MIN else 0.0
                    if abs(v["contrast"] - 1) >= 0.03 or abs(v["gamma"] - 1) >= 0.03 or abs(v["shadows"]) >= 0.05:
                        g["contrast"], g["gamma"], g["shadows"] = round(v["contrast"], 3), round(v["gamma"], 2), round(v["shadows"], 2)
                    if v["dark_sat"] >= 0.1:
                        g["dark_sat"] = _round(v["dark_sat"], 0.05)
                rep.update(ref_ratio=g.get("sat", 1.0), ref_hue_shift=g.get("hue", 0.0))
            ref_stats = None if fit else ref_stats
    if ref_stats:
        # Same scenes (a better release of this episode, screenshots of the same moments): everything can be
        # matched. Other scenes (another episode): brightness, hue and dark-tone colour depend on each scene's
        # lighting and paints more than on the transfer (other episodes: 17-40 levels apart before any fault,
        # hue within 4 degrees in 29-43 % only - bench/ref_sweep.py), so only a clear fade is matched.
        # pair_stats: the source's own frames that show the same picture as the reference frames (found by
        # measure_levels in a thumbnail index of the whole source) - same scenes by construction
        if pair_stats:
            share, same, cmp = 1.0, True, pair_stats
        else:
            share = content_match(stats, ref_stats)
            same, cmp = share >= REF_SAME_SHARE, stats
        rep.update(ref_same=same, ref_match=round(100 * share), ref_pairs=len(pair_stats or []))
        min_ratio = REF_MIN if same else REF_MIN_OTHER
    if ref_stats and same:
        # brightness: contrast, midtones, shadow lift
        t = ref_tone(cmp, g, ref_stats, gr)
        if t:
            gam, sh, con, e0, e1 = t
            rep.update(ref_tone_err=round(e0 * 219, 1), ref_tone_err_fit=round(e1 * 219, 1))
            if e1 < REF_TONE_GAIN * e0 and (abs(gam - 1) >= 0.03 or abs(sh) >= 0.05 or abs(con - 1) >= 0.03):
                g["gamma"], g["shadows"], g["contrast"] = round(gam, 2), round(sh, 2), round(con, 3)
    if ref_stats and same:
        # hue: one rotation for every colour (NTSC phase)
        h = ref_hue(cmp, g, ref_stats, gr)
        if h:
            rep.update(ref_hue_shift=h[0], ref_hue_gain=round(h[1], 3))
            if abs(h[0]) >= REF_HUE_MIN and h[1] >= REF_HUE_GAIN:
                g["hue"] = float(h[0])
    if ref_stats:
        # colour strength in the mid tones -> saturation; in the dark tones -> shadow colour (same scenes only)
        ratio, nb = ref_ratio(cmp, g, ref_stats, gr)
        rep.update(ref_frames=len(ref_stats), ref_ratio=round(ratio, 3) if ratio else None, ref_hues=nb)
        if ratio:
            g["sat"] = 1.0 if abs(np.log(ratio)) < np.log(min_ratio) else _round(min(REF_MAX, max(1 / REF_MAX, ratio)), 0.05)
            dark, _ = ref_ratio(cmp, g, ref_stats, gr, 0.05, 0.3) if same else (None, 0)
            if dark:
                rel = dark / ratio                       # how much more the darks lost than the mid tones
                rep["ref_dark_ratio"] = round(rel, 3)
                # grade: dark gain 1 + dark_sat * (1 - Y / 0.4)^2, ~0.25 at the dark tones measured (Y ~ 0.2)
                if rel >= REF_DARK_MIN:
                    g["dark_sat"] = _round(min(2.0, (rel - 1) / 0.25), 0.05)

    # --- no reference: the colour prior (a CNN trained on many good transfers, colourprior.py) for colour strength,
    # midtones and hue. It is a learned guess: on shows it never saw it under-calls faults and leans on its own
    # idea of a show's colour (README "COLOUR PRIOR"), so only what is beyond what it says about good transfers is
    # corrected (PRIOR_*_GATE), and only part of the way (PRIOR_*_GAIN).
    if not ref_stats and not mono and shots >= PRIOR_SHOTS:
        pr = prior_check(stats, g)
        if pr:
            k, gam, hue, mad = pr
            rep.update(prior_sat=round(1 / k, 3), prior_gamma=round(gam, 3), prior_hue=round(-hue, 1))
            ls = -(np.log(k) - PRIOR_SAT_BIAS)               # correction (log), the prior's own lean removed
            gate = PRIOR_SAT_GATE if ls > 0 else PRIOR_SAT_GATE_DOWN
            if abs(ls) > gate:
                f = float(np.exp(np.sign(ls) * (abs(ls) - gate) * PRIOR_SAT_GAIN + np.sign(ls) * gate))
                g["sat"] = _round(min(SAT_MAX, max(1 / SAT_MAX, max(g["sat"], f) if f > 1 else f)), 0.05)
                rep["prior_used"] = True
            lg = np.log(gam)
            if abs(lg) > PRIOR_GAMMA_GATE:
                g["gamma"] = round(float(np.exp(np.sign(lg) * PRIOR_GAMMA_GATE + (lg - np.sign(lg) * PRIOR_GAMMA_GATE) * PRIOR_GAMMA_GAIN)), 2)
                rep["prior_used"] = True
            if abs(hue) > PRIOR_HUE_GATE:
                g["hue"] = float(round(-hue * PRIOR_HUE_GAIN))
                rep["prior_used"] = True

    # --- gamut
    og = float(np.percentile([s["gamut"] for s in stats], 90))
    rep.update(gamut_out=round(og, 2), clip_flat=round(float(np.percentile([s["flat"] for s in stats], 90)), 2))
    g["gamut"] = "compress" if og >= 1.0 else "off"
    return {"report": rep, "grade": g}


def measure(frames):
    """frame_stats of every frame that is not nearly black (fades, title cards)."""
    out = []
    for Y, U, V in frames:
        s = frame_stats(Y, U, V)
        if s["mean"] >= 14 / 219:
            out.append(s)
    return out


def analyze(frames, ref_frames=None, pair_frames=None):
    """frames: iterable of (Y, U, V) uint8 arrays (limited range), the source. ref_frames: the same from a
    reference of the same show (optional). pair_frames: the source frames that show the same pictures as
    ref_frames (optional; then ref_frames should be only the reference frames that found a twin)."""
    stats = measure(frames)
    if not stats:
        return {}
    ref = measure(ref_frames) if ref_frames is not None else None
    pairs = measure(pair_frames) if pair_frames is not None else None
    return solve(stats, ref or None, pairs or None)
