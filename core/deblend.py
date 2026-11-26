"""Blend reduction: finds frames that are a mix of the frame before and the frame after (ghost / double
image left by standards conversion, bad field handling or a blending deinterlacer) and replaces them.

Test per frame, on a small luma copy:  c = p + a (q - p)   (p = previous, q = next, least-squares a)
  a blend has 0.2 < a < 0.8, the mix explains the frame almost completely (residual / |q - p|^2 small) and
  p and q really differ (there is motion to mix).
A fade or cross-fade passes that test too - every frame of a fade is the mix of its neighbours - so a frame
only counts when it stands alone: its neighbours must not be blends of *their* neighbours. Slow pans and
zooms do not pass: a moved picture is not a mix of two pictures (double outlines are, a shifted outline
is not), the residual stays high.
Replacement:
  dup     the nearer clean neighbour (a > 0.5 -> next, else previous). No new picture is invented; before
          decimation the duplicate is what the decimator removes anyway.
  interp  motion-interpolated frame between the two neighbours at position a (MVTools): keeps motion
          smooth, can smear on fast or complex motion.
"""
import numpy as np
import vapoursynth as vs

core = vs.core
A_LO, A_HI = 0.2, 0.8
RES_MAX = 0.06            # residual of the mix / |q - p|^2
MOTION_MIN = 3e-5         # mean (q - p)^2 on 0..1 luma: below this the neighbours are the same picture
ISOLATION = 2.5           # a neighbour with a residual within this factor makes it a fade, not a blend
MOTION_CHECK = True       # a frame that motion explains (pan / zoom) as well as a mix is not a blend
MC_RATIO = 1.0            # blend only if the mix explains the frame better than motion by this factor
HOLD_AWARE = True         # pick the replacement from the drawing holds around the blend (see _choice)
HELD = 0.15               # a neighbour pair differing by < this share of the blend's motion is one held drawing
HOLD_R = 6                # frames looked at on each side for the hold pattern
HOLD_OFFS = (-6, -5, -4, -3, -2, 2, 3, 4, 5, 6)
CUT = 0.12                # mean change (0..1) above which neighbours are taken as a cut, not motion
AREAS = True              # also blends in only a PART of the picture (area test, masked repair)
AREA = 15                 # area size on the quarter-size test copy (= 60 px of a DVD frame)
AREA_MOTION = 1e-3        # mean (q - p)^2 an area needs (~8 levels rms): a blend there is a visible double image
AREA_A = (0.3, 0.7)       # mix weight band for an area (narrower than a whole frame's: small areas are noisier)
AREA_RES = 0.02           # area residual limit (stricter than RES_MAX); AREA_MC: motion must explain it this much worse.
AREA_MC = 2.0             # Blends put into real frames (+1 level noise): 79 % of their areas pass, 37 % of the areas
                          # flagged on real discs with the looser limits (in-between drawings, gradients) - bench_blend
FADE_R = 0.3              # whole-frame mix residual below which the frame is a fade / whole blend (no area test)
DUP = 0.5 / 255           # a frame this close (mean abs) to a neighbour is a repeat (telecine / held drawing): no area test
AREA_MIN = 2              # areas that must be blended (fewer: noise)
AREA_MAX = 0.6            # share of the picture above which it is a whole-frame case (left to is_blend)


def _score(p, c, q):
    d = q - p
    den = float((d * d).sum())
    if den / d.size < MOTION_MIN:
        return 0.0, 9.0, den / d.size
    a = float(((c - p) * d).sum() / den)
    res = float(((c - p - a * d) ** 2).sum()) / den
    return a, res, den / d.size


def _mid(clip):
    """Motion-compensated picture half-way between frame n-1 and n+1 (MVTools), for frame n."""
    y = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    sup = core.mv.Super(y, pel=2, sharp=2)
    bv = core.mv.Analyse(sup, isb=True, delta=2, blksize=16, overlap=8, search=5, truemotion=True)
    fv = core.mv.Analyse(sup, isb=False, delta=2, blksize=16, overlap=8, search=5, truemotion=True)
    mid = core.mv.FlowInter(y, sup, bv, fv, time=50, ml=70)
    return mid[:1] + mid[:-1]


def analyse(clip):
    """Adds props BlendA (mix weight), BlendR (residual of the mix, small = well explained), BlendM (motion),
    BlendRM (residual of the motion-compensated in-between picture, same scale as BlendR)."""
    small = lambda c: core.resize.Bilinear(core.std.ShufflePlanes(c, 0, vs.GRAY), max(64, clip.width // 4 // 2 * 2),
                                           max(48, clip.height // 4 // 2 * 2), format=vs.GRAYS)
    y = small(clip)
    prv, nxt = y[:1] + y[:-1], y[1:] + y[-1:]
    def sh(o):                                           # frame n -> frame n + o (edges repeated)
        return y[o:] + y[-1] * o if o > 0 else y[0] * (-o) + y[:o]
    clips = [prv, y, nxt] + [sh(o) for o in HOLD_OFFS]
    if MOTION_CHECK:
        clips.append(small(_mid(clip)))

    def sel(n, f):
        p, c, q = (np.asarray(f[i][0]) for i in range(3))
        a, r, m = _score(p, c, q)
        out = f[1].copy()
        out.props["BlendA"], out.props["BlendR"], out.props["BlendM"] = a, r, m
        # drawing holds around the frame: change between successive frames n-6 .. n-1 and n+1 .. n+6 (mean abs, 0..1)
        w = {o: np.asarray(f[3 + i][0]) for i, o in enumerate(HOLD_OFFS)}
        w[-1], w[1] = p, q
        out.props["HoldDiff"] = [float(np.abs(w[o + 1] - w[o]).mean()) for o in range(-HOLD_R, -1)] + \
                                [float(np.abs(w[o + 1] - w[o]).mean()) for o in range(1, HOLD_R)]
        out.props["BlendD"] = float(np.abs(q - p).mean())
        out.props["BlendNear"] = float(min(np.abs(c - p).mean(), np.abs(c - q).mean()))
        mid = np.asarray(f[3 + len(HOLD_OFFS)][0]) if MOTION_CHECK else None
        if MOTION_CHECK:
            d = q - p
            den = max(float((d * d).sum()), 1e-9)
            out.props["BlendRM"] = float(((c - mid) ** 2).sum()) / den
        if AREAS:
            _area_props(out.props, p, c, q, mid)
        return out
    return core.std.ModifyFrame(y, clips=clips, selector=sel)


def _area_props(props, p, c, q, mid):
    """the mix test per area (AREA x AREA on the test copy): AreaA (weight), AreaR (residual / |q-p|^2), AreaM (mean
    (q-p)^2), AreaRM (residual of the motion-compensated in-between), flattened row by row; AreaGrid = [cols, rows]"""
    gh, gw = c.shape[0] // AREA, c.shape[1] // AREA
    blk = lambda x: x[:gh * AREA, :gw * AREA].reshape(gh, AREA, gw, AREA).sum(axis=(1, 3))
    d, e = q - p, c - p
    dd, cd, cc = blk(d * d), blk(e * d), blk(e * e)
    den = np.maximum(dd, 1e-9)
    a = cd / den
    props["AreaA"] = a.ravel().tolist()
    props["AreaR"] = np.maximum(cc - a * cd, 0).ravel().__truediv__(den.ravel()).tolist()
    props["AreaM"] = (dd / (AREA * AREA)).ravel().tolist()
    props["AreaRM"] = (blk((c - mid) ** 2) / den).ravel().tolist() if mid is not None else [9.0] * (gh * gw)
    props["AreaGrid"] = [gw, gh]


def area_blends(fp, fc, fn):
    """flags (rows x cols, bool) of the areas of frame c that are a blend of the same area in the frame before and
    after - while the frame as a whole is not (a layer composited from two frames: a moving part of the picture shows a
    double image). Same tests as is_blend, per area: weight 0.2-0.8, mix explains it, motion does not explain it as well,
    the area really moves, and the same area of a neighbour is not a mix too (a local fade / dissolve)."""
    if not AREAS or "AreaA" not in fc.props:
        return None
    gw, gh = fc.props["AreaGrid"]
    g = lambda f, k: np.asarray(f.props[k], dtype=np.float64).reshape(gh, gw)
    a, r, m, rm = g(fc, "AreaA"), g(fc, "AreaR"), g(fc, "AreaM"), g(fc, "AreaRM")
    if fc.props.get("BlendNear", 1.0) < DUP:              # a repeat of a neighbour (the duplicate before decimation)
        return None
    if A_LO < fc.props["BlendA"] < A_HI and fc.props["BlendR"] < FADE_R:
        return None              # the whole frame reads as a mix: a fade / glow (is_blend judges whole-frame blends).
                                 # Before decimation a fade's neighbour can be a telecine duplicate, which never looks
                                 # like a mix, so the per-area isolation test alone let fades through (Naruto's opening)
    ok = (a > AREA_A[0]) & (a < AREA_A[1]) & (r < AREA_RES) & (m >= AREA_MOTION)
    if MOTION_CHECK:
        ok &= rm > r * AREA_MC
    for f in (fp, fn):
        if "AreaA" in f.props:
            an, rn = g(f, "AreaA"), g(f, "AreaR")
            ok &= ~((an > A_LO) & (an < A_HI) & (rn < np.maximum(RES_MAX, r * ISOLATION)))
    n = int(ok.sum())
    if n < AREA_MIN or n > AREA_MAX * ok.size:
        return None
    return ok


def is_blend(fp, fc, fn):
    a, r = fc.props["BlendA"], fc.props["BlendR"]
    if not (A_LO < a < A_HI and r < RES_MAX):
        return False
    if MOTION_CHECK and "BlendRM" in fc.props and fc.props["BlendRM"] <= r * MC_RATIO:
        return False                                     # motion explains it as well: pan / zoom, not a blend
    for f in (fp, fn):                                  # a neighbour that is itself a mix -> fade / dissolve
        if A_LO < f.props["BlendA"] < A_HI and f.props["BlendR"] < max(RES_MAX, r * ISOLATION):
            return False
    return True


def _runs(held):
    """lengths of runs of one drawing from a list of 'this pair is the same drawing' flags"""
    out, r = [], 1
    for h in held:
        if h:
            r += 1
        else:
            out.append(r)
            r = 1
    out.append(r)
    return out


def _choice(fc):
    """which neighbour the blended frame's own picture most likely was: 'p' / 'n', or None (no evidence -> mix weight).
    Animation holds each drawing for H frames (twos, threes), so the missing picture is very often an exact copy of a
    neighbour. H = the typical run length of drawings in +-6 frames; the run ending at n-1 (before) and the one starting
    at n+1 (after) tell which drawing is short of its H frames: before >= H, after < H -> the next drawing (n+1);
    after >= H, before < H -> the previous one. Not across cuts (the neighbours then belong to other shots)."""
    if not HOLD_AWARE or "HoldDiff" not in fc.props:
        return None
    dd, D = list(fc.props["HoldDiff"]), fc.props["BlendD"]
    if D > CUT:
        return None
    lim = max(0.6 / 255, HELD * D)
    nb = HOLD_R - 1                                      # pairs before: (n-6,n-5) .. (n-2,n-1); after: (n+1,n+2) ..
    if max(dd) > CUT:
        return None
    hb, ha = [x < lim for x in dd[:nb]], [x < lim for x in dd[nb:]]
    before = 1
    for h in reversed(hb):
        if not h:
            break
        before += 1
    after = 1
    for h in ha:
        if not h:
            break
        after += 1
    runs = _runs(hb)[1:-1] + _runs(ha)[1:-1]             # complete runs only (both ends seen)
    if not runs:
        return None
    H = int(np.median(runs))
    if H < 2:
        return None                                      # drawn on ones: every frame new, no copy to take
    if before >= H and after < H:
        return "n"
    if after >= H and before < H:
        return "p"
    return None


def deblend(clip, mode="dup"):
    """clip: progressive frames (after field matching). Returns the clip with blended frames replaced and
    the prop _Deblended = 1 on them."""
    if mode not in ("dup", "interp"):
        return clip
    st = analyse(clip)
    sp, sn = st[:1] + st[:-1], st[1:] + st[-1:]
    prv, nxt = clip[:1] + clip[:-1], clip[1:] + clip[-1:]
    mark = lambda c: core.std.SetFrameProps(c, _Deblended=1)
    sw = max(64, clip.width // 4 // 2 * 2)                 # the test copy's width (analyse)
    sc = clip.width / sw                                  # test-copy px -> frame px

    def area_mask(n, f):
        """the frame's blended areas as a soft GRAY8 mask (grown by one area, feathered); empty when there are none"""
        out = f[0].copy()
        ok = area_blends(f[1], f[2], f[3]) if not is_blend(f[1], f[2], f[3]) else None
        arr = np.asarray(out[0])
        arr[:] = 0
        if ok is not None:
            from scipy.ndimage import binary_dilation, uniform_filter
            ok = binary_dilation(ok, iterations=1)
            cell = int(round(AREA * sc))
            big = np.kron(ok.astype(np.float32), np.ones((cell, cell), np.float32))[:arr.shape[0], :arr.shape[1]]
            big = uniform_filter(big, size=max(3, cell // 2))
            arr[:big.shape[0], :big.shape[1]] = np.clip(big * 255 + 0.5, 0, 255).astype(np.uint8)
            out.props["_AreaBlend"] = 1
        return out
    blank = core.std.BlankClip(clip, format=vs.GRAY8, color=0)
    amask = core.std.ModifyFrame(blank, clips=[blank, sp, st, sn], selector=area_mask)

    def mean_a(fc, ok):
        gw, gh = fc.props["AreaGrid"]
        return float(np.asarray(fc.props["AreaA"]).reshape(gh, gw)[ok].mean())

    if mode == "interp":
        sup = core.mv.Super(clip, pel=2, sharp=2)
        bv = core.mv.Analyse(sup, isb=True, delta=2, blksize=16, overlap=8, search=5, truemotion=True)
        fv = core.mv.Analyse(sup, isb=False, delta=2, blksize=16, overlap=8, search=5, truemotion=True)
        # FlowInter(clip, time) at frame n gives the picture between n and n+delta: shift by one to get p..q
        mid = {t: mark(core.mv.FlowInter(clip, sup, bv, fv, time=t, ml=70)) for t in (30, 40, 50, 60, 70)}
        mid = {t: c[:1] + c[:-1] for t, c in mid.items()}          # frame n now = between n-1 and n+1

        dp, dn = mark(prv), mark(nxt)
        mm = lambda r: mark(core.std.MaskedMerge(clip, r, amask, first_plane=True))
        part = {id(r): mm(r) for r in [dp, dn] + list(mid.values())}

        def repl(fc, a):
            ch = _choice(fc)
            if ch:                                       # a held drawing: its copy, not an invented in-between
                return dn if ch == "n" else dp
            return mid[min(mid, key=lambda k: abs(k - 100 * a))]

        def pick(n, f):
            if is_blend(f[0], f[1], f[2]):
                return repl(f[1], f[1].props["BlendA"])
            ok = area_blends(f[0], f[1], f[2])
            if ok is None:
                return clip
            # inside an area the interpolated picture measured worse than leaving the blend (bench_blend: 34.28 vs
            # 34.61 dB): the area takes the neighbour's copy (hold pattern, else the mix weight), as in 'dup'
            ch = _choice(f[1]) or ("n" if mean_a(f[1], ok) > 0.5 else "p")
            return part[id(dn if ch == "n" else dp)]
    else:
        dp, dn = mark(prv), mark(nxt)

        def repl(fc, a):
            ch = _choice(fc)
            if ch:
                return dn if ch == "n" else dp
            return dn if a > 0.5 else dp

        part = {"n": mark(core.std.MaskedMerge(clip, dn, amask, first_plane=True)),
                "p": mark(core.std.MaskedMerge(clip, dp, amask, first_plane=True))}

        def pick(n, f):
            if is_blend(f[0], f[1], f[2]):
                return repl(f[1], f[1].props["BlendA"])
            ok = area_blends(f[0], f[1], f[2])
            if ok is None:
                return clip
            return part["n" if repl(f[1], mean_a(f[1], ok)) is dn else "p"]
    return core.std.FrameEval(clip, pick, prop_src=[sp, st, sn])
