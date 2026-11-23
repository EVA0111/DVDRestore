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
    clips = [prv, y, nxt]
    if MOTION_CHECK:
        clips.append(small(_mid(clip)))

    def sel(n, f):
        p, c, q = (np.asarray(f[i][0]) for i in range(3))
        a, r, m = _score(p, c, q)
        out = f[1].copy()
        out.props["BlendA"], out.props["BlendR"], out.props["BlendM"] = a, r, m
        if MOTION_CHECK:
            d = q - p
            den = max(float((d * d).sum()), 1e-9)
            out.props["BlendRM"] = float(((c - np.asarray(f[3][0])) ** 2).sum()) / den
        return out
    return core.std.ModifyFrame(y, clips=clips, selector=sel)


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


def deblend(clip, mode="dup"):
    """clip: progressive frames (after field matching). Returns the clip with blended frames replaced and
    the prop _Deblended = 1 on them."""
    if mode not in ("dup", "interp"):
        return clip
    st = analyse(clip)
    sp, sn = st[:1] + st[:-1], st[1:] + st[-1:]
    prv, nxt = clip[:1] + clip[:-1], clip[1:] + clip[-1:]
    mark = lambda c: core.std.SetFrameProps(c, _Deblended=1)
    if mode == "interp":
        sup = core.mv.Super(clip, pel=2, sharp=2)
        bv = core.mv.Analyse(sup, isb=True, delta=2, blksize=16, overlap=8, search=5, truemotion=True)
        fv = core.mv.Analyse(sup, isb=False, delta=2, blksize=16, overlap=8, search=5, truemotion=True)
        # FlowInter(clip, time) at frame n gives the picture between n and n+delta: shift by one to get p..q
        mid = {t: mark(core.mv.FlowInter(clip, sup, bv, fv, time=t, ml=70)) for t in (30, 40, 50, 60, 70)}
        mid = {t: c[:1] + c[:-1] for t, c in mid.items()}          # frame n now = between n-1 and n+1

        def pick(n, f):
            if not is_blend(f[0], f[1], f[2]):
                return clip
            t = min(mid, key=lambda k: abs(k - 100 * f[1].props["BlendA"]))
            return mid[t]
    else:
        dp, dn = mark(prv), mark(nxt)

        def pick(n, f):
            if not is_blend(f[0], f[1], f[2]):
                return clip
            return dn if f[1].props["BlendA"] > 0.5 else dp
    return core.std.FrameEval(clip, pick, prop_src=[sp, st, sn])
