"""Guided temporal fix: removes the shimmer / line wobble / flicker that a frame-by-frame AI model adds.

Why it is different from vs_temporalfix: that filter decides what to average by looking at the flickering
picture itself, so its thresholds must be loose enough to accept the shimmer - and then they also accept small
real changes (ghosting), and wherever motion is unsure it reverts to the unfixed frame. Here the decision is made
on the GUIDE, the picture the model was given (it has no model shimmer):

  B = guide resized to the output size (the base picture)      R = output - B (everything the model added)
  motion vectors from B (stable), neighbours n-r .. n+r motion-compensated
  per pixel and neighbour: weight from how well the compensated GUIDE matches (low-passed difference, so grain
  cancels; plus an edge-alignment term) - a mismatch means motion error, occlusion or a cut -> weight 0
  output = B + weighted average of the compensated R (the current frame always weight 1)

Only what the model added is averaged; the base picture of the frame itself (and its grain, brightness and real
motion) is never blended, so nothing can ghost beyond the model's own residual, and weight 0 = the exact output.
Without a guide (unguided=True) the guide is the output itself, low-passed: high frequencies (line wobble,
shimmer) are averaged under weights from the low-passed picture.
"""
import vapoursynth as vs

core = vs.core

PRESETS = {"light": 0.6, "normal": 1.0, "strong": 1.6}


def _u16(clip):
    if clip.format.id == vs.YUV444P16:
        return clip
    if clip.format.color_family == vs.RGB:
        return core.resize.Bicubic(clip, format=vs.YUV444P16, matrix_s="709")
    return core.resize.Bicubic(clip, format=vs.YUV444P16)


def _lowpass(c, r):
    for _ in range(2):
        c = core.std.BoxBlur(c, hradius=r, vradius=r)
    return c


def gtfix(out, guide=None, radius=3, strength=1.0, unguided_sigma=2, rguard=None, debug=False):
    """out: the model's output (YUV or RGB). guide: the model's input (any size; None = unguided).
    strength scales the match thresholds (higher = averages more). rguard: local mean |compensated R - R| (8-bit
    levels, x strength) that also drops the weight to 0 - shimmer is small, a misaligned edge of real detail is
    not (default: off when guided, 4 unguided - moving-shot bench_stable: Y 42.59 -> 43.10 dB, no clip worse
    than without a stabilizer; a clean guide needs no guard). Returns YUV444P16 at the output size."""
    import os
    if rguard is None:
        rguard = float(os.environ.get("GTFIX_RGUARD", 0 if guide is not None else 4))
    O = _u16(out)
    W, H = O.width, O.height
    if guide is None:
        B = _lowpass(O, unguided_sigma)
    else:
        g = _u16(guide) if guide.format.color_family == vs.RGB else guide
        B = core.resize.Spline36(g, W, H, format=vs.YUV444P16) if (g.width, g.height, g.format.id) != (W, H, vs.YUV444P16) else g
    R = core.std.Expr([O, B], "x y - 32768 +")
    big = max(W, H) > 1280
    blk, ov = (32, 16) if big else (16, 8)
    # motion search on a lightly smoothed base (stable, fine detail would only add noise to the search)
    pre = core.std.BoxBlur(B, hradius=1, vradius=1)
    sup_a = core.mv.Super(pre, pel=2, sharp=2, rfilter=4)
    sup_b = core.mv.Super(B, pel=2, sharp=2, levels=1)
    sup_r = core.mv.Super(R, pel=2, sharp=2, levels=1)
    kw = dict(blksize=blk, overlap=ov, truemotion=True, search=4, searchparam=2, chroma=True)
    t1 = 1.2 * strength * 256            # low-passed guide mismatch (16-bit) that drops the weight to 0
    t2 = 7.0 * strength * 256            # local mean absolute mismatch (edge alignment) that drops it to 0
    t3 = rguard * strength * 256
    num = [R]
    ws = []
    for d in range(1, radius + 1):
        for back in (True, False):
            v = core.mv.Analyse(sup_a, isb=back, delta=d, **kw)
            v = core.mv.Recalculate(sup_a, v, blksize=blk // 2, overlap=ov // 2, truemotion=True, thsad=100000)
            bc = core.mv.Compensate(B, sup_b, v, thsad=100000, thscd1=16320, thscd2=255)
            rc = core.mv.Compensate(R, sup_r, v, thsad=100000, thscd1=16320, thscd2=255)
            dif = core.std.Expr([bc, B], "x y - 32768 +")
            lp = _lowpass(dif, 2)
            ad = core.std.BoxBlur(core.std.Expr(dif, "x 32768 - abs"), hradius=2, vradius=2)
            # one weight for all planes: luma + half of each colour plane
            pl = lambda c, i: core.std.ShufflePlanes(c, i, vs.GRAY)
            w = core.std.Expr([pl(lp, 0), pl(lp, 1), pl(lp, 2), pl(ad, 0), pl(ad, 1), pl(ad, 2)],
                              f"x 32768 - abs y 32768 - abs z 32768 - abs + 0.5 * + {t1} / "
                              f"a b c + 0.5 * + {t2} / max 1 swap - 0 max dup * 65535 *")
            if t3 > 0:
                ar = core.std.BoxBlur(core.std.Expr([pl(rc, 0), pl(R, 0)], "x y - abs"), hradius=2, vradius=2)
                w = core.std.Expr([w, ar], f"y {t3} / 1 swap - 0 max dup * x *")
            ws.append(core.std.ShufflePlanes([w, w, w], [0, 0, 0], vs.YUV))
            num.append(rc)
    n = len(ws)
    # src0 = O, src1 = B, src2..src(1+n) = compensated R, then the n weights
    terms_num = " ".join(f"src{2 + i} 32768 - src{2 + n + i} 65535 / * +" for i in range(n))
    terms_den = " ".join(f"src{2 + n + i} 65535 / +" for i in range(n))
    expr = f"src0 src1 - {terms_num} 1 {terms_den} / src1 +"
    res = core.akarin.Expr([O, B] + num[1:] + ws, expr)
    if debug:                                        # luma = how much was averaged (0 black .. all white)
        tot = core.akarin.Expr(ws, " ".join(f"src{i}" for i in range(n)) + " +" * (n - 1) + f" {n} /")
        return core.std.ShufflePlanes(tot, 0, vs.GRAY)
    return core.std.CopyFrameProps(res, out)
