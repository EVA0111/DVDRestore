"""Steady chain, detail- and line-safe: the Steady chain's result, but each pixel may only move a little from its input
- shimmer, flicker and wobble are small changes (0.5-1.2 levels on real discs), smeared detail and hammered lines are
big ones. The allowed change is tighter on lines / edges / texture (edge mask of the input) than on flat areas.
    steady_safe(clip, job, level="normal", flat=3.0, edge=1.0, chroma=3.0)    levels on the 8-bit scale"""
import vapoursynth as vs

core = vs.core


def parse(spec):
    """'level=normal/flat=3/edge=1' (or commas) -> dict"""
    out = {}
    for kv in spec.replace("/", ",").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip() if k.strip() in ("level", "method") else float(v)
    return out


def edge_mask(clip, t=20.0):
    """0..65535 on lines / edges / texture of the luma (>= t levels per pixel step = full), widened by a pixel"""
    y = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    s = core.std.Sobel(y)
    m = core.std.Expr(s, f"x {65535 / (t * 256)} * 65535 min")
    return core.std.Inflate(core.std.Maximum(m))


def limit(inp, out, mask, flat=3.0, edge=1.0, chroma=3.0):
    """out, but each pixel at most `flat` levels from inp (`edge` on the mask), colour at most `chroma`"""
    lf, le, lc = flat * 256, edge * 256, chroma * 256
    ly = f"{lf} {lf - le} z 65535 / * -"
    expr_y = f"x y x - {ly} -1 * max {ly} min +"
    expr_c = f"x y x - {-lc} max {lc} min +"
    y = core.std.Expr([core.std.ShufflePlanes(inp, 0, vs.GRAY), core.std.ShufflePlanes(out, 0, vs.GRAY), mask], expr_y)
    uv = core.std.Expr([inp, out], ["", expr_c, expr_c])
    return core.std.CopyFrameProps(core.std.ShufflePlanes([y, uv, uv], [0, 1, 2], vs.YUV), out)


def steady_safe(clip, job, level="normal", flat=3.0, edge=1.0, chroma=3.0, method="chain", t=20.0):
    import vsgraph
    inp = core.resize.Bicubic(clip, format=vs.YUV444P16) if clip.format.id != vs.YUV444P16 else clip
    st = vsgraph.stabilize(inp, dict(job, stab=method, stab_level=level))
    if st.format.id != vs.YUV444P16:
        st = core.resize.Bicubic(st, format=vs.YUV444P16)
    return limit(inp, st, edge_mask(inp, t), flat, edge, chroma)


def flicker_map(inp, radius=1):
    """per pixel |input - median(input, its motion-compensated neighbours)| (16-bit, luma and colour): how much the
    input itself jumps around there. Real detail and lines follow the motion and stay; shimmer, flicker, boiling do not"""
    sup = core.mv.Super(inp, pel=2, sharp=2, rfilter=4)
    sup_r = core.mv.Super(inp, pel=2, sharp=2, levels=1)
    kw = dict(blksize=16, overlap=8, truemotion=True, search=4, searchparam=2)
    comp = []
    for d in range(1, radius + 1):
        for back in (True, False):
            v = core.mv.Analyse(sup, isb=back, delta=d, **kw)
            comp.append(core.mv.Compensate(inp, sup_r, v, thsad=10000, thscd1=16320, thscd2=255))
    if radius == 1:
        med = core.std.Expr([comp[0], inp, comp[1]], "x y z max min x y z min max max")   # median of 3
    else:
        med = core.zsmooth.Median(inp) if False else core.std.Expr([comp[0], inp, comp[1]], "x y z max min x y z min max max")
    return core.std.Expr([inp, med], "x y - abs")


def limit_adaptive(inp, out, mask, flick, flat=1.0, edge=0.5, chroma=1.0, gain=1.5, cmax=12.0):
    """change allowed per pixel = max(the fixed cap (tighter on edges), gain x how much the input flickers there),
    at most cmax levels; colour likewise"""
    lf, le, lc, cm = flat * 256, edge * 256, chroma * 256, cmax * 256
    base_y = f"{lf} {lf - le} z 65535 / * -"
    ly = f"{base_y} a {gain} * max {cm} min"
    expr_y = f"x y x - {ly} -1 * max {ly} min +"
    lcx = f"{lc} z {gain} * max {cm} min"
    expr_c = f"x y x - {lcx} -1 * max {lcx} min +"
    P = lambda c, i: core.std.ShufflePlanes(c, i, vs.GRAY)
    y = core.akarin.Expr([P(inp, 0), P(out, 0), mask, P(flick, 0)], expr_y)
    u = core.akarin.Expr([P(inp, 1), P(out, 1), P(flick, 1)], expr_c)
    v = core.akarin.Expr([P(inp, 2), P(out, 2), P(flick, 2)], expr_c)
    return core.std.CopyFrameProps(core.std.ShufflePlanes([y, u, v], [0, 0, 0], vs.YUV), out)


def steady_adaptive(clip, job, level="normal", flat=1.0, edge=0.5, chroma=1.0, gain=1.5, cmax=12.0, method="chain", t=20.0):
    """the Steady chain where the input flickers, the input where it is steady (detail, lines)"""
    import vsgraph
    inp = core.resize.Bicubic(clip, format=vs.YUV444P16) if clip.format.id != vs.YUV444P16 else clip
    st = vsgraph.stabilize(inp, dict(job, stab=method, stab_level=level))
    if st.format.id != vs.YUV444P16:
        st = core.resize.Bicubic(st, format=vs.YUV444P16)
    return limit_adaptive(inp, st, edge_mask(inp, t), flicker_map(inp), flat, edge, chroma, gain, cmax)


def comb3d(clip, t=1.5, soft=1.5, r=2, frames=1):
    """3D-comb style: dot crawl / rainbows flip every frame, so where the picture is still, averaging a frame with its
    neighbours cancels them and keeps all real detail (the detail is the same in both). Stillness is judged on a blurred
    copy (crawl is fine-grained, so blurring removes it from the test): neighbour used fully when the blurred luma differs
    < t levels, faded out over the next `soft` levels. YUV444P16 in/out."""
    inp = core.resize.Bicubic(clip, format=vs.YUV444P16) if clip.format.id != vs.YUV444P16 else clip
    y = core.std.ShufflePlanes(inp, 0, vs.GRAY)
    b = core.std.BoxBlur(y, hradius=r, vradius=r, hpasses=2, vpasses=2)
    nb = []
    for d in range(1, int(frames) + 1):
        for s in (-d, d):
            c = inp[0] * (-s) + inp[:s] if s < 0 else inp[s:] + inp[-1] * s
            bb = b[0] * (-s) + b[:s] if s < 0 else b[s:] + b[-1] * s
            nb.append((c, bb))
    T, S = t * 256, soft * 256
    planes = []
    for p in range(3):
        P = lambda c: core.std.ShufflePlanes(c, p, vs.GRAY)
        clips = [P(inp), b] + [P(c) for c, _ in nb] + [bb for _, bb in nb]
        n = len(nb)
        v = [chr(ord("a") + i) for i in range(len(clips))]
        v = ["x", "y", "z"] + [f"src{i}" for i in range(3, len(clips))]
        num, den = ["x"], ["1"]
        for i in range(n):
            ci, bi = v[2 + i], v[2 + n + i]
            w = f"1 {bi} y - abs {T} - {S} / 0 max 1 min -"
            num.append(f"{ci} {w} *"); den.append(w)
        expr = " ".join(num) + " +" * n + " " + " ".join(den) + " +" * n + " /"
        planes.append(core.akarin.Expr(clips, expr))
    return core.std.CopyFrameProps(core.std.ShufflePlanes(planes, [0, 0, 0], vs.YUV), inp)


def motion_mask(clip, px=0.5, sad=None):
    """GRAYS 1.0 where motion tracking finds the picture moved by more than `px` pixels (to the previous OR next frame)"""
    y = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    y8 = core.resize.Bicubic(y, format=vs.GRAY8)
    sup = core.mv.Super(y8, pel=2, sharp=2, rfilter=4)
    kw = dict(blksize=16, overlap=8, truemotion=True, search=4, searchparam=2)
    ms = []
    for back in (True, False):
        v = core.mv.Analyse(sup, isb=back, delta=1, **kw)
        ms.append(core.std.ShufflePlanes(core.mv.Mask(y8, v, kind=0, ml=px * 2, gamma=1.0, ysc=255), 0, vs.GRAY))
    m = core.std.Expr(ms, "x y max 127 > 1 0 ?", format=vs.GRAYS)
    return m


def wavelet_color_fix(clip, ref, sigma=4.0, luma=False):
    """colour (and optionally luma) at coarse scale from `ref`, fine scale from `clip`: keeps clip's cleaned fine colour
    but ref's overall colour / saturation. YUV444P16 both."""
    b = lambda c: core.std.Convolution(core.std.Convolution(c, [1] * 9), [1] * 9) if sigma <= 0 else         core.tcanny.TCanny(c, sigma=sigma, mode=-1) if hasattr(core, "tcanny") else core.std.BoxBlur(c, hradius=int(sigma), vradius=int(sigma), hpasses=3, vpasses=3)
    out = []
    for p_ in range(3):
        P = lambda c: core.std.ShufflePlanes(c, p_, vs.GRAY)
        if p_ == 0 and not luma:
            out.append(P(clip)); continue
        out.append(core.std.Expr([P(clip), b(P(clip)), b(P(ref))], "x y - z +"))
    return core.std.CopyFrameProps(core.std.ShufflePlanes(out, [0, 0, 0], vs.YUV), clip)


def steady_plus(clip, job, level="normal", thr=4.0, s=3.0, keep=0.97, px=0.5, anchor=1, cap=0.0, fade=1, chroma=1, wcf=0.0):
    """Steady chain, then a still lock that is looser on shimmer (thr / s) but never locks anything that moved"""
    import tstab, vsgraph
    st = vsgraph.stabilize(clip, dict(job, stab="chain", stab_level=level))
    st = core.resize.Bicubic(st, format=vs.YUV444P16) if st.format.id != vs.YUV444P16 else st
    out = tstab.still_lock(st, keep=keep, thr=thr, s=s, motion=motion_mask(st, px) if px > 0 else None, anchor=anchor, fade=fade, chroma=chroma)
    if wcf > 0:                       # wavelet colour fix: broad colour of the current frame (never washed out by the memory)
        out = core.resize.Bicubic(out, format=vs.YUV444P16) if out.format.id != vs.YUV444P16 else out
        out = wavelet_color_fix(out, st, wcf)
    if cap > 0:                       # the lock may only take out the small leftover shimmer: each pixel at most cap levels
        out = core.resize.Bicubic(out, format=vs.YUV444P16) if out.format.id != vs.YUV444P16 else out
        out = limit(st, out, core.std.BlankClip(st, format=vs.GRAY16, color=0), cap, cap, cap)
    return out
