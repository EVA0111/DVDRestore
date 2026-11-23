"""Universal temporal stabilizer: steadies flicker, leftover crawl, line wobble and noise in every kind of shot -
still, pans, zooms, rotation, cuts, crossfades, fades to black, flashes, lighting changes, glows, credits scrolling
over motion, fast objects - without dragging the previous picture into the next (bench/bench_cases.py).

Why the older stabilizers failed on effects: they average whole pixels whenever neighbours look alike, so a fade,
a dissolve or a light ramp (a small change every frame) passes their thresholds and the picture lags behind -
the Steady chain was worse than no stabilizer on cuts, crossfades, light ramps and glows.

Here every frame is split in two bands and each is averaged under its own rule, over n-R .. n+R, motion-compensated:
  smooth band (brightness, colour, gradients; box low-pass)  - averaged only where the compensated neighbour's band
     is within `t_low` levels: frame-to-frame flicker (< 1 level) is averaged out, a fade / flash / glow /
     dissolve changes faster and is left alone. The window is symmetric, so a slow LINEAR change (light ramp)
     averages to the frame's own value - no lag.
  detail band (lines, texture, noise)                         - averaged only where the compensated neighbour's
     detail lines up: local mean |difference| below (noise allowance + 0.6 px x local edge slope). Line wobble and
     crawl (sub-pixel, flipping) pass, a motion error or a different picture does not.
The noise allowance comes from the clip itself (measured on sampled frames), so the same settings fit a clean
digital master and a grainy film transfer. MVTools' scene-change test makes a neighbour across a cut the frame
itself (weight without effect)."""
import numpy as np
import vapoursynth as vs

core = vs.core

LEVELS = {"light": 0.7, "normal": 1.0, "strong": 1.5}


def _u16(c):
    return c if c.format.id == vs.YUV444P16 else core.resize.Bicubic(c, format=vs.YUV444P16)


def _lp(c, r):
    for _ in range(2):
        c = core.std.BoxBlur(c, hradius=r, vradius=r)
    return c


def noise_sigma(clip, samples=8):
    """luma noise of the clip (8-bit levels): median |fine detail| in flat areas of sampled frames"""
    y = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    n = y.num_frames
    vals = []
    for t in np.linspace(0, n - 1, min(samples, n)).astype(int):
        a = np.asarray(y.get_frame(int(t))[0], dtype=np.float32) / 257
        b = a.copy()
        b[1:-1, 1:-1] = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]) / 4
        hp = (a - b)[2:-2, 2:-2]
        g = np.abs(np.diff(a, axis=1))[2:-2, 2:-1].astype(np.float32)
        flat = g < np.percentile(g, 40)
        if flat.sum() > 1000:
            vals.append(float(np.median(np.abs(hp[flat]))) * 1.4826 / 0.9)
    return float(np.median(vals)) if vals else 1.5


def ustab(clip, strength=1.0, radius=4, t_low=3.0, slope_px=0.25, sigma=None, noise_k=5.0, pix_px=0.35, ref="center", center_min=0.1, bands="both", debug=False):
    """clip: YUV (any; 4:4:4 16-bit inside). strength scales both thresholds. Returns YUV444P16."""
    O = _u16(clip)
    W, H = O.width, O.height
    # floor 1.5: clean anime fills are perfectly flat, the measured noise reads 0 there (bench_stable: 21 of 24 real
    # clips) and the allowances collapse - real clips Y 42.79 -> 43.02 dB, wobble 1.53 -> 1.29 with the floor
    sig = max(1.5, noise_sigma(O)) if sigma is None else float(sigma)
    s = float(strength)
    L = _lp(O, 4)                                        # smooth band
    big = max(W, H) > 1280
    blk, ov = (32, 16) if big else (16, 8)
    pre = _lp(O, 1)
    sup_a = core.mv.Super(pre, pel=2, sharp=2, rfilter=4)
    sup_o = core.mv.Super(O, pel=2, sharp=2, levels=1)
    kw = dict(blksize=blk, overlap=ov, truemotion=True, search=4, searchparam=2, chroma=True)
    pl = lambda c, i: core.std.ShufflePlanes(c, i, vs.GRAY)
    # local edge slope (8-bit levels per pixel), for the detail band's alignment allowance
    Ly = pl(O, 0)
    slope_pix = core.akarin.Expr(Ly, "x[1,0] x[-1,0] - abs x[0,1] x[0,-1] - abs max 0.5 *")
    slope = core.std.BoxBlur(slope_pix, hradius=2, vradius=2)
    slope_pix = core.std.Maximum(slope_pix)
    tl = t_low * s * 256
    pix_allow = 6.0 * sig * s * 256                      # one pixel of two noisy frames: < ~4 sigma
    noise_allow = noise_k * sig * s * 256                # two noisy frames differ by ~1.6 sigma on average -> weight ~0.9
    comps, wl_, wh_ = [], [], []
    for d in range(1, radius + 1):
        for back in (True, False):
            v = core.mv.Analyse(sup_a, isb=back, delta=d, **kw)
            v = core.mv.Recalculate(sup_a, v, blksize=blk // 2, overlap=ov // 2, truemotion=True, thsad=100000)
            oc = core.mv.Compensate(O, sup_o, v, thsad=100000)          # default scene-change test: cut -> the frame itself
            lc = _lp(oc, 4)
            # smooth band: local mean |difference| of the bands (luma + half of each colour plane)
            dl = core.akarin.Expr([lc, L], "x y - abs")
            dlm = core.std.BoxBlur(dl, hradius=3, vradius=3)
            wl = core.akarin.Expr([pl(dlm, 0), pl(dlm, 1), pl(dlm, 2)], f"x y z + 0.5 * + {tl} / dup * 1 swap - 0 max 65535 *")
            # detail band: local mean |difference of the details|
            dh = core.akarin.Expr([oc, lc, O, L], "x y - z a - - abs")
            dhm = core.std.BoxBlur(dh, hradius=2, vradius=2)
            wh = core.akarin.Expr([pl(dhm, 0), pl(dhm, 1), pl(dhm, 2), slope],
                                  f"x y z + 0.5 * + {noise_allow} a {slope_px * s} * + / dup * 1 swap - 0 max 65535 *")   # slope: 16-bit units already
            # per-pixel guard: the box mean dilutes one misaligned thin line (crossfade: 1241 pixels > 25 levels off,
            # input 10) - any pixel (3x3 max) whose detail differs by more than the noise + a 1 px shift of its edge
            # drops out (0.7 px: the wobble it should average is +-0.35 px a frame; a block-motion error on text is whole pixels)
            dpx = core.std.Maximum(pl(dh, 0))
            wh = core.akarin.Expr([wh, dpx, slope_pix], f"y {pix_allow} z {pix_px * s} * + / dup * 1 swap - 0 max x *")
            comps.append((oc, lc))
            wl_.append(core.std.ShufflePlanes([wl, wl, wl], [0, 0, 0], vs.YUV))
            wh_.append(core.std.ShufflePlanes([wh, wh, wh], [0, 0, 0], vs.YUV))
    n = len(comps)
    if ref in ("median", "hybrid"):
        # Detail band referenced to the per-pixel MEDIAN of the window (frame n + its 2R compensated neighbours), and
        # frame n weighted like the others: frame-by-frame filters (the Precise model on SCA1) re-draw lines a little
        # differently every frame - "boiling", 3.95 levels on still lines - and measured against frame n itself every
        # neighbour looked mismatched, so nothing was averaged (2.36 left). Against the median, frame n's own boil is
        # the outlier; a neighbour with a motion error or another picture (a minority) is out as before. The smooth
        # band (brightness: fades, flashes, glows) keeps its rule against frame n.
        det_c = [core.akarin.Expr([O, L], "x y - 32768 +")] + [core.akarin.Expr([oc, lc], "x y - 32768 +") for oc, lc in comps]
        m = len(det_c)
        if m == 9:
            med = core.akarin.Expr(det_c, " ".join(f"src{i}" for i in range(m)) + " sort9 drop4 swap4 drop4")
        else:                                            # any odd count
            k = m // 2
            med = core.akarin.Expr(det_c, " ".join(f"src{i}" for i in range(m)) + f" sort{m} drop{k} swap{k} drop{k}")
        wd = []
        for i, dc in enumerate(det_c):
            dev = core.akarin.Expr([dc, med], "x y - abs")
            dbm = core.std.BoxBlur(dev, hradius=2, vradius=2)
            w = core.akarin.Expr([pl(dbm, 0), pl(dbm, 1), pl(dbm, 2), slope],
                                 f"x y z + 0.5 * + {noise_allow} a {slope_px * s} * + / dup * 1 swap - 0 max 65535 *")
            dpx = core.std.Maximum(pl(dev, 0))
            w = core.akarin.Expr([w, dpx, slope_pix], f"y {pix_allow} z {pix_px * s} * + / dup * 1 swap - 0 max x *"
                                 + (f" {center_min * 65535} max" if i == 0 else ""))
            wd.append(core.std.ShufflePlanes([w, w, w], [0, 0, 0], vs.YUV))
        # src0 O, src1 L, n compensated L, n low weights, then m detail clips, m detail weights
        lc_i = lambda i: f"src{2 + i}"
        wl_i = lambda i: f"src{2 + n + i} 65535 /"
        dd_i = lambda i: f"src{2 + 2 * n + i} 32768 -"
        dw_i = lambda i: f"src{2 + 2 * n + m + i} 65535 /"
        low = "src1 " + " ".join(f"{lc_i(i)} {wl_i(i)} * +" for i in range(n)) + " 1 " + " ".join(f"{wl_i(i)} +" for i in range(n)) + " /"
        det = " ".join(f"{dd_i(i)} {dw_i(i)} *" + (" +" if i else "") for i in range(m)) + " " +               " ".join(f"{dw_i(i)}" + (" +" if i else "") for i in range(m)) + " /"
        clips = [O, L] + [c[1] for c in comps] + wl_ + det_c + wd
        if ref == "hybrid":
            # Trust the median only where frame n is within boil size of it (a pixel deviation up to the per-pixel
            # allowance, fully; to 3x it, partly). Where it is far off, the median itself is wrong - motion search
            # failed for most neighbours (credits scrolling over another motion: median-only lost 2.8 dB there, cuts
            # and crossfades 4) - and the detail is averaged against frame n as before.
            d0 = core.std.Maximum(core.akarin.Expr([pl(det_c[0], 0), pl(med, 0)], "x y - abs"))
            alpha = core.akarin.Expr([d0, slope_pix], f"{3 * pix_allow} y {3 * pix_px * s} * + x - "
                                     f"{2 * pix_allow} y {2 * pix_px * s} * + / 0 max 1 min 65535 *")
            base = len(clips)
            oc_j = lambda i: f"src{base + i}"
            wh_j = lambda i: f"src{base + n + i} 65535 /"
            al = f"src{base + 2 * n} 65535 /"
            det_cen = "src0 src1 - " + " ".join(f"{oc_j(i)} {lc_i(i)} - {wh_j(i)} * +" for i in range(n)) +                       " 1 " + " ".join(f"{wh_j(i)} +" for i in range(n)) + " /"
            det = f"{det} {al} * {det_cen} 1 {al} - * +"
            clips = clips + [c[0] for c in comps] + wh_ + [core.std.ShufflePlanes([alpha] * 3, [0, 0, 0], vs.YUV)]
        if bands == "low":
            det = "src0 src1 -"
        elif bands == "det":
            low = "src1"
        res = core.akarin.Expr(clips, f"{low} {det} +")
        return core.std.CopyFrameProps(res, clip)
    # src0 O, src1 L, then n compensated O, n compensated L, n low weights, n detail weights
    oc_i = lambda i: f"src{2 + i}"
    lc_i = lambda i: f"src{2 + n + i}"
    wl_i = lambda i: f"src{2 + 2 * n + i} 65535 /"
    wh_i = lambda i: f"src{2 + 3 * n + i} 65535 /"
    low = "src1 " + " ".join(f"{lc_i(i)} {wl_i(i)} * +" for i in range(n)) + " 1 " + " ".join(f"{wl_i(i)} +" for i in range(n)) + " /"
    det = "src0 src1 - " + " ".join(f"{oc_i(i)} {lc_i(i)} - {wh_i(i)} * +" for i in range(n)) + " 1 " + " ".join(f"{wh_i(i)} +" for i in range(n)) + " /"
    clips = [O, L] + [c[0] for c in comps] + [c[1] for c in comps] + wl_ + wh_
    if debug:                                            # Y = share of the smooth band averaged, U = of the detail
        a = core.akarin.Expr(wl_, " ".join(f"src{i}" for i in range(n)) + " +" * (n - 1) + f" {n} /")
        b = core.akarin.Expr(wh_, " ".join(f"src{i}" for i in range(n)) + " +" * (n - 1) + f" {n} /")
        return core.std.ShufflePlanes([a, b, b], [0, 0, 0], vs.YUV)
    if bands == "low":                                   # diagnostics: average one band only
        det = "src0 src1 -"
    elif bands == "det":
        low = "src1"
    res = core.akarin.Expr(clips, f"{low} {det} +")
    return core.std.CopyFrameProps(res, clip)


def still_smooth(clip, reach=10, thr=1.0, pair=0.0, thr_lo=None, pair_lo=None, med_ref=False, hold=None, debug=False):
    """Stage 2: a long, two-sided average where the picture stands still (static camera, held drawings): frames
    n-reach .. n+reach, each weighted by a gate CHAINED frame to frame from n outwards - the chain stops at the first
    real change between two neighbours (motion, cut, a fade step, a flash), so nothing beyond it is used. Two-sided:
    a slow linear change (light ramp) averages to the frame's own value, where a one-way memory (the Steady chain's
    still lock) lags behind it. thr: local mean |change| between two neighbours (8-bit levels, luma + half chroma)
    that closes the gate. clip: YUV444P16."""
    import torch
    import torch.nn.functional as F
    c = _u16(clip)
    n_all = c.num_frames
    shifted = []
    for k in range(-reach, reach + 1):
        if k < 0:
            s_ = c[0] * (-k) + c[:n_all + k] if n_all + k > 0 else c[0] * n_all
        elif k > 0:
            s_ = c[k:] + c[-1] * k if k < n_all else c[-1] * n_all
        else:
            s_ = c
        shifted.append(s_)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = thr * 257
    # gate shape: flat top up to *_lo (noise / boil-level differences keep FULL weight - with a smooth curve the
    # chained product decayed even on a perfectly still picture: SCA1's logo still shimmered 0.72 levels), then
    # falling linearly to 0 at thr / pair. None = the old curve 1 - (m / thr)^2
    tlo = None if thr_lo is None else thr_lo * 257
    plo = None if pair_lo is None else pair_lo * 257

    def gate(m, hi, lo):
        if lo is None:
            return (1 - (m / hi) ** 2).clamp(min=0)
        return ((hi - m) / (hi - lo)).clamp(0, 1)

    def box(x, r=2):
        return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), 2 * r + 1, 1)

    def proc(n, f):
        fr = lambda i: torch.from_numpy(np.stack([np.asarray(f[i][p], dtype=np.float32) for p in range(3)])).to(dev)
        with torch.no_grad():
            frames = [fr(i) for i in range(2 * reach + 1)]
            valid = [0 <= n + k < n_all for k in range(-reach, reach + 1)]
            ctr = frames[reach]
            num = ctr.clone()
            den = torch.ones_like(ctr[:1])
            lm = lambda d: box(d[0:1][None])[0] + 0.5 * (box(d[1:2][None])[0] + box(d[2:3][None])[0])
            if pair > 0:
                # symmetric pairs: (n-k, n+k) together, only where their average is frame n's picture (any LINEAR
                # change - ramp, slow zoom, fade - passes unbiased; a real difference does not) and both chains hold
                gb = torch.ones_like(ctr[:1]); gf = torch.ones_like(ctr[:1])
                tp = pair * 257
                # pair reference: frame n, or the median of the 5 nearest frames (med_ref) - a frame-by-frame filter's
                # boil makes frame n itself the odd one (SCA1 logo), and against it no pair passed. Anything that moves
                # or flashes closes the chained gate at the first step, so the median cannot act there.
                ref_ = ctr
                if med_ref and reach >= 2 and all(valid[reach - 2:reach + 3]):
                    ref_ = torch.stack(frames[reach - 2:reach + 3]).median(dim=0).values
                # holds: a neighbour showing the SAME drawing (step change at boil level, chained from n) is averaged
                # even from one side. Drawings on twos: the pair (n-1, n+1) is already another drawing, so the two
                # copies of one drawing were never averaged - SCA1's moving petals boiled 1.12 levels copy to copy
                # (Steady chain 0.37). A held drawing does not change, so one side cannot lag; a fade / ramp changes
                # faster than boil or is handled by the pairs.
                if hold:
                    h_lo, h_hi = hold[0] * 257, hold[1] * 257
                    s_lo, s_hi = (hold[2] * 257, hold[3] * 257) if len(hold) > 3 else (0.4 * 257, 1.2 * 257)
                    hb = torch.ones_like(ctr[:1]); hf = torch.ones_like(ctr[:1])
                for k in range(1, reach + 1):
                    ib, i_f = reach - k, reach + k
                    okb, okf = valid[ib], valid[i_f]
                    if not (okb or okf):
                        break
                    sb = lm((frames[ib] - frames[ib + 1]).abs()) if okb else None
                    sf = lm((frames[i_f] - frames[i_f - 1]).abs()) if okf else None
                    gp = None
                    if okb and okf:
                        gb = gb * gate(sb, t, tlo)
                        gf = gf * gate(sf, t, tlo)
                        pa = 0.5 * (frames[ib] + frames[i_f])
                        gp = torch.minimum(gb, gf) * gate(lm((pa - ref_).abs()), tp, plo)
                    else:
                        gb = gb * 0; gf = gf * 0
                    wb = wf = gp if gp is not None else torch.zeros_like(ctr[:1])
                    if hold:
                        # tested against frame n (not step by step): motion adds up frame after frame, boil does not
                        # (chained steps let slow rotation / light ramps through: rotate 49.6 -> 43.6 dB); and the
                        # SIGNED local mean must stay near 0 - a fade / ramp moves a whole area one way, boil both ways
                        # against frame n itself, never the median: next to a flash the median is the unflashed
                        # picture and the flash was averaged away (bench: 22 dB)
                        def hold_w(fr_):
                            dd = fr_ - ctr
                            ab = lm(dd.abs())
                            sg = box(dd[0:1][None])[0].abs()
                            return gate(ab, h_hi, h_lo) * gate(sg, s_hi, s_lo)
                        if okb:
                            hb = torch.minimum(hb, hold_w(frames[ib]))
                            wb = torch.maximum(wb, hb)
                        else:
                            hb = hb * 0
                        if okf:
                            hf = torch.minimum(hf, hold_w(frames[i_f]))
                            wf = torch.maximum(wf, hf)
                        else:
                            hf = hf * 0
                        done = float(hb.max()) <= 0 and float(hf.max()) <= 0
                    else:
                        done = True
                    if done and float(wb.max()) <= 0 and float(torch.minimum(gb, gf).max()) <= 0:
                        break
                    if okb:
                        num += frames[ib] * wb
                        den += wb
                    if okf:
                        num += frames[i_f] * wf
                        den += wf
                steps = ()
            else:
                steps = (-1, 1)
            for step in steps:
                g = torch.ones_like(ctr[:1])
                for k in range(1, reach + 1):
                    i = reach + step * k
                    if not valid[i]:
                        break
                    a, b = frames[i], frames[i - step]
                    m = lm((a - b).abs())
                    g = g * (1 - (m / t) ** 2).clamp(min=0)
                    if float(g.max()) <= 0:
                        break
                    num += a * g
                    den += g
            out = (num / den).clamp(0, 65535)
            if debug:
                out = torch.cat([(den - 1) / (2 * reach) * 65535] * 3)
            out = out.round().to(torch.int32).cpu().numpy().astype(np.uint16)
        dst = f[reach].copy()
        for p in range(3):
            np.copyto(np.asarray(dst[p]), out[p])
        return dst
    res = core.std.ModifyFrame(c, shifted, proc)
    return core.std.CopyFrameProps(res, clip)


HOLDS = {"light": (1.5, 3.5, 0.3, 0.9), "normal": (2.0, 5.0, 0.4, 1.2), "strong": (3.0, 7.0, 0.6, 1.8)}


def stabilize(clip, level="normal"):
    """the app's stabilizer "Universal": motion-compensated two-band average (+-4 frames), then the two-sided still
    stage (+-10 frames): balanced pairs (flat-top chained gates 2-6 levels, pair test 1-2.5 levels against
    frame n) plus HOLDS - one-sided averaging of neighbours that show the same drawing
    (light / normal / strong: HOLDS[level]).
    bench_cases (13 kinds of shot, normal): Y 44.65 dB, wobble 0.41, error on the effect frames 1.01 - input 41.27 /
    2.59 / 1.55, MC degrain 44.09 / 0.92 / 1.11, Steady chain 42.64 / 0.37 / 1.35 (worse than none on cuts,
    crossfades, light ramps, glows); better than the input on every case.
    SCA1 (Precise model, title logo + petals on twos): boil between the two copies of a held drawing 2.81 -> 0.65
    levels (first Universal 1.19, Steady chain 0.37), still lines 2.16 -> 0.07 (chain 0.06), real drawing changes
    kept 98 % (chain 93 %)."""
    if not isinstance(level, str):                       # older settings: a number
        level = "light" if float(level) < 1.25 else "strong" if float(level) > 2.0 else "normal"
    # med_ref off: the median pair reference cost 0.23 dB on the real clips for a hardly visible gain on SCA1
    return still_smooth(ustab(clip), thr_lo=2.0, thr=6.0, pair_lo=1.0, pair=2.5, reach=10, med_ref=False,
                        hold=HOLDS.get(level, HOLDS["normal"]))
