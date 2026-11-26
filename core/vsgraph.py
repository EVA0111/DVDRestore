"""VapourSynth processing graph for DVD Restore.

Pipeline per episode:
    source (pulldown flags applied, colour tagged) -> trim to episode
    -> composite chroma re-decode -> motion-compensated crawl/rainbow cancellation (stage A)
    -> IVTC (TFM + TDecimate, CFR or two-pass VFR) / PAL handling -> comb safety net
    -> motion-compensated averaging (stage B) -> ghost/halo removal (LGhost-style)
    -> chroma shift + bleed fix -> crop -> resize (selectable kernel / NNEDI3) -> colour -> YUV420P10

Command line (used by the job runner):
    python vsgraph.py pass1 job.json      VFR analysis pass (writes tfm/tdec files)
    python vsgraph.py measure job.json    auto-detect everything (artifacts, ghosts, chroma, crop) as JSON
"""
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

# The app starts this file with python -I (isolated), which leaves the script folder off sys.path;
# the neural modules next to it (dvdnet, chromanet, dotnet) are imported lazily, so add it here.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import vapoursynth as vs

import nnbackend
import paths  # noqa: E402  (core/paths.py: every folder in one place)

core = vs.core
# VapourSynth keeps rendered frames up to max_cache_size - by default HALF the RAM (15.9 GB here), so the preview
# backend grew to 8 GB and stayed there. Frames of a 480p chain are ~2 MB: 4 GB is far more than any filter reaches back.
_CACHE_MB = int(__import__("os").environ.get("DVDRESTORE_CACHE_MB", 4096))
if core.max_cache_size > _CACHE_MB:
    core.max_cache_size = _CACHE_MB

FPS_NUM, FPS_DEN = 30000, 1001

# Tunable defaults (the GUI's Advanced dialog can override every one of these per job)
DEFAULTS = {
    "temporal_guard": "off", # experimental (temporal_guard): measured no gain - crawl and MPEG corrections must change per frame
    "backend": "pytorch",  # where the neural models run: pytorch / tensorrt (nnbackend.py; engines cached in ./engines)
    "order": "tff",        # field order: tff / bff
    "vfr_video": "on",     # VFR: sections without a telecine field pattern stay 30p (video_sections) - off: TDecimate alone
    "cthresh": 9,          # TFM: per-pixel comb threshold (lower = more sensitive)
    "mi": 80,              # TFM: combed pixels per 16x16 block before a frame counts as combed
    "ivtc_decide": "clean",  # raw IVTC (FieldNet): match / decimate on a crawl-free copy ("clean") or the frames ("plain")
    "pp": 6,               # TFM: post-processing of combed frames (0 off, 1 detect only, 6 motion-adaptive)
    "dupthresh": 1.1,      # TDecimate: below this a frame counts as a duplicate (%)
    "vidthresh": 1.1,      # TDecimate VFR: cycles whose duplicate is weaker than this are kept as 30p video
    "composite": "off",    # composite re-decode (vapoursynth-composite): off / chroma (rainbows) / full / clean (full, no sharpening)
    "comp_dims": 3,        # composite: 3 = spatio-temporal 3D comb/transform (best), 2 = spatial only (2x faster)
    "comp_refine": 1,      # composite: luma refinement passes (0 = off for grainy/flat material, 2-4 = detailed)
    "crawl": "off",        # dot crawl / line flicker (luma): off light normal strong
    "rainbow": "off",      # rainbows / cross-colour (chroma): off light normal strong
    "rainbow_fill": "off", # on = colour stains left where a very strong rainbow was are filled from the surroundings.
                           # Off by default: on One Piece / Courage it also repaints 5-6 % of the picture (fine coloured
                           # texture), Dominion needs it (grey-green blotch on hatched shading)
    "ghost": "off",        # ghost / halo removal: off auto
    "ghost_strength": 1.0, # halo removal: AI 0.25-1 = blend, above 1 = part of a second pass; classic = passes
    "halo_method": "ai",   # ai (HaloNet: halos + ringing + short echoes, needs models/halonet.pt) classic (DeHalo)
    "vs_filters_pos": "repairs",   # where Repair -> VapourSynth filters run: source (before IVTC, also what the VFR
                           # analysis sees), ivtc (INSTEAD of the app's IVTC / PAL handling: your own field matching),
                           # after_ivtc, repairs (after the colour alignment), end (after crop, stabilizer, colour, size)
    "repairnet": "on",     # on = AI halos and AI colour bleed in ONE RepairNet pass (models/repairnet.pt) in place of
                           # HaloNet + ChromaNet; off = the two separate models
    "deblock": "off",      # MPEG blocking / mosquito noise / blocky colour (BlockNet, models/blocknet.pt): off light normal
    "pal_mode": "auto",    # PAL: auto progressive fieldmatch deint50 deint25
    "pal_film": "on",      # PAL deint50 / deint25: film frames in a mixed source are field-matched whole, not bobbed
    "kernel": "spline36",  # resizer for the output size (see KERNELS)
    "ai_model": "temporal", # neural dots + rainbows cleanup: temporal (5-frame, best) precise (single frame) off
    "ivtc_repair": "ai",   # frames field matching cannot clean: ai (CombNet in the combed areas) / masked (NNEDI3 +
                           # Bwdif in the combed areas) / frame (whole frame deinterlaced)
    "chroma_unmix": "off", # on = colour of frames IVTC assembled from two source frames is unmixed (chroma_unmix)
    "deblend": "off",      # frames that are a mix of their neighbours (ghost / double image): off dup interp (deblend.py)
    "aa": "off",           # AANet anti-aliasing (models/aanet.pt): off / repair (only areas rebuilt by the comb
                           # repair and deinterlaced PAL frames) / all (whole picture, every frame)
    "comb": "nnedi3",      # leftover combing after IVTC: off safe (Bwdif whole frame) nnedi3 (masked) strong (+vinverse)
    "stab": "off",         # temporal stabilizer: off chain plus gtfix tfix_ai tfix_classic mc ttemp
    "stab_level": "normal",# light normal strong
}

# temporal stabilizer strength per method and level
STAB = {
    # gtfix.py strength (match thresholds; unguided here, residual guard 4). Moving shots (bench_stable, 24 clips):
    # 3 = best Y / T; 6 = steadier, a little less fine detail
    "gtfix":        {"light": 2.0, "normal": 3.0, "strong": 6.0},
    # Universal (ustab.py): pair tolerance of the still stage. bench_cases, 13 kinds of shot incl. cuts, crossfades,
    # fades, flashes, light ramps, glows, zoom, rotation, credits, fast objects: better than no stabilizer on all
    "ustab":        {"light": "light", "normal": "normal", "strong": "strong"},     # ustab.HOLDS
    "tfix_ai":      {"light": 1.0, "normal": 2.0, "strong": 3.0},          # vs_temporalfix.model strength 0-3
    "tfix_classic": {"light": 400, "normal": 550, "strong": 700},          # vs_temporalfix.classic strength (tr 6)
    "mc":           {"light": (2, 150), "normal": (2, 300), "strong": (3, 450)},   # (radius, thSAD) MDegrain
    # TTempSmooth (thresh, mdiff), 8-bit scale. Lodoss / Courage still-area flicker: 6/3 -28/-34 %,
    # 9/4 -36/-41 %, 12/5 -41/-47 %; edge contrast unchanged, moving areas changed < 0.6 (8-bit)
    "ttemp":        {"light": (6, 3), "normal": (9, 4), "strong": (12, 5)},
    # the chain measured with FieldNet v4 on the real Courage disc (door / rug 1.11 / 1.02 -> 0.10 / 0.10, WELCOME
    # letters shimmer 0.65 -> 0.03, wobble 1.12 -> 0.34, sharpness 7.51 -> 7.40, colour unchanged): level of each part
    # (mc, gtfix, ttemp) and the still lock's memory (keep)
    "chain":        {"light": ("normal", "normal", "normal", 0.95), "normal": ("strong", "strong", "strong", 0.97),
                     "strong": ("strong", "strong", "strong", 0.985)},
    # Steady plus (steady.py): the Steady chain at this level, then a looser still lock that tracks motion (never
    # locks what moved), holds through fades and steadies colour too. Real Courage disc with Precise v2 + full
    # re-decode: door / rug 0.76 / 0.94 -> 0.25 / 0.20, WELCOME letters shimmer 0.24 -> 0.03, door mesh colour flicker
    # 1.36 -> 0.07, detail unchanged. Grainy moving film: ~1 dB below the Steady chain (grain read as still) - cartoons
    "plus":         {"light": "light", "normal": "normal", "strong": "strong"},
}

# name: (zimg resizer, extra parameters) - "nnedi3" doubles with NNEDI3 first, then Spline36 to size
KERNELS = {
    "spline36": ("Spline36", {}),
    "spline64": ("Spline64", {}),
    "spline16": ("Spline16", {}),
    "lanczos3": ("Lanczos", {"filter_param_a": 3}),
    "lanczos4": ("Lanczos", {"filter_param_a": 4}),
    "catmull-rom": ("Bicubic", {"filter_param_a": 0, "filter_param_b": 0.5}),
    "mitchell": ("Bicubic", {"filter_param_a": 1 / 3, "filter_param_b": 1 / 3}),
    "hermite": ("Bicubic", {"filter_param_a": 0, "filter_param_b": 0}),
    "bilinear": ("Bilinear", {}),
    "point": ("Point", {}),
    "nnedi3": ("Spline36", {}),
    "lanczos2": ("Lanczos", {"filter_param_a": 2}),
    "bspline": ("Bicubic", {"filter_param_a": 1, "filter_param_b": 0}),
    "bicubic_sharp": ("Bicubic", {"filter_param_a": 0, "filter_param_b": 1}),
    "robidoux": ("Bicubic", {"filter_param_a": 0.37822, "filter_param_b": 0.31089}),
    "robidoux_sharp": ("Bicubic", {"filter_param_a": 0.26201, "filter_param_b": 0.36899}),
    "robidoux_soft": ("Bicubic", {"filter_param_a": 0.67962, "filter_param_b": 0.16019}),
    "bicubic": ("Bicubic", None),          # b / c from the job (kernel_b, kernel_c)
    "mks2013": ("custom", "mks2013"),      # Magic Kernel Sharp 2013 / 2021 (John Costella), resize2.Custom
    "mks2021": ("custom", "mks2021"),
    "ewa_lanczos": ("placebo", "ewa_lanczos"),            # elliptical (EWA) resamplers from libplacebo (mpv), GPU
    "ewa_lanczossharp": ("placebo", "ewa_lanczossharp"),
    "ewa_robidouxsharp": ("placebo", "ewa_robidouxsharp"),
}


def _mks2013(x):
    x = abs(x)
    return 17 / 16 - 7 / 4 * x * x if x <= 0.5 else (0.25 * (4 * x * x - 11 * x + 7) if x <= 1.5 else
                                                       (-0.125 * (x - 2.5) ** 2 if x <= 2.5 else 0.0))


def _mks2021(x):
    """Magic Kernel Sharp 2021 (libvips' coefficients). Checked: weights sum to 1 at every offset, continuous."""
    x = abs(x)
    if x <= 0.5:
        return 577 / 576 - 239 / 144 * x * x
    if x <= 1.5:
        return (140 * x * x - 379 * x + 239) / 144
    if x <= 2.5:
        return -(24 * x * x - 113 * x + 130) / 144
    if x <= 3.5:
        return (4 * x * x - 27 * x + 45) / 144
    if x <= 4.5:
        return -(4 * x * x - 36 * x + 81) / 1152
    return 0.0


def _resize(clip, w, h, kernel, bc=(1 / 3, 1 / 3), **kw):
    name, params = KERNELS.get(kernel, KERNELS["spline36"])
    if name == "custom":
        fn, taps = (_mks2013, 3) if params == "mks2013" else (_mks2021, 5)
        return core.resize2.Custom(clip, custom_kernel=fn, taps=taps, width=w, height=h, **kw)
    if name == "placebo":
        # EWA at 16 bit 4:4:4; colour / format conversion of kw afterwards at the new size. ewa.py (PyTorch) instead of
        # vs-placebo: only one libplacebo Vulkan device can exist per process here, a second one segfaults the backend
        import ewa
        c = core.resize.Bicubic(clip, format=clip.format.replace(subsampling_w=0, subsampling_h=0, bits_per_sample=16,
                                                                  sample_type=vs.INTEGER).id)
        c = ewa.resample(c, w, h, params)
        return core.resize.Spline36(c, w, h, **kw)
    if params is None:
        params = {"filter_param_a": float(bc[0]), "filter_param_b": float(bc[1])}
    return getattr(core.resize, name)(clip, w, h, **params, **kw)


def nnedi3_double(clip):
    """Double width and height with NNEDI3 (edge-directed interpolation, ideal for line art).
    The result is +0.25 source px off on both axes (measured), corrected in the following resize."""
    c = core.znedi3.nnedi3(clip, field=1, dh=True)
    c = core.std.Transpose(c)
    c = core.znedi3.nnedi3(c, field=1, dh=True)
    return core.std.Transpose(c)

# per level: stage A = (stillness threshold, max artifact amplitude) for the cancellation before IVTC,
# motion = motion-compensated cancellation, stage B = MC averaging after IVTC (luma change limit / chroma SAD)
CRAWL = {"light": dict(t=(5, 24), motion=False, limit=0),
         "normal": dict(t=(8, 64), motion=True, limit=2),
         "strong": dict(t=(10, 96), motion=True, limit=4)}
RAINBOW = {"light": dict(t=(4, 16), motion=False, thsadc=0),
           "normal": dict(t=(6, 48), motion=True, thsadc=500),
           "strong": dict(t=(8, 64), motion=True, thsadc=900)}


def opt(job, key):
    v = job.get(key)
    return DEFAULTS[key] if v is None or v == "" else v


def is_pal(job):
    return abs(job.get("fps_num", 30000) / job.get("fps_den", 1001) - 25) < 0.5


# ---------------------------------------------------------------- dot crawl / rainbow

def _cancel(cur, prv, nxt, ty, tc):
    """Temporal comb filter: NTSC crosstalk flips sign every frame, so where prev and next agree the
    current pixel is averaged with them - (2*cur + prev + next) / 4 - cancelling the alternating part
    exactly. ty / tc = (agreement threshold, max artifact amplitude) for luma / chroma; None = untouched."""
    d = core.std.Maximum(core.akarin.Expr([prv, nxt], "x y - abs"))
    ex = []
    for t in (ty, tc, tc):
        if t is None:
            ex.append("x")
        else:
            t1, t2 = t
            ex.append(f"a D! x y z + 2 / - A! {t1} D@ - {t1 / 2} / 0 max 1 min W! A@ abs {t2} < x A@ 2 / W@ * - x ?")
    return core.akarin.Expr([cur, prv, nxt, d], ex)


def _mc_pair(S, blksize=8):
    """Motion-compensated previous / next frame (search on a blurred copy so the artifact can't steer it)."""
    pre = core.std.BoxBlur(S, hradius=1, vradius=1)
    sa = core.mv.Super(pre, pel=2, sharp=2)
    sc = core.mv.Super(S, pel=2, sharp=2, levels=1)
    kw = dict(blksize=blksize, overlap=blksize // 2, truemotion=1, search=3)
    bv = core.mv.Analyse(sa, isb=True, delta=1, **kw)
    fv = core.mv.Analyse(sa, isb=False, delta=1, **kw)
    return core.mv.Compensate(S, sc, bv, thsad=400), core.mv.Compensate(S, sc, fv, thsad=400)


def _lowpass_h(y, fc=0.31, taps=15):
    """horizontal windowed-sinc low-pass (cutoff fc cycles/px), integer kernel for std.Convolution"""
    import math
    h = taps // 2
    k = [(2 * fc if i == 0 else math.sin(2 * math.pi * fc * i) / (math.pi * i)) * (0.42 + 0.5 * math.cos(math.pi * i / (h + 1))
         + 0.08 * math.cos(2 * math.pi * i / (h + 1))) for i in range(-h, h + 1)]
    s = sum(k); k = [round(1024 * v / s) for v in k]
    return core.std.Convolution(y, matrix=k, mode="h", divisor=sum(k), saturate=False) if min(k) < 0 else core.std.Convolution(y, matrix=k, mode="h")


def _above_carrier(src, r, amount=1.0):
    """luma of r + amount x (the band of src above the colour carrier)"""
    ys = core.std.ShufflePlanes(src, 0, vs.GRAY); yr = core.std.ShufflePlanes(r, 0, vs.GRAY)
    if ys.format.bits_per_sample < 16:
        ys16 = core.resize.Point(ys, format=vs.GRAY16); yr16 = core.resize.Point(yr, format=vs.GRAY16)
    else:
        ys16, yr16 = ys, yr
    # replace r's band above the carrier by the original's (amount 1 = fully), not add to it
    out = core.std.Expr([yr16, ys16, _lowpass_h(ys16), _lowpass_h(yr16)], f"x y z - x a - - {amount} * +")
    return core.resize.Point(out, format=ys.format.id) if out.format.id != ys.format.id else out


def composite_restore(src, job):
    """vapoursynth-composite: re-encode the picture into the composite signal the master went through and
    decode it again with a proper motion-adaptive 3D comb / transform separator - removes cross-colour
    (rainbows) and cross-luma (dot crawl) at the source. Runs on the telecined/interlaced stream."""
    tff = _order(job) == 1
    if opt(job, "composite") == "clean":
        # full re-decode without its sharpening: refine off (no halos / ringing; truth clips: mid detail x1.28 and
        # crawl band x1.69 -> x0.99 / x0.91, Y 37.6 -> 46.7 dB with Precise v2), the original's detail above the
        # carrier put back, colour from the original decode (re-decode colour was ~6 dB less accurate)
        job = dict(job, comp_refine=0, comp_hf=job.get("comp_hf") or 1, comp_luma_only=1)
    r = core.composite.Restore(core.std.SetFieldBased(src, 2 if tff else 1),
                               standard="pal" if is_pal(job) else "ntsc",
                               dimensions=int(opt(job, "comp_dims")), refine=int(opt(job, "comp_refine")),
                               **dict(job.get("comp_opts") or {}))   # extra plugin options (transform, eq, cti, level, threshold...)
    r = core.resize.Bicubic(r, format=src.format.id, dither_type="error_diffusion")
    hf = float(job.get("comp_hf", 0) or 0)
    if hf > 0 and opt(job, "composite") != "chroma":
        # the re-decode (refine 0) is the most accurate luma but softens the finest detail; detail ABOVE the colour
        # carrier (> ~0.33 cycles/px horizontally) carries no dot crawl, so take that band from the original decode
        r = core.std.ShufflePlanes([_above_carrier(src, r, hf), r], [0, 1, 2], vs.YUV)
    if job.get("comp_luma_only"):                      # luma from the re-decode, colour from the original decode
        r = core.std.ShufflePlanes([r, src], [0, 1, 2], vs.YUV)
    wr = int(job.get("comp_wcf", 0) or 0)
    if wr > 0:
        # wavelet colour fix: broad colour from the original decode (the re-decode's overall colour is less accurate),
        # fine colour (where rainbows live) from the re-decode
        bl = lambda c: core.std.BoxBlur(c, hradius=wr, vradius=wr, hpasses=3, vpasses=3)
        P = lambda c, i: core.std.ShufflePlanes(c, i, vs.GRAY)
        uv = [core.std.Expr([P(r, i), bl(P(r, i)), bl(P(src, i))], "x y - z +") for i in (1, 2)]
        r = core.std.ShufflePlanes([r] + uv, [0, 0, 0], vs.YUV)
    if opt(job, "composite") == "chroma":
        # chroma only: its re-separated colour removes ~70% of rainbows / cross-colour (even static
        # patterns), while its luma path sharpens and adds halos on DVD masters - keep the original luma
        r = core.std.ShufflePlanes([src, r], [0, 1, 2], vs.YUV)
    return core.std.CopyFrameProps(r, src)


GUARD = {"on": False, "radius": 2, "tau": 3.0}


def temporal_guard(src, out, radius=None, tau=None):
    """Keep a neural model's correction steady over time. Single-frame models decide afresh on every frame,
    so fine texture they are unsure about (wallpaper, bark, grain) is changed a little differently each time:
    it shimmers / wobbles. What flickers is the CORRECTION (out - src), not the picture - so only the
    correction is stabilised: the corrections of the frames around (up to `radius` on each side) are motion-
    compensated onto this frame and averaged with it, each weighted by how well the motion compensation fits
    the picture there (weight 1 when the compensated picture matches within ~0 levels, 0 from `tau` levels
    mismatch: scene cuts, occlusions, bad vectors). The picture itself is never averaged over time, so nothing
    smears; a correction that really changes (a moving artifact) follows the motion.
    src / out: same size, YUV. Returns out's format."""
    if not GUARD["on"]:
        return out
    radius = GUARD["radius"] if radius is None else radius
    tau = GUARD["tau"] if tau is None else tau
    fmt = out.format
    if src.format.id != fmt.id:
        src = core.resize.Bicubic(src, format=fmt.id)
    hi = fmt.replace(bits_per_sample=16, sample_type=vs.INTEGER)
    s16, o16 = core.resize.Point(src, format=hi.id), core.resize.Point(out, format=hi.id)
    R = core.std.Expr([o16, s16], "x y - 32768 +")                          # correction, centred at 32768
    pre = core.std.BoxBlur(s16, hradius=1, vradius=1)
    sa = core.mv.Super(pre, pel=2, sharp=2)
    ss, sr = core.mv.Super(s16, pel=2, sharp=2, levels=1), core.mv.Super(R, pel=2, sharp=2, levels=1)
    kw = dict(blksize=16, overlap=8, truemotion=1, search=3)
    t = tau * 256
    clips = [R]
    names = "yzabcdefghijklmnopqrstuvw"
    k = 0
    for d in range(1, radius + 1):
        for isb in (True, False):
            v = core.mv.Analyse(sa, isb=isb, delta=d, **kw)
            rc = core.mv.Compensate(R, sr, v, thsad=100000)
            sc = core.mv.Compensate(s16, ss, v, thsad=100000)
            err = core.std.BoxBlur(core.std.Expr([s16, sc], "x y - abs"), hradius=2, vradius=2)
            clips += [rc, err]
            k += 1
    # num/den over the centred values: ((R-32768) + sum w*(Rc-32768)) / (1 + sum w) + 32768
    num = "x 32768 -"
    den = "1"
    for j in range(k):
        a, e = names[2 * j], names[2 * j + 1]
        num += f" {a} 32768 - 1 {e} {t} / - 0 max * +"
        den += f" 1 {e} {t} / - 0 max +"
    Rs = core.akarin.Expr(clips, f"{num} {den} / 32768 +")
    res = core.std.Expr([s16, Rs], "x y + 32768 -")
    res = core.resize.Point(res, format=fmt.id) if res.format.id != fmt.id else res
    return core.std.CopyFrameProps(res, out)


def _ai_cleaner(job):
    """True when the neural dots + rainbows model will run (xtalk_stage_b). The classic LUMA filters are then
    left out: on the end-to-end benchmark (bench/, 11 clips of anime and live action through a simulated NTSC
    chain) they kept only 60 % of the finest detail and cost 4.7 dB, while the model alone kept 93 % and
    removed the dots better. The classic chroma (rainbow) cleaning still helps a little and stays."""
    if opt(job, "ai_model") in ("uni", "field"):       # the all-in-one models run whenever selected
        try:
            import stablenet
            if opt(job, "ai_model") == "field":
                import fieldnet
                return fieldnet.available(job.get("field_model"))
            return stablenet.available(stablenet.UNI_MODEL)
        except Exception:                                # noqa: BLE001
            return False
    if opt(job, "ai_model") == "off" or (opt(job, "crawl") == "off" and opt(job, "rainbow") == "off"):
        return False
    try:
        if opt(job, "ai_model") in ("stable", "uni"):
            import stablenet
            return stablenet.available(stablenet.UNI_MODEL if opt(job, "ai_model") == "uni" else None)
        import dvdnet
        return dvdnet.available(opt(job, "ai_model"))
    except Exception:                                    # noqa: BLE001
        return False


def _stablenet_on(job):
    """StableNet (stablenet.py) is the cleaner: it was trained on the raw IVTC output, so no classic crawl /
    rainbow filter may run before it; it removes dots, rainbows, MPEG flicker and noise and is steady by itself."""
    return opt(job, "ai_model") in ("stable", "uni", "field") and _ai_cleaner(job)


def uni_dials(job):
    """All-in-one model (ai_model "uni") dials from the job, 0..1: uni_crawl, uni_rainbow, uni_mpeg, uni_halo,
    uni_chroma (1 = remove), uni_grain (1 = keep grain / noise). Defaults: remove everything, keep no grain."""
    d = {"crawl": 1.0, "rainbow": 1.0, "mpeg": 1.0, "halo": 1.0, "chroma": 1.0, "grain_keep": 0.0, "detail": 1.0}
    for k, jk in (("crawl", "uni_crawl"), ("rainbow", "uni_rainbow"), ("mpeg", "uni_mpeg"), ("halo", "uni_halo"),
                  ("chroma", "uni_chroma"), ("grain_keep", "uni_grain"), ("detail", "uni_detail")):
        if job.get(jk) is not None:
            d[k] = max(0.0, min(1.0, float(job[jk])))
    return d


def repair_mode(job):
    """Comb repair actually used. With StableNet / the all-in-one model as cleaner: 'masked' instead of 'ai'
    (CombNet) - the models were trained on masked repair, and CombNet rebuilt fine still patterns (Courage's
    mesh door) differently every few frames: door flicker with the all-in-one model 2.35 -> 1.48 levels."""
    m = opt(job, "ivtc_repair")
    return "masked" if m == "ai" and opt(job, "ai_model") in ("stable", "uni", "field") else m


def xtalk_stage_a(src, job):
    """Before IVTC: cancel dot crawl (luma) and rainbows (chroma). 'light' = still areas only,
    'normal'/'strong' = field-based motion-compensated, so moving objects and pans are cleaned too."""
    if _stablenet_on(job):
        return src
    cr, rb = opt(job, "crawl"), opt(job, "rainbow")
    if opt(job, "composite") in ("chroma", "full", "clean", "on"):
        rb = "off"                       # chroma already re-separated by the composite decoder
    ty = CRAWL[cr]["t"] if cr != "off" and not _ai_cleaner(job) else None
    tc = RAINBOW[rb]["t"] if rb != "off" else None
    if ty is None and tc is None:
        return src
    motion = (cr != "off" and CRAWL[cr]["motion"]) or (rb != "off" and RAINBOW[rb]["motion"])
    if not motion:
        out = _cancel(src, src[:1] + src[:-1], src[1:] + src[-1:], ty, tc)
        return core.std.CopyFrameProps(out, src)
    tff = _order(job) == 1
    fl = core.std.SeparateFields(core.std.SetFieldBased(src, 2 if tff else 1), tff=tff)
    parts = []
    for S in (fl[0::2], fl[1::2]):
        cp, cn = _mc_pair(S)
        parts.append(_cancel(S, cp, cn, ty, tc))
    w = core.std.DoubleWeave(core.std.Interleave(parts), tff=tff)[::2]
    return core.std.CopyFrameProps(core.std.SetFieldBased(w, 2 if tff else 1), src)


# ---- colour stains where a strong rainbow was
FSC = 0.2652            # NTSC colour carrier in cycles per pixel at 720 px (3.58 / 13.5 MHz)


def carrier_mask(y, t_lo=4.0, t_hi=10.0):
    """y: GRAYS 0..1. Soft mask (GRAYS 0..1) of places whose luma has energy at the colour carrier: there a
    composite decoder cannot tell luma from colour, whatever colour it shows is not to be trusted."""
    taps = []
    for k in range(-6, 7):
        w = math.exp(-0.5 * (k / 2.6) ** 2)
        taps.append((k, w * math.cos(2 * math.pi * FSC * k), w * math.sin(2 * math.pi * FSC * k)))
    mc = sum(c for _, c, _ in taps) / len(taps)
    ec = " ".join(f"x[{k},0] {c - mc:.5f} *" for k, c, _ in taps) + " +" * (len(taps) - 1)
    es = " ".join(f"x[{k},0] {s:.5f} *" for k, _, s in taps) + " +" * (len(taps) - 1)
    e = core.akarin.Expr([y], f"{ec} dup * {es} dup * + sqrt 255 *")            # carrier amplitude, 8-bit levels
    e = core.std.BoxBlur(core.std.BoxBlur(e, hradius=3, vradius=2), hradius=3, vradius=2)
    return core.std.Expr([e], f"x {t_lo} - {t_hi - t_lo} / 0 max 1 min")


def stain_fill(clip, ref, strength=1.0, d_lo=3.0, d_hi=7.0, radius=8, v_lo=0.8, v_hi=2.0):
    """Colour stains left where a strong rainbow was. Fine luma hatching at the colour carrier (pencil shading,
    fabric texture) makes a rainbow so strong that after cleaning a grey-green blotch remains instead of the
    colour of the surface (Dominion Tank Police LD capture: the shading on the nurse's wrist).
    clip: cleaned YUV, ref: the same frames before the rainbow cleaning. A pixel is filled when all three hold:
      - the cleaning removed a strong rainbow there (blurred colour change d_lo..d_hi levels),
      - luma has energy at the colour carrier (so no decoder could have known the colour),
      - the colour that is left is blotchy (v_lo..v_hi); a flat painted area such as a nail is left alone.
    The fill colour comes from trusted pixels around the mask that have a similar brightness (5 soft bands):
    skin is filled from skin, a white bandage from white, dark lines from dark lines. Luma is not touched."""
    f = core.resize.Bicubic(clip, format=vs.YUV444PS, range_in=0, range=1)
    r = core.resize.Bicubic(ref, format=vs.YUV444PS, range_in=0, range=1)
    Y, U, V = (core.std.ShufflePlanes(f, i, vs.GRAY) for i in range(3))
    d = core.std.Expr([U, core.std.ShufflePlanes(r, 1, vs.GRAY), V, core.std.ShufflePlanes(r, 2, vs.GRAY)], "x y - abs z a - abs + 255 *")
    d = core.std.BoxBlur(core.std.BoxBlur(d, hradius=4, vradius=3), hradius=4, vradius=3)
    # ... AND the colour that is left there is blotchy (a stain); a flat fill such as a painted nail is not
    rough = core.std.Expr([U, core.std.BoxBlur(U, hradius=3, vradius=3), V, core.std.BoxBlur(V, hradius=3, vradius=3)], "x y - abs z a - abs + 255 *")
    rough = core.std.BoxBlur(core.std.BoxBlur(rough, hradius=4, vradius=3), hradius=4, vradius=3)
    m = core.std.Expr([d, carrier_mask(Y), rough], f"x {d_lo} - {d_hi - d_lo} / 0 max 1 min y * z {v_lo} - {v_hi - v_lo} / 0 max 1 min *")
    m = core.std.BoxBlur(core.std.Maximum(m), hradius=2, vradius=2)
    bad = core.std.Expr([core.std.BoxBlur(m, hradius=3, vradius=2)], "x 0.05 > 1 0 ?")
    w = core.std.Expr([bad], "1 x -")
    blur = lambda c, q: core.std.BoxBlur(core.std.BoxBlur(c, hradius=q, vradius=q), hradius=q, vradius=q)
    # The fill colour comes from trusted pixels of similar brightness (5 soft brightness bands): a pink nail
    # is filled from pink, the white bandage next to it from white, a dark line from dark lines.
    bands = [(c, core.std.Expr([Y, w], f"x {c} - dup * -{1 / (2 * 0.11 ** 2)} * exp y *")) for c in (0.1, 0.3, 0.5, 0.7, 0.9)]
    dens = [(c, wb, blur(wb, radius * 2)) for c, wb in bands]
    den0 = blur(w, radius * 3)
    outs = []
    for C in (U, V):
        num0 = blur(core.std.Expr([C, w], "x y *"), radius * 3)
        wide = core.std.Expr([num0, den0, blur(C, radius * 3)], "y 0.05 > x y / z ?")
        acc_n, acc_d = None, None
        for c, wb, db in dens:
            fb = core.std.Expr([blur(core.std.Expr([C, wb], "x y *"), radius * 2), db, wide], "y 0.02 > x y / z ?")
            # weight of this band for the pixel: its own brightness, and how much trusted support the band has
            ab = core.std.Expr([Y, db], f"x {c} - dup * -{1 / (2 * 0.11 ** 2)} * exp y 0.02 > 1 0.05 ? *")
            n = core.std.Expr([fb, ab], "x y *")
            acc_n = n if acc_n is None else core.std.Expr([acc_n, n], "x y +")
            acc_d = ab if acc_d is None else core.std.Expr([acc_d, ab], "x y +")
        fill = core.std.Expr([acc_n, acc_d, wide], "y 0.0001 > x y / z ?")
        outs.append(core.std.Expr([C, fill, m], f"x y x - z {strength} * 1 min * +"))
    out = core.std.ShufflePlanes([Y] + outs, [0, 0, 0], vs.YUV)
    out = core.resize.Bicubic(out, format=clip.format.id, range_in=1, range=0)
    return core.std.CopyFrameProps(out, clip)


def xtalk_stage_b(clip, job):
    """After IVTC: motion-compensated averaging over +-2 film frames. Chroma (rainbows) at full strength;
    luma (crawl/flicker) with a hard per-pixel change limit so detail and edges cannot be softened."""
    cr, rb = opt(job, "crawl"), opt(job, "rainbow")
    if _stablenet_on(job):
        import stablenet
        lv = max(DOTNET_STRENGTH.get(cr, 0.0), DOTNET_STRENGTH.get(rb, 0.0))
        if opt(job, "ai_model") == "field":            # FieldNet: also the raw telecined frames (before IVTC)
            import fieldnet
            fm = job.get("field_model") or fieldnet.MODEL
            kw = dict(dials=uni_dials(job), raw=source(job, clean=False), raw_start=int(job.get("start", 0)), pal=is_pal(job))
            import fieldnet5_vs
            if fieldnet5_vs.is_v5(fm):                 # v5: + backward pass, NeuFlow, TStab cascade, still lock (fieldnet5_vs.py)
                res = fieldnet5_vs.fieldnet5(clip, model=fm, lock=str(job.get("still_lock", "on")) != "off", **kw)
            elif fieldnet.is_v3(fm):                   # v3: motion-aligned memory, scene-cut handling (fieldnet.py)
                res = fieldnet.fieldnet(clip, model=fm, **kw)
                # steadiness stages after FieldNet (job "field_stab": filters (default) | off). Real Courage disc, v4:
                # door / rug flicker 1.11 / 1.02 -> 0.11 / 0.11, WELCOME letters shimmer 0.65 -> 0.04, wobble 1.12 -> 0.38,
                # sharpness 7.51 -> 7.45, colour 29.6 -> 29.5 (the user's pick: "great results, still area is the best")
                if str(job.get("field_stab", "filters")) == "filters":
                    # kept at 16-bit 4:4:4 from here on (finalize converts for the encoder): rounding the smoothed
                    # picture back to 8 bits between stages put the flicker back (door / rug 0.11 -> 0.31)
                    import tstab
                    res = core.resize.Bicubic(res, format=vs.YUV444P16)
                    for m_, lv_ in (("mc", "strong"), ("gtfix", "strong"), ("ttemp", "strong")):
                        res = stabilize(res, dict(job, stab=m_, stab_level=lv_))
                        if res.format.id != vs.YUV444P16:
                            res = core.resize.Bicubic(res, format=vs.YUV444P16)
                    res = tstab.still_lock(res)
                # TStab + still lock after FieldNet (tstab.py): Courage door / rug flicker 1.12 / 1.03 -> ~0.2,
                # WELCOME letters shimmer 0.67 -> 0.04, sharpness unchanged (measured with mc+gtfix+ttemp+lock; TStab is
                # the learned version). job "tstab": lock | stab | off (default off until the trained TStab is verified)
                ts_mode = str(job.get("tstab", "off"))
                if ts_mode in ("lock", "stab"):
                    import tstab
                    res = tstab.tstab(res, model=job.get("tstab_model"), lock=ts_mode == "lock")
            else:
                res = stablenet.stablenet(clip, 1.0, model=fm, **kw)
        elif opt(job, "ai_model") == "uni":
            res = stablenet.stablenet(clip, 1.0, model=stablenet.UNI_MODEL, dials=uni_dials(job))
        else:
            res = stablenet.stablenet(clip, strength=min(1.0, lv) if lv else 1.0)
        rb_on = rb != "off" or (opt(job, "ai_model") in ("uni", "field") and uni_dials(job)["rainbow"] > 0)   # all-in-one: its dial
        if rb_on and opt(job, "rainbow_fill") != "off":
            res = stain_fill(res, clip)
        return res
    lim = CRAWL[cr]["limit"] if cr != "off" and not _ai_cleaner(job) else 0
    tsc = RAINBOW[rb]["thsadc"] if rb != "off" else 0
    if not lim and not tsc and not _ai_cleaner(job):
        return clip
    pre = core.std.BoxBlur(clip, hradius=1, vradius=1)
    sa = core.mv.Super(pre, pel=2, sharp=2)
    sc = core.mv.Super(clip, pel=2, sharp=2, levels=1)
    kw = dict(blksize=16, overlap=8, truemotion=1, search=3)
    v = []
    for d in (1, 2):
        v += [core.mv.Analyse(sa, isb=True, delta=d, **kw), core.mv.Analyse(sa, isb=False, delta=d, **kw)]
    out = clip
    if tsc:
        out = core.mv.Degrain2(out, sc, *v, thsad=400, thsadc=tsc, plane=3)
    if lim:
        y = core.mv.Degrain2(clip, sc, *v, thsad=400, thsadc=400, plane=0, limit=lim)
        out = core.std.ShufflePlanes([y, out], [0, 1, 2], vs.YUV)
    import dvdnet
    ai = opt(job, "ai_model")
    if ai == "off":
        return out
    if (cr != "off" or rb != "off") and dvdnet.available(ai):
        # combined neural dots + rainbows remover (see dvdnet.py); strength follows the stronger setting
        lv = max(DOTNET_STRENGTH.get(cr, 0.0), DOTNET_STRENGTH.get(rb, 0.0))
        # no temporal_guard here: dot crawl flips every frame, so its correction must flip too (bench: guard -1.7 dB)
        res = dvdnet.dvdnet(out, strength=lv, mode=ai)
        if opt(job, "rainbow") != "off" and opt(job, "rainbow_fill") != "off":
            res = stain_fill(res, clip)
        return res
    if cr != "off":
        import dotnet
        if dotnet.available():          # neural dot remover: removes dots, leaves detail (see dotnet.py)
            out = dotnet.dotnet(out, strength=DOTNET_STRENGTH.get(cr, 1.0))
        elif DEDOT_PASSES.get(cr, 0):
            out = dedot_spatial(out, passes=DEDOT_PASSES[cr])
    return out


# spatial outline filter: OFF below "strong". On a clean (crawl-free) Blu-ray 2 passes changed the picture
# to 39.9 dB PSNR - it removes fine detail in the same ~4 px band and softened Lodoss / Courage, while a
# spatial+temporal consensus test showed most of what it removed there was static detail, not crawl.
# Leftover crawl / flicker is handled by the TTempSmooth stabilizer instead (55-59 dB on clean content).
DEDOT_PASSES = {"light": 0, "normal": 0, "strong": 1}
DOTNET_STRENGTH = {"light": 0.6, "normal": 1.0, "strong": 1.3}


def dedot_spatial(clip, passes=2, t=12):
    """Spatial comb dot-crawl removal for what the temporal stages cannot reach (dots crawling along
    moving outlines). NTSC has 227.5 subcarrier cycles per line, so residual chroma in luma flips sign on
    every line of a field (rows y-2, y, y+2 of the frame) and repeats every ~3.8 px horizontally.
    The same-field second difference, band-passed horizontally at ~4 px, isolates exactly that pattern;
    it is clamped to +-t (8-bit scale) so real edges cannot be changed much, and subtracted from luma.
    Used only at crawl "strong" (1 pass): it also takes fine detail from the same band (softens)."""
    k = 1 << (clip.format.bits_per_sample - 8) if clip.format.sample_type == vs.INTEGER else 1 / 255
    d = lambda dx: f"x[{dx},0] 2 * x[{dx},-2] - x[{dx},2] - 4 /"
    expr = f"{d(0)} 2 * {d(-2)} - {d(2)} - 4 / C! x C@ {-t * k} max {t * k} min -"
    y = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    for _ in range(passes):
        y = core.akarin.Expr([y], expr)
    return core.std.ShufflePlanes([y, clip], [0, 1, 2], vs.YUV)


# ---------------------------------------------------------------- ghosting / ringing (LGhost-style)

GHOST_SMAX = 48


def edge_profiles(clip, samples=50):
    """Average luma step response around strong, isolated vertical edges, polarity-normalised.
    Returns (right, left, noise): right[s] / left[s] = response s pixels after / before the edge."""
    n = clip.num_frames
    L = GHOST_SMAX + 2
    right = np.zeros(L); left = np.zeros(L); cnt = 0
    for t in np.linspace(n * 0.05, n * 0.95, samples).astype(int):
        Y = np.asarray(clip.get_frame(int(t))[0], dtype=np.float32)
        g = np.zeros_like(Y); g[:, 1:] = Y[:, 1:] - Y[:, :-1]
        thr = np.percentile(np.abs(g), 99.7)
        ys, xs = np.nonzero(np.abs(g) > thr)
        for y, x in zip(ys[::3], xs[::3]):
            if x < L + 4 or x + L + 4 >= Y.shape[1]:
                continue
            if np.abs(g[y, x - 3:x]).max() > thr * 0.5 or np.abs(g[y, x + 1:x + 4]).max() > thr * 0.9:
                continue                                    # isolated edges only
            right += g[y, x:x + L] / g[y, x]
            left += g[y, x - L + 1:x + 1][::-1] / g[y, x]
            cnt += 1
    if not cnt:
        return None, None, 1.0, 0
    right /= cnt; left /= cnt
    noise = float(np.std(np.diff(right[10:])) + 1e-6)
    return right, left, noise, cnt


def ghost_taps(right, left, noise, max_taps=6):
    """Echo taps (shift, amplitude): shift > 0 = to the right, < 0 = to the left. The edge's own blur is
    skipped (until the response first drops below 10%); after that anything clearly above noise is echo."""
    taps = []
    for side, prof in ((1, right), (-1, left)):
        start = next((s for s in range(1, len(prof)) if prof[s] < 0.10), len(prof))
        for s in range(max(2, start), len(prof) - 1):
            a = float(prof[s])
            if abs(a) > max(0.035, 5 * noise):
                taps.append((side * s, a))
    taps.sort(key=lambda t: -abs(t[1]))
    return taps[:max_taps]


def lghost(clip, taps, strength=1.0):
    """Remove echoes (ghosts) and halos from luma. Model: seen = true + sum(a * true shifted by s).
    Inverse FIR on the full signal: out = (y - sum(a * y[x - s])) / (1 - sum(a)) - the division keeps
    flat-area brightness exactly unchanged. Taps come from detect_ghost (s > 0 = echo to the right)."""
    taps = [(s, a * strength) for s, a in taps if a]
    if not taps:
        return clip
    c16 = core.resize.Point(clip, format=clip.format.replace(bits_per_sample=16).id) if clip.format.bits_per_sample < 16 else clip
    y = core.std.ShufflePlanes(c16, 0, vs.GRAY)
    norm = 1.0 - sum(a for _, a in taps)
    terms = " ".join(f"x[{-s},0] {a:.5f} * -" for s, a in taps)
    out = core.akarin.Expr([y], f"x {terms} {norm:.5f} /")
    out = core.std.ShufflePlanes([out, c16], [0, 1, 2], vs.YUV)
    return core.resize.Point(out, format=clip.format.id) if out.format.id != clip.format.id else out


def dehalo(clip, rx=2.0, ry=1.0, strength=1.0, lowsens=50, highsens=50):
    """Halo removal without smearing (DeHalo_alpha method by Didee), luma only.
    A smoothed copy (down/up-scaled by the halo radius) defines what the picture 'should' look like next to
    edges. Where the original has much more local contrast than the smoothed copy (fine detail, thin line
    art) the original is kept; elsewhere each pixel is clamped to the local min/max of that reference, which
    cuts off the overshoot / undershoot (the halo) while the edge itself - present in both - stays sharp."""
    if strength <= 0:
        return clip
    y = core.resize.Point(core.std.ShufflePlanes(clip, 0, vs.GRAY), format=vs.GRAYS)
    y = core.std.Expr([y], "x 255 *")                             # 8-bit-equivalent scale for the thresholds
    W, H = y.width, y.height
    sw, sh = max(4, int(round(W / rx / 4) * 4)), max(4, int(round(H / ry / 4) * 4))
    smooth = core.resize.Bicubic(core.resize.Bicubic(y, sw, sh), W, H, filter_param_a=1, filter_param_b=0)
    rng = lambda c: core.std.Expr([core.std.Maximum(c), core.std.Minimum(c)], "x y -")
    are, ugly = rng(y), rng(smooth)
    keep = core.std.Expr([ugly, are], f"y x - y 0.001 + / 255 * {lowsens} - y 256 + 512 / {highsens / 100} + * "
                                      f"0 max 255 min 255 /")
    lets = core.std.Expr([smooth, y, keep], "x 1 z - * y z * +")
    clamped = core.std.Expr([y, core.std.Minimum(lets), core.std.Maximum(lets)], "x y max z min")
    out = core.std.Expr([y, clamped], f"x y - {strength} * x swap - 255 / 0 max 1 min")
    out = core.resize.Point(out, format=clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id)
    return core.std.ShufflePlanes([out, clip], [0, 1, 2], vs.YUV)


DEBLOCK = {"light": 0.6, "normal": 1.0, "strong": 1.0}


def deblock(clip, job):
    """MPEG compression repair (blocknet.py). Runs first in the cleanup chain, on the progressive frames:
    compression was the last thing done to the picture, so it is the first thing to undo."""
    lv = opt(job, "deblock")
    if lv not in DEBLOCK:
        return clip
    try:
        import blocknet
        if blocknet.available():
            out = blocknet.blocknet(clip, strength=DEBLOCK[lv])
            return temporal_guard(clip, out) if GUARD.get("block", True) else out
    except Exception as e:                                # noqa: BLE001
        print(f"BlockNet not usable ({e}); compression repair skipped", file=sys.stderr)
    return clip


def ghost_halo(clip, job):
    """Halo removal without smearing. strength = number of passes (1.0 = one pass, 2.0 = two passes,
    fractions = a partial extra pass). Radius 2.0 measured best on 8 discs: 10-35% less halo per pass with
    the edges as sharp as the original (larger radii soften edges). The old inverse-echo method is not used
    for display any more - it widened edges by 60% and no disc had a real long-distance ghost."""
    if not job.get("halo_rx"):
        return clip
    s = max(0.0, min(2.0, float(opt(job, "ghost_strength"))))
    if opt(job, "halo_method") == "ai":
        # HaloNet (halonet.py): halos, ringing after edges and echoes up to 16 px in one pass, luma only.
        # Golden Boy / Courage: overshoot 42 -> 31 % / 18 -> 9 % of edge height (DeHalo 33 / 11), and it
        # changes a halo-free picture less than DeHalo does (47 vs 43 dB on simulated clean frames).
        try:
            import halonet
            if halonet.available():
                orig = clip
                clip = halonet.halonet(clip, strength=min(s, 1.0))
                if s > 1.01:
                    clip = halonet.halonet(clip, strength=s - 1.0)
                return temporal_guard(orig, clip)
        except Exception as e:                           # noqa: BLE001  (no torch / no weights -> classic)
            print(f"HaloNet not usable ({e!r}), using DeHalo", file=sys.stderr)
    while s >= 1.0:
        clip = dehalo(clip, rx=2.0)
        s -= 1.0
    if s > 0.01:
        clip = dehalo(clip, rx=2.0, strength=s)
    return clip


def detect_ghost(clip, iterations=8, samples=36, progress=None):
    """Find echo taps iteratively: measure the edge response, cancel the strongest remaining echo outside
    the edge's own blur, re-measure the corrected picture, repeat. Neighbouring echoes cannot double-count."""
    right, left, noise, cnt = edge_profiles(clip, samples)
    if right is None:
        return {"taps": [], "edges": 0, "halo": 0.0, "ghost": 0.0}
    before = (right, left)
    taps = {}
    for it in range(iterations):
        if progress:
            progress(it, iterations)
        cand = ghost_taps(right, left, noise, max_taps=1)
        if not cand:
            break
        s, a = cand[0]
        taps[s] = taps.get(s, 0.0) + 0.85 * a
        right, left, noise, _ = edge_profiles(lghost(clip, list(taps.items())), samples)
    t = sorted(taps.items(), key=lambda kv: -abs(kv[1]))
    halo = max([abs(v) for v in list(before[0][2:6]) + list(before[1][2:6]) if v < 0] or [0])
    ghost = max([abs(a) for s, a in t if abs(s) > 6] or [0])
    # halo distance = strongest negative lobe 2..6 px from the edge (either side) -> DeHalo radius
    lobes = [(before[0][k], k) for k in range(2, 7)] + [(before[1][k], k) for k in range(2, 7)]
    hv, hpos = min(lobes)
    halo_rx = 2.0 if hv < -0.05 else 0.0                       # radius 2.0 = best measured balance
    return {"taps": [(int(s), round(float(a), 4)) for s, a in t], "edges": cnt, "halo": round(float(halo), 3),
            "ghost": round(float(ghost), 3), "halo_rx": halo_rx, "halo_pos": int(hpos),
            "right": [round(float(v), 3) for v in before[0][:16]], "left": [round(float(v), 3) for v in before[1][:16]],
            "right_after": [round(float(v), 3) for v in right[:16]], "left_after": [round(float(v), 3) for v in left[:16]]}


# ---------------------------------------------------------------- source / IVTC

def fix_fps(src, job):
    """Raw MPEG-2 streams (.m2v) carry no timestamps: BestSource guessed 25.37 fps for SCA1 (29.97), and the VFR
    timecodes made from that rate stretched a 102 s clip to 120 s. Only a rate that is no video rate at all is
    replaced (pulldown applied, a DVD is 29.97 or 25 - a soft-telecine disc probes 23.976 but plays 29.97 here)."""
    f = float(src.fps)
    if any(abs(f - r) < 0.01 for r in (24000 / 1001, 24, 25, 30000 / 1001, 30, 50, 60000 / 1001, 60)):
        return src
    pal = job.get("fps_num") and abs(job["fps_num"] / job["fps_den"] - 25) < 0.5
    return core.std.AssumeFPS(src, fpsnum=25 if pal else 30000, fpsden=1 if pal else 1001)


_BAD_RE = re.compile(r"No frame returned for frame number (\d+)")


def _bad_file(path):
    """where the undecodable frames of this source are listed (temp\\bad_frames, keyed by path + size + date)"""
    import hashlib
    st = os.stat(path)
    key = hashlib.sha1(f"{os.path.abspath(path).lower()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]
    return os.path.join(paths.TEMP, "bad_frames", key + ".json")


def _bad_info(path):
    """{"bad": [frames], "scanned": True once the whole file was checked} (an older file is a plain list)"""
    try:
        with open(_bad_file(path), encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, list):
            d = {"bad": d}
        return {"bad": sorted({int(n) for n in d.get("bad", [])}), "scanned": bool(d.get("scanned"))}
    except Exception:                                  # noqa: BLE001  (none known, or no file)
        return {"bad": [], "scanned": False}


def _save_bad(path, info):
    f = _bad_file(path)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(info, fh)


def bad_frames(path):
    return _bad_info(path)["bad"]


def note_bad_frame(path, err):
    """An undecodable frame reported by BestSource ("No frame returned for frame number N": a damaged picture in the
    file - Kishin Douji Zenki 01 frame 7532, even with threads=1): list it, so source() replaces it from now on.
    Returns the frame number, or None when err is a different error."""
    m = _BAD_RE.search(str(err))
    if not m:
        return None
    n = int(m.group(1))
    info = _bad_info(path)
    info["bad"] = sorted(set(info["bad"]) | {n})
    _save_bad(path, info)
    print(f"Source frame {n} cannot be decoded (damaged in the file): it is replaced by its neighbour", file=sys.stderr, flush=True)
    return n


def _skip_bad(src, bad):
    """every undecodable frame becomes a copy of the nearest decodable one before it (after it at the start);
    FreezeFrames never asks the decoder for the replaced frame"""
    bad = [b for b in bad if 0 <= b < src.num_frames]
    if not bad:
        return src
    bs, first, last, repl = set(bad), [], [], []
    for b in bad:
        r = b - 1
        while r in bs:
            r -= 1
        if r < 0:
            r = b + 1
            while r in bs:
                r += 1
        first.append(b); last.append(b); repl.append(min(r, src.num_frames - 1))
    return core.std.FreezeFrames(src, first=first, last=last, replacement=repl)


def source(job, clean=True):
    src = core.bs.VideoSource(job["source"], rff=True)
    if not job.get("image"):
        src = _skip_bad(src, bad_frames(job["source"]))
    if job.get("image"):                               # a still: a short clip of identical frames (runner.IMAGE_FRAMES)
        n = int(job.get("end") or 24)
        src = core.std.AssumeFPS(src[0] * max(1, n), fpsnum=job.get("fps_num", 30000), fpsden=job.get("fps_den", 1001))
    src = fix_fps(src, job)
    tff = opt(job, "order") != "bff"
    # an image / progressive source has no fields: marked field-based, the 4:2:0 conversion below resized each FIELD
    # (an image 708x530 = two fields of 265 lines, odd -> "Resize error 1027")
    interlaced = not (job.get("image") or job.get("progressive"))
    src = core.std.SetFieldBased(src, (2 if tff else 1) if interlaced else 0)
    start, end = job.get("start", 0), job.get("end") or src.num_frames
    src = src[start:end]
    # DVDs often carry no colour tags ("unspecified") - tag them so viewers and resizers know the space
    m, p, t = sd_colour(job)
    if src.format.color_family == vs.RGB or src.format.id != vs.YUV420P8:
        # captures / intermediates (RGB Lagarith, 4:2:2 ...) instead of a DVD stream: bring them to the DVD
        # format the rest of the chain (field matching, the neural models) is built for
        kw = {"matrix": m, "range": vs.RANGE_LIMITED} if src.format.color_family == vs.RGB else {}
        vm = 4                                           # 4:2:0 needs even sizes, the field-based crawl stage even fields: an image
        if src.width % 2 or src.height % vm:             # of 708x531 failed with "Resize error 1027" - a few px off
            src = core.std.Crop(src, right=src.width % 2, bottom=src.height % vm)
        src = core.resize.Bicubic(src, format=vs.YUV420P8, dither_type="error_diffusion", **kw)
    src = core.std.SetFrameProps(src, _Matrix=m, _Primaries=p, _Transfer=t, _Range=vs.RANGE_LIMITED)
    GUARD["on"] = opt(job, "temporal_guard") != "off"
    src = core.akarin.PropExpr(src, lambda: {"_SrcN": f"N {int(start)} +"})      # scene grades (Colour step)
    if clean and opt(job, "composite") in ("chroma", "full", "clean", "on") and not _stablenet_on(job):
        src = composite_restore(src, job)            # StableNet does the composite repair itself (trained on raw input)
    if clean and not is_pal(job):
        src = xtalk_stage_a(src, job)
    if clean and vsf_at(job, "source"):                # your filters on the raw source (field matching / VFR analysis see them)
        src = vs_filters(src, job, "source")
    return src


def _order(job):
    return 0 if opt(job, "order") == "bff" else 1


def _deblend(clip, job):
    mode = opt(job, "deblend")
    if mode not in ("dup", "interp"):
        return clip
    import deblend
    return deblend.deblend(clip, mode)


def _aa(clip, job, strength=1.0):
    """AANet on the luma of whole frames (aanet.py). Returns clip unchanged if the model is not there."""
    try:
        import aanet
        if aanet.available():
            return aanet.aanet(clip, strength)
    except Exception as e:                                # noqa: BLE001
        print(f"AANet not usable ({e}); anti-aliasing skipped", file=sys.stderr)
    return clip


def _tfm(clip, job, **kw):
    """Field matching + repair of what cannot be matched + blend reduction (before decimation, so a blend
    replaced by its neighbour is the duplicate the decimator removes)."""
    return _deblend(_tfm_match(clip, job, **kw), job)


def raw_ivtc(job):
    """Field matching + decimation only, no repair of any kind (TFM PP=0, no comb repair, no safety net): for a
    FieldNet trained on such input (it repairs combing itself from the raw fields), and for making its data."""
    if job.get("raw_ivtc"):
        return True
    if opt(job, "ai_model") != "field":
        return False
    try:
        import fieldnet
        return fieldnet.wants_raw_ivtc(job.get("field_model"))
    except Exception:                                    # noqa: BLE001
        return False


def _tfm_match(clip, job, **kw):
    order = _order(job)
    if raw_ivtc(job):
        return core.tivtc.TFM(clip, order=order, cthresh=int(opt(job, "cthresh")), MI=int(opt(job, "mi")), PP=0, **kw)
    if repair_mode(job) in ("masked", "ai") and int(opt(job, "pp")) > 1:
        # TFM only marks the frames it cannot match cleanly (PP=1); comb_repair rebuilds the combed areas
        t = core.tivtc.TFM(clip, order=order, cthresh=int(opt(job, "cthresh")), MI=int(opt(job, "mi")), PP=1, **kw)
        return comb_repair(t, job, src=clip)
    deint = core.bwdif.Bwdif(clip, field=order)       # used by TFM to rebuild frames it finds combed
    return core.tivtc.TFM(clip, order=order, cthresh=int(opt(job, "cthresh")), MI=int(opt(job, "mi")),
                          PP=int(opt(job, "pp")), clip2=deint, **kw)


def comb_pixels(clip, cth=9, grow=2):
    """Pixel mask (GRAY8, soft edges) of combing: TFM's own test - a pixel differs from the lines above and
    below in the same direction by more than cth, and the 5-line pattern confirms it. Single pixels are
    dropped (combing comes in horizontal runs), then the mask is grown so the rebuilt area covers the
    whole comb tooth and blends in softly."""
    y = core.resize.Point(core.std.ShufflePlanes(clip, 0, vs.GRAY), format=vs.GRAY8)
    m = core.akarin.Expr([y], f"x x[0,-1] - A! x x[0,1] - B! A@ {cth} > B@ {cth} > and A@ -{cth} < B@ -{cth} < and or "
                              f"x[0,-2] x[0,2] + x 4 * + x[0,-1] x[0,1] + 3 * - abs {6 * cth} > and 255 0 ?")
    m = core.std.Expr(core.std.BoxBlur(m, hradius=2, vradius=0), "x 100 > 255 0 ?")     # at least 2 of 5 in the row
    for _ in range(grow):
        m = core.std.Maximum(m)
    return core.std.BoxBlur(m, hradius=2, vradius=2)


def _deint_frame(clip, order):
    """Same-rate deinterlace of single frames: NNEDI3 for the spatial part (clean lines), Bwdif decides
    per pixel whether the other field can be kept (no motion) or has to be interpolated."""
    nn = core.znedi3.nnedi3(clip, field=order, nsize=0, nns=3, qual=2)
    bw = core.bwdif.Bwdif(core.std.SetFieldBased(clip, 2 if order else 1), field=order, edeint=nn)
    return core.std.SetFieldBased(bw, 0)


def _combnet_deint(src, order):
    """Frames rebuilt by CombNet (combnet.py): the source bobbed field by field (NNEDI3 + Bwdif, double
    rate - exactly what the model was trained on), CombNet corrects the rebuilt lines of each field-frame
    from its neighbours, every second one = the frame rebuilt from field `order` of source frame n (the
    field TFM keeps)."""
    import combnet
    il = core.std.SetFieldBased(src, 2 if order else 1)
    nn = core.znedi3.nnedi3(il, field=order + 2, nsize=0, nns=3, qual=2)
    bob = core.std.SetFieldBased(core.bwdif.Bwdif(il, field=order + 2, edeint=nn), 0)
    cn = combnet.combnet(bob, lambda k: (k % 2 == 0) == bool(order))
    return cn[::2]


def comb_repair(clip, job, prop="_Combed", flagged=None, src=None):
    """Repair of frames that are still combed after field matching (30i titles over film, orphan fields
    at cuts, bad edits). Old behaviour: the whole frame was deinterlaced - half the vertical detail gone
    and stair-steps on every line of the frame, although usually only a part of it is combed.
    Now only the combed areas are rebuilt (mask from comb_pixels); everything else stays the matched,
    full-resolution frame. Faint leftover combing around the rebuilt area is flattened with Vinverse
    under a wider, more sensitive mask. Frames that are not flagged are passed through bit-exact."""
    order = _order(job)
    cth = int(opt(job, "cthresh"))
    base = core.std.SetFieldBased(clip, 0)
    deint = None
    if src is not None and repair_mode(job) == "ai":
        try:
            import combnet
            if combnet.available():
                deint = _combnet_deint(src, order)
        except Exception as e:                            # noqa: BLE001
            print(f"CombNet not usable ({e}); NNEDI3 + Bwdif repair", file=sys.stderr)
    if deint is None:
        deint = _deint_frame(clip, order)
    # Combing only exists where something moves: in a still area both fields show the same picture, even if they
    # were matched wrongly. Fine still patterns (a mesh door, hatching, small text) look like combing to the
    # detector; rebuilding them from one field every few frames made them jump (Courage's door: a visible jump
    # every 5-9 frames). So only pixels that are combed AND move (blurred frame differs from a neighbour by
    # more than ~2.5 levels; the blur hides crawl and the pattern itself) are rebuilt.
    sc = (1 << base.format.bits_per_sample) / 256.0 if base.format.sample_type == vs.INTEGER else 1 / 255.0
    yb = core.std.BoxBlur(core.std.ShufflePlanes(base, 0, vs.GRAY), hradius=3, vradius=3)
    prv, nxt = yb[0] + yb[:-1], yb[1:] + yb[-1]
    dif = core.std.Expr([yb, prv, nxt], "x y - abs x z - abs max")
    dif = core.std.BoxBlur(dif, hradius=4, vradius=4)         # the change over an area, not single crawl-hit pixels
    mot = core.std.Expr([dif], f"x {3.0 * sc} > {255 * sc if sc >= 1 else 1} 0 ?")
    mot = core.std.Maximum(core.std.Maximum(mot))
    mot = core.resize.Point(mot, format=vs.GRAY8) if mot.format.id != vs.GRAY8 else mot
    strong = core.std.Expr([comb_pixels(base, cth), mot], "x y min")
    if opt(job, "aa") == "repair":
        # the rebuilt lines come from one field only: stair-steps on diagonals. AANet smooths them; it runs
        # only on flagged frames and only its result inside the rebuilt area is used.
        deint = _aa(deint, job)
    fixed = core.std.MaskedMerge(base, deint, strong, first_plane=True)
    weak = core.std.Expr([comb_pixels(fixed, max(4, cth // 2), grow=1), mot], "x y min")
    fixed = core.std.MaskedMerge(fixed, vinverse(fixed), weak, first_plane=True)
    if flagged is None:
        sel = lambda n, f: fixed if f.props.get(prop, 0) else base
        return core.std.FrameEval(base, sel, prop_src=clip)
    return core.std.FrameEval(base, lambda n, f: fixed if flagged(f) else base, prop_src=clip)


def _tdec(clip, job, **kw):
    return core.tivtc.TDecimate(clip, dupThresh=float(opt(job, "dupthresh")),
                                vidThresh=float(opt(job, "vidthresh")), **kw)


def vinverse(clip, sstr=2.7, scl=0.25):
    """Vinverse (Didee): removes residual combing from progressive frames; only vertical oscillations
    that look like combing are flattened, real detail is kept."""
    neutral = 1 << (clip.format.bits_per_sample - 1) if clip.format.sample_type == vs.INTEGER else 0
    vblur = core.std.Convolution(clip, [50, 99, 50], mode="v")
    vblur_d = core.std.MakeDiff(clip, vblur)
    vshrp = core.std.Expr([vblur, core.std.Convolution(vblur, [1, 4, 6, 4, 1], mode="v")], f"x x y - {sstr} * +")
    vshrp_d = core.std.MakeDiff(vshrp, vblur)
    n = neutral
    expr = (f"x {n} - y {n} - * 0 < x {n} - abs y {n} - abs < x y ? {n} - {scl} * {n} + "
            f"x {n} - abs y {n} - abs < x y ? ?")
    return core.std.MergeDiff(vblur, core.std.Expr([vshrp_d, vblur_d], expr))


SPIKE_MIN = 30        # fine-comb value (motion-gated interlace signature, densest 5x5 area, 0-255) a frame needs ...
SPIKE_RATIO = 3.0     # ... and how many times the largest of its 2 + 2 neighbours'


def fine_comb(clip, prop="CF"):
    """per frame <prop>Max: densest 5x5 area (0-255) of MOVING pixels with the interlace signature (_combmask's test),
    the picture's 8-px edges left out"""
    y = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    y = core.resize.Point(y, format=vs.GRAY8) if y.format.id != vs.GRAY8 else y
    sig = core.akarin.Expr([y], "x x[0,-1] - x x[0,1] - * 400 > x x[0,-2] - abs 12 < x x[0,2] - abs 12 < and and 255 0 ?")
    yb = core.std.BoxBlur(y, hradius=3, vradius=3)
    dif = core.std.BoxBlur(core.std.Expr([yb, yb[0] + yb[:-1], yb[1:] + yb[-1]], "x y - abs x z - abs max"), hradius=4, vradius=4)
    m = core.std.BoxBlur(core.std.Expr([sig, dif], "y 3 > x 0 ?"), hradius=2, vradius=2)
    return core.std.PlaneStats(core.std.Crop(m, 8, 8, 8, 8), prop=prop)


def comb_spikes(clip):
    """per frame: CombFine = densest 5x5 area of moving pixels with the interlace signature (_combmask's test), and
    CombNb = the largest of the 2 frames before and after. A frame field matching left combed stands out against its
    neighbours; drawn detail (hatching, mesh, fine lines) is there in every frame, so it does not."""
    fine = fine_comb(clip)
    nb = [_shift(fine, o) for o in (-2, -1, 1, 2)]
    return core.akarin.PropExpr([clip, fine] + nb, lambda: {"CombFine": "y.CFMax", "CombNb": "z.CFMax a.CFMax max b.CFMax c.CFMax max max"})


def is_spike(f, overlay=None):
    if overlay and f.props.get("SrcN", -1) in overlay:   # a frame of a combed overlay (overlay_comb_frames)
        return True
    v = f.props.get("CombFine", 0)
    return v >= SPIKE_MIN and v >= SPIKE_RATIO * max(1, f.props.get("CombNb", 0))


def comb_safety_net(clip, job):
    """Last line of defence after IVTC: combing that field matching could not fix (true interlaced
    footage, bad edits, orphan fields).
      safe   - frames whose detector max > 110 are fully deinterlaced with Bwdif (original behaviour)
      nnedi3 - same detection, rebuilt with Bwdif using NNEDI3 for the spatial part (sharper lines)
      strong - more sensitive detection (> 60) + NNEDI3 repair + Vinverse on every frame (faint combing)"""
    mode = opt(job, "comb")
    if mode == "off":
        return clip
    order = _order(job)
    ov = job.get("_overlay") or None
    spike = lambda f: is_spike(f, ov)
    clip = core.std.SetFieldBased(clip, 0)
    st = core.std.PlaneStats(_combmask(clip), plane=0)
    st = core.std.CopyFrameProps(comb_spikes(clip), st, props=["PlaneStatsMax"])
    if mode == "safe":
        deint = core.bwdif.Bwdif(core.std.SetFieldBased(clip, 2 if order else 1), field=order)
        deint = core.std.SetFieldBased(deint, 0)
        out = core.std.FrameEval(clip, lambda n, f: deint if f.props["PlaneStatsMax"] > 110 or spike(f) else clip, prop_src=st)
        return core.std.SetFieldBased(out, 0)
    limit = 60 if mode == "strong" else 110
    if opt(job, "ivtc_repair") in ("masked", "ai"):
        # same detection, but only the combed areas of the frame are rebuilt
        stc = core.std.CopyFrameProps(clip, st, props=["PlaneStatsMax", "CombFine", "CombNb"] + (["SrcN"] if ov else []))
        out = comb_repair(stc, job, flagged=lambda f: f.props["PlaneStatsMax"] > limit or spike(f))
        if mode == "strong":
            out = vinverse(out)
        return core.std.SetFieldBased(out, 0)
    # NNEDI3 interpolates the dominant field; Bwdif's temporal check is used where there is motion
    nn = core.znedi3.nnedi3(clip, field=order, nsize=0, nns=3, qual=2)
    bw = core.std.SetFieldBased(core.bwdif.Bwdif(core.std.SetFieldBased(clip, 2 if order else 1), field=order, edeint=nn), 0)
    # whole frame: a comb mask misses low-contrast combing (e.g. colour-only lines in flat areas)
    out = core.std.FrameEval(clip, lambda n, f: bw if f.props["PlaneStatsMax"] > limit or spike(f) else clip, prop_src=st)
    if mode == "strong":
        out = vinverse(out)
    return core.std.SetFieldBased(out, 0)


# ---------------------------------------------------------------- colour of frames IVTC assembled from two
def _align_score(y, ym1, yp1, u, v):
    """How well the colour edges sit on the brightness edges in the moving parts (y at chroma resolution)."""
    def grad(a):
        return np.abs(np.diff(a, axis=1, prepend=a[:, :1])) + np.abs(np.diff(a, axis=0, prepend=a[:1]))
    mov = (np.abs(y - ym1) > 12) | (np.abs(y - yp1) > 12)
    if mov.mean() < 0.02:
        return None
    ey = grad(y) * mov
    ec = (grad(u) + grad(v)) * mov
    return float((ey * ec).sum() / np.sqrt((ey ** 2).sum() * (ec ** 2).sum() + 1e-9))


def chroma_unmix(dec, src, job):
    """Telecined discs whose colour was halved vertically as if the video were progressive: every source frame's
    colour is the mix of its two fields' film frames. IVTC rebuilds the film frames that only exist split over
    two source frames (match p / n) - and their colour carries the film frames before and after it (coloured
    ghosts trailing moving objects; Transformers DVD: 27 % of moving frames, 2-4 % on clean discs).
    Undone: source colour X_s = (C + B)/2, X_s2 = (C + D)/2 with B, D = the colour of the neighbouring output
    frames, so C = ((2 X_s - B) + (2 X_s2 - D)) / 2, never outside the colours the three frames have at that
    spot (no new colours), and only kept when the colour then sits better on the outlines than before.
    Frames IVTC took whole (match c) are not touched."""
    if opt(job, "chroma_unmix") != "on" or dec.format.color_family != vs.YUV:
        return dec
    from scipy.ndimage import maximum_filter, minimum_filter
    fmt = dec.format
    cw, ch = dec.width >> fmt.subsampling_w, dec.height >> fmt.subsampling_h
    yl = core.resize.Bilinear(core.std.ShufflePlanes(dec, 0, vs.GRAY), cw, ch, format=vs.GRAYS)
    sc = 255.0 / ((1 << fmt.bits_per_sample) - 1)
    src = core.resize.Point(src, format=fmt.id) if src.format.id != fmt.id else src
    nsrc = src.num_frames

    def shifted(other):
        # clip whose frame n is source frame s (+1 / -1 for the partner field's frame) of output frame n
        def ev(n, f):
            s = int(f.props.get("TDecimateOriginalFrame", n))
            m = int(f.props.get("TFMMatch", 1))
            if other:
                s = s + 1 if m == 2 else s - 1
            s = max(0, min(nsrc - 1, s))
            return src[s - n:] if s >= n else (src[0] * (n - s) + src)
        return core.std.FrameEval(dec, ev, prop_src=dec)
    sa, sb = shifted(False), shifted(True)
    prv, nxt = dec[:1] + dec[:-1], dec[1:] + dec[-1:]
    yp, yn = yl[:1] + yl[:-1], yl[1:] + yl[-1:]

    def proc(n, f):
        cur = f[0]
        if int(cur.props.get("TFMMatch", 1)) == 1 or n == 0 or n == dec.num_frames - 1:
            return cur
        Y = [np.asarray(x[0]) * 255 for x in (f[5], f[6], f[7])]          # prev, cur, next (0..255)
        P = lambda fr, p: np.asarray(fr[p]).astype(np.float32) * sc
        o = [P(cur, 1), P(cur, 2)]
        base = _align_score(Y[1], Y[0], Y[2], *o)
        if base is None:
            return cur
        best, bs = None, base + 0.02
        for a, b in ((f[1], f[2]), (f[2], f[1])):
            res = []
            for p in (1, 2):
                pa, pb = P(a, p), P(b, p)
                c = ((2 * P(f[3], p) - pa) + (2 * P(f[4], p) - pb)) / 2
                lo = minimum_filter(np.minimum(np.minimum(o[p - 1], pa), pb), 3) - 2
                hi = maximum_filter(np.maximum(np.maximum(o[p - 1], pa), pb), 3) + 2
                res.append(np.clip(c, lo, hi))
            s_ = _align_score(Y[1], Y[0], Y[2], *res)
            if s_ is not None and s_ > bs:
                best, bs = res, s_
        if best is None:
            return cur
        dst = cur.copy()
        for p in (1, 2):
            np.copyto(np.asarray(dst[p]), (best[p - 1] / sc + 0.5).astype(np.asarray(dst[p]).dtype))
        dst.props["ChromaUnmixed"] = 1
        return dst
    return core.std.ModifyFrame(dec, clips=[dec, prv, nxt, sa, sb, yp, yl, yn], selector=proc)


def measure_chroma_unmix(job, samples=160):
    """Analyze: how often frames IVTC assembled from two source frames have colour from a neighbouring frame,
    against frames taken whole. On when assembled frames are off far more often (Transformers: 39 % vs 6 %)."""
    src = source(job, clean=False)
    dec = _tdec(_tfm(src, dict(job, ivtc_repair="masked")), job, mode=0)
    fmt = dec.format
    cw, ch = dec.width >> fmt.subsampling_w, dec.height >> fmt.subsampling_h
    yl = core.resize.Bilinear(core.std.ShufflePlanes(dec, 0, vs.GRAY), cw, ch, format=vs.GRAYS)
    sc = 255.0 / ((1 << fmt.bits_per_sample) - 1)
    cnt = {1: [0, 0], 2: [0, 0]}
    n = dec.num_frames
    for t in np.linspace(n * 0.05, n * 0.95, samples * 3).astype(int):
        t = int(t)
        if t < 2 or t > n - 3:
            continue
        m = int(dec.get_frame(t).props.get("TFMMatch", 1))
        m = 1 if m == 1 else 2
        Y = [np.asarray(yl.get_frame(t + d)[0]) * 255 for d in (-1, 0, 1)]
        sco = []
        for k in (-1, 0, 1):
            fr = dec.get_frame(t + k)
            s_ = _align_score(Y[1], Y[0], Y[2], np.asarray(fr[1]) * sc, np.asarray(fr[2]) * sc)
            sco.append(s_)
        if sco[1] is None:
            continue
        cnt[m][0] += 1
        cnt[m][1] += int(max(sco[0], sco[2]) > sco[1] + 0.02)
        if cnt[1][0] + cnt[2][0] >= samples:
            break
    ra = cnt[2][1] / max(cnt[2][0], 1)
    rc = cnt[1][1] / max(cnt[1][0], 1)
    on = cnt[2][0] >= 10 and ra >= 0.2 and ra >= 3 * max(rc, 0.02)
    return {"chroma_unmix": "on" if on else "off", "unmix_assembled": round(100 * ra, 1), "unmix_whole": round(100 * rc, 1),
            "unmix_frames": cnt[2][0]}


def decision_clip(clip, strength=1):
    """What field matching and decimation should LOOK at: dot crawl and rainbows flip from field to field, so a
    repeated field can differ from its twin only by its inverted crawl and mislead both. Luma smoothed along the
    lines (the dot pattern is horizontal, the vertical structure combing detection needs stays), colour flattened.
    strength 2: a wider smoothing (heavy crawl)."""
    y = core.std.Convolution(clip, matrix=[1, 2, 3, 2, 1], mode="h", planes=[0])
    for _ in range(max(0, int(strength) - 1)):
        y = core.std.Convolution(y, matrix=[1, 4, 6, 4, 1], mode="h", planes=[0])
    mid = 1 << (clip.format.bits_per_sample - 1) if clip.format.sample_type == vs.INTEGER else 0.0
    return core.std.Expr([y], ["", str(mid), str(mid)])


def _matched_from(m_cl, clip, job):
    """the untouched frames, field-matched the way TFM decided on the crawl-free copy: per source frame k, its kept
    field with the other field of k (c), k-1 (p) or k+1 (n), woven from the real fields. (TFM's clip2 is not honoured
    here either: it returned the grey decision copy.)"""
    order = _order(job)
    tff = order == 1
    f = core.std.SeparateFields(core.std.SetFieldBased(clip, 2 if tff else 1), tff=tff)
    first, second = f[0::2], f[1::2]
    prv, nxt = second[0] + second[:-1], second[1:] + second[-1]
    weave = lambda a, b: core.std.SetFieldBased(core.std.DoubleWeave(core.std.Interleave([a, b]), tff=tff)[::2], 0)
    var = {1: core.std.SetFieldBased(clip, 0), 0: weave(first, prv), 2: weave(first, nxt)}

    def sel(n, f):
        return var.get(int(f.props.get("TFMMatch", 1)), var[1])[n]
    out = core.std.FrameEval(core.std.BlankClip(var[1], length=m_cl.num_frames), sel, prop_src=m_cl)
    return core.std.CopyFrameProps(out, m_cl)


def _apply_decisions(dec_cl, matched, job):
    """frames of `matched` (the field-matched untouched frames, one per source frame) in the order the decimation of
    the crawl-free copy chose them: output frame n = matched[_SrcN of n - start]. (TDecimate's own clip2 is not
    honoured here - it returned the grey decision copy.)"""
    s0 = int(job.get("start", 0))
    nm = matched.num_frames

    def sel(n, f):
        k = int(f.props.get("_SrcN", s0 + n)) - s0
        return matched[min(max(k, 0), nm - 1)]
    return core.std.FrameEval(core.std.BlankClip(matched, length=dec_cl.num_frames), sel, prop_src=dec_cl)


def _raw_matched(clip, job, mode):
    """raw IVTC: field matching + decimation only. ivtc_decide 'clean' (default): decided on decision_clip(),
    applied to the untouched frames."""
    if opt(job, "ivtc_decide") != "clean":
        return _tdec(_tfm(clip, job), job, mode=mode)
    order = _order(job)
    cl = core.std.SetFieldBased(decision_clip(clip, int(job.get("ivtc_decide_strength", 2))), 2 if order else 1)
    kw = dict(order=order, cthresh=int(opt(job, "cthresh")), MI=int(opt(job, "mi")), PP=0, **dict(job.get("ivtc_tfm_kw") or {}))
    m_cl = core.tivtc.TFM(cl, **kw)
    m_raw = _matched_from(m_cl, clip, job)
    return _apply_decisions(_tdec(m_cl, job, mode=mode, **dict(job.get("ivtc_tdec_kw") or {})), m_raw, job)


def film_rate(clip):
    """A 23.976p source with no pulldown (MakeMKV / pre-IVTC rips: Ken, HxH temp1.mkv): already film frames - field
    matching + decimation would drop every 5th REAL frame (TDecimate removed 16 % of HxH's frames)."""
    return abs(float(clip.fps) - 24000 / 1001) < 0.01 or abs(float(clip.fps) - 24) < 0.01


def ivtc_cfr(clip, job):
    if film_rate(clip) or job.get("progressive"):
        return clip
    if raw_ivtc(job):
        return _raw_matched(clip, job, 0)
    return comb_safety_net(chroma_unmix(_tdec(_tfm(clip, job), job, mode=0), clip, job), job)


def _clean_decisions(job):
    """raw IVTC deciding on the crawl-free copy (ivtc_decide 'clean')"""
    return raw_ivtc(job) and opt(job, "ivtc_decide") == "clean"


def _decision_tfm(clip, job, **kw):
    """TFM deciding on the crawl-free copy; clip2 (if given) gets the decisions applied"""
    order = _order(job)
    cl = core.std.SetFieldBased(decision_clip(clip, int(job.get("ivtc_decide_strength", 2))), 2 if order else 1)
    base = dict(order=order, cthresh=int(opt(job, "cthresh")), MI=int(opt(job, "mi")), PP=0, **dict(job.get("ivtc_tfm_kw") or {}))
    return core.tivtc.TFM(cl, **base, **kw)


def _with_diff(tfm):
    """field-matched clip + prop "VDiffDiff": mean change from the previous frame (small luma copy, 0..1) - the
    clean motion measure video_sections() reads (TDecimate's own block metric is dominated by MPEG / crawl noise:
    SCA1 held frames 40, real changes 60, a slow pan 45)"""
    y = core.resize.Bilinear(core.std.ShufflePlanes(tfm, 0, vs.GRAY), 240, 160, format=vs.GRAYS)
    st = core.std.PlaneStats(y, y[0] + y[:-1], prop="VDiff")
    # + "VBlkMax": the change of the most-changed area (15 x 10 areas of ~48 px): credits / captions moving over a film
    # picture change one strip only - the frame mean showed the film's repeat and the section was decimated to 24p
    d = core.std.Expr([y, y[0] + y[:-1]], "x y - abs")
    b = core.resize.Bilinear(d, 15, 10)
    blk = core.std.PlaneStats(b, prop="VBlk")
    # + "VLocRMax": per area, this frame's change against the larger change of its two neighbours (areas changing
    # >= 6 levels only), the largest over the frame. An overlay moving at video rate changes as much at the film's
    # repeat as around it (Bebop's titles 0.90-1.02); a blended / noisy part of the film picture about half as much
    # (GTO's arm 0.64-0.82)
    t = 6 / 255
    lr = core.std.Expr([b, b[0] + b[:-1], b[1:] + b[-1]], f"x {t} >= x y z max 0.0001 max / 0 ?")
    loc = core.std.PlaneStats(lr, prop="VLocR")
    st = core.std.CopyFrameProps(st, blk, props=["VBlkMax"])
    st = core.std.CopyFrameProps(st, loc, props=["VLocRMax"])
    # + "VCombMax": combing left in the matched frame (fine_comb) - an overlay laid on at 60 fields/s stays combed in
    # every frame (Bleach 190's credits), a blend (GTO's arm) does not
    st = core.std.CopyFrameProps(st, fine_comb(tfm, prop="VComb"), props=["VCombMax"])
    return core.std.CopyFrameProps(tfm, st, props=["VDiffDiff", "VBlkMax", "VLocRMax", "VCombMax"])


SEEK_PREROLL = 20          # BestSource's default seekpreroll (frames decoded before a seek target)


_SCAN_LOCKS = {}


def scan_source(path, progress=True):
    """(once per file; a second caller - the background scan started when the file was opened, and an analysis that
    needs it - waits for the first instead of scanning again)"""
    import threading
    lk = _SCAN_LOCKS.setdefault(os.path.normcase(os.path.abspath(path)), threading.Lock())
    with lk:
        return _scan_source(path, progress)


def _scan_source(path, progress=True):
    """Decode every frame of the source once (BestSource alone, no filters) and list the frames that cannot be decoded.
    Done once per file (remembered in temp\bad_frames). Frame by frame on ONE decoder: BestSource reports a damaged
    frame at once while it reads straight on, but asking again for it - or restarting a read inside the damaged
    stretch - sends it back to the start of the file to recover, ~150 s each (Zenki 01: 36 min -> about a minute)."""
    import time
    info = _bad_info(path)
    if info["scanned"]:
        return info["bad"]
    src = core.bs.VideoSource(path, rff=True)
    n, bad, t0 = src.num_frames, set(info["bad"]), time.time()
    if progress:
        print("Checking the source for damaged frames (once per file)...", file=sys.stderr, flush=True)
    for k in range(n):
        try:
            src.get_frame(k)
        except vs.Error as e:
            if not _BAD_RE.search(str(e)):
                raise
            # the 20 frames after it too: reaching any of them by a jump (a filter that skips the replaced frame)
            # makes BestSource seek 20 frames back (its seekpreroll) and decode through the damage again - each such
            # miss cost about a minute in the VFR pass 1 (Zenki 01: 7532 -> 7533..7552 all failed that way)
            bad.update(range(k, min(n, k + SEEK_PREROLL + 1)))
        if progress and k % 5000 == 0:
            print(f"PROGRESS {k} {n}", flush=True)
    info = {"bad": sorted(bad), "scanned": True}
    _save_bad(path, info)
    if progress:
        print(f"Source checked in {time.time() - t0:.0f} s" + (f": {len(bad)} damaged frame(s), each replaced by its neighbour" if bad else ""),
              file=sys.stderr, flush=True)
    return info["bad"]


def vfr_pass1(job):
    """VFR pass 1. Damaged frames are found first by one fast scan of the file (scan_source) and replaced by their
    neighbours; if one still turns up, it is listed and the pass starts again."""
    try:
        scan_source(job["source"])
    except Exception as e:                             # noqa: BLE001  (the scan is a speed-up: pass 1 still copes)
        print(f"Damaged-frame scan skipped: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    for _ in range(200):
        try:
            return _vfr_pass1(job)
        except vs.Error as e:
            n = note_bad_frame(job["source"], e)
            if n is None:
                raise
            _scan_bad_run(job["source"], n)
    raise RuntimeError("too many undecodable frames in the source")


def _scan_bad_run(path, n, reach=600):
    """a damaged GOP usually loses a run of frames (Zenki 01: 7532-7552): list the whole run now, so the pass
    restarts once instead of once per frame. Stops at the first frame that decodes."""
    src = core.bs.VideoSource(path, rff=True)
    for k in range(n + 1, min(src.num_frames, n + reach)):
        try:
            src.get_frame(k)
            return
        except vs.Error as e:
            if note_bad_frame(path, e) is None:
                return


def _vfr_pass1(job):
    clip = source(job)
    if _clean_decisions(job):                         # VFR pass 1 (metrics) on the crawl-free copy too
        tfm = _decision_tfm(clip, job, output=job["work"] + "/tfm.txt")
        dec = _tdec(_with_diff(tfm), job, mode=4, output=job["work"] + "/tdec.txt", **dict(job.get("ivtc_tdec_kw") or {}))
    else:
        tfm = _tfm(clip, job, output=job["work"] + "/tfm.txt")
        dec = _tdec(_with_diff(tfm), job, mode=4, output=job["work"] + "/tdec.txt")
    total = dec.num_frames
    diff, blk, loc, comb = [], [], [], []
    for n, f in enumerate(dec.frames(close=True)):
        diff.append(float(f.props.get("VDiffDiff", -1)) * 255)
        blk.append(float(f.props.get("VBlkMax", -1)) * 255)
        loc.append(float(f.props.get("VLocRMax", 0)))
        comb.append(float(f.props.get("VCombMax", 0)))
        if n % 500 == 0:
            print(f"PROGRESS {n} {total}", flush=True)
    print(f"PROGRESS {total} {total}", flush=True)
    with open(job["work"] + "/diff.txt", "w") as fh:
        fh.writelines("%.3f\n" % d for d in diff)
    with open(job["work"] + "/blk.txt", "w") as fh:
        fh.writelines("%.3f\n" % d for d in blk)
    with open(job["work"] + "/loc.txt", "w") as fh:
        fh.writelines("%.3f\n" % d for d in loc)
    with open(job["work"] + "/comb.txt", "w") as fh:
        fh.writelines("%.1f\n" % d for d in comb)


def _local_video(blk, diff, cyc, floor_min=4.0, focus=10.0, loc=None, keep=0.85):
    """per 5-frame cycle: a PART of the picture moves at video rate over telecined film - credits or a caption
    scrolling over a film picture (blk: VBlkMax = change of the most-changed of 150 areas, diff: VDiffDiff = mean
    change of the frame, both levels). Telecined film repeats every area once per cycle; at that repeat frame the
    overlay still changes in its own area while the rest of the picture stands still, so the change is concentrated:
    most-changed area / frame mean >= focus. Measured at the repeat: synthetic credits over a film pan ~32, Bebop's
    video-rate opening titles 13-19. MPEG noise on the duplicate of a fast film pan is spread over the whole picture
    and can be strong in the busiest area (GTO opening 10-23 levels) but stays at 3-7.5 (GTO, Angel Densetsu,
    Shin Hakkenden film parts, Golden Boy), so those stay film. The overlay also has to change visibly in every
    frame of the cycle (>= floor_min levels and 4x the source's floor).
    loc (VLocRMax): and the overlay changes at the repeat about as much as in the frames around it (>= keep). GTO's
    opening has an arm that is a double image (blend) at the face's repeat: concentrated (ratio ~15) but only half
    the change -> kept 30p it only brought the uneven 3:2 rhythm back to the face. Without loc.txt (older pass 1):
    off - the rule does nothing."""
    if not blk or not diff or not loc:
        return [False] * cyc
    floor = max(0.1, float(np.percentile(blk, 5)))
    out = []
    for k in range(cyc):
        v, d = blk[5 * k:5 * k + 5], diff[5 * k:5 * k + 5]
        if len(v) < 5 or len(d) < 5:
            out.append(False)
            continue
        r = int(np.argmin(d))                             # the film's repeat (smallest frame change)
        lr = loc[5 * k + r] if 5 * k + r < len(loc) else 0.0
        out.append(min(v) >= max(floor_min, 4 * floor) and v[r] >= focus * max(d[r], 0.05) and lr >= keep)
    return out


OVERLAY_COMB = 11     # median fine-comb value (0-255) of a cycle's matched frames: the overlay is interlaced
OVERLAY_RUN = 6       # cycles (~1 s) in a row
OVERLAY_GAP = 3       # cycles without it (text standing still, between two cards) bridged inside a run


def overlay_comb_frames(blk, diff, comb, floor_min=4.0, focus=0.0):
    """source frames (pass 1 numbering) of a COMBED video-rate overlay over film: credits / captions laid on at 60
    fields/s. Like _local_video, the overlay changes in every frame of the cycle (>= floor_min levels; its other test,
    concentration at the film's repeat, passed only half of Bleach's credit cycles and is not needed: changing in
    every frame AND combed marked 0 cycles of the rest of the episode) - but it fails _local_video's
    repeat test (it changes only about half as much there: 0.56 on Bleach 190's ending, where a 30p overlay keeps
    >= 0.85), and keeping it 30p would not help: it is combed in every field-matched frame. So it stays film (24p)
    and comb_safety_net rebuilds its combed, moving pixels. GTO's blended arm is concentrated too, but not combed."""
    if not blk or not diff or not comb:
        return set()
    cyc = min(len(blk), len(diff), len(comb)) // 5
    floor = max(0.1, float(np.percentile(blk, 5)))
    hit = []
    for k in range(cyc):
        v, d, c = blk[5 * k:5 * k + 5], diff[5 * k:5 * k + 5], comb[5 * k:5 * k + 5]
        r = int(np.argmin(d))
        hit.append(min(v) >= max(floor_min, 4 * floor) and v[r] >= focus * max(d[r], 0.05) and float(np.median(c)) >= OVERLAY_COMB)
    idx = [k for k in range(cyc) if hit[k]]               # short gaps (text standing still, a cut) bridged
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= OVERLAY_GAP + 1:
            for k in range(a + 1, b):
                hit[k] = True
    out, a = set(), None
    for k in range(cyc + 1):
        h = k < cyc and hit[k]
        if h and a is None:
            a = k
        elif not h and a is not None:
            if k - a >= OVERLAY_RUN:
                out.update(range(5 * a, 5 * k))
            a = None
    return out


def _overlay_frames(job):
    """overlay_comb_frames from the work folder's pass 1 (comb.txt only from pass 1 version 7 on)"""
    w = job.get("work") or ""
    try:
        if not os.path.exists(w + "/comb.txt"):
            return set()
        return overlay_comb_frames(_read_diff(w + "/blk.txt"), _read_diff(w + "/diff.txt"), _read_diff(w + "/comb.txt"))
    except Exception as e:                               # noqa: BLE001
        print(f"overlay check failed: {e}", file=sys.stderr)
        return set()


def video_sections(diff, matches=None, run=4, window=6, shifted=0.6, blk=None, loc=None):
    """Sections (of whole 5-frame cycles) that are NOT telecined film, from the frame changes of pass 1 (levels).
    After field matching, telecined film repeats one frame per cycle AT THE SAME POSITION cycle after cycle (hard
    or soft pulldown). Animation made at 29.97p (SCA1) also repeats frames - drawings held 3 or 2 frames - but the
    repeats move through the cycle (period 3 or 2 against 5), and pans change every frame. TDecimate alone took
    those held drawings for film: short 24p runs all through SCA1 (38 with raw IVTC, up to 148 frames with
    FieldNet), a real frame dropped each time and the pans stuttered.
    Per moving cycle: the repeated positions (change near the cycle's floor); film = a run of cycles sharing one
    (below), or field matches that keep needing the other field (`matches`, pass 1 'c' / 'p' / 'n': >= `shifted`
    per cycle within +-window cycles). Angel Densetsu is telecined film whose cadence shifts almost every cycle
    (repeat at position 2, 1, 0, 3, 1, 4 ...): no shared position, but a 'p' match in nearly every cycle, while
    29.97p progressive needs none (SCA1: 0.17 per cycle, cut and noise matches). Every other cycle stays video.
    Returns [(a, b)] frames, b exclusive."""
    n = len(diff)
    cyc = n // 5
    # the source's noise floor (a repeated frame / still picture): MPEG + crawl noise, grain - 0.15 levels on SCA1
    floor = max(0.05, float(np.percentile(diff, 5))) if n else 0.05
    kind, reps = [], []
    for k in range(cyc):
        v = diff[5 * k:5 * k + 5]
        lo, hi = min(v), max(v)
        if lo >= 2 * floor and lo >= 0.6 * hi:           # every frame changes (slow pan: SCA1 1.3 levels each)
            kind.append("v"); reps.append(()); continue  # - film always has its repeat at the floor
        if hi < 3 * floor + 0.3:                         # still
            kind.append("s"); reps.append(()); continue
        r = tuple(j for j in range(5) if v[j] < max(2 * floor + 0.3, 1.6 * lo) and v[j] < 0.3 * hi)
        kind.append("m" if r else "v"); reps.append(r)
    # film: a run of >= `run` moving cycles (still cycles in between don't count) sharing one repeated position.
    # Cuts made after the telecine move the position (Idaten Jump: every scene), so the run restarts there.
    # 29.97p drawings held h frames share a position over at most h-1 cycles (threes: 2, twos: 1).
    film = [False] * cyc
    for i in range(cyc):
        if kind[i] != "m":
            continue
        common, members, k = set(reps[i]), [i], i + 1
        while k < cyc and kind[k] != "v":
            if kind[k] == "m":
                nxt = common & set(reps[k])
                if not nxt:
                    break
                common = nxt
                members.append(k)
            k += 1
        if len(members) >= run:
            for m in range(members[0], members[-1] + 1):
                film[m] = True
    if matches:
        nonc = [0] * cyc
        for i, m in enumerate(matches[:5 * cyc]):
            nonc[i // 5] += m not in ("c", "C")
        for k in range(cyc):
            win = nonc[max(0, k - window):k + window + 1]
            if sum(win) >= shifted * len(win):
                film[k] = True
    # everything without that evidence stays 30p: a still shot looks the same either way, and SCA1's near-still
    # slow pans (changes 0.6-1.0 levels, at its noise floor) were decimated by TDecimate otherwise.
    # A part of the picture moving at video rate (credits / captions over film, blk) stays 30p even over film: the film
    # evidence (its repeat, its p / n matches) comes from the rest of the picture
    loc = _local_video(blk, diff, cyc, loc=loc)
    video = [not f or l_ for f, l_ in zip(film, loc)]
    out, a = [], None
    for k in range(cyc + 1):
        v = k < cyc and video[k]
        if v and a is None:
            a = k
        elif not v and a is not None:
            if k - a >= 2:                               # a lone cycle: noise, leave it to TDecimate
                out.append((5 * a, 5 * k))
            a = None
    return out


def _read_diff(path):
    with open(path) as f:
        return [float(x) for x in f if x.strip()]


def _read_matches(path):
    """TFM's log (tfm.txt) -> match letter per frame ('c' where a frame has no entry)"""
    m = {}
    try:
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 2 and p[0].isdigit():
                    m[int(p[0])] = p[1]
    except OSError:
        return []
    return [m.get(i, "c") for i in range(max(m) + 1)] if m else []


def _video_ovr(job):
    """TDecimate override file forcing the video sections (video_sections) to stay 30p, or None."""
    if opt(job, "vfr_video") == "off":
        return None
    w = job["work"]
    try:
        sec = (video_sections(_read_diff(w + "/diff.txt"), _read_matches(w + "/tfm.txt"),
                              blk=_read_diff(w + "/blk.txt") if os.path.exists(w + "/blk.txt") else None,
                              loc=_read_diff(w + "/loc.txt") if os.path.exists(w + "/loc.txt") else None)
               if os.path.exists(w + "/diff.txt") else [])
    except Exception as e:                               # noqa: BLE001
        print(f"video section check failed: {e}", file=sys.stderr)
        return None
    if not sec:
        return None
    with open(w + "/tdec_ovr.txt", "w") as f:
        f.writelines(f"{a},{b - 1} v\n" for a, b in sec)
    return w + "/tdec_ovr.txt"


def ivtc_vfr(clip, job):
    if film_rate(clip) or job.get("progressive"):      # no telecine to undo (and no pass 1 to read)
        return clip
    w = job["work"]
    ovr = _video_ovr(job)
    kw = {"ovr": ovr} if ovr else {}
    if _clean_decisions(job):                         # pass 2: the logged decisions, applied to the untouched frames
        m_cl = _decision_tfm(clip, job, input=w + "/tfm.txt")
        m_raw = _matched_from(m_cl, clip, job)
        dec = _tdec(m_cl, job, mode=5, hybrid=2, input=w + "/tdec.txt", tfmIn=w + "/tfm.txt",
                    mkvOut=w + "/timecodes.txt", tcfv1=0, **{**kw, **dict(job.get("ivtc_tdec_kw") or {})})
        return _apply_decisions(dec, m_raw, job)
    ov = _overlay_frames(job)
    if ov and not raw_ivtc(job):                      # combed credits / captions: their source frames are marked
        clip = core.akarin.PropExpr([clip], lambda: {"SrcN": "N"})
        job = dict(job, _overlay=ov)
        print(f"Combed video-rate overlay (credits / captions over film): {len(ov)} frames - its combing is rebuilt",
              file=sys.stderr, flush=True)
    tfm = _tfm(clip, job, input=w + "/tfm.txt")
    dec = _tdec(tfm, job, mode=5, hybrid=2, input=w + "/tdec.txt", tfmIn=w + "/tfm.txt",
                mkvOut=w + "/timecodes.txt", tcfv1=0, **kw)
    if raw_ivtc(job):
        return dec
    return comb_safety_net(chroma_unmix(dec, clip, job), job)


# ---------------------------------------------------------------- PAL

def _ok_frames(clip, picks, path=None):
    """the sample frames that can be decoded (a damaged frame is listed for the source and left out, so an analysis
    that samples frames does not stop at it - Zorro DVD 01: "No frame returned for frame number 1360")"""
    ok = []
    for k in picks:
        try:
            clip.get_frame(int(k))
            ok.append(int(k))
        except vs.Error as e:
            if not _BAD_RE.search(str(e)):
                raise
            if path:
                note_bad_frame(path, e)
    return ok


def pal_detect(job, samples=240):
    """Classify a PAL source: progressive, field-shifted (fixable by field matching) or interlaced.
    Field matching is tried with both field orders; the one that removes the combing wins."""
    if job.get("source") and not job.get("image"):
        try:
            scan_source(job["source"], progress=False)   # damaged frames known first: jumping onto one costs minutes
        except Exception as e:                           # noqa: BLE001
            print(f"Damaged-frame scan skipped: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    src = source(job)
    n = src.num_frames
    picks = np.linspace(min(500, n // 10), n - 1 - min(500, n // 10), min(samples, n)).astype(int)
    moving = core.std.PlaneStats(src, src[1:] + src[-1:], plane=0)
    picks = _ok_frames(moving, [k for k in picks if k + 2 < n], job.get("source"))
    picks = [k for k in picks if moving.get_frame(k).props["PlaneStatsDiff"] >= 0.004]
    if not picks:
        return {"mode": "progressive", "order": opt(job, "order"), "raw_combed": 0.0, "matched_combed": 0.0}

    def combed_rate(clip):
        st = core.std.PlaneStats(_combmask(clip), plane=0)
        return sum(st.get_frame(k).props["PlaneStatsMax"] > 110 for k in picks) / len(picks)

    raw_c = combed_rate(src)
    best = None
    for order in (1, 0):
        c = core.std.SetFieldBased(src, 2 if order else 1)
        m_c = combed_rate(core.vivtc.VFM(c, order=order, mode=1, cthresh=9, mi=80, micmatch=0))
        if best is None or m_c < best[0]:
            best = (m_c, "tff" if order else "bff")
    m_c, order = best
    if raw_c < 0.05:
        mode = "progressive"
    elif m_c < 0.10:
        mode = "fieldmatch"
    else:
        # truly interlaced: with the wrong field order the bobbed frames run back and forth in time
        # (1,0,3,2,...), so the step from the 2nd field of one frame to the 1st field of the next
        # (odd bob index -> even) is much bigger than with the right order (0,1,2,3,...).
        mode = "deint50"
        step = {}
        for o in (1, 0):
            b = core.bwdif.Bwdif(core.std.SetFieldBased(src, 2 if o else 1), field=o + 2)
            d = core.std.PlaneStats(b, b[1:] + b[-1:])
            step[o] = sum(d.get_frame(2 * k + 1).props["PlaneStatsDiff"] for k in picks[:80])
        order = "tff" if step[1] <= step[0] else "bff"
    return {"mode": mode, "order": order, "raw_combed": raw_c, "matched_combed": m_c}


def _combmask(clip):
    """Interlace-signature detector: pixel differs from both vertical neighbours in the same direction
    but matches the rows two above/below. Block-averaged; frame max > 110 = visibly combed."""
    y = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    y = core.resize.Point(y, format=vs.GRAY8)
    c = core.akarin.Expr([y], "x x[0,-1] - x x[0,1] - * 400 > x x[0,-2] - abs 12 < x x[0,2] - abs 12 < and and 255 0 ?")
    return core.std.BoxBlur(c, hradius=4, vradius=4)


_PAL_CACHE = {}


def _shift(c, o):
    """frame k of the result = frame k + o of c (edges repeated)"""
    if o > 0:
        return c[o:] + c[-1] * o
    if o < 0:
        return c[0] * (-o) + c[:o]
    return c


def pal_film_flags(m, radius=1):
    """Per source frame of the field-matched clip m (TFM, PP=1): film (25p progressive or field-shifted) or true 50i
    video? Film always has a clean match (TFM's comb value of the chosen match 0-5 on the synthetic mix), 50i video
    in motion has none (150+). A frame counts as film when it and its +-radius neighbours matched clean, so a single
    clean frame inside moving video (motion stopping for a moment) does not switch back and forth. -> prop _PalFilm"""
    clips = [_shift(m, o) for o in range(-radius, radius + 1)]

    def proc(n, f):
        out = f[radius].copy()
        out.props["_PalFilm"] = int(all(not x.props.get("_Combed", 0) for x in f))
        return out
    return core.std.ModifyFrame(m, clips=clips, selector=proc)


def _pal_keep_film(bob, clip, job, mode):
    """deint50 / deint25 on a source that mixes film with 50i video (PAL anime: field-shifted film + video-made credits
    or CG): the film frames are field-matched whole (full vertical detail) instead of being rebuilt from single fields;
    in 50p each is shown twice. Video frames stay QTGMC (bench_ivtc2, pal_* clips: see README "PAL")."""
    order = _order(job)
    m = core.tivtc.TFM(core.std.SetFieldBased(clip, 2 if order else 1), order=order, cthresh=int(opt(job, "cthresh")),
                       MI=int(opt(job, "mi")), PP=1)
    flags = pal_film_flags(m)
    matched = core.std.SetFieldBased(m, 0)
    matched = core.resize.Point(matched, format=bob.format.id) if matched.format.id != bob.format.id else matched
    if mode == "deint25":
        base = bob[::2]
        out = core.std.FrameEval(base, lambda n, f: matched if f.props["_PalFilm"] else base, prop_src=flags)
        return core.std.CopyFrameProps(out, base)
    m2 = core.std.Interleave([matched, matched])
    f2 = core.std.Interleave([flags, flags])
    out = core.std.FrameEval(bob, lambda n, f: m2 if f.props["_PalFilm"] else bob, prop_src=f2)
    return core.std.CopyFrameProps(out, bob)


def pal_process(clip, job):
    mode = job.get("pal_mode_resolved") or opt(job, "pal_mode")
    if mode == "auto":                       # preview / vsview / auto-detect: classify once and reuse
        key = (job["source"], job.get("start"), job.get("end"))
        if key not in _PAL_CACHE:
            _PAL_CACHE[key] = pal_detect(job)
        det = _PAL_CACHE[key]
        mode, job = det["mode"], dict(job, order=det["order"])
    order = _order(job)
    if mode == "progressive":
        return _deblend(core.std.SetFieldBased(clip, 0), job)
    if mode == "fieldmatch":
        if _clean_decisions(job):                     # PAL dot / cross-colour patterns mislead matching too
            return core.std.SetFieldBased(_matched_from(_decision_tfm(clip, job), clip, job), 0)
        return core.std.SetFieldBased(_tfm(clip, job), 0)
    try:
        from vsdeinterlace import QTempGaussMC
        bob = QTempGaussMC().deinterlace(core.std.SetFieldBased(clip, 2 if order else 1))
    except Exception:
        bob = core.bwdif.Bwdif(clip, field=order + 2)
    bob = core.std.SetFieldBased(bob, 0)
    # source frame and field of every output frame (FieldNet reads the raw fields around it): bob frame N shows
    # field N % 2 of source frame N // 2
    s0 = int(job.get("start", 0))
    bob = core.akarin.PropExpr(bob, lambda: {"_SrcN": f"N 2 / floor {s0} +", "_Phase": "N 2 %"})
    if opt(job, "aa") == "repair":                        # every bobbed frame is a deinterlaced one
        bob = _aa(bob, job)
    if opt(job, "pal_film") != "off" and not raw_ivtc(job):
        out = _pal_keep_film(bob, clip, job, mode)
    else:
        out = bob[::2] if mode == "deint25" else bob
    return _deblend(out, job)


# ---------------------------------------------------------------- chroma

EXTREME_RELAX = 0.0       # how much the colour check relaxes for near-white / near-black pixels (0 = off)
AI_BLEED = {"light": (0.6, 1), "normal": (1.0, 1), "strong": (1.0, 2)}       # ChromaNet (blend, passes)
REBUILD = {"normal": dict(radius=4, sig_s=3.0, sig_y=0.06, sig_c=0.05),
           "strong": dict(radius=5, sig_s=4.0, sig_y=0.08, sig_c=0.06)}


def _trilateral_expr(radius, sig_s, sig_y, sig_c, plane):
    """akarin Expr: inputs x=U y=V z=Y a=confidence. Weighted mean of the chroma plane over a window,
    weights = spatial * luma similarity * chroma similarity * confidence of the neighbour."""
    import math
    sy2 = 2 * sig_y ** 2
    cur = "x" if plane == "u" else "y"
    # colour-similarity width: strict for mid-tones (keeps e.g. yellow vs lavender of equal brightness apart),
    # relaxed for brightness extremes - white text / eye whites / black outlines are identified by
    # brightness alone, so their fringe colour (bled in from neighbours) can be replaced by their real colour
    e = [f"z 0.72 - 0.20 / 0 max 0.22 z - 0.12 / 0 max + 1 min {EXTREME_RELAX} * {sig_c} + dup * 2 * SC2!",
         "0 num! 0 den!"]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ws = math.exp(-(dx * dx + dy * dy) / (2 * sig_s ** 2))
            if ws < 0.02:
                continue
            o = f"[{dx},{dy}]"
            w = f"z{o} z - dup * {sy2} / x{o} x - dup * y{o} y - dup * + SC2@ / + -1 * exp {ws:.5f} * a{o} *"
            e.append(f"{w} w! num@ w@ {cur}{o} * + num! den@ w@ + den!")
    e.append("num@ den@ 1e-9 + /")
    return " ".join(e)


def chroma_rebuild(clip444, strength="normal", iters=2):
    """Edge-aware chroma reconstruction (confidence-weighted trilateral filter).
    DVD chroma is quarter resolution and blurred, so next to line art it is a blend of the fill colour
    and the line/neighbour colour. Each chroma pixel is rebuilt from nearby pixels that have similar
    luma AND similar colour (same paint region), trusting pixels away from edges most. Fill colour is
    carried right up to the lines without leaking between regions of similar brightness."""
    p = REBUILD[strength]
    f = core.resize.Point(clip444, format=vs.YUV444PS)
    Y, U, V = (core.std.ShufflePlanes(f, i, vs.GRAY) for i in range(3))
    grad = core.akarin.Expr([Y], "x[1,0] x[-1,0] - abs x[0,1] x[0,-1] - abs max")
    conf = core.std.Maximum(core.std.Maximum(grad))
    conf = core.akarin.Expr([conf], "1 x 20 * - 0.02 max dup *")      # 1 in flat areas, ~0 at edges
    eu = _trilateral_expr(p["radius"], p["sig_s"], p["sig_y"], p["sig_c"], "u")
    ev = _trilateral_expr(p["radius"], p["sig_s"], p["sig_y"], p["sig_c"], "v")
    for _ in range(iters):
        U, V = core.akarin.Expr([U, V, Y, conf], eu), core.akarin.Expr([U, V, Y, conf], ev)
        conf = core.akarin.Expr([conf], "x 0.5 + 1 min")
    out = core.std.ShufflePlanes([Y, U, V], [0, 0, 0], vs.YUV)
    out = core.resize.Point(out, format=clip444.format.id)
    return core.std.CopyFrameProps(out, clip444)


def neutralize(clip444):
    """Remove the colour cast that DVD's half-resolution chroma leaves on thin white and black detail
    (lyrics / subtitles, eye whites, highlights, outlines): near-white or near-black pixels that are only
    slightly tinted get the tint removed (soft ramps). Real colours - bright yellow, sky, saturated pastels -
    are left alone because they are either not that bright/dark or clearly saturated."""
    f = core.resize.Point(clip444, format=vs.YUV444PS)
    # luma 0..1 (limited range 16..235 -> ~0.06..0.92); chroma -0.5..0.5
    # weight: brightness ramp (white 0.80->0.88, black 0.18->0.12) x low-saturation ramp (0.10 -> 0.05)
    # white side: bright (0.80 -> 0.88) and only lightly tinted (sat 0.10 -> 0.05)
    # black side: only true black line art - very dark (0.12 -> 0.09) and almost colourless (0.05 -> 0.025),
    # so dark navy skies, shadows and dark clothing keep their colour
    # a = gate for the white side (see below): 0 where the bright area around is itself tinted (pale skin)
    w = ("y dup * z dup * + sqrt S! "
         "x 0.80 - 0.08 / 0 max 1 min 0.10 S@ - 0.05 / 0 max 1 min * a * "
         "0.12 x - 0.03 / 0 max 1 min 0.05 S@ - 0.025 / 0 max 1 min * max")
    Y, U, V = (core.std.ShufflePlanes(f, i, vs.GRAY) for i in range(3))
    # only on thin detail next to strong edges (text strokes, outlines, eye whites): large pale or dark
    # areas have enough chroma resolution and carry intentional tints - they are left alone
    grad = core.akarin.Expr([Y], "x[1,0] x[-1,0] - abs x[0,1] x[0,-1] - abs max")
    near = core.std.Maximum(core.std.Maximum(grad))
    edge = core.std.Expr([near], "x 0.08 - 0.08 / 0 max 1 min")
    # A cast is bleed only if the bright surface the pixel belongs to is colourless. Take the colour of the
    # flat bright pixels around (away from edges, where DVD chroma is reliable): if that colour is a real tint
    # (pale skin, cream walls, pastel sky) the pixel next to the line keeps it - removing it painted a white
    # outline around every line on pale skin (Blue Dragon). Thin detail with no flat bright area nearby
    # (text strokes, small highlights) is still neutralised.
    B = core.akarin.Expr([Y, near], "x 0.78 - 0.04 / 0 max 1 min 1 y 20 * - 0 max *")
    blur = lambda c: core.std.BoxBlur(c, hradius=8, vradius=6, hpasses=2, vpasses=2)
    Bb, Ub, Vb = blur(B), blur(core.std.Expr([U, B], "x y *")), blur(core.std.Expr([V, B], "x y *"))
    gate = core.akarin.Expr([Ub, Vb, Bb], "z 0.03 < 1 0.045 x z / dup * y z / dup * + sqrt - 0.02 / 0 max 1 min ?")
    wt = core.std.Expr([core.akarin.Expr([Y, U, V, gate], w), edge], "x y *")
    U2 = core.std.Expr([U, wt], "x 1 y - *")
    V2 = core.std.Expr([V, wt], "x 1 y - *")
    out = core.std.ShufflePlanes([Y, U2, V2], [0, 0, 0], vs.YUV)
    return core.std.CopyFrameProps(core.resize.Point(out, format=clip444.format.id), clip444)


def chroma_fix(clip, shift_x=0.0, shift_y=0.0, warp=0, method=None, strength="normal"):
    """Return YUV444P16 with chroma moved by shift_x/shift_y LUMA pixels (+ = right/down), then bleed fix:
    method "ai" = ChromaNet (neural, colour only), "rebuild" = edge-aware reconstruction, "awarp" = warp chroma edges onto luma edges
    (fast, depth = warp), "off" = none."""
    if method is None:
        method = "awarp" if warp else "off"
    clip = core.resize.Point(clip, format=clip.format.replace(bits_per_sample=16, sample_type=vs.INTEGER).id)
    y = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    u = core.std.ShufflePlanes(clip, 1, vs.GRAY)
    v = core.std.ShufflePlanes(clip, 2, vs.GRAY)
    w, h = clip.width, clip.height
    sl = -shift_x / 2 - 0.25          # MPEG-2 left chroma siting -> centred, then user shift
    st = -shift_y / 2
    u = core.resize.Spline36(u, w, h, src_left=sl, src_top=st)
    v = core.resize.Spline36(v, w, h, src_left=sl, src_top=st)
    out = core.std.ShufflePlanes([y, u, v], [0, 0, 0], vs.YUV)
    out = core.std.CopyFrameProps(out, clip)
    if method == "ai":                                   # neural bleed repair; falls back to the rebuild filter
        import chromanet
        if chromanet.available():
            blend, passes = AI_BLEED.get(strength, (1.0, 1))
            before = out
            for _ in range(passes):
                out = chromanet.chromanet(out, blend)
            return temporal_guard(before, out)
        method = "rebuild"
    if method == "rebuild":
        return chroma_rebuild(out, strength if strength in REBUILD else "normal")
    if method == "awarp" and warp:
        mask = core.std.Sobel(y, scale=2)
        mask = core.std.BoxBlur(mask, hradius=2, vradius=2, hpasses=2, vpasses=2)
        mask = core.std.ShufflePlanes([mask, mask, mask], [0, 0, 0], vs.YUV)
        out = core.awarp.AWarp(out, mask, depth_h=[0, warp, warp], depth_v=[0, warp, warp], planes=[1, 2])
    return out


def _grad(a):
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    return gx, gy


def _best_shift(gl, gc, axis):
    m = 6
    cl = np.abs(gl[m:-m, m:-m])
    sc = np.array([np.sum(cl * np.abs(np.roll(gc, -s, axis=axis)[m:-m, m:-m])) for s in range(-4, 5)])
    i = int(np.argmax(sc))
    d = (sc[i - 1] - sc[i + 1]) / (2 * (sc[i - 1] - 2 * sc[i] + sc[i + 1])) if 0 < i < 8 else 0.0
    return i - 4 + d


def _edge_width(p):
    lo, hi = p.min(), p.max()
    if hi - lo < 0.02:
        return None
    t = (p - lo) / (hi - lo)
    return int(np.sum((t > 0.1) & (t < 0.9)))


def _edge_width_plateau(p):
    """Like _edge_width, but the step is taken between the plateau levels (median of the outer 3 samples on
    each side) instead of min / max, so a halo's over- and undershoot does not count as part of the edge."""
    lo, hi = np.median(p[:3]), np.median(p[-3:])
    if abs(hi - lo) < 0.02:
        return None
    t = (p - lo) / (hi - lo)
    return int(np.sum((t > 0.1) & (t < 0.9)))


def chroma_stats(clip, samples=30, progress=None):
    """Measure chroma drift (luma px) and bleed (px wider than luma edges) on a 4:4:4 clip."""
    c = core.resize.Point(clip, format=vs.YUV444PS)
    n_total = c.num_frames
    margin = min(1500, n_total // 10)
    picks = np.linspace(margin, n_total - 1 - margin, samples).astype(int)
    xs, ys, bleed, bleed_p = [], [], [], []
    for k, n in enumerate(picks):
        if progress:
            progress(k, len(picks))
        f = c.get_frame(int(n))
        Y, U, V = (np.asarray(f[i], dtype=np.float32) for i in range(3))
        gx, gy = _grad(Y)
        if np.mean(np.abs(gx)) < 0.004:          # flat / black frame
            continue
        strong = np.abs(gx) > np.percentile(np.abs(gx), 99.5)
        yy, xx = np.nonzero(strong[8:-8, 8:-8])
        pts = list(zip(yy + 8, xx + 8))[::25]
        for C in (U, V):
            cx, cy = _grad(C)
            xs.append(_best_shift(gx, cx, 1))
            ys.append(_best_shift(gy, cy, 0))
            wd = []
            for py, px in pts:
                a, b = _edge_width(C[py, px - 6:px + 7]), _edge_width(Y[py, px - 6:px + 7])
                if a is not None and b is not None:
                    wd.append(a - b)
            if wd:
                bleed.append(float(np.median(wd)))
            wp = []
            for py, px in pts:
                a, b = _edge_width_plateau(C[py, px - 6:px + 7]), _edge_width_plateau(Y[py, px - 6:px + 7])
                if a is not None and b is not None:
                    wp.append(a - b)
            if wp:
                bleed_p.append(float(np.mean(wp)))
    if not xs:
        return None
    # Two estimates. The min/max one is what the thresholds were calibrated on, but strong halos make luma edges
    # look as wide as the colour edges (Golden Boy: luma 8.0 px, chroma 7.7 px -> "-0.1, no bleed"). The plateau
    # one ignores halos and reads 2.7 there, 3.1-3.8 on the other bleeding discs; minus 1.5 it sits at or below
    # the first on those, so taking the larger changes nothing for them and catches the halo case.
    b0 = float(np.mean(bleed)) if bleed else 0.0
    bp = float(np.mean(bleed_p)) if bleed_p else 0.0
    return {"x": float(np.median(xs)), "y": float(np.median(ys)), "bleed": max(b0, bp - 1.5),
            "bleed_minmax": b0, "bleed_plateau": bp}


def measure_xtalk(job, samples=100, progress=None):
    """Strength of NTSC composite crosstalk, luma (dot crawl / line flicker) and chroma (rainbows)
    separately. In still areas a frame is compared with the next one (artifact flips sign) and the one
    after (same sign): ratio ~1.0 = clean, 1.5+ = visible, 2.5+ = heavy. Returns (luma, chroma)."""
    src = source(job, clean=False)
    n = src.num_frames
    vals = {"Y": [], "C": []}
    for k, t in enumerate(np.linspace(40, n - 41, samples).astype(int)):
        if progress:
            progress(k, samples)
        f = [src.get_frame(int(t + d)) for d in (-2, -1, 0)]
        for key, planes, thr in (("Y", [0], 2.0), ("C", [1, 2], 1.5)):
            A, B = [], []
            for p in planes:
                a2, a1, a0 = (np.asarray(x[p], dtype=np.float32) for x in f)
                A.append(np.abs(a0 - a1))
                B.append(np.abs(a0 - a2))
            A, B = np.mean(A, 0), np.mean(B, 0)
            still = B < thr
            if still.mean() >= 0.3:
                vals[key].append(A[still].mean() / max(B[still].mean(), 0.05))
    return tuple(float(np.median(vals[k])) if vals[k] else 1.0 for k in ("Y", "C"))


def measure_composite(job, samples=80, progress=None, block=(16, 128)):
    """NTSC composite crosstalk measured from first principles - no thresholds taken from particular discs.

    Dot crawl (luma) and rainbows (chroma) are the colour subcarrier, whose phase turns 180 degrees from one
    video frame to the next. In still picture areas the difference of consecutive frames d = a0 - a1 therefore
    holds twice the artifact, while the difference of frames two apart c = a0 - a2 holds none (same phase).
    Grain, sensor noise and MPEG noise enter both differences with the same power. So, per horizontal
    frequency, (power(d) - power(c)) / 4 is the power spectrum of the artifact itself: zero for a clean
    source at every frequency, however grainy.
    Only still areas can be measured: the carrier is fixed to the screen grid, so motion compensation would
    shift its phase. A whole episode has enough still moments (80 samples spread over it).
    Still blocks: OUTSIDE the carrier band (below 0.07 and above 0.36 cycles/px) d and c must have the same
    power (within 25 %) and the block mean must not change - true only when nothing moves. Chroma uses the
    luma blocks (same place). Returns RMS amplitudes in 8-bit levels: luma in the carrier band (0.10-0.34
    cycles/px) with the carrier frequency, chroma over all frequencies; whole-picture value and 90th
    percentile over still blocks (artifacts only on fine detail), plus the share of the picture measured."""
    raw = source(job, clean=False)
    n = raw.num_frames
    sc = (1 << raw.format.bits_per_sample) / 256.0
    bh, bw = block
    fy_ = np.fft.rfftfreq(bw)
    band_y = (fy_ >= 0.10) & (fy_ <= 0.34)
    ctrl_y = (fy_ > 0.02) & ((fy_ < 0.07) | (fy_ > 0.36))
    fc_ = np.fft.rfftfreq(bw // 2)
    band_c = fc_ > 0.01
    norm = lambda w_: float((np.hanning(w_) ** 2).sum()) * w_
    per = {"Y": [], "C": []}
    sumA = {"Y": 0.0, "C": 0.0}
    spec_acc = 0.0
    nblk = tot = 0

    def blocks_of(a, h_, w_, ny, nx):
        return a[8:8 + ny * h_, 8:8 + nx * w_].reshape(ny, h_, nx, w_).transpose(0, 2, 1, 3)

    def spec(x, w_):
        win = np.hanning(w_).astype(np.float32)
        return (np.abs(np.fft.rfft((x - x.mean(-1, keepdims=True)) * win, axis=-1)) ** 2).mean(2)     # [ny,nx,F]

    for k, t in enumerate(np.linspace(20, max(21, n - 23), samples).astype(int)):
        if progress:
            progress(k, samples)
        fr = [raw.get_frame(int(t + i)) for i in range(3)]
        a0, a1, a2 = (np.asarray(x[0], dtype=np.float32) / sc for x in fr)
        H, W = a0.shape
        ny, nx = (H - 16) // bh, (W - 16) // bw
        d, c = blocks_of(a0 - a1, bh, bw, ny, nx), blocks_of(a0 - a2, bh, bw, ny, nx)
        Pd, Pc = spec(d, bw), spec(c, bw)
        cd, cc = Pd[..., ctrl_y].sum(-1), Pc[..., ctrl_y].sum(-1)
        still = (np.abs(c.mean(axis=(2, 3))) < 0.5) & (np.abs(d.mean(axis=(2, 3))) < 0.5)
        still &= (cc < 1.25 * cd + 1e-3) & (cd < 1.25 * cc + 1e-3)
        tot += ny * nx
        nblk += int(still.sum())
        if not still.any():
            continue
        A = (Pd - Pc)[still] / 4                                         # [m,F]
        sumA["Y"] = sumA["Y"] + A[:, band_y].sum()
        spec_acc = spec_acc + A.sum(0)
        per["Y"].append(A[:, band_y].sum(1))
        Ac = 0.0
        for p in (1, 2):
            b0, b1, b2 = (np.asarray(x[p], dtype=np.float32) / sc for x in fr)
            dc, cc2 = blocks_of(b0 - b1, bh // 2, bw // 2, ny, nx), blocks_of(b0 - b2, bh // 2, bw // 2, ny, nx)
            Ac = Ac + ((spec(dc, bw // 2) - spec(cc2, bw // 2))[still] / 4)[:, band_c].sum(1) / 2
        sumA["C"] = sumA["C"] + Ac.sum()
        per["C"].append(Ac)
    res = {"still_share": round(nblk / max(1, tot), 3), "still_blocks": nblk}
    for key, w_ in (("Y", bw), ("C", bw // 2)):
        if not per[key]:
            res.update({f"{key}_amp": None, f"{key}_p90": None})
            continue
        allb = np.concatenate(per[key])
        # Parseval with the Hann window: rms^2 = 2 * (one-sided power summed over the band) / (sum(win^2) * width)
        res[f"{key}_amp"] = round(float(np.sqrt(max(0.0, 2 * sumA[key] / len(allb)) / norm(w_))), 2)
        res[f"{key}_p90"] = round(float(np.sqrt(max(0.0, 2 * np.percentile(allb, 90)) / norm(w_))), 2)
    if per["Y"]:
        sm = np.convolve(np.maximum(spec_acc, 0), np.ones(3) / 3, "same")
        res["carrier"] = round(float(fy_[band_y][np.argmax(sm[band_y])]), 3) if sm[band_y].max() > 0 else None
    return res


def _box(a, h, w):
    """Mean over a h x w window (odd sizes), edges wrapped - only used on difference pictures."""
    if h == 1 and w == 1:
        return a
    out = np.zeros_like(a)
    for dy in range(-(h // 2), h // 2 + 1):
        for dx in range(-(w // 2), w // 2 + 1):
            out += np.roll(a, (dy, dx), (0, 1))
    return out / (h * w)


def measure_flip(job, samples=100, progress=None):
    """Local composite crosstalk: share of pixels (per mille) that alternate from frame to frame while the
    picture itself stands still. Over 4 source frames, alt = (a0 - a1 + a2 - a3) / 2 is the alternating part;
    a pixel counts when |alt| is above the threshold and a0 = a2, a1 = a3 within a third of it. Returns the
    90th percentile over the samples for (luma, chroma): the whole-frame ratio of measure_xtalk misses
    rainbows that only sit on fine detail (Golden Boy: ratio 0.98 = "clean", but 6 per mille of chroma pixels
    flip in its worst tenth of shots; a clean digital master such as Blue Dragon has 0.08).
        p90 luma / chroma:  Blue Dragon 0.15 / 0.08   Golden Boy 2.8 / 6.1   Cowboy Bebop 1.1 / 7.0
        Courage 4.5 / 0.84  Lodoss 7.9 / 6.9          One Piece 16.8 / 44     Naruto 22.5 / 23"""
    src = source(job, clean=False)
    n = src.num_frames
    vals = {"Y": [], "C": []}
    for k, t in enumerate(np.linspace(40, n - 41, samples).astype(int)):
        if progress:
            progress(k, samples)
        f = [src.get_frame(int(t - d)) for d in range(4)]
        sc = (1 << src.format.bits_per_sample) / 256.0
        for key, planes, T, sz in (("Y", [0], 6.0, (1, 1)), ("C", [1, 2], 3.0, (3, 5))):
            v = []
            for p in planes:
                a = [np.asarray(x[p], dtype=np.float32) / sc for x in f]
                alt = np.abs(_box((a[0] - a[1] + a[2] - a[3]) / 2, *sz))
                s1, s2 = np.abs(_box(a[0] - a[2], *sz)), np.abs(_box(a[1] - a[3], *sz))
                v.append(1000 * np.mean((alt > T) & (s1 < T / 3) & (s2 < T / 3)))
            vals[key].append(float(np.mean(v)))
    return tuple(float(np.percentile(vals[k], 90)) if vals[k] else 0.0 for k in ("Y", "C"))


def measure_block(job, samples=60, progress=None):
    """MPEG blocking: how much larger the luma step across an 8 px block border is than a step inside a
    block, in flat areas (steps under 12 levels, so real lines do not count). Measured on the source
    frames as coded, where the grid sits at multiples of 8. 1.00 = no grid. Returns the 90th percentile
    over the samples (blocking comes and goes with motion) and the median."""
    src = source(job, clean=False)
    n = src.num_frames
    sc = (1 << src.format.bits_per_sample) / 256.0
    v, ex = [], []
    for k, t in enumerate(np.linspace(40, max(41, n - 41), samples).astype(int)):
        if progress:
            progress(k, samples)
        a = np.asarray(src.get_frame(int(min(t, n - 1)))[0], dtype=np.float32) / sc
        g = np.abs(np.diff(a[:, 16:a.shape[1] - 16 + 1], axis=1))
        ok = g < 12
        num = (g * ok).sum(0)
        den = ok.sum(0).clip(min=1)
        col = num / den                                   # col[i] = step between x = 16 + i and 17 + i
        prof = np.array([col[i::8].mean() for i in range(8)])
        if a.mean() > 30 * 1 and np.median(prof) > 0.05:
            v.append(float(prof[7] / np.median(prof)))
            ex.append(float(prof[7] - np.median(prof)))         # extra step at the grid, 8-bit levels
    if not v:
        return 1.0, 1.0, 0.0
    return float(np.percentile(v, 90)), float(np.median(v)), float(np.percentile(ex, 90))


def _level_block(excess):
    """MPEG repair from the physical step the block grid adds (8-bit levels, strongest tenth of the frames).
    Every MPEG-2 DVD has a faint grid (clean digital master 0.17, ordinary DVDs 0.5-0.6); it is only worth
    repairing where it is plainly visible - the repair also costs detail and film grain (bench_real.py:
    'light' keeps 76-90 % of the finest detail of a clean film DVD)."""
    return "off" if excess < BLOCK_T[0] else ("light" if excess < BLOCK_T[1] else "normal")


BLOCK_T = (0.9, 1.6)            # extra step at the 8-px grid in levels: off below 0.9, light below 1.6
LEVELS = ("off", "light", "normal", "strong")


def _level(ratio):
    return "off" if ratio < 1.2 else ("light" if ratio < 1.4 else ("normal" if ratio < 3.0 else "strong"))


# Composite levels from the physical artifact size (measure_composite: RMS in 8-bit levels, strongest tenth of the
# still picture). Steps follow visibility, not particular discs: under ~0.4 level RMS a pattern is below the
# noise of any DVD; ~1 level is plainly visible on flat colour; 2+ is strong crawl.
COMPOSITE_STEPS = {"Y": (0.4, 0.9, 2.0), "C": (0.35, 0.8, 1.8)}


def _level_phys(v, key):
    a, b, c = COMPOSITE_STEPS[key]
    return "off" if v is None or v < a else ("light" if v < b else ("normal" if v < c else "strong"))


def composite_auto(job, progress=None):
    """Dot crawl / rainbow suggestion from measure_composite (see there). Too little still picture in the
    sampled moments = no reliable measurement: say so instead of guessing."""
    m = measure_composite(job, progress=progress)
    sure = m["still_blocks"] >= 1500
    cr, rb = _level_phys(m.get("Y_p90"), "Y"), _level_phys(m.get("C_p90"), "C")
    # composite re-decode is never switched on by Auto: on the end-to-end benchmark it cost 7 dB of colour
    # accuracy (UV 53.7 -> 46.5 dB) and changed clean pictures too (55.1 -> 52.4 dB); it assumes the
    # textbook carrier (0.265 cycles/px) while real discs measure 0.18-0.26
    out = {"crawl": cr, "rainbow": rb, "composite": "off",
           "crawl_level": m.get("Y_p90"), "rainbow_level": m.get("C_p90"), "carrier": m.get("carrier"),
           "still_share": m["still_share"], "composite_sure": sure}
    if not sure:
        out["note"] = (f"Only {m['still_share']:.0%} of the sampled picture stood still - the dot crawl / rainbow "
                       "measurement is uncertain; check the picture.")
    return out


def _level_flip(share, key):
    """Level from the local flip share (per mille). Clean masters stay below 0.2."""
    a, b, c = (1.0, 3.0, 20.0) if key == "Y" else (0.5, 3.0, 20.0)
    return "off" if share < a else ("light" if share < b else ("normal" if share < c else "strong"))


def measure_cadence(job, windows=320, cycles=8):
    """Film or video? After field matching, film (3:2 pulldown) has one repeated frame in every cycle of 5;
    material made at video rate (scrolling captions, video titles, 30p animation) has none. Sampled windows of
    `cycles` x 5 frames; only cycles where something moves count, and a video section needs two video cycles in a
    row (single cycles are noise: fades, film grain). Any video section -> VFR (film parts stay 24p in VFR, so it
    costs nothing but the analysis pass); none -> CFR."""
    src = source(job, clean=False)
    tfm = core.tivtc.TFM(src, order=_order(job), PP=1)
    y = core.resize.Bilinear(core.std.ShufflePlanes(tfm, 0, vs.GRAY), 192, 128, format=vs.GRAYS)
    n = y.num_frames
    film = video = still = sections = held = 0
    # change between two frames = the most changed 16x16 block (of 12 x 8): a caption scrolling over a still
    # picture moves in one strip only - averaged over the frame it looked like a still shot (Idaten Jump)
    blk = lambda a: np.abs(a).reshape(8, 16, 12, 16).mean(axis=(1, 3)).max()
    span = cycles * 5 + 1
    for w in np.linspace(n * 0.02, n * 0.98 - 21, windows).astype(int):   # positions as with 4 cycles
        w = max(0, min(int(w), n - span))
        fr = [np.asarray(y.get_frame(int(w + i))[0]) * 255 for i in range(span)]
        d = np.array([blk(fr[i + 1] - fr[i]) for i in range(span - 1)])
        kinds = []
        for c in range(cycles):
            cd = d[c * 5:(c + 1) * 5]
            med = float(np.median(cd))
            if med < 2.0:
                still += 1; kinds.append("s"); continue
            r = float(cd.min()) / med
            # film repeats a frame (smallest change near 0); video-rate parts change every frame - also a
            # caption scrolling over a film picture (Idaten Jump: smallest 14 vs typical 49 on the repeat)
            k = "f" if r < 0.2 and float(cd.min()) < 4 else ("v" if r > 0.25 and float(cd.min()) > 4 else "?")
            film += k == "f"; video += k == "v"; kinds.append(k)
        # two video cycles in a row: judged on the first 4 cycles as before (on 8, Courage's grain made 2 sections)
        if "vv" in "".join(kinds[:4]):
            sections += 1
        elif _held_video(np.array([np.abs(fr[i + 1] - fr[i]).mean() for i in range(span - 1)])):
            held += 1                                    # 29.97p held drawings (SCA1: 225 windows, Courage: 1)
    moving = film + video
    sections += held if held >= 2 else 0
    return {"mode": "vfr" if sections >= 1 else "cfr", "video_share": round(100 * video / max(moving, 1), 1),
            "video_sections": sections, "film_cycles": film, "video_cycles": video, "windows": int(windows)}


def _held_video(d, run=4, min_moving=5):
    """One window of frame changes (mean, levels): 29.97p animation with held drawings (SCA1: held 3 / 2 frames)?
    Its frames repeat like film, so the repeat test above calls it film - but the repeats move through the cycle.
    True when >= min_moving cycles move with repeats and no `run` of them shares a repeated position (telecined
    film always has one; a cut splits an 8-cycle window so that one side still has 4)."""
    reps = []
    for c in range(len(d) // 5):
        v = d[5 * c:5 * c + 5]
        lo, hi = float(v.min()), float(v.max())
        if hi < 3.0:
            continue
        r = {j for j in range(5) if v[j] < max(1.0, 1.6 * lo) and v[j] < 0.3 * hi}
        if r:
            reps.append(r)
    if len(reps) < min_moving:
        return False
    for i in range(len(reps) - run + 1):
        if set.intersection(*reps[i:i + run]):
            return False
    return True


def measure_order(job, samples=120):
    """Field order from the picture (NTSC; the PAL detector has its own): every field bobbed to a frame, the step from
    one bobbed field to the next. With the right order the fields run forward in time; with the wrong one they run
    1, 0, 3, 2 ... and every second step spans three field times. Telecined film repeats fields, so its steps are
    judged the same way (a repeated field is a zero step in either order). Only moving frames count.
    -> {"order": "tff" / "bff" / None (undecided), "order_ratio": wrong-order step / right-order step}"""
    src = source(job, clean=False)
    n = src.num_frames
    if n < 20:
        return {"order": None, "order_ratio": 1.0}
    sm = core.resize.Bilinear(core.std.ShufflePlanes(src, 0, vs.GRAY), 360, 480, format=vs.GRAY8)
    mv = core.std.PlaneStats(sm, sm[1:] + sm[-1:])
    picks = [int(k) for k in np.linspace(n * 0.03, n * 0.97 - 2, samples * 3).astype(int)]
    picks = [k for k in picks if mv.get_frame(k).props["PlaneStatsDiff"] >= 0.01][:samples]
    if len(picks) < 8:
        return {"order": None, "order_ratio": 1.0}
    step = {}
    for o in (1, 0):
        b = core.bwdif.Bwdif(core.std.SetFieldBased(sm, 2 if o else 1), field=o + 2)
        d = core.std.PlaneStats(b, b[1:] + b[-1:])
        # cross-frame step (2nd field of frame k -> 1st field of k+1) against the in-frame step (1st -> 2nd of k)
        step[o] = sum(d.get_frame(2 * k + 1).props["PlaneStatsDiff"] + d.get_frame(2 * k).props["PlaneStatsDiff"] for k in picks)
    good, bad = (1, 0) if step[1] <= step[0] else (0, 1)
    ratio = step[bad] / max(step[good], 1e-9)
    return {"order": ("tff" if good == 1 else "bff") if ratio >= 1.15 else None, "order_ratio": round(float(ratio), 3)}


def measure_unmatched(job, samples=400):
    """What field matching leaves: share of frames TFM cannot match cleanly (they get rebuilt from one field),
    how strong their combing is, and whether the source was line-doubled before it got to the disc (every
    second line a copy of the one above - jaggies on every outline). -> Combing repair, Unmatched frames,
    Anti-aliasing suggestions."""
    src = source(job, clean=False)
    tfm = core.tivtc.TFM(src, order=_order(job), cthresh=int(opt(job, "cthresh")), MI=int(opt(job, "mi")), PP=1)
    st = core.std.PlaneStats(comb_pixels(core.std.SetFieldBased(tfm, 0), int(opt(job, "cthresh"))))
    y = core.std.ShufflePlanes(src, 0, vs.GRAY)
    n = tfm.num_frames
    flagged, comb, doubled, checked = 0, [], 0, 0
    for t in np.linspace(n * 0.02, n * 0.98, samples).astype(int):
        f = st.get_frame(int(t))
        if f.props.get("_Combed", 0):
            flagged += 1
            comb.append(float(f.props["PlaneStatsAverage"]))
        if True:
            a = np.asarray(y.get_frame(int(t))[0], dtype=np.float32)
            if a.std() > 8:
                checked += 1
                pair = np.abs(a[0::2] - a[1::2]).mean()           # line and the one below
                cross = np.abs(a[1:-1:2] - a[2::2]).mean()         # line and the next pair
                doubled += int(pair < 0.25 * cross)
    share = 100 * flagged / samples
    strength = 100 * float(np.median(comb)) if comb else 0.0         # % of the picture combed in flagged frames
    dbl = 100 * doubled / max(checked, 1)
    comb_mode = "strong" if share > 5 and strength > 8 else "nnedi3"
    aa = "all" if dbl > 40 else ("repair" if share > 1 else "off")
    return {"comb": comb_mode, "ivtc_repair": "ai", "aa": aa,
            "unmatched_share": round(share, 1), "comb_strength": round(strength, 1), "line_doubled": round(dbl, 1)}


def auto_group(kind, job, progress=None):
    """The Auto button of one Adjust group: only that group's detector runs. Values are suggestions."""
    if kind == "frames":
        if is_pal(job):
            d = pal_detect(job)
            return {"pal_mode": d["mode"], "order": d["order"], "raw_combed": d["raw_combed"], "matched_combed": d["matched_combed"]}
        od = measure_order(job)
        if od["order"] and od["order"] != opt(job, "order"):
            # the file's field order is wrong (ffmpeg-made MPEG-2 files, mislabelled rips): every match would be wrong
            job = dict(job, order=od["order"])
        res = {**measure_chroma_unmix(job), **measure_cadence(job), **measure_unmatched(job)}
        res.update(order=opt(job, "order"), order_measured=od["order"], order_ratio=od["order_ratio"])
        return res
    if kind == "composite":
        if is_pal(job):
            return {"crawl": "off", "rainbow": "off", "composite": "off", "note": "PAL: no NTSC dot crawl / rainbows"}
        return composite_auto(job, progress)
    if kind == "block":
        bp, bm, be = measure_block(job)
        return {"deblock": _level_block(be), "block_step": round(be, 2), "block_p90": round(bp, 3), "block_median": round(bm, 3)}
    if kind == "chroma":
        return auto_chroma(job, progress)
    if kind == "stab":
        if is_pal(job):
            return {"stab": "off", "stab_level": "normal", "crawl_level": 0.0}
        m = measure_composite(job, samples=40)
        lv = _level_phys(m.get("Y_p90"), "Y")
        # with the AI cleaner on: Temporal fix (guided). Real still shots (Courage x4, Lodoss x2, One Piece,
        # bench_still.py): steady-reference 45.5 -> 51.9 dB, luma wobble 1.86 -> 0.55, line detail kept;
        # TTempSmooth 49.8 dB / 0.83, TemporalFix AI 49.1 dB / 0.64 but detail 0.85 (Lodoss 0.37)
        ai = _ai_cleaner(job)                          # the AI cleanup really runs (not only selected)
        if opt(job, "ai_model") == "field":            # FieldNet runs its own steadiness stages
            return {"stab": "off", "stab_level": "normal", "crawl_level": m.get("Y_p90"), "ai_cleaner": ai, "stable_net": True}
        # Universal (ustab.py) for everything else: better than no stabilizer on every kind of shot (bench_cases),
        # never drags fades / cuts / flashes (the Steady chain did), holds drawings on twos. Precise single-frame
        # boils the most (SCA1: 2.81 levels between copies of a held drawing) -> Strong; with the AI cleaner on it
        # runs even when little crawl is left (the frame-by-frame model itself adds boil and shimmer).
        return {"stab": auto_stab(job, lv)[0], "stab_level": auto_stab(job, lv)[1], "crawl_level": m.get("Y_p90"), "ai_cleaner": ai}
    raise ValueError(f"unknown detector {kind}")


def auto_stab(job, crawl):
    """Auto's stabilizer: (method, level). crawl: the measured dot crawl level (off / light / normal / strong).
    Universal whenever there is something to steady; FieldNet runs its own steadiness stages."""
    m = opt(job, "ai_model")
    if job.get("image"):
        return "off", "normal"
    if m == "field":
        return "off", "normal"
    # the AI cleanup's own boil / shimmer only counts when it runs (dot crawl or rainbows on, or an all-in-one
    # model): with both off the Precise model is idle and a clean film source got Universal strong for nothing
    if m in ("precise", "temporal", "stable", "uni") and _ai_cleaner(job):
        return "ustab", "strong" if m == "precise" else "normal"
    if crawl in ("normal", "strong", "light"):                # classic filters only: leftover crawl / flicker
        return "ustab", "strong" if crawl == "strong" else "normal"
    return "off", "normal"


def auto_detect(job, progress=None):
    """Auto-detect everything: dot crawl + rainbow levels, ghost/halo echo taps, chroma drift and bleed."""
    res_x = {}
    if not is_pal(job):
        res_x = composite_auto(job, progress)
        job = dict(job, crawl=res_x["crawl"], rainbow=res_x["rainbow"], composite=res_x["composite"])
    bp, bm, be = measure_block(job)
    res_x.update({"deblock": _level_block(be), "block_step": round(be, 2), "block_p90": round(bp, 3), "block_median": round(bm, 3)})
    if not is_pal(job):
        try:
            res_x.update(measure_chroma_unmix(job))
        except Exception as e:                            # noqa: BLE001
            print(f"colour timing check failed: {e}", file=sys.stderr)
    g = detect_ghost(source(job, clean=False))
    on = g.get("halo_rx", 0) > 0
    res_x.update({"ghost": "auto" if on else "off", "ghost_taps": g["taps"], "halo": g["halo"],
                  "ghost_amp": g["ghost"], "halo_rx": g.get("halo_rx", 0), "halo_pos": g.get("halo_pos", 0)})
    # leftover crawl and composite flicker are temporal: TTempSmooth is the measured best stabilizer for it
    st, sl = auto_stab(job, res_x.get("crawl", "off"))      # Universal (see auto_stab)
    res_x.update(stab=st, stab_level=sl)
    if job.get("image"):
        # a still: the crawl / rainbow detectors need the pattern flipping between frames, which a still cannot show.
        # A DVD-sized still gets dot crawl + rainbow removal on (the Precise model is spatial and leaves a clean picture
        # practically as it is: 59-63 dB on clean Blu-ray frames); nothing to steady in a still.
        sd = (job.get("src_h") or 0) <= 576
        res_x.update(crawl="normal" if sd else "off", rainbow="normal" if sd else "off", stab="off", stab_level="normal",
                     note=("Still image: dot crawl / rainbow removal set for a DVD-sized picture (a still cannot show the "
                           "flicker the detector measures) - check the preview." if sd else
                           "Still image: dot crawl / rainbow detection needs video - set them by hand if the picture has them."))
    # crop has its own detector ('crop' command / its own section)
    return auto_chroma(dict(job, ghost=res_x["ghost"], ghost_taps=g["taps"], halo_rx=res_x["halo_rx"]), progress, res_x)


def auto_chroma(job, progress=None, res_x=None):
    """Find shift_x/shift_y and bleed fix that put chroma on the luma edges (after cleaning and IVTC)."""
    res_x = res_x or {}
    # measure drift / bleed on the DVD's own chroma: the composite decoder smears chroma edges
    # asymmetrically, which reads as extra 'drift' (One Piece: 0.7 -> 1.4 px) and causes over-correction
    mj = dict(job, composite="off")
    src = source(mj)
    base = pal_process(src, mj) if is_pal(mj) else ivtc_cfr(src, mj)
    raw = chroma_stats(chroma_fix(base), progress=progress)
    if raw is None:
        return {"shift_x": 0.0, "shift_y": 0.0, "warp": 0, "measured": None, **res_x}
    sx, sy = round(-raw["x"] * 4) / 4, round(-raw["y"] * 4) / 4      # quarter-pixel steps
    # bleed: ~1 px extra width -> warp 6 (calibrated on DVD anime); none below 0.3 px
    warp = 0 if raw["bleed"] < 0.3 else int(min(12, max(2, round(raw["bleed"] * 6))))
    # ChromaNet (neural, colour only) when its model and PyTorch are there, else the rebuild filter.
    # It is safe on clean discs too, but below 0.3 px there is nothing to gain. Two passes above 3 px.
    import chromanet
    ai = chromanet.available()
    method = "off" if raw["bleed"] < 0.3 else ("ai" if ai else "rebuild")
    strength = "strong" if raw["bleed"] > 3.0 else "normal"
    return {"shift_x": sx, "shift_y": sy, "warp": warp, "method": method, "strength": strength, "measured": raw, **res_x}


# ---------------------------------------------------------------- output

def crop_values(job):
    """Crop L/R/T/B, forced to even numbers (4:2:0 output needs even sizes)."""
    c = job.get("crop", {}) or {}
    return tuple(max(0, int(c.get(k, 0) or 0)) // 2 * 2 for k in ("l", "r", "t", "b"))


def _torch_cuda_ok():
    """True when PyTorch can actually run kernels on the GPU (a build without kernels for the card fails here)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        (torch.ones(8, device="cuda") * 2).sum().item()
        return True
    except Exception:                                    # noqa: BLE001
        return False


def stabilize(clip, job):
    """Temporal stabilizer (after cleaning and crop, before resizing): removes line wobble, shimmer and
    flicker that frame-by-frame filters leave. clip: YUV 4:4:4 16-bit."""
    method, level = opt(job, "stab"), opt(job, "stab_level")
    if method not in STAB:
        return clip
    p = STAB[method].get(level, STAB[method]["normal"])
    if method == "plus":
        import steady
        return steady.steady_plus(core.resize.Bicubic(clip, format=vs.YUV444P16), job, level=p)
    if method == "chain":
        # every part at 16-bit 4:4:4: rounding the smoothed picture back to 8 bits between parts puts flicker back
        import tstab
        c = core.resize.Bicubic(clip, format=vs.YUV444P16)
        for m_, lv_ in zip(("mc", "gtfix", "ttemp"), p[:3]):
            c = stabilize(c, dict(job, stab=m_, stab_level=lv_))
            if c.format.id != vs.YUV444P16:
                c = core.resize.Bicubic(c, format=vs.YUV444P16)
        return tstab.still_lock(c, keep=p[3])
    if method == "ustab":
        import ustab
        return ustab.stabilize(clip, level=p)
    if method == "gtfix":
        # no clean guide exists at this point (the cleaner's input still has the crawl), so the guide is the
        # low-passed picture itself: only fine detail (line wobble, shimmer) is averaged, under weights from it
        import gtfix
        return gtfix.gtfix(clip, None, strength=p)
    if method == "tfix_ai":
        p = float(job.get("tfix_strength", p))                      # 0.0-3.0 (slider), 0 = off
        if p <= 0:
            return clip
        import vs_temporalfix
        import trt_temporalfix
        mat = sd_colour(job)[0]
        rgb = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in=mat)
        say = lambda m: print(m, file=sys.stderr, flush=True)      # stdout is the app's message channel
        if trt_temporalfix.available():                            # TensorRT: ~3x PyTorch (64 vs 19 fps, 720x480)
            rgb = trt_temporalfix.temporalfix(rgb, strength=p, log=say)
        else:
            backend = "cuda" if _torch_cuda_ok() else "cpu"
            if backend == "cpu":
                say("TemporalFix AI: no usable GPU - running on the CPU (very slow)")
            rgb = vs_temporalfix.model(rgb, strength=p, backend=backend)
        out = core.resize.Bicubic(rgb, format=clip.format.id, matrix=mat)
        return core.std.CopyFrameProps(out, clip)
    if method == "tfix_classic":
        import vs_temporalfix
        return vs_temporalfix.classic(clip, strength=p, tr=6)
    if method == "mc":
        tr, thsad = p
        sup = core.mv.Super(clip, pel=2, sharp=2, rfilter=4)
        vec = {}
        for d in range(1, tr + 1):
            for back in (True, False):
                vec[(d, back)] = core.mv.Analyse(sup, isb=back, delta=d, blksize=16, overlap=8, truemotion=True)
        args = [vec[(d, b)] for d in range(1, tr + 1) for b in (True, False)]
        deg = core.mv.Degrain3 if tr >= 3 else core.mv.Degrain2
        return deg(clip, sup, *args[:2 * min(tr, 3)], thsad=thsad, thsadc=thsad)
    if method == "ttemp":
        th, md = p
        # scene cuts marked by mvtools (_SceneChangePrev/Next), so nothing is blended across a cut
        sup = core.mv.Super(clip, pel=1)
        clip = core.mv.SCDetection(clip, core.mv.Analyse(sup, isb=False, blksize=16), thscd1=400, thscd2=130)
        return core.zsmooth.TTempSmooth(clip, maxr=3, scthresh=0, thresh=[th, max(1, th - 1), max(1, th - 1)],   # 8-bit scale
                                        mdiff=[md, max(1, md - 1), max(1, md - 1)])
    return clip


# Colour correction (GUI step "Colour"). All values at their default = the stage is skipped.
GRADE = {
    "black": 16,        # luma code value of the source that becomes black (16). Above 16 = washed-out source,
                        # below 16 = picture detail under legal black that players cut off
    "white": 235,       # luma code value that becomes white (235)
    "shadows": 0.0,     # -0.5..1.5 lift of the dark tones only (opens crushed blacks), midtones / whites stay
    "gamma": 1.0,       # 0.6..1.8 midtone brightness
    "contrast": 1.0,    # 0.6..1.6 around mid grey
    "sat": 1.0,         # 0..2 saturation
    "dark_sat": 0.0,    # 0..2 extra saturation in the dark tones only (colour that was crushed with the blacks)
    "temp": 0.0,        # -2..2 blue <-> amber
    "tint": 0.0,        # -2..2 green <-> magenta
    "gamut": "off",     # off / compress (colour stored beyond the screen range is squeezed in, not cut flat)
                        # / ai (ClipNet, models/clipnet.pt: rebuilds colour that is already clipped flat)
    "vibrance": 0.0,    # -1..1 saturation that acts mostly on weakly coloured areas (skin and saturated colour kept)
    "hue": 0.0,         # -180..180 degrees, rotates every colour
    # colour wheels (Lift / Gamma / Gain): [brightness, Cb, Cr] offsets, 0 = neutral
    "lift": [0.0, 0.0, 0.0],
    "gamma_w": [0.0, 0.0, 0.0],
    "gain_w": [0.0, 0.0, 0.0],
    "curves": {},       # {"m" | "r" | "g" | "b": [[x, y], ...]} control points 0..1 (monotone cubic through them)
    "luts": [],         # [{"path": .cube (3D or 1D), "mix": 0..1, "on": bool}], applied last, in this order (lut.py)
    "lut": "",          # older settings: one LUT (read as the first entry of "luts")
    "lut_mix": 1.0,
}


def grade_settings(job, src=None):
    g = dict(GRADE)
    g.update({k: v for k, v in ((job.get("grade") or {}) if src is None else src).items() if k in GRADE and v is not None})
    luts = [dict(l) for l in (g.get("luts") or []) if isinstance(l, dict)]
    if g.get("lut"):
        luts.insert(0, {"path": g["lut"], "mix": g.get("lut_mix", 1.0), "on": True})
    g["luts"], g["lut"] = luts, ""
    return g


def grade_layers(job):
    """Adjustment layers (Colour step), bottom first: [(first source frame, end source frame, opacity, grade)].
    Source frames are counted from the start of the title (frame property _SrcN), end exclusive. Like an
    adjustment layer in an editor, each one grades the picture below it (the base grade and the layers under
    it) for its time range; opacity blends its result with that picture."""
    out = []
    for l in job.get("grade_layers") or []:
        if l.get("on", True) and l.get("grade") and int(l["end"]) > int(l["start"]) and float(l.get("opacity", 1)) > 0:
            g = grade_settings(job, l["grade"])
            if grade_active(g):
                out.append((int(l["start"]), int(l["end"]), min(1.0, float(l.get("opacity", 1))), g))
    return out


def gamut_mode(g):
    v = g.get("gamut")
    return "compress" if v is True else (v if v in ("compress", "ai") else "off")


SCALAR_GRADE = ("black", "white", "shadows", "gamma", "contrast", "sat", "dark_sat", "temp", "tint", "vibrance", "hue")


def _curve_active(pts):
    return bool(pts) and any(abs(float(x) - float(y)) > 1e-4 for x, y in pts)


def wheels_active(g):
    return any(abs(float(v)) > 1e-6 for k in ("lift", "gamma_w", "gain_w") for v in (g.get(k) or [0, 0, 0])) or \
        any(_curve_active((g.get("curves") or {}).get(c)) for c in "mrgb")


def active_luts(g):
    import os
    return [l for l in (g.get("luts") or []) if l.get("on", True) and l.get("path") and os.path.isfile(str(l["path"]))
            and float(l.get("mix", 1)) > 0]


def lut_active(g):
    return bool(active_luts(g))


def grade_active(g):
    return any(abs(float(g[k]) - float(GRADE[k])) > 1e-6 for k in SCALAR_GRADE) or gamut_mode(g) != "off" or \
        wheels_active(g) or lut_active(g)


def monotone_curve(pts, x):
    """Monotone cubic (Fritsch-Carlson) through control points [[x, y], ...] in 0..1, evaluated at x (array).
    Same rule as the Colour step's curve editor, so what you draw is what you get."""
    import numpy as np
    p = sorted((float(a), float(b)) for a, b in (pts or []))
    if not p or p[0][0] > 0:
        p = [(0.0, 0.0)] + p
    if p[-1][0] < 1:
        p = p + [(1.0, 1.0)]
    xs, ys = np.array([a for a, _ in p]), np.array([b for _, b in p])
    keep = np.concatenate([[True], np.diff(xs) > 1e-6])
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 2:
        return np.clip(x, 0, 1)
    h = np.diff(xs); d = np.diff(ys) / h
    m = np.zeros_like(xs)
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, len(xs) - 1):
        m[i] = 0.0 if d[i - 1] * d[i] <= 0 else 3 * (h[i - 1] + h[i]) / ((2 * h[i] + h[i - 1]) / d[i - 1] + (h[i] + 2 * h[i - 1]) / d[i])
    k = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    t = (x - xs[k]) / h[k]
    t2, t3 = t * t, t * t * t
    y = (2 * t3 - 3 * t2 + 1) * ys[k] + (t3 - 2 * t2 + t) * h[k] * m[k] + (-2 * t3 + 3 * t2) * ys[k + 1] + (t3 - t2) * h[k] * m[k + 1]
    return np.clip(y, 0, 1)


def wheel_luts(g):
    """One 16-bit table per RGB channel: Lift (shadows), Gain (highlights), Gamma (midtones), master curve, channel
    curve - all functions of one channel, so they collapse into a lookup (exact, fast)."""
    import numpy as np
    x = np.arange(65536, dtype=np.float64) / 65535

    def rgb(w):
        lum, u, v = (float(a) for a in (w or [0, 0, 0]))
        return np.array([lum + 1.402 * v, lum - 0.344136 * u - 0.714136 * v, lum + 1.772 * u])
    L, G, M = rgb(g.get("lift")) * 0.25, rgb(g.get("gain_w")) * 0.5, rgb(g.get("gamma_w")) * 0.5
    cv = g.get("curves") or {}
    luts = []
    for c, key in enumerate("rgb"):
        y = x + L[c] * (1 - x)
        y = y * (1 + G[c])
        y = np.clip(y, 0, 1) ** (1 / max(0.1, 1 + M[c]))
        if _curve_active(cv.get("m")):
            y = monotone_curve(cv["m"], y)
        if _curve_active(cv.get(key)):
            y = monotone_curve(cv[key], y)
        luts.append(np.clip(np.round(y * 65535), 0, 65535).astype(np.int64).tolist())
    return luts


def gamut_compress(yuv, matrix, knee=0.8):
    """yuv: YUV444PS full range. A DVD stores colours a screen cannot show (strong reds, blues): on screen
    they are cut off at the RGB limit and the detail inside them goes flat. Every pixel keeps its
    brightness and hue; its saturation is scaled so that anything beyond 80 % of the way to the limit is
    squeezed smoothly into the remaining 20 %. Pixels inside the limit are not touched."""
    rgb = core.resize.Point(yuv, format=vs.RGBS, matrix_in=matrix)
    R, G, B = (core.std.ShufflePlanes(rgb, i, vs.GRAY) for i in range(3))
    lum = "x 0.299 * y 0.587 * + z 0.114 * + K! K@ 0.002 max 0.998 min L! "     # K = brightness, L = same within 0..1
    exc = lambda c: f"{c} L@ > {c} L@ - 1 L@ - / L@ {c} - L@ / ?"          # 1.0 = exactly at the RGB limit
    k = knee
    S = core.akarin.Expr([R, G, B], lum + f"{exc('x')} {exc('y')} max {exc('z')} max M! "
                         f"M@ {k} <= 1 {k} {1 - k} 1 2 M@ {k} - {1 - k} / 2 * exp 1 + / - * + M@ / ?")
    ch = lambda c: core.akarin.Expr([R, G, B, S], lum + f"K@ {c} K@ - a * +")
    out = core.std.ShufflePlanes([ch("x"), ch("y"), ch("z")], [0, 0, 0], vs.RGB)
    return core.resize.Point(out, format=vs.YUV444PS, matrix=matrix)


def grade(clip, job):
    """Colour correction: levels, shadow lift, gamma, contrast, saturation, shadow saturation, white balance,
    gamut compression, wheels, curves, LUTs. clip: YUV limited range; returns YUV 4:4:4 16-bit limited range.
    Adjustment layers (job["grade_layers"]): every combination of layers that occurs is its own chain (base,
    then each layer on the result of the one below); every frame is taken from the chain of the layers over
    its source frame (_SrcN), so only that chain renders it."""
    base = grade_settings(job)
    layers = grade_layers(job)
    if not layers:
        return _grade_one(clip, base, job) if grade_active(base) else clip
    if clip.format.id != vs.YUV444P16:
        clip = core.std.CopyFrameProps(core.resize.Bicubic(clip, format=vs.YUV444P16), clip)
    memo = {(): _grade_one(clip, base, job) if grade_active(base) else clip}

    def chain(t):
        if t not in memo:
            below = chain(t[:-1])
            g = _grade_one(below, layers[t[-1]][3], job)
            op = layers[t[-1]][2]
            memo[t] = g if op >= 0.999 else core.std.Merge(below, g, op)
        return memo[t]
    bounds = sorted({f for l in layers for f in l[:2]})
    segs = []
    for a, b in zip(bounds, bounds[1:]):
        t = tuple(i for i, l in enumerate(layers) if l[0] <= a and l[1] >= b)
        if t:
            chain(t)
            segs.append((a, b, list(memo).index(t)))
    chains = list(memo.values())
    expr = "0"
    for a, b, i in segs:
        expr = f"x._SrcN {a} >= x._SrcN {b} < and {i} {expr} ?"
    return core.akarin.Select(chains, [clip], expr)


def _grade_one(clip, g, job):
    m = sd_colour(job)[0]
    props = clip
    f = clip if clip.format.subsampling_w == 0 and clip.format.subsampling_h == 0 else \
        core.resize.Bicubic(clip, format=clip.format.replace(subsampling_w=0, subsampling_h=0).id)
    f = core.resize.Point(f, format=vs.YUV444PS, range_in=0, range=1)
    Y, U, V = (core.std.ShufflePlanes(f, i, vs.GRAY) for i in range(3))
    bp, wp = (float(g["black"]) - 16) / 219, (float(g["white"]) - 16) / 219
    wp = max(wp, bp + 0.2)
    con, gam, sh = float(g["contrast"]), max(0.2, float(g["gamma"])), float(g["shadows"])
    # n = after levels + contrast (linear); c = after gamma + shadow lift. Below black stays linear.
    lin = f"x {bp} - {wp - bp} / 0.5 - {con} * 0.5 +"
    cur = (f"{lin} N! N@ 0.000001 max 1 min T! T@ {1 / gam} pow C! "      # pow(0, x) is NaN in Expr
           f"C@ {sh * 0.12} T@ 1 T@ - 5 pow * 14.93 * * + N@ T@ - +")       # lift peaks at 1/6 of the range, +26 levels per 1.0
    Yn = core.akarin.Expr([Y], cur)
    # Chroma follows what was done to luma (a lifted shadow with unchanged chroma would look grey):
    # linear gain of levels + contrast, times the local gain of the curve, times saturation.
    lvl = con / (wp - bp)
    gain = core.akarin.Expr([Y, Yn], f"{lin} 0.03 max N! y 0.03 max N@ / 0.5 max 3 min "
                            f"{lvl * float(g['sat'])} * 1 {float(g['dark_sat'])} 1 y 0.4 / - 0 max dup * * + *")
    t, ti = float(g["temp"]) * 0.05, float(g["tint"]) * 0.035
    Un = core.akarin.Expr([U, gain, Yn], f"x y * z 0 max {ti - t} * +")
    Vn = core.akarin.Expr([V, gain, Yn], f"x y * z 0 max {ti + t} * +")
    vib, hue = float(g.get("vibrance", 0)), float(g.get("hue", 0))
    if abs(vib) > 1e-6 or abs(hue) > 1e-6:
        # hue: rotation of every colour; vibrance: saturation weighted towards weakly coloured pixels
        cs, sn = math.cos(math.radians(hue)), math.sin(math.radians(hue))
        fac = f"x x * y y * + sqrt 0.25 / 1 min 1 swap - {vib} * 1 +"
        Un, Vn = (core.akarin.Expr([Un, Vn], f"x {cs} * y {sn} * - {fac} *"),
                  core.akarin.Expr([Un, Vn], f"x {sn} * y {cs} * + {fac} *"))
    out = core.std.ShufflePlanes([Yn, Un, Vn], [0, 0, 0], vs.YUV)
    gm = gamut_mode(g)
    if gm == "ai":
        try:
            import clipnet
            if not clipnet.available():
                raise RuntimeError("models/clipnet.pt or PyTorch missing")
            rgb = clipnet.clipnet(core.resize.Point(out, format=vs.RGBS, matrix_in=m))
            out = core.resize.Point(rgb, format=vs.YUV444PS, matrix=m)
        except Exception as e:                            # noqa: BLE001
            print(f"ClipNet not usable ({e}); using gamut compression", file=sys.stderr)
            gm = "compress"
    if gm == "compress":
        out = gamut_compress(out, m)
    if wheels_active(g):
        rgb = core.resize.Point(out, format=vs.RGB48, matrix_in=m, range_in=1)
        for c, lut in enumerate(wheel_luts(g)):
            rgb = core.std.Lut(rgb, planes=[c], lut=lut)
        out = core.resize.Point(rgb, format=vs.YUV444PS, matrix=m, range=1)
    if lut_active(g):
        import lut
        rgb = core.resize.Point(out, format=vs.RGBS, matrix_in=m, range_in=1)
        for l in active_luts(g):                          # each LUT on the result of the one before
            rgb = lut.apply(core.std.Expr(rgb, "x 0 max 1 min"), str(l["path"]), float(l.get("mix", 1)))
        out = core.resize.Point(rgb, format=vs.YUV444PS, matrix=m, range=1)
    out = core.resize.Point(out, format=vs.YUV444P16, range_in=1, range=0)
    return core.std.CopyFrameProps(out, props)


def measure_levels(job, samples=60):
    """Auto colour (Colour step): samples frames of the source and lets autocolour.py measure them and solve the
    grade - black / white point, white cast (Highlights wheel), black cast (Shadows wheel), fade (saturation),
    clipped colour. Returns autocolour's report (levels, whites, blacks, ...) plus "grade": the settings it sets."""
    import autocolour
    src = source(job, clean=False)
    if src.format.id != vs.YUV420P8:
        src = core.resize.Bicubic(src, format=vs.YUV420P8, dither_type="error_diffusion")
    n = src.num_frames

    seen, own = set(), []
    for t in np.linspace(min(40, n // 10), max(min(40, n // 10), n - 41), samples).astype(int):
        t = int(min(max(t, 0), n - 1))
        if t in seen:
            continue
        seen.add(t)
        f = src.get_frame(t)
        own.append((np.asarray(f[0]).copy(), np.asarray(f[1]).copy(), np.asarray(f[2]).copy()))
    paths = job.get("colour_ref") or []
    ref = ref_frames(paths, samples)
    pair = None
    if ref:
        ref, pair = match_reference(src, own, ref, paths)
    r = autocolour.analyze(own, ref, pair)
    if not r:
        return {}
    return {**r["report"], "grade": r["grade"]}


def _thumb_index(clip, step):
    """thumb_z of every `step`-th frame of a clip (~20 s for an episode at step 15)."""
    import autocolour
    small = core.resize.Bilinear(core.std.SelectEvery(clip, step, 0) if step > 1 else clip, 80, 60, format=vs.GRAY8)
    return np.stack([autocolour.thumb_z(np.asarray(f[0])) for f in small.frames()])


def _yuv(f):
    return tuple(np.asarray(f[p]).copy() for p in range(3))


def _exact_twin(clip, around, target_y, step):
    """The frame of `clip` within +-step/2 of `around` that best matches the picture target_y (the index found
    the right half second; motion inside the shot would still make the two sides differ)."""
    import autocolour
    t = autocolour.thumb_z(target_y)
    small = core.resize.Bilinear(clip, 80, 60, format=vs.GRAY8)
    lo, hi = max(0, around - step // 2), min(clip.num_frames - 1, around + step // 2)
    best = max(range(lo, hi + 1), key=lambda k: float(autocolour.thumb_z(np.asarray(small.get_frame(k)[0])) @ t))
    return _yuv(clip.get_frame(best))


def match_reference(src, own, ref, paths):
    """Pairs of frames that show the same picture in the source and in the reference (another release is timed
    differently, may be a different length or cut). Both ways: the reference frames searched in a thumbnail index
    of the whole source, and the source's own sampled frames searched in an index of each reference video - a short
    clip against a whole episode only matches the second way. Returns (reference frames, source frames) of the
    matched pairs when enough match (then auto colour compares exactly the same pictures), else (all reference
    frames, None): "other scenes"."""
    import os
    import autocolour
    n = src.num_frames
    step = 15 if n > 300 else 1
    twin, _ = autocolour.find_twins(_thumb_index(src, step), np.stack([autocolour.thumb_z(f[0]) for f in ref]))
    hit = [i for i in range(len(ref)) if twin[i] >= 0]
    pairs_a = ([ref[i] for i in hit], [_exact_twin(src, int(min(twin[i] * step, n - 1)), ref[i][0], step) for i in hit])
    share_a = len(hit) / max(1, len(ref))
    best = (share_a, pairs_a)
    own_t = np.stack([autocolour.thumb_z(f[0]) for f in own])
    for p in paths:
        if not (p and os.path.isfile(p)):
            continue
        c = _ref_clip(p)
        if c.num_frames <= 2:
            continue
        st = 15 if c.num_frames > 300 else 1
        tw, _ = autocolour.find_twins(_thumb_index(c, st), own_t)
        got = [i for i in range(len(own)) if tw[i] >= 0]
        share = len(got) / max(1, len(own))
        if share > best[0]:
            best = (share, ([_exact_twin(c, int(min(tw[i] * st, c.num_frames - 1)), own[i][0], st) for i in got],
                            [own[i] for i in got]))
    share, (rf, sf) = best
    print(f"colour reference: {round(100 * share)} % of the frames found on the other side ({len(rf)} pairs)", file=sys.stderr)
    if len(rf) >= 3 and share >= autocolour.REF_SAME_SHARE:
        return rf, sf
    return ref, None


def _ref_clip(p):
    c = core.bs.VideoSource(p)
    if c.format.color_family == vs.RGB:
        c = core.resize.Bicubic(c, format=vs.YUV444P8, matrix=6, range=0, dither_type="error_diffusion")
    elif c.format.id not in (vs.YUV420P8, vs.YUV444P8):
        c = core.resize.Bicubic(c, format=vs.YUV444P8 if c.format.subsampling_w == 0 else vs.YUV420P8,
                                dither_type="error_diffusion")
    return c


def ref_frames(paths, samples=60):
    """Frames of the colour reference (images and / or videos of the same show, a better release or a good
    episode): every image once, the videos sampled evenly with what is left of `samples`. None without one."""
    import os
    paths = [p for p in paths if p and os.path.isfile(p)]
    if not paths:
        return None
    clips = [_ref_clip(p) for p in paths]
    stills = [c for c in clips if c.num_frames <= 2]
    vids = [c for c in clips if c.num_frames > 2]
    per = max(8, (samples - len(stills)) // max(1, len(vids))) if vids else 0
    out = []
    for c in stills:
        f = c.get_frame(0)
        out.append(tuple(np.asarray(f[i]).copy() for i in range(3)))
    for c in vids:
        n = c.num_frames
        for t in np.linspace(0.04 * n, 0.96 * n, per).astype(int):
            f = c.get_frame(int(min(t, n - 1)))
            out.append(tuple(np.asarray(f[i]).copy() for i in range(3)))
    return out


def crop(clip, job):
    l, r, t, b = crop_values(job)
    if not (l or r or t or b):
        return clip
    sw, sh = job.get("src_w") or clip.width, job.get("src_h") or clip.height
    if (clip.width, clip.height) != (sw, sh) and sw and sh:   # an AI upscale in a filter group: crop at that scale
        kx, ky = clip.width / sw, clip.height / sh
        l, r, t, b = (int(round(v * k / 2)) * 2 for v, k in ((l, kx), (r, kx), (t, ky), (b, ky)))
    return core.std.Crop(clip, l, r, t, b)


# Order of the Colour and Crop & size steps (the GUI's rail can reorder them). Crop itself always runs first (it is in
# source pixels); "size" = output size + line darkening + sharpening. (AI upscaling was removed 2026-11-22; older
# settings that still list "upscale" are read without it.)
STAGES = ("colour", "size")


def stage_order(job):
    o = [s for s in (job.get("stage_order") or []) if s in STAGES]
    return o if sorted(o) == sorted(STAGES) else list(STAGES)


def resize_size(clip, job):
    """Output size only (Crop & size step placed before another step): 16-bit 4:4:4, the DVD's colour
    space kept; the output format and colour tags follow at the end (output_format)."""
    ow, oh = job.get("out_w") or clip.width, job.get("out_h") or clip.height
    ow, oh = ow // 2 * 2, oh // 2 * 2
    kw = dict(format=vs.YUV444P16)
    kernel = opt(job, "kernel")
    if kernel == "nnedi3" and (ow > clip.width * 1.05 or oh > clip.height * 1.05):
        out = core.resize.Spline36(nnedi3_double(clip), ow, oh, src_left=-0.5, src_top=-0.5, **kw)
    else:
        out = _resize(clip, ow, oh, kernel, bc=(job.get("kernel_b", 1 / 3), job.get("kernel_c", 1 / 3)), **kw)
    return core.std.CopyFrameProps(out, clip)


def output_format(clip, job):
    """Output format and colour tags at the size the picture has now (no resize)."""
    return resize_output(clip, dict(job, out_w=clip.width, out_h=clip.height))


VSF_POINTS = ("source", "ivtc", "after_ivtc", "after_composite", "after_cleanup", "after_halos", "repairs", "after_stab",
              "after_crop", "after_size", "after_sharpen", "end")
STAB_POINTS = ("after_ivtc", "after_cleanup", "after_halos", "default", "after_size", "end")   # job "stab_pos" (older jobs)

# Repair steps after Telecine and Composite artifacts, in the order they run (job "repair_order", dragged in the app)
REPAIR_STEPS = ("compression", "halos", "colour", "crop", "stab", "size", "sharpen")
STEP_POINT = {"compression": "after_cleanup", "halos": "after_halos", "colour": "repairs", "crop": "after_crop",
              "stab": "after_stab", "size": "after_size", "sharpen": "after_sharpen"}   # the filter groups placed after it
_STAB_AFTER = {"after_ivtc": None, "after_cleanup": "compression", "after_halos": "halos", "default": "crop",
               "after_size": "size", "end": "sharpen"}


def repair_order(job):
    """the movable Repair steps in their order; a missing step goes back to its usual place. Older jobs: the
    default order with the stabilizer where "stab_pos" put it."""
    o = job.get("repair_order")
    if isinstance(o, (list, tuple)) and o:
        out = [x for x in dict.fromkeys(o) if x in REPAIR_STEPS]
        for i, x in enumerate(REPAIR_STEPS):
            if x not in out:
                out.insert(min(i, len(out)), x)
        return out
    out = [x for x in REPAIR_STEPS if x != "stab"]
    a = _STAB_AFTER[stab_pos(job)]
    out.insert(0 if a is None else out.index(a) + 1, "stab")
    return out


def stab_pos(job):
    p = str(job.get("stab_pos") or "default")
    return p if p in STAB_POINTS else "default"


def _stab_ok(job):
    """FieldNet has its own steadiness stages (mc + gtfix + ttemp + still lock, or v5's TStab): no second stabilizer"""
    return not (opt(job, "ai_model") == "field" and str(job.get("field_stab", "filters")) != "off")


def vsf_groups(job):
    """[(point, [filters])]: the user's VapourSynth filter groups (Repair step, each dragged to its place), in order.
    The older single list (job "vs_filters" + "vs_filters_pos") counts as one more group."""
    ok = lambda fl: [f for f in (fl or []) if isinstance(f, dict) and f.get("on", True) and f.get("type") in ("vs", "jet", "ai", "my", "code")]
    out = []
    for g in job.get("vs_groups") or []:
        if isinstance(g, dict) and g.get("on", True) and ok(g.get("filters")):
            out.append((g.get("pos") if g.get("pos") in VSF_POINTS else "repairs", ok(g.get("filters"))))
    if ok(job.get("vs_filters")):
        p = opt(job, "vs_filters_pos")
        out.append((p if p in VSF_POINTS else "repairs", ok(job.get("vs_filters"))))
    return out


def vsf_at(job, point):
    """the filters that run at this point (all groups placed there, in order)"""
    return [f for p, fl in vsf_groups(job) if p == point for f in fl]


def vsf_pos(job):
    """compatibility: the first point that has filters, or None"""
    g = vsf_groups(job)
    return g[0][0] if g else None


def vs_filters(clip, job, point="repairs"):
    """Repair -> VapourSynth filters: any installed plugin function or vs-jetpack function, in the user's order, after the
    colour alignment and before crop / stabilizer. Each entry {"type": "vs" | "jet", "p": {"call", "args"}, "on"} runs
    through blocks.block (the EVA toolbox: typed arguments, the picture converted to a format the filter accepts and
    back). Whatever the filters do to the format, the chain continues in its own format; a changed size is kept."""
    fl = vsf_at(job, point)
    if not fl:
        return clip
    import blocks
    fmt = clip.format
    ctx = {"sar": 1.0, "group_in": clip, "prev_in": clip}   # @group / @prev in a filter's clip settings
    last_in = clip
    for f in fl:
        ctx["prev_in"], last_in = last_in, clip
        try:
            clip = blocks.block(clip, job, f["type"], f.get("p") or {}, ctx)
        except Exception as e:                           # noqa: BLE001
            raise RuntimeError(f"VapourSynth filter {f.get('p', {}).get('call', '?')}: {type(e).__name__}: {e}") from e
    if clip.format.id != fmt.id:                         # back to the chain's format (RGB / grey results: BT.601 YUV)
        kw = {"matrix_s": "170m"} if clip.format.color_family != vs.YUV and fmt.color_family == vs.YUV else {}
        clip = core.resize.Bicubic(clip, format=fmt.id, dither_type="error_diffusion", **kw)
    return clip


# Output sharpening and line darkening (Crop & size step), at the output size, after resizing.
SHARP = {
    "luma": 0.0,        # 0..2 luma sharpening
    "chroma": 0.0,      # 0..2 colour sharpening (colour edges, e.g. after ChromaNet or a soft upscale)
    "mode": "faithful", # faithful = never brighter / darker than the pixels around it (no halos, no ringing);
                        # unsharp = classic unsharp mask (crisper, can halo)
    "radius": "fine",   # fine (3x3, detail) / wide (5x5, outlines)
    "lines": 0.0,       # 0..1 line darkening: dark outlines thinner than ~3 px become darker
}


def sharp_settings(job):
    g = dict(SHARP)
    g.update({k: v for k, v in (job.get("sharp") or {}).items() if k in SHARP and v is not None})
    return g


def line_darken(clip, job):
    """Darkens thin dark lines (outlines) only: a pixel darker than the brightest pixel around it by more
    than a small threshold is pulled further down by `lines` x that difference (at most 1/4 of the range
    for strength 1). Flat areas, gradients and wide dark areas are unchanged (the 3x3 max is the area
    itself there). Luma only."""
    st = float(sharp_settings(job)["lines"])
    if st <= 0:
        return clip
    bits = clip.format.bits_per_sample
    k = (1 << bits) / 256.0
    y = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    mx = core.std.Maximum(y)
    thr, lim = 4 * k, 64 * k
    d = core.std.Expr([y, mx], f"y x - {thr} - 0 max {lim} min {st} * x swap - {16 * k} max")
    return core.std.ShufflePlanes([d, clip], [0, 1, 2], vs.YUV)


def _sharp_plane(p, amount, mode, radius):
    if amount <= 0:
        return p
    if radius == "wide":
        blur = core.std.Convolution(p, [1, 4, 6, 4, 1], mode="hv")
    else:
        blur = core.std.Convolution(p, [1, 2, 1, 2, 4, 2, 1, 2, 1])
    if mode == "unsharp":
        return core.std.Expr([p, blur], f"x x y - {amount} * +")
    lo, hi = core.std.Minimum(p), core.std.Maximum(p)
    return core.std.Expr([p, blur, lo, hi], f"x x y - {amount} * + z max a min")


def sharpen(clip, job):
    """Luma and colour sharpening at the output size. Faithful: the sharpened value is kept between the
    darkest and brightest pixel of its 3x3 neighbourhood, so edges get steeper without the bright / dark
    rims (halos) a plain unsharp mask adds."""
    g = sharp_settings(job)
    lu, cr = float(g["luma"]), float(g["chroma"])
    if lu <= 0 and cr <= 0:
        return clip
    planes = [core.std.ShufflePlanes(clip, i, vs.GRAY) for i in range(3)]
    planes[0] = _sharp_plane(planes[0], lu, g["mode"], g["radius"])
    planes[1] = _sharp_plane(planes[1], cr, g["mode"], g["radius"])
    planes[2] = _sharp_plane(planes[2], cr, g["mode"], g["radius"])
    return core.std.CopyFrameProps(core.std.ShufflePlanes(planes, [0, 0, 0], vs.YUV), clip)


def resize_output(clip, job):
    """Output size, resizer, 10-bit 4:2:0 and the colour tags of the output (BT.709 from 720 lines up)."""
    ow, oh = job.get("out_w") or clip.width, job.get("out_h") or clip.height
    ow, oh = ow // 2 * 2, oh // 2 * 2
    hd = oh >= 720
    sd = sd_colour(job)                     # (matrix, primaries, transfer) of the DVD
    import encoding
    fmt = encoding.vs_format(job)
    kw = dict(format=fmt, matrix_in=sd[0], primaries_in=sd[1], transfer_in=sd[2], range_in=0, range=0)
    if core.get_video_format(fmt).bits_per_sample == 8:
        kw["dither_type"] = "error_diffusion"
    out = (1, 1, 1) if hd else sd
    kw.update(matrix=out[0], primaries=out[1], transfer=out[2])
    kernel = opt(job, "kernel")
    if (ow, oh) == (clip.width, clip.height):
        clip = _resize(clip, ow, oh, kernel, bc=(job.get("kernel_b", 1 / 3), job.get("kernel_c", 1 / 3)), **kw)
    elif kernel == "nnedi3" and (ow > clip.width * 1.05 or oh > clip.height * 1.05):
        d = nnedi3_double(clip)
        clip = core.resize.Spline36(d, ow, oh, src_left=-0.5, src_top=-0.5, **kw)     # undo the +0.25 px offset
    else:
        clip = _resize(clip, ow, oh, kernel, bc=(job.get("kernel_b", 1 / 3), job.get("kernel_c", 1 / 3)), **kw)
    return core.std.SetFrameProps(clip, _Matrix=out[0], _Primaries=out[1], _Transfer=out[2], _Range=vs.RANGE_LIMITED)


def detect_crop(job, samples=80):
    """Find black borders (letterbox / pillarbox / the 8-px side bars of most DVDs). A row or column
    counts as border when almost all its pixels stay near black across many frames (fades excluded)."""
    src = source(job, clean=False)
    n = src.num_frames
    acc = []
    for t in np.linspace(n * 0.05, n * 0.95, samples).astype(int):
        Y = np.asarray(src.get_frame(int(t))[0], dtype=np.float32)
        if Y.mean() < 30:                    # black / fade frame says nothing about borders
            continue
        acc.append(Y)
    if not acc:
        return {"l": 0, "r": 0, "t": 0, "b": 0}
    Y = np.stack(acc)                        # frames x H x W
    black = 16 + 12
    col = np.percentile(Y, 98, axis=(0, 1))  # per column: brightest typical value
    row = np.percentile(Y, 98, axis=(0, 2))
    col_med = np.median(Y, axis=(0, 1)) - 16  # typical brightness above black
    row_med = np.median(Y, axis=(0, 2)) - 16

    def run(v, med):
        k = 0
        while k < len(v) // 4 and v[k] <= black:
            k += 1
        # soft analog edge: up to 4 more lines that are clearly darker than the picture just inside
        extra = 0
        while extra < 4 and k + 16 < len(med):
            ref = np.median(med[k + 6:k + 16])
            if ref <= 4 or med[k] >= 0.85 * ref:
                break
            k += 1
            extra += 1
        return k
    l, r = run(col, col_med), run(col[::-1], col_med[::-1])
    t, b = run(row, row_med), run(row[::-1], row_med[::-1])
    up = lambda x: (x + 1) // 2 * 2          # round up to even so no dark line survives
    return {"l": up(l), "r": up(r), "t": up(t), "b": up(b)}


def sd_colour(job):
    """DVD colour: NTSC = BT.601-525 (SMPTE-C primaries), PAL = BT.601-625 (EBU primaries)."""
    return (5, 5, 6) if is_pal(job) else (6, 6, 6)


def anti_alias(clip, job):
    return _aa(clip, job) if opt(job, "aa") == "all" else clip


def repair_dials(job, ch=None):
    """(halo, bleed) dials when RepairNet does the AI halos and / or AI bleed, else None (the separate models run).
    halo = the halo strength (capped at 1) when halo removal is on with the AI method; bleed = the ChromaNet strength
    (light 0.6, normal / strong 1.0) when the bleed fix is AI. One of the two may be 0."""
    if opt(job, "repairnet") != "on":
        return None
    ch = job.get("chroma", {}) if ch is None else ch
    halo_on = opt(job, "ghost") != "off" and bool(job.get("halo_rx") or job.get("ghost_taps")) and opt(job, "halo_method") == "ai"
    bleed_on = ch.get("method", "off") == "ai"
    if not (halo_on or bleed_on):
        return None
    import repairnet
    if not repairnet.available():
        return None
    h = max(0.0, min(1.0, float(opt(job, "ghost_strength")))) if halo_on else 0.0
    b = {"light": 0.6}.get(ch.get("strength", "normal"), 1.0) if bleed_on else 0.0
    return h, b


def repair_colour(clip, job, ch, dials):
    """colour shift (as before), then one RepairNet pass for halos + bleed (dials), steadied like the other models"""
    import repairnet
    before = chroma_fix(clip, ch.get("shift_x", 0.0), ch.get("shift_y", 0.0), 0, "off")
    return temporal_guard(before, repairnet.repairnet(before, halo=dials[0], bleed=dials[1]))


def halos(clip, job):
    if repair_dials(job) and opt(job, "halo_method") == "ai":
        return clip                                    # RepairNet does the halos in the colour step
    if opt(job, "ghost") != "off" and (job.get("halo_rx") or job.get("ghost_taps")):
        return ghost_halo(clip, job)
    return clip


def colour_fix(clip, job):
    ch = job.get("chroma", {})
    dials = repair_dials(job, ch)
    if dials:
        clip = repair_colour(clip, job, ch, dials)
    else:
        clip = chroma_fix(clip, ch.get("shift_x", 0.0), ch.get("shift_y", 0.0), int(ch.get("warp", 0)),
                          ch.get("method", "off"), ch.get("strength", "normal"))
    return neutralize(clip) if ch.get("neutral", False) else clip


def to_display(clip, job):
    """RGB24 at square-pixel display size (what the preview shows)."""
    dar = job.get("dar_display") or job.get("dar", 4 / 3)
    h = clip.height
    w = int(round(h * dar / 2) * 2) if not job.get("out_w") else clip.width
    if job.get("chain"):                                # the chain's own pixel shape (after its crops / resizes / upscales)
        try:
            pr = clip.get_frame(0).props
            sar = float(pr.get("_SARNum", 1)) / float(pr.get("_SARDen", 1) or 1)
        except Exception:                                # noqa: BLE001
            sar = 1.0
        w = int(round(clip.width * sar / 2) * 2)
    mat = 1 if clip.height >= 720 else sd_colour(job)[0]
    return core.resize.Spline36(clip, w, h, format=vs.RGB24, matrix_in=mat)


# keys that belong to one episode; the rest of the job are settings shared by every episode
EPISODE_KEYS = ("source", "start", "end", "fps_num", "fps_den", "work", "output")


def script_text(job, title=""):
    """The job as an editable VapourSynth script: every stage that is switched on, in processing order,
    one line each. Running it (vspipe, vsview, the app's Script step) gives exactly build(job)."""
    import datetime
    import pprint
    ep = {k: job[k] for k in EPISODE_KEYS if k in job}
    st = {k: v for k, v in job.items() if k not in EPISODE_KEYS and k != "chain"}   # Chain step: the dr.block lines are the chain
    pal, ntsc = is_pal(job), not is_pal(job)
    L = [
        "# DVD Restore - VapourSynth script" + (f" for {title}" if title else ""),
        f"# Made {datetime.datetime.now():%Y-%m-%d %H:%M}. Edit freely: the Script step previews and exports this text.",
        "# Every stage reads its settings from `job`; change a value there, or add / remove / reorder lines below.",
        "# Run outside the app:  vspipe -c y4m script.vpy - | ffmpeg -i - ...   or open it in vsview.",
        "import sys",
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})",
        "import vapoursynth as vs",
        "import vsgraph as dr",
        "core = vs.core",
        "",
        "# ---- settings ----",
        "job = " + pprint.pformat(st, indent=1, width=110, sort_dicts=True).replace("\n", "\n      "),
        "",
        "# ---- episode (DVD Restore fills this block in for every episode it exports) ----",
        "EPISODE = " + pprint.pformat(ep, indent=1, width=110, sort_dicts=False).replace("\n", "\n          "),
        "# ---- end of episode ----",
        "job.update(EPISODE)",
        "",
        "dr.nnbackend.set_backend(job.get('backend', 'pytorch'))   # neural models: 'pytorch' or 'tensorrt'",
        "",
    ]
    if job.get("chain"):                                # Chain step: one dr.block(...) line per block, exactly the chain
        import blocks
        L[2] = "# Every block is one dr.block(...) line with its settings written out - this is exactly the chain."
        L += blocks.chain_lines(job)
        L += ["", "clip.set_output()", ""]
        return "\n".join(L)
    clean = lambda key: opt(job, key) != "off"
    L.append("clip = dr.source(job)" + ("                          # load; dot crawl / rainbow cancellation before IVTC" if ntsc else "                          # load"))
    vn = lambda pt: ", ".join(f.get("p", {}).get("call", "?") for f in vsf_at(job, pt))

    def vline(pt, what="your VapourSynth filters"):
        if vsf_at(job, pt):
            L.append(f'clip = dr.vs_filters(clip, job, "{pt}")'.ljust(47) + f"  # {what}: {vn(pt)}")

    if vsf_at(job, "source"):
        L[-1] += f"   (+ your VapourSynth filters on the source: {vn('source')})"
    own_ivtc = bool(vsf_at(job, "ivtc"))
    if own_ivtc:
        vline("ivtc", "your own IVTC (instead of the app's)")
    elif pal:
        L.append("clip = dr.pal_process(clip, job)               # PAL: progressive / field-shifted / interlaced")
    elif job.get("mode") == "vfr":
        L.append("clip = dr.ivtc_vfr(clip, job)                  # inverse telecine, VFR (needs the app's pass 1 files in job['work'])")
    else:
        L.append("clip = dr.ivtc_cfr(clip, job)                  # inverse telecine to 23.976 fps")
    if opt(job, "deblend") == "ai" and not own_ivtc:
        L.append("clip = dr.deblend_ai(clip, job)                # AI blend removal (DeblendNet)")
    vline("after_ivtc")
    if opt(job, "aa") == "all":
        L.append("clip = dr.anti_alias(clip, job)                # AANet on every frame")
    order = repair_order(job)
    if order != [x for x in REPAIR_STEPS]:
        L.append(f"# Repair order: {' -> '.join(order)} (job['repair_order'])")
    grading = bool(grade_active(grade_settings(job)) or grade_layers(job))
    gline = ("clip = dr.grade(clip, job)                     # colour correction" +
             (f" + {len(grade_layers(job))} adjustment layers (job['grade_layers'])" if grade_layers(job) else ""))

    def compression():
        if clean("deblock"):
            L.append(f"clip = dr.deblock(clip, job)                   # MPEG repair: {opt(job, 'deblock')}")
    early = order[0] == "compression"
    if early:
        compression()
    if ntsc and (clean("crawl") or clean("rainbow")):
        L.append(f"clip = dr.xtalk_stage_b(clip, job)             # dots + rainbows after IVTC: {opt(job, 'crawl')} / {opt(job, 'rainbow')}")
    vline("after_composite")
    out_done = False
    for i, step in enumerate(order):
        rest = order[i + 1:]
        if step == "compression" and not early:
            compression()
        elif step == "halos":
            if opt(job, "ghost") != "off" and (job.get("halo_rx") or job.get("ghost_taps")):
                L.append(f"clip = dr.halos(clip, job)                     # halo removal: {opt(job, 'halo_method')}")
        elif step == "colour":
            L.append("clip = dr.colour_fix(clip, job)                # colour shift, bleed, clean white / black")
        elif step == "crop":
            if any(crop_values(job)):
                L.append("clip = dr.crop(clip, job)")
        elif step == "stab":
            if clean("stab") and _stab_ok(job):
                L.append(f"clip = dr.stabilize(clip, job)                 # temporal stabilizer: {opt(job, 'stab')}")
        elif step == "size":
            if not grading and all(r == "sharpen" for r in rest):
                L.append("clip = dr.resize_output(clip, job)             # output size, 10-bit, colour tags")
                out_done = True
            else:
                L.append("clip = dr.resize_size(clip, job)               # output size (format and tags at the end)")
        elif step == "sharpen":
            sg = sharp_settings(job)
            if float(sg["lines"]) > 0:
                L.append(f"clip = dr.line_darken(clip, job)               # line darkening {sg['lines']}")
            if float(sg["luma"]) > 0 or float(sg["chroma"]) > 0:
                L.append(f"clip = dr.sharpen(clip, job)                   # sharpening ({sg['mode']}): luma {sg['luma']}, colour {sg['chroma']}")
        vline(STEP_POINT[step])
    if grading:
        L.append(gline)
    if not out_done:
        L.append("clip = dr.output_format(clip, job)             # 10-bit, colour tags (size as it is now)")
    vline("end", "your VapourSynth filters at the end")
    L += ["", "clip.set_output()", ""]
    return "\n".join(L)


def set_episode(text, ep):
    """Replace the EPISODE block of a (possibly edited) script with another episode's values."""
    import pprint
    a = text.find("EPISODE = ")
    b = text.find("# ---- end of episode ----")
    if a < 0 or b < a:
        raise ValueError("the script has no EPISODE block (keep the lines 'EPISODE = ...' and '# ---- end of episode ----')")
    return text[:a] + "EPISODE = " + pprint.pformat(ep, indent=1, width=110, sort_dicts=False).replace("\n", "\n          ") + "\n" + text[b:]


def run_script(text, name="script.vpy"):
    """Execute a script and return its output 0."""
    vs.clear_outputs()
    ns = {"__name__": "__vapoursynth__", "__file__": name}
    exec(compile(text, name, "exec"), ns)
    out = vs.get_output(0)
    vs.clear_outputs()
    return out.clip if hasattr(out, "clip") else out


def deblend_ai(clip, job):
    """Blend removal = AI: DeblendNet on the finished frame sequence (deblendnet.py)."""
    if opt(job, "deblend") != "ai":
        return clip
    try:
        import deblendnet
        if deblendnet.available():
            return deblendnet.deblendnet(clip, sd_colour(job)[0])
        print("models/deblendnet.pt missing: blend removal off", file=sys.stderr)
    except Exception as e:                                # noqa: BLE001
        print(f"DeblendNet not usable ({e})", file=sys.stderr)
    return clip


def block(clip, job, type_, p, ctx=None):
    """one block of the Chain step (blocks.py); the Chain script calls this for every block"""
    import blocks
    return blocks.block(clip, job, type_, p, ctx)


def chain_context(job, before=False):
    import blocks
    return blocks.chain_context(job, before)


def chain_finish(clip, ctx):
    import blocks
    return blocks.chain_finish(clip, ctx)


def build(job, ivtc=True, chroma=True, clean=True):
    if job.get("chain"):                                # the Chain step: the blocks, in their order
        import blocks
        return blocks.build_chain(job, before=not clean)
    nnbackend.set_backend(opt(job, "backend"))
    clip = source(job, clean=clean)
    if ivtc and job.get("progressive"):
        pass                                           # progressive source / image: nothing to field-match or deinterlace
    elif ivtc and vsf_at(job, "ivtc"):
        clip = vs_filters(clip, job, "ivtc")           # your own IVTC / deinterlacing instead of the app's
    elif ivtc:
        if is_pal(job):
            clip = pal_process(clip, job)
        else:
            clip = ivtc_vfr(clip, job) if job.get("mode") == "vfr" else ivtc_cfr(clip, job)
        clip = deblend_ai(clip, job)
    if clean:
        clip = vs_filters(clip, job, "after_ivtc")
    if opt(job, "aa") == "all" and not (opt(job, "ai_model") == "field" and raw_ivtc(job)):
        clip = _aa(clip, job)                          # not before FieldNet: it repairs from the raw IVTC itself
    # the all-in-one model (ai_model "uni") also does MPEG repair, halos and colour alignment: the single-frame
    # stages for an artifact its dial handles are left out (no double correction)
    # FieldNet v3 also knows PAL (its own composite patterns, 2:2 film, bobbed 50i): it runs on PAL discs too
    uni = opt(job, "ai_model") in ("uni", "field") and _ai_cleaner(job) and (not is_pal(job) or opt(job, "ai_model") == "field")
    ud = uni_dials(job) if uni else {}
    order = repair_order(job)
    # the Colour step's grade runs after every Repair step (the app's order: Repair, then Colour). The older
    # "stage_order" (Colour before / after a separate Crop & size step) is no longer read: that step is part of
    # Repair now and the setting could not be seen or changed any more
    grading = bool(grade_active(grade_settings(job)) or grade_layers(job))

    def compression(c):
        return c if (uni and ud["mpeg"] > 0) else deblock(c, job)
    early = clean and order[0] == "compression"        # its usual place: MPEG repair before the dot crawl clean-up
    if early:
        clip = compression(clip)
    if clean:
        if not is_pal(job) or uni:
            clip = xtalk_stage_b(clip, job)
        clip = vs_filters(clip, job, "after_composite")
    out_done = False
    for i, step in enumerate(order):
        rest = order[i + 1:]
        if step == "compression":
            if clean and not early:
                clip = compression(clip)
        elif step == "halos":
            rn = repair_dials(job) if not uni else None
            if clean and opt(job, "ghost") != "off" and (job.get("halo_rx") or job.get("ghost_taps")) \
                    and not (uni and ud["halo"] > 0) and not (rn and opt(job, "halo_method") == "ai"):
                clip = ghost_halo(clip, job)
        elif step == "colour":
            ch = job.get("chroma", {}) if chroma else {}
            if uni and ud["chroma"] > 0:
                ch = dict(ch, method="off", shift_x=0.0, shift_y=0.0)
            dials = repair_dials(job, ch) if clean and not uni else None
            if dials:                                  # RepairNet: halos + bleed in one pass (after the colour shift)
                clip = repair_colour(clip, job, ch, dials)
            else:
                clip = chroma_fix(clip, ch.get("shift_x", 0.0), ch.get("shift_y", 0.0), int(ch.get("warp", 0)),
                                  ch.get("method", "off"), ch.get("strength", "normal"))
            if chroma and ch.get("neutral", False):
                clip = neutralize(clip)
        elif step == "crop":
            clip = crop(clip, job)
        elif step == "stab":
            if _stab_ok(job):
                clip = stabilize(clip, job)
        elif step == "size":
            if not grading and all(r == "sharpen" for r in rest):
                clip = resize_output(clip, job)        # nothing but sharpening after it: straight to the output format
                out_done = True
            else:
                clip = resize_size(clip, job)
        elif step == "sharpen":
            clip = sharpen(line_darken(clip, job), job)
        if clean:
            clip = vs_filters(clip, job, STEP_POINT[step])
    if grading:
        clip = grade(clip, job)
    if not out_done:
        clip = output_format(clip, job)
    if clean:
        clip = vs_filters(clip, job, "end")
    return clip


def detect_scenes(job, progress=None, cancelled=None, min_len=12):
    """Scene cuts of the episode, as source frames counted from the start of the title. Small greyscale
    frames are compared with the one before; a cut is a difference that stands out against the frames
    around it (so pans and fades are not cut every frame), at least min_len frames after the last cut.
    The repeated field of telecine and a flash count as no cut."""
    src = source(job, clean=False)
    start = int(job.get("start", 0))
    sm = core.resize.Bilinear(src, 128, 96, format=vs.GRAY8)
    st = core.std.PlaneStats(sm, sm[0] + sm[:-1])
    n = st.num_frames
    d = np.zeros(n, np.float32)
    for i, f in enumerate(st.frames(prefetch=16, backlog=32)):
        d[i] = f.props["PlaneStatsDiff"]
        if i % 2000 == 0:
            if progress:
                progress(i, n)
            if cancelled and cancelled():
                raise RuntimeError("Stopped")
    cuts = []
    last = -min_len
    for i in range(1, n - 1):
        around = np.concatenate([d[max(0, i - 6):i], d[i + 1:i + 7]])
        ref = float(np.percentile(around, 75)) if len(around) else 0.0
        if d[i] > 0.09 and d[i] > 3.0 * ref + 0.02 and i - last >= min_len:
            # a flash: the frame after it is close to the frame before it again
            if i + 2 < n:
                a = np.asarray(sm.get_frame(i - 1)[0], np.float32)
                b = np.asarray(sm.get_frame(min(n - 1, i + 2))[0], np.float32)
                if float(np.abs(a - b).mean()) / 255 < 0.35 * d[i]:
                    continue
            cuts.append(start + i)
            last = i
    return cuts


def locate_source_frame(job, src_n):
    """Preview frame (the CFR or VFR frames the Colour step shows) that holds source frame src_n."""
    tm = "vfr" if job.get("mode") == "vfr" and job.get("preview_vfr") and not is_pal(job) else "cfr"
    j = dict(job, mode=tm, grade={}, grade_layers=[], sharp={}, upscale={}, stab="off",
             deblend="off" if opt(job, "deblend") == "ai" else opt(job, "deblend"), aa="off", ivtc_repair="off")
    if is_pal(j) or j.get("chain"):
        light = build(j, ivtc=True, chroma=False, clean=False)
    else:
        # only field matching + decimation, on the SAME source the VFR analysis saw: with the classic crawl cancellation
        # before IVTC (crawl / rainbow on) a source without it made TFM refuse the pass-1 files ("crc32 in input file
        # does not match that of the current clip" - original.mkv)
        src = source(j, clean=True)
        light = ivtc_vfr(src, j) if tm == "vfr" else ivtc_cfr(src, j)
    lo, hi = 0, light.num_frames - 1
    while lo < hi:                                   # first frame whose source frame is >= src_n
        mid = (lo + hi) // 2
        if int(light.get_frame(mid).props.get("_SrcN", 0)) < src_n:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _timecode_durations(path, n_frames):
    """Frame durations (seconds) from an mkv timecodes v2 file."""
    with open(path, encoding="utf-8") as f:
        ts = [float(l) for l in f if l.strip() and not l.startswith("#")]
    d = [(b - a) / 1000 for a, b in zip(ts, ts[1:])]
    d.append(1001 / 24000)
    return d[:n_frames] if len(d) >= n_frames else None


def preview_outputs(job):
    """Register comparison outputs for vsview (tabs). VFR timing is used when pass 1 already ran."""
    import os
    from vsview import set_output
    w = job.get("work", "")
    import pass1sig                                     # only a pass 1 made for THIS clip (else TFM: crc32 mismatch)
    have_pass1 = bool(w) and pass1sig.matches(job, w)
    vfr = job.get("mode") == "vfr" and have_pass1
    pj = dict(job, mode="vfr" if vfr else "cfr")
    ch = dict(job.get("chroma", {}))

    raw = source(job, clean=False)
    set_output(raw, 0, "Source" + (" (telecined)" if not is_pal(job) else ""))
    if job.get("chain"):                                # Chain step: untouched (field matched, cropped, sized) and final
        import blocks
        before = blocks.build_chain(pj, before=True)
        steps, final = blocks.chain_steps(pj)            # one tab per block: the picture after it
        fd = _timecode_durations(os.path.join(w, "timecodes.txt"), final.num_frames) if vfr else None
        tag = " [VFR]" if fd else (" [PAL]" if is_pal(job) else " [CFR]")

        def dur(c):                                      # VFR timing only where the frames are the export's
            return fd if fd and c.num_frames == len(fd) else None
        set_output(before, 1, "Before (field matched)" + tag, framedurs=dur(before))
        set_output(final, 2, "FINAL (chain)" + tag, framedurs=fd)
        for i, (name, c) in enumerate(steps, 1):
            set_output(c, 2 + i, f"{i}. after {name}", framedurs=dur(c))
        return
    idx = 1
    if (opt(job, "crawl") != "off" or opt(job, "rainbow") != "off") and not is_pal(job):
        cleaned = source(job)
        set_output(cleaned, idx, "Dot crawl/rainbow cleaned (before IVTC)")
        # what was removed, x4 on grey: crawl shows as a fine checkerboard, rainbows as colour
        rem = core.std.Expr([raw, cleaned], ["x y - 4 * 128 +"])
        set_output(rem, idx + 1, "Removed crawl/rainbow x4")
        idx += 2
    before = build(pj, chroma=False)
    shift_only = build(dict(pj, chroma=dict(ch, method="off")))
    final = build(pj)
    fd = _timecode_durations(os.path.join(w, "timecodes.txt"), final.num_frames) if vfr else None
    tag = " [VFR]" if fd else (" [PAL]" if is_pal(job) else " [CFR]")
    set_output(before, idx, "IVTC, no chroma fix" + tag, framedurs=fd)
    set_output(shift_only, idx + 1, "Chroma shift only" + tag, framedurs=fd)
    set_output(final, idx + 2, "FINAL" + tag, framedurs=fd)
    # colour change x4 on a dimmed picture: grey = unchanged, colour = what the fix added/removed
    diff = core.std.Expr([final, before], ["x 2 / 256 +", "x y - 4 * 512 +", "x y - 4 * 512 +"])
    set_output(diff, idx + 3, "Chroma change x4" + tag, framedurs=fd)


def preview(job, after=True):
    """Preview converted to RGB24 at square-pixel display size. CFR (no VFR pass needed) - unless the job is VFR
    and its pass 1 exists (job["preview_vfr"]): then the frames and timing are exactly the export's (30p video
    sections such as scrolling captions play evenly instead of skipping a step every 5 frames)."""
    tm = "vfr" if job.get("mode") == "vfr" and job.get("preview_vfr") and not is_pal(job) else "cfr"
    # "Before" = the untouched DVD, only field-matched, cropped and sized like "After" so Split lines up:
    # no colour correction, stabilizer, AI blend removal, anti-aliasing, AI upscale, sharpening or line darkening
    pjob = dict(job, mode=tm) if after else dict(job, mode=tm, grade={}, grade_layers=[], sharp={}, upscale={}, stab="off",
                                                 deblend="off", aa="off")
    if job.get("chain"):
        import blocks
        return to_display(blocks.build_chain(dict(job, mode=tm), before=not after), job)
    try:
        clip = build(pjob, ivtc=True, chroma=after, clean=after)
    except vs.Error as e:
        # a VFR analysis made for a slightly different clip (TDecimate: "crc32 in input file does not match"): set it
        # aside so it is made again, and show 24p meanwhile instead of failing the preview
        if "crc32" not in str(e) or tm != "vfr" or not job.get("work"):
            raise
        sig = os.path.join(job["work"], "pass1.sig")
        try:
            os.replace(sig, sig + ".stale")
        except OSError:
            pass
        print("The saved VFR analysis no longer matches this clip: it is made again (24p preview meanwhile)", file=sys.stderr, flush=True)
        clip = build(dict(pjob, mode="cfr", preview_vfr=False), ivtc=True, chroma=after, clean=after)
    return to_display(clip, job)


def _save_rgb(clip, n, path):
    from PIL import Image
    if clip.format.color_family != vs.RGB:
        clip = core.resize.Point(clip, format=vs.RGB24)        # uses the clip's own _Matrix tag
    f = clip.get_frame(int(n))
    Image.fromarray(np.dstack([np.asarray(f[p]) for p in range(3)])).save(path, "PNG", compress_level=6)


def _edge_profile(job, lines=24, samples=60):
    """Brightness of the outer rows/columns (98th percentile and median): shows borders and soft edges."""
    src = source(job, clean=False)
    n = src.num_frames
    acc = [np.asarray(src.get_frame(int(t))[0], dtype=np.float32)
           for t in np.linspace(n * 0.05, n * 0.95, samples).astype(int)]
    acc = np.stack([a for a in acc if a.mean() >= 30] or acc)
    out = {}
    for name, axis, sl in (("left", (0, 1), slice(0, lines)), ("right", (0, 1), slice(-lines, None)),
                           ("top", (0, 2), slice(0, lines)), ("bottom", (0, 2), slice(-lines, None))):
        p98 = np.percentile(acc, 98, axis=axis)[sl]
        med = np.median(acc, axis=axis)[sl]
        if name in ("right", "bottom"):
            p98, med = p98[::-1], med[::-1]                # outermost line first
        out[name] = {"p98": [round(float(v), 1) for v in p98], "median": [round(float(v), 1) for v in med]}
    return out


def report(job, out_dir, frames=6, progress=None):
    """Collect everything needed to tune a source: all raw measurements and sample frames."""
    import platform
    import time
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {"job": job, "errors": {}, "timing": {}}
    try:
        res["env"] = {"python": sys.version, "platform": platform.platform(), "vapoursynth": str(core.version()),
                      "plugins": sorted(f"{p.namespace} {getattr(p, 'version', '')}" for p in core.plugins())}
    except Exception as e:                                # noqa: BLE001
        res["errors"]["env"] = repr(e)
    steps = [("auto_detect", lambda: auto_detect(job)),
             ("crop", lambda: detect_crop(job)),
             ("edge_profile", lambda: _edge_profile(job)),
             ("ghost_raw", lambda: detect_ghost(source(job, clean=False))),
             ("chroma_current", lambda: chroma_stats(preview_chroma(job)))]
    if is_pal(job):
        steps.insert(0, ("pal_detect", lambda: pal_detect(job)))
    else:
        steps.insert(0, ("xtalk", lambda: dict(zip(("luma", "chroma"), measure_xtalk(job)))))
    total = len(steps) + frames
    for k, (name, fn) in enumerate(steps):
        if progress:
            progress(k, total)
        print(f"Report: {name}...", flush=True)
        t = time.time()
        try:
            res[name] = fn()
        except Exception as e:                            # noqa: BLE001
            res["errors"][name] = repr(e)
        res["timing"][name] = round(time.time() - t, 1)
    # sample frames: untouched source (still interlaced), before (IVTC only) and after (current settings)
    print("Report: sample frames...", flush=True)
    try:
        raw = source(job, clean=False)
        before, after = preview(job, after=False), preview(job, after=True)
        res["frames"] = []
        for i, fr in enumerate(np.linspace(0.08, 0.92, frames)):
            if progress:
                progress(len(steps) + i, total)
            nb = int(fr * (before.num_frames - 1))
            nr = int(fr * (raw.num_frames - 1))
            _save_rgb(raw, nr, out / f"frame{i + 1}_source.png")
            _save_rgb(before, nb, out / f"frame{i + 1}_before.png")
            _save_rgb(after, nb, out / f"frame{i + 1}_after.png")
            res["frames"].append({"pos": round(float(fr), 3), "source_frame": nr, "ivtc_frame": nb})
    except Exception as e:                                # noqa: BLE001
        res["errors"]["frames"] = repr(e)
    (out / "measurements.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return {"dir": str(out), "errors": res["errors"]}


def preview_chroma(job):
    """The clip that chroma is measured on, with the current settings applied (to check the result)."""
    pjob = dict(job, mode="cfr")
    return build(pjob, ivtc=True, chroma=True, clean=True)


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    with open(path, encoding="utf-8") as f:
        job = json.load(f)
    if cmd == "pass1":
        vfr_pass1(job)
    elif cmd == "ghost":
        g = detect_ghost(source(job, clean=False),
                         progress=lambda k, n: print(f"PROGRESS {k} {n}", flush=True))
        g["ghost_on"] = "auto" if g.get("halo_rx", 0) > 0 else "off"
        print("RESULT " + json.dumps(g), flush=True)
    elif cmd == "crop":
        print("RESULT " + json.dumps(detect_crop(job)), flush=True)
    elif cmd == "levels":
        print("RESULT " + json.dumps(measure_levels(job)), flush=True)
    elif cmd == "paldetect":
        print("RESULT " + json.dumps(pal_detect(job)), flush=True)
    elif cmd == "report":
        r = report(job, job["report_dir"], progress=lambda k, n: print(f"PROGRESS {k} {n}", flush=True))
        print("RESULT " + json.dumps(r), flush=True)
    elif cmd.startswith("auto_"):
        res = auto_group(cmd[5:], job, progress=lambda k, n: print(f"PROGRESS {k} {n}", flush=True))
        print("RESULT " + json.dumps(res), flush=True)
    elif cmd == "geometry":
        c = build(job)
        pr = c.get_frame(0).props
        print("RESULT " + json.dumps({"width": c.width, "height": c.height, "frames": c.num_frames,
                                      "sar": float(pr.get("_SARNum", 1)) / float(pr.get("_SARDen", 1) or 1)}), flush=True)
    elif cmd == "image":
        # a still (runner.probe_image): the repaired picture's middle frame as 16-bit PNG. Converted to RGB here with
        # the matrix the chain tagged (an HD-size result is BT.709, SD BT.601) - ffmpeg alone would guess 601
        import subprocess
        c = build(job)
        n = c.num_frames // 2
        rgb = core.resize.Bicubic(c, format=vs.RGB48, dither_type="error_diffusion")
        f = rgb.get_frame(n)
        a = np.dstack([np.asarray(f[p]) for p in range(3)]).astype("<u2")
        ff = shutil.which("ffmpeg") or r"C:fmpeginfmpeg.exe"
        try:
            import runner as _r
            ff = _r.TOOLS.get("ffmpeg") or ff
        except Exception:                                # noqa: BLE001
            pass
        r = subprocess.run([ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{rgb.width}x{rgb.height}",
                            "-i", "-", "-frames:v", "1", job["output"]], input=a.tobytes(), capture_output=True)
        if r.returncode:
            raise RuntimeError(r.stderr.decode(errors="replace")[-300:])
        print("RESULT " + json.dumps({"output": job["output"], "width": rgb.width, "height": rgb.height}), flush=True)
    elif cmd == "measure":
        res = auto_detect(job, progress=lambda k, n: print(f"PROGRESS {k} {n}", flush=True))
        print("RESULT " + json.dumps(res), flush=True)
