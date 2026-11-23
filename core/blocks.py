"""Processing blocks for the Chain step: the whole processing as an ordered list of blocks the user adds from a library
(the "+" box), each with its own settings.

job["chain"] = [{"type": <block id>, "on": bool, "p": {settings}}, ...]   (first block: "source", last: "output")

One function runs a block - block(clip, job, type, params, ctx) - and both the engine (build_chain) and the Script step
(chain_script) go through it, so the script is the chain exactly: every enabled block is one `dr.block(...)` line with
its settings written out. ctx carries the pixel shape (SAR) through the chain, so "square pixels" resizing knows the
shape after crops and upscales.

Library: the app's own stages (source, IVTC, PAL, AI cleanups, MPEG / halo / colour repair, stabilizers incl. the Steady
chain, still lock, crop, resize, sharpening, colour), ready-made vs-jetpack tools (QTGMC, DFTTest, NNEDI3 /
EEDI3 / SangNom anti-aliasing), every installed VapourSynth plugin function, every vs-jetpack function that takes and
returns a clip, and a custom-code block."""
import importlib
import inspect
import json
from fractions import Fraction

import vapoursynth as vs

core = vs.core


def _v():
    import vsgraph
    return vsgraph


# ---------------------------------------------------------------- parameter helpers
def P(key, label, kind, default, **kw):
    """one setting of a block. kind: choice (options or opts = a list name of the app's options), int, float, bool,
    text, code; show = rule on the block's own settings (key=value, key!value, | or, & and)"""
    d = {"key": key, "label": label, "kind": kind, "default": default}
    d.update(kw)
    return d


LEVELS = [["off", "Off"], ["light", "Light"], ["normal", "Normal"], ["strong", "Strong"]]
LEVELS_ON = [["light", "Light"], ["normal", "Normal"], ["strong", "Strong"]]
FORMATS = [["YUV420P8", "YUV 4:2:0 8-bit"], ["YUV420P10", "YUV 4:2:0 10-bit"], ["YUV420P16", "YUV 4:2:0 16-bit"],
           ["YUV444P8", "YUV 4:4:4 8-bit"], ["YUV444P16", "YUV 4:4:4 16-bit"], ["YUV444PS", "YUV 4:4:4 float"],
           ["GRAY8", "Gray 8-bit"], ["GRAY16", "Gray 16-bit"], ["RGB24", "RGB 8-bit"], ["RGB48", "RGB 16-bit"], ["RGBS", "RGB float"]]


def _even(v):
    return max(0, int(v or 0)) // 2 * 2


# ---------------------------------------------------------------- the app's blocks
def run_source(clip, job, p, ctx):
    v = _v()
    j = dict(job, ai_model="off", crawl=p["crawl"], rainbow=p["rainbow"], composite=p["composite"])
    return v.source(j, clean=not ctx.get("before"))


def run_ivtc(clip, job, p, ctx):
    v = _v()
    mode = "vfr" if p["mode"] == "vfr" and job.get("mode") == "vfr" else "cfr"     # preview / CFR export: CFR
    j = dict(job, ai_model="off", mode=mode, order=p["order"], comb=p["comb"], ivtc_repair=p["ivtc_repair"],
             chroma_unmix=p["chroma_unmix"], deblend=p["blend"], aa="repair" if p["aa_repair"] else "off",
             raw_ivtc=bool(p["raw"]), cthresh=int(p["cthresh"]), mi=int(p["mi"]), pp=int(p["pp"]),
             dupthresh=float(p["dupthresh"]), vidthresh=float(p["vidthresh"]))
    return v.ivtc_vfr(clip, j) if mode == "vfr" else v.ivtc_cfr(clip, j)


def run_pal(clip, job, p, ctx):
    return _v().pal_process(clip, dict(job, ai_model="off", pal_mode=p["pal_mode"], order=p["order"]))


def run_deblend_ai(clip, job, p, ctx):
    return _v().deblend_ai(clip, dict(job, deblend="ai"))


def run_aa(clip, job, p, ctx):
    return _v()._aa(clip, job, float(p["strength"]))


def run_cleanup(clip, job, p, ctx):
    j = dict(job, ai_model=p["model"], crawl=p["crawl"], rainbow=p["rainbow"], rainbow_fill="on" if p["rainbow_fill"] else "off",
             **{k: float(p[k]) for k in ("uni_crawl", "uni_rainbow", "uni_mpeg", "uni_halo", "uni_chroma", "uni_grain")})
    return _v().xtalk_stage_b(clip, j)


def run_xtalk_classic(clip, job, p, ctx):
    return _v().xtalk_stage_b(clip, dict(job, ai_model="off", crawl=p["crawl"], rainbow=p["rainbow"]))


def run_deblock(clip, job, p, ctx):
    return _v().deblock(clip, dict(job, deblock=p["level"]))


def run_halos(clip, job, p, ctx):
    # ghost_halo works at the measured best radius 2 (halo_rx only switches it on)
    return _v().ghost_halo(clip, dict(job, ghost="auto", halo_rx=2.0, ghost_strength=float(p["strength"]), halo_method=p["method"]))


def run_chroma(clip, job, p, ctx):
    method, strength = (p["bleed"] or "off|normal").split("|")
    return _v().chroma_fix(clip, float(p["shift_x"]), float(p["shift_y"]), int(p["warp"]), method, strength)


def run_neutral(clip, job, p, ctx):
    return _v().neutralize(clip)


def run_stab(clip, job, p, ctx):
    return _v().stabilize(clip, dict(job, stab=p["method"], stab_level=p["level"], tfix_strength=float(p["tfix_strength"])))


def run_still_lock(clip, job, p, ctx):
    import tstab
    return tstab.still_lock(clip, keep=float(p["keep"]))


def run_crop(clip, job, p, ctx):
    l, r, t, b = (_even(p[k]) for k in ("left", "right", "top", "bottom"))
    if not (l or r or t or b):
        return clip
    if l + r >= clip.width - 16 or t + b >= clip.height - 16:
        raise ValueError(f"Crop {l}/{r}/{t}/{b} leaves no picture of {clip.width}x{clip.height}")
    return core.std.Crop(clip, l, r, t, b)


def resize_target(clip, p, ctx):
    """output size of a Resize block for this clip and pixel shape"""
    sar = ctx.get("sar", 1.0)
    if p["mode"] == "square":
        h = _even(p["height"])
        return int(round(h * clip.width * sar / clip.height / 2)) * 2, h
    if p["mode"] == "scale":
        f = float(p["factor"])
        return _even(round(clip.width * f)), _even(round(clip.height * f))
    return _even(p["width"]), _even(p["height"])


def run_resize(clip, job, p, ctx):
    w, h = resize_target(clip, p, ctx)
    if w < 16 or h < 16:
        raise ValueError(f"Resize to {w}x{h}: too small")
    if (w, h) == (clip.width, clip.height):
        return clip
    return _v().resize_size(clip, dict(job, out_w=w, out_h=h, kernel=p["kernel"], kernel_b=float(p["kernel_b"]),
                                       kernel_c=float(p["kernel_c"])))


def run_sharpen(clip, job, p, ctx):
    return _v().sharpen(clip, dict(job, sharp={"luma": float(p["luma"]), "chroma": float(p["chroma"]), "mode": p["mode"],
                                                "radius": p["radius"]}))


def run_lines(clip, job, p, ctx):
    return _v().line_darken(clip, dict(job, sharp={"lines": float(p["lines"])}))


def run_colour(clip, job, p, ctx):
    return _v().grade(clip, job)                     # the Colour step's settings (job["grade"], job["grade_layers"])


def run_format(clip, job, p, ctx):
    fmt = getattr(vs, p["format"])
    v = _v()
    kw = {}
    to = core.get_video_format(fmt)
    rgb_in, rgb_out = clip.format.color_family == vs.RGB, to.color_family == vs.RGB
    mat = 1 if clip.height >= 720 else v.sd_colour(job)[0]          # as the output: BT.709 from 720 lines up
    if rgb_out and not rgb_in and clip.format.color_family == vs.YUV:
        kw["matrix_in"] = mat
    if rgb_in and not rgb_out and to.color_family == vs.YUV:
        kw["matrix"] = mat
    if core.get_video_format(fmt).bits_per_sample <= 8 and clip.format.bits_per_sample > 8:
        kw["dither_type"] = "error_diffusion"
    return core.resize.Bicubic(clip, format=fmt, **kw)


def to_yuv(clip):
    """whatever a filter left (gray, RGB, odd sizes of subsampling) as YUV 4:4:4 16-bit for the output: a gray picture
    becomes the luma with neutral colour, RGB is converted with the matrix the output uses"""
    if clip.format is None:
        raise ValueError("the clip has a changing format - add a Convert format block before Output")
    cf = clip.format.color_family
    if cf == vs.GRAY:
        y = core.resize.Bicubic(clip, format=vs.GRAY16) if clip.format.id != vs.GRAY16 else clip
        uv = core.std.BlankClip(y, color=[32768])
        out = core.std.CopyFrameProps(core.std.ShufflePlanes([y, uv, uv], [0, 0, 0], vs.YUV), clip)
    elif cf == vs.RGB:
        out = core.std.CopyFrameProps(core.resize.Bicubic(clip, format=vs.YUV444P16, matrix=_mat(clip)), clip)
    else:
        return clip
    # the tags of a YUV picture (an RGB / gray clip carries matrix "RGB", which a YUV clip may not have)
    return core.std.SetFrameProps(out, _Matrix=_mat(clip), _Range=vs.RANGE_LIMITED)


def run_output(clip, job, p, ctx):
    return _v().output_format(to_yuv(clip), job)   # 10-bit (or the encoder's), colour tags; size as it is


ADAPT_FORMATS = ("RGB24", "RGB48", "RGBS", "GRAY8", "GRAY16", "GRAYS", "YUV420P16", "YUV444P8", "YUV444P16", "YUV444PS")
# every format the chain can hand a block (the scan tries each and records those that work)
SCAN_FORMATS = ("YUV420P8", "YUV420P10", "YUV420P16", "YUV444P8", "YUV444P16", "YUV444PS",
                "GRAY8", "GRAY16", "GRAYS", "RGB24", "RGB48", "RGBS")


def _mat(clip):
    return 1 if clip.height >= 720 else 6                # BT.709 from 720 lines up, else BT.601 (as the app's output)


def adapt_in(clip, name):
    """the clip in the format a filter needs (EVA toolbox adapter): RGB / gray (the luma) / another depth"""
    to = core.get_video_format(getattr(vs, name))
    if clip.format.color_family == vs.GRAY and to.color_family != vs.GRAY:
        clip = to_yuv(clip)                              # a grey picture (e.g. after get_r): luma + neutral colour, YUV tags
        if clip.format.id == to.id:
            return clip
    if to.color_family == vs.GRAY and clip.format.color_family != vs.GRAY:
        y = core.std.ShufflePlanes(clip, 0, vs.GRAY)
        return core.resize.Bicubic(y, format=to.id) if y.format.id != to.id else y
    kw = {}
    if to.color_family == vs.RGB and clip.format.color_family == vs.YUV:
        kw["matrix_in"] = _mat(clip)
    elif to.color_family == vs.YUV and clip.format.color_family == vs.RGB:
        kw["matrix"] = _mat(clip)
    if to.bits_per_sample < clip.format.bits_per_sample:
        kw["dither_type"] = "error_diffusion"
    return core.resize.Bicubic(clip, format=to.id, **kw)


def adapt_out(out, orig, name):
    """back to the chain's format where that is possible (same size; RGB -> YUV; a luma result becomes the luma)"""
    if (out.width, out.height) != (orig.width, orig.height) or out.format is None:
        return out
    to = core.get_video_format(getattr(vs, name))
    if out.format.color_family == vs.GRAY and to.color_family == vs.GRAY and orig.format.color_family == vs.YUV:
        g = core.get_video_format(orig.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id)
        y = core.resize.Bicubic(out, format=g.id) if out.format.id != g.id else out
        return core.std.CopyFrameProps(core.std.ShufflePlanes([y, orig], [0, 1, 2], vs.YUV), orig)
    kw = {}
    if out.format.color_family == vs.RGB and orig.format.color_family == vs.YUV:
        kw["matrix"] = _mat(orig)
    elif out.format.color_family != orig.format.color_family:
        return out                                       # e.g. a colour map of a gray clip: keep what the filter made
    if orig.format.bits_per_sample < out.format.bits_per_sample:
        kw["dither_type"] = "error_diffusion"
    return core.std.CopyFrameProps(core.resize.Bicubic(out, format=orig.format.id, **kw), orig)


def _prep(clip, p):
    """EVA toolbox preparation before a function (catalog "prep"): scene_detect = mark scene changes
    (_SceneChangePrev / _SceneChangeNext, vs-jetpack's sc_detect) for filters that need them"""
    if p.get("prep") == "scene_detect":
        import vstools
        clip = vstools.sc_detect(clip)
    return clip


def pick_format(clip, formats):
    """the accepted format closest to the clip's: same colour family first (YUV keeps its subsampling if it can), then
    the same or a higher bit depth, integer before float"""
    if not formats or clip.format is None:
        return None
    if clip.format.name in formats:
        return clip.format.name
    f0 = clip.format

    def cost(n):
        f = core.get_video_format(getattr(vs, n))
        c = 0 if f.color_family == f0.color_family else (1 if f0.color_family == vs.YUV and f.color_family == vs.RGB else 2) * 100
        c += 0 if (f.subsampling_w, f.subsampling_h) == (f0.subsampling_w, f0.subsampling_h) else 10
        c += 0 if f.bits_per_sample >= f0.bits_per_sample else 5
        c += 0 if f.sample_type == f0.sample_type else 3
        return c
    return min(formats, key=cost)


def _formats_of(p):
    """the formats a plugin / jetpack block accepts: its own setting (written when it was added), else the scan's"""
    if p.get("formats"):
        return list(p["formats"])
    try:
        comp, _ = compat()
        r = comp.get(("vs:" if "." in p.get("call", "") and p.get("call", "").split(".")[0] in _vs_namespaces() else "jet:") + p.get("call", ""))
        return list((r or {}).get("formats") or [])
    except Exception:                                    # noqa: BLE001
        return []


_VS_NS = set()


def _vs_namespaces():
    if not _VS_NS:
        _VS_NS.update(pl.namespace for pl in core.plugins())
    return _VS_NS


def _with_adapt(clip, p, fn):
    """run fn on the clip in a format it accepts: the picture is converted only when its own format is not one of them,
    and converted back afterwards (EVA toolbox adapter)"""
    clip = _prep(clip, p)
    if p.get("native"):                                  # the compatibility scan: the filter must see the format as it is
        return fn(clip)
    a = p.get("adapt")                                   # older blocks: one fixed format
    if not a:
        a = pick_format(clip, _formats_of(p))
    if not a or a not in SCAN_FORMATS + ADAPT_FORMATS or (clip.format and clip.format.name == a):
        return fn(clip)
    return adapt_out(fn(adapt_in(clip, a)), clip, a)


def run_vs(clip, job, p, ctx):
    ns, fn = p["call"].split(".", 1)
    f = getattr(getattr(core, ns), fn)
    args = _vs_args(p["call"], p.get("args") or {})
    return _with_adapt(clip, p, lambda c: f(c, **args))


def run_jet(clip, job, p, ctx):
    _fix_dfttest2()
    mod, fn = p["call"].rsplit(".", 1)
    M = importlib.import_module(mod)
    f = getattr(M, fn)
    args = {k: _eval_expr(e, mod) for k, e in (p.get("args") or {}).items() if str(e).strip() != ""}

    def call(c):
        out = f(c, **args)
        if not isinstance(out, vs.VideoNode):
            raise TypeError(f"{p['call']} returned {type(out).__name__}, not a clip")
        return out
    return _with_adapt(clip, p, call)


def run_code(clip, job, p, ctx):
    ns = {"clip": clip, "core": core, "vs": vs, "job": job, "dr": _v()}
    for m in ("vstools", "vsdenoise", "vsaa", "vsdehalo", "vsdeinterlace", "vsscale", "vsmasktools", "vsrgtools", "vsdeband",
              "vskernels", "vsexprtools"):
        try:
            ns[m] = importlib.import_module(m)
        except Exception:                                # noqa: BLE001
            pass
    _fix_dfttest2()
    exec(compile(p["code"] or "", "custom block", "exec"), ns)
    out = ns.get("clip")
    if not isinstance(out, vs.VideoNode):
        raise TypeError("the custom block must leave a clip in `clip`")
    return out


# ---- ready-made vs-jetpack tools (JET: github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack); their plain functions are
# in the library automatically (jet_catalogue), these are the class-based ones with proper settings
def run_jet_qtgmc(clip, job, p, ctx):
    _fix_dfttest2()
    from vsdeinterlace import QTempGaussMC as Q
    kw = dict(prefilter_tr=int(p["tr0"]), basic_tr=int(p["tr1"]), final_tr=int(p["tr2"]),
              source_match_mode=Q.SourceMatchMode[p["source_match"]], lossless_mode=Q.LosslessMode[p["lossless"]])
    if float(p["sharpen"]) >= 0:
        kw["sharpen_strength"] = float(p["sharpen"])
    if p["rate"] == "single":
        kw["motion_blur_fps_divisor"] = 2
    q = Q(**kw)
    tff = None if p["tff"] == "auto" else p["tff"] == "tff"
    return q.bob(clip, tff=tff) if p["rate"] == "double" else q.deinterlace(clip, tff=tff)


def _fix_dfttest2():
    """the EVA toolbox's runtime fixes for bugs in installed filter packages (eva_toolbox/shims.py: dfttest2's TypeIs,
    vinverse / deblend's amnt, fine_dehalo2's coord ...) - applied once per process, before toolbox blocks run"""
    try:
        from eva_toolbox import shims
        shims.apply()
    except Exception:                                    # noqa: BLE001
        pass


def run_jet_dfttest(clip, job, p, ctx):
    _fix_dfttest2()
    from vsdenoise import DFTTest
    planes = {"all": None, "luma": [0], "chroma": [1, 2]}[p["planes"]]
    return DFTTest().denoise(clip, float(p["sigma"]), tr=int(p["tr"]), planes=planes)


def run_jet_aa(clip, job, p, ctx):
    import vsaa
    A = {"nnedi3": vsaa.NNEDI3, "eedi3": vsaa.EEDI3, "sangnom": vsaa.SangNom}[p["method"]]
    return A().antialias(clip)


# geometry: how a size change affects the pixel shape (SAR). "crop": same pixel shape (Crop, AddBorders);
# "scale" (default): same picture shape (resizers, upscalers keep the picture's proportions)
BLOCKS = {}


def _def(id_, name, cat, run, params=(), desc="", auto=None, before=False, geometry="scale", ntsc=None, pinned=None):
    BLOCKS[id_] = {"id": id_, "name": name, "cat": cat, "run": run, "params": list(params), "desc": desc, "auto": auto,
                   "before": before, "geometry": geometry, "ntsc": ntsc, "pinned": pinned}


_def("source", "Source", "Source", run_source, [
    P("crawl", "Dot crawl (before IVTC)", "choice", "off", options=LEVELS,
      tip="Classic dot crawl cancellation on the telecined frames. Leave off before an AI cleanup (UniNet / StableNet were trained on untouched frames)."),
    P("rainbow", "Rainbows (before IVTC)", "choice", "off", options=LEVELS),
    P("composite", "Composite decode", "choice", "off", opts="composite"),
], desc="Loads the episode. Optional classic dot crawl / rainbow cancellation before field matching.", auto="composite",
   before=True, pinned="first")
_def("ivtc", "Inverse telecine (NTSC)", "Telecine", run_ivtc, [
    P("mode", "Frame rate", "choice", "vfr", options=[["vfr", "VFR (24p + 30p)"], ["cfr", "CFR 23.976"]]),
    P("order", "Field order", "choice", "tff", options=[["tff", "Top first"], ["bff", "Bottom first"]]),
    P("comb", "Combing repair", "choice", "nnedi3", opts="comb"),
    P("ivtc_repair", "Unmatched frames", "choice", "ai", opts="ivtc_repair"),
    P("chroma_unmix", "ICP fix", "choice", "off", options=[["off", "Off"], ["on", "Repair"]]),
    P("blend", "Blended frames (during IVTC)", "choice", "off", options=[["off", "Off"], ["dup", "Replace by the nearer clean frame"], ["interp", "Replace by a motion-interpolated frame"]]),
    P("aa_repair", "AI anti-aliasing on repaired areas", "bool", False),
    P("raw", "Raw IVTC (field matching + decimation only)", "bool", False, tip="For FieldNet-type models that repair combing themselves."),
    P("cthresh", "Comb threshold (cthresh)", "int", 9, min=1, max=255, step=1),
    P("mi", "Combed pixels per block (MI)", "int", 80, min=0, max=255, step=1),
    P("pp", "Combed-frame repair (PP)", "int", 6, min=0, max=7, step=1),
    P("dupthresh", "Duplicate threshold (%)", "float", 1.1, min=0, max=10, step=0.1),
    P("vidthresh", "Video threshold (%)", "float", 1.1, min=0, max=10, step=0.1),
], desc="Field matching and decimation to film frames (TIVTC), with repair of what cannot be matched.", auto="frames",
   before=True, ntsc=True)
_def("pal", "PAL processing", "Telecine", run_pal, [
    P("pal_mode", "PAL type", "choice", "auto", opts="pal_modes"),
    P("order", "Field order", "choice", "tff", options=[["tff", "Top first"], ["bff", "Bottom first"]]),
], desc="Progressive / field-shifted / interlaced PAL.", auto="frames", before=True, ntsc=False)
_def("deblend_ai", "AI blend removal (DeblendNet)", "Telecine", run_deblend_ai, [],
     desc="Frames that are a mix of their neighbours are rebuilt as the picture of their own moment.")
_def("aa", "AI anti-aliasing (AANet)", "Telecine", run_aa, [P("strength", "Strength", "float", 1.0, min=0, max=1, step=0.05)],
     desc="Removes the stair-steps bad deinterlacing leaves on outlines (luma, whole frame).")
_def("cleanup", "AI cleanup", "Cleanup", run_cleanup, [
    P("model", "Model", "choice", "uni", options=[["uni", "All-in-one (UniNet)"], ["stable", "StableNet"],
                                                ["temporal", "Precise temporal (5 frames)"], ["precise", "Precise single-frame"]]),
    P("uni_crawl", "Dot crawl", "float", 1.0, min=0, max=1, step=0.05, show="model=uni"),
    P("uni_rainbow", "Rainbows", "float", 1.0, min=0, max=1, step=0.05, show="model=uni"),
    P("uni_mpeg", "MPEG", "float", 1.0, min=0, max=1, step=0.05, show="model=uni"),
    P("uni_halo", "Halos", "float", 1.0, min=0, max=1, step=0.05, show="model=uni"),
    P("uni_chroma", "Colour bleed", "float", 1.0, min=0, max=1, step=0.05, show="model=uni"),
    P("uni_grain", "Grain keep", "float", 0.0, min=0, max=1, step=0.05, show="model=uni"),
    P("crawl", "Dot crawl", "choice", "normal", options=LEVELS, show="model!uni"),
    P("rainbow", "Rainbows", "choice", "normal", options=LEVELS, show="model!uni"),
    P("rainbow_fill", "Stain fill (blotches after rainbow removal)", "bool", False),
], desc="Neural dots + rainbows cleanup; UniNet also does MPEG, halos and colour bleed (dials: 1 = remove fully).", auto="cleanup")
_def("xtalk", "Dot crawl / rainbow filter (classic)", "Cleanup", run_xtalk_classic, [
    P("crawl", "Dot crawl", "choice", "normal", options=LEVELS), P("rainbow", "Rainbows", "choice", "normal", options=LEVELS)],
    desc="Motion-compensated classic filter after IVTC (no neural model).", auto="composite")
_def("deblock", "MPEG repair (BlockNet)", "Cleanup", run_deblock, [
    P("level", "Strength", "choice", "normal", options=[["light", "Light (60 %)"], ["normal", "Normal"]])],
    desc="Blocking, mosquito noise around lines, blocky colour.", auto="block")
_def("halos", "Halo removal", "Cleanup", run_halos, [
    P("method", "Method", "choice", "ai", options=[["ai", "AI (HaloNet)"], ["classic", "Classic (DeHalo)"]]),
    P("strength", "Strength", "float", 1.0, min=0.25, max=2, step=0.25, tip="AI: 0.25-1 blend, above 1 part of a second pass; classic: passes")],
    desc="Halos, ringing and short echoes next to edges.", auto="ghost")
_def("chroma", "Colour alignment", "Cleanup", run_chroma, [
    P("shift_x", "Shift right (px)", "float", 0.0, min=-4, max=4, step=0.25),
    P("shift_y", "Shift down (px)", "float", 0.0, min=-4, max=4, step=0.25),
    P("bleed", "Bleed fix", "choice", "ai|normal", opts="bleed"),
    P("warp", "AWarp depth", "int", 7, min=1, max=16, step=1, show="bleed=awarp|normal")],
    desc="Puts the colour back on the outlines and removes colour bleed. Also brings the DVD's half-size colour to full size (4:4:4 16-bit) with the right siting - keep it in, even with nothing to fix.", auto="chroma")
_def("neutral", "Clean white / black", "Cleanup", run_neutral, [],
     desc="Removes the colour tint DVD colour leaves on thin white and black detail (text, eyes, outlines).")
_def("stab", "Temporal stabilizer", "Temporal", run_stab, [
    P("method", "Method", "choice", "chain", opts="stab_methods"),
    P("level", "Strength", "choice", "normal", opts="stab_levels"),
    P("tfix_strength", "TemporalFix strength", "float", 2.0, min=0, max=3, step=0.05, show="method=tfix_ai")],
    desc="Removes shimmer, flicker and wobble left after cleaning. Steady chain = MC + fine detail + still smoothing + still lock.",
    auto="stab")
_def("still_lock", "Still-area lock", "Temporal", run_still_lock, [
    P("keep", "Memory", "float", 0.97, min=0.9, max=0.995, step=0.005)],
    desc="Holds still detail completely steady (long memory, reset by any real change).")
_def("crop", "Crop", "Crop & size", run_crop, [
    P("left", "Left", "int", 0, min=0, max=400, step=2, unit="px"), P("right", "Right", "int", 0, min=0, max=400, step=2, unit="px"),
    P("top", "Top", "int", 0, min=0, max=400, step=2, unit="px"), P("bottom", "Bottom", "int", 0, min=0, max=400, step=2, unit="px")],
    desc="Cuts borders (even numbers).", auto="crop", before=True, geometry="crop")
_def("resize", "Resize", "Crop & size", run_resize, [
    P("mode", "Size", "choice", "square", options=[["square", "Square pixels at a height"], ["custom", "Width x height"], ["scale", "Scale factor"]]),
    P("height", "Height", "int", 720, min=64, max=4320, step=2, unit="px", show="mode!scale"),
    P("width", "Width", "int", 960, min=64, max=7680, step=2, unit="px", show="mode=custom"),
    P("factor", "Factor", "float", 2.0, min=0.25, max=8, step=0.25, show="mode=scale"),
    P("kernel", "Resizer", "choice", "spline36", opts="kernels"),
    P("kernel_b", "Bicubic b", "float", 0.33, min=-1, max=2, step=0.01, show="kernel=bicubic"),
    P("kernel_c", "Bicubic c", "float", 0.33, min=-1, max=2, step=0.01, show="kernel=bicubic")],
    desc="Output size. Square pixels: the width follows the picture's real shape (after crops and upscales).", before=True)
_def("sharpen", "Sharpen", "Crop & size", run_sharpen, [
    P("luma", "Luma", "float", 0.5, min=0, max=2, step=0.1), P("chroma", "Colour", "float", 0.0, min=0, max=2, step=0.1),
    P("mode", "Method", "choice", "faithful", options=[["faithful", "Faithful (no halos)"], ["unsharp", "Unsharp mask"]]),
    P("radius", "Radius", "choice", "fine", options=[["fine", "Fine detail"], ["wide", "Outlines"]])],
    desc="Sharpening; faithful never goes brighter or darker than the pixels around (no halos).")
_def("lines", "Line darkening", "Crop & size", run_lines, [P("lines", "Strength", "float", 0.3, min=0, max=1, step=0.05)],
     desc="Dark outlines thinner than ~3 px become darker.")
_def("colour", "Colour correction", "Colour", run_colour, [],
     desc="The Colour step's correction and adjustment layers, at this point of the chain.")
_def("format", "Convert format", "Utility", run_format, [P("format", "Format", "choice", "YUV444P16", options=FORMATS)],
     desc="Bit depth / colour family for filters that need one.")
_def("code", "Custom code", "Utility", run_code, [P("code", "Python", "code", "clip = clip")],
     desc="Any VapourSynth / vs-jetpack code working on `clip` (core, vs, vstools, vsdenoise, vsaa, ... are imported).")
_def("jet_qtgmc", "QTGMC deinterlace", "JET tools", run_jet_qtgmc, [
    P("rate", "Output rate", "choice", "double", options=[["double", "Double rate (59.94 / 50 fps, every field)"], ["single", "Single rate (29.97 / 25 fps)"]]),
    P("tff", "Field order", "choice", "auto", options=[["auto", "From the clip"], ["tff", "Top first"], ["bff", "Bottom first"]]),
    P("tr0", "Prefilter radius", "int", 2, min=0, max=3, step=1),
    P("tr1", "Basic radius", "int", 2, min=0, max=3, step=1),
    P("tr2", "Final radius", "int", 1, min=0, max=3, step=1),
    P("source_match", "Source match", "choice", "NONE", options=[["NONE", "Off"], ["BASIC", "Basic"], ["REFINED", "Refined"], ["TWICE_REFINED", "Twice refined (most faithful)"]]),
    P("lossless", "Lossless", "choice", "NONE", options=[["NONE", "Off"], ["PRESHARPEN", "Before sharpening"], ["POSTSMOOTH", "After smoothing"]]),
    P("sharpen", "Sharpening (-1 = default)", "float", -1.0, min=-1, max=2, step=0.1)],
    desc="vs-jetpack's QTempGaussMC: motion-compensated deinterlacing for interlaced video (60i / 50i). Use it instead of Inverse telecine for video-rate material.",
    before=True)
_def("jet_dfttest", "DFTTest denoise", "JET tools", run_jet_dfttest, [
    P("sigma", "Strength (sigma)", "float", 4.0, min=0.5, max=64, step=0.5),
    P("tr", "Temporal radius", "int", 1, min=0, max=3, step=1),
    P("planes", "Planes", "choice", "all", options=[["all", "All"], ["luma", "Luma only"], ["chroma", "Colour only"]])],
    desc="vs-jetpack's DFTTest: frequency-domain denoiser (grain, noise), spatial + temporal.")
_def("jet_aa", "Anti-aliasing (NNEDI3 / EEDI3 / SangNom)", "JET tools", run_jet_aa, [
    P("method", "Method", "choice", "nnedi3", options=[["nnedi3", "NNEDI3"], ["eedi3", "EEDI3 (sharper lines, slower)"], ["sangnom", "SangNom (strong)"]])],
    desc="vs-jetpack's anti-aliasers: smooths jagged outlines by re-interpolating the picture.")
_def("output", "Output", "Output", run_output, [],
     desc="Encoder format and colour tags (always last).", before=True, pinned="last")
_def("vs", "VapourSynth filter", "VapourSynth", run_vs, [], desc="A plugin function.")
_def("jet", "vs-jetpack function", "vs-jetpack", run_jet, [], desc="A vs-jetpack function.")


# ---------------------------------------------------------------- EVA toolbox: your own blocks (eva_toolbox/recipes)
RECIPE_ERRORS = {}


def load_recipes():
    """register eva_toolbox/recipes/*.py as blocks "recipe:<file>" (again after a recipe was added / changed)"""
    try:
        import eva_toolbox
        found = eva_toolbox.recipes()
    except Exception as e:                               # noqa: BLE001
        found = {"(eva_toolbox)": f"{type(e).__name__}: {e}"}
    for k in [k for k in BLOCKS if k.startswith("recipe:")]:
        del BLOCKS[k]
    RECIPE_ERRORS.clear()
    for stem, mod in found.items():
        if isinstance(mod, str):
            RECIPE_ERRORS[stem] = mod
            continue
        B = mod.BLOCK

        def run(clip, job, p, ctx, _m=mod):
            _fix_dfttest2()
            out = _m.run(clip, p, job)
            if not isinstance(out, vs.VideoNode):
                raise TypeError(f"recipe {_m.__name__} returned no clip")
            return out
        _def("recipe:" + stem, str(B.get("name") or stem), str(B.get("cat") or "My blocks"), run,
             [dict(q) for q in B.get("params", [])], desc=str(B.get("desc") or ""), geometry=B.get("geometry", "scale"))


# ---------------------------------------------------------------- VapourSynth plugins / vs-jetpack catalogue
_VS_SIGS = {}


def _parse_sig(sig):
    out = []
    for a in [s for s in sig.split(";") if s]:
        parts = a.split(":")
        out.append({"name": parts[0], "type": parts[1] if len(parts) > 1 else "any", "opt": "opt" in parts[2:]})
    return out


def vs_catalogue():
    """[{call, plugin, args: [{name, type, opt}]}] of every plugin function whose first argument is the video clip"""
    res = []
    for pl in core.plugins():
        for f in pl.functions():
            args = _parse_sig(f.signature)
            if not args or args[0]["type"] != "vnode":
                continue
            call = f"{pl.namespace}.{f.name}"
            _VS_SIGS[call] = args
            res.append({"call": call, "plugin": pl.name, "args": args[1:]})
    res.sort(key=lambda r: r["call"].lower())
    return res


def _vs_args(call, args):
    """settings (strings / numbers from the form) -> typed arguments for the plugin function"""
    if call not in _VS_SIGS:
        vs_catalogue()
    sig = {a["name"]: a for a in _VS_SIGS.get(call, [])}
    out = {}
    for k, v in args.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue
        t = sig.get(k, {}).get("type", "data")
        arr = t.endswith("[]")
        base = t[:-2] if arr else t
        vals = [x.strip() for x in str(v).split(",")] if arr and isinstance(v, str) else (v if isinstance(v, list) else [v])
        conv = []
        for x in vals:
            if base == "int":
                conv.append(int(float(x)))
            elif base == "float":
                conv.append(float(x))
            elif base == "vnode":
                raise ValueError(f"{call}: argument {k} needs a second clip - use a Custom code block")
            else:
                conv.append(str(x))
        out[k] = conv if arr else conv[0]
    return out


JET_MODULES = ("vstools", "vsdenoise", "vsaa", "vsdehalo", "vsdeinterlace", "vsscale", "vsmasktools", "vsrgtools", "vsdeband",
               "vskernels", "vsexprtools")


def jet_catalogue():
    """[{call, doc, args: [{name, default (repr), type}]}] of the vs-jetpack functions that take a clip first and return one"""
    res = []
    for m in JET_MODULES:
        try:
            M = importlib.import_module(m)
        except Exception:                                # noqa: BLE001
            continue
        for name in sorted(getattr(M, "__all__", dir(M))):
            if name.startswith("_"):
                continue
            f = getattr(M, name, None)
            if not inspect.isfunction(f):
                continue
            try:
                sig = inspect.signature(f)
            except Exception:                            # noqa: BLE001
                continue
            ps = list(sig.parameters.values())
            if not ps or "VideoNode" not in str(sig.return_annotation):
                continue
            p0 = ps[0]
            if not ("VideoNode" in str(p0.annotation) or p0.name in ("clip", "src")):
                continue
            args = []
            for q in ps[1:]:
                if q.kind in (q.VAR_POSITIONAL, q.VAR_KEYWORD):
                    continue
                args.append({"name": q.name, "type": str(q.annotation) if q.annotation is not q.empty else "",
                             "default": "" if q.default is q.empty else repr(q.default), "required": q.default is q.empty})
            doc = (inspect.getdoc(f) or "").strip().split("\n\n")[0][:300]
            res.append({"call": f"{m}.{name}", "doc": doc, "args": args})
    return res


def _eval_expr(expr, mod):
    """a jet argument as typed in the form: a Python expression (numbers, strings in quotes, enums such as
    vsdenoise.Prefilter.DFTTEST), evaluated with vs, core and the vs-jetpack modules available"""
    ns = {"vs": vs, "core": core}
    for m in JET_MODULES:
        try:
            ns[m] = importlib.import_module(m)
        except Exception:                                # noqa: BLE001
            pass
    try:
        M = importlib.import_module(mod)
        ns.update({k: getattr(M, k) for k in getattr(M, "__all__", []) if hasattr(M, k)})
    except Exception:                                    # noqa: BLE001
        pass
    return eval(str(expr), ns)                           # noqa: S307  (the user's own settings, like the Script step)


# ---------------------------------------------------------------- running a chain
def params(type_, p):
    """the block's settings with the defaults filled in"""
    T = BLOCKS[type_]
    out = {q["key"]: q["default"] for q in T["params"]}
    out.update({k: v for k, v in (p or {}).items()})
    return out


def sar0(job):
    """pixel shape of the source: display aspect ratio of the whole frame / its storage shape"""
    w, h = job.get("src_w") or 720, job.get("src_h") or (576 if _v().is_pal(job) else 480)
    return float(job.get("dar", 4 / 3)) * h / w


def chain_context(job, before=False):
    return {"sar": sar0(job), "before": before}


def block(clip, job, type_, p, ctx=None):
    """run one block (the engine and every line of the Chain script call this)"""
    ctx = ctx if ctx is not None else chain_context(job)
    T = BLOCKS[type_]
    q = dict(p or {}) if type_ in ("vs", "jet") else params(type_, p)
    if (clip is not None and clip.format is not None and clip.format.color_family != vs.YUV
            and type_ not in ("vs", "jet", "code", "format", "output") and not type_.startswith("recipe:")):
        clip = to_yuv(clip)                              # the app's own blocks work on YUV: a grey / RGB picture becomes YUV first
    w0, h0 = (clip.width, clip.height) if clip is not None else (None, None)
    out = T["run"](clip, job, q, ctx)
    if not isinstance(out, vs.VideoNode):
        raise TypeError(f"block {type_} returned no clip")
    if w0 and (out.width, out.height) != (w0, h0):
        geo = T["geometry"]
        if type_ == "vs" and q.get("call", "").split(".")[-1] in ("Crop", "CropAbs", "CropRel", "AddBorders"):
            geo = "crop"
        if type_ == "resize" and q.get("mode") == "square":
            ctx["sar"] = 1.0
        elif geo != "crop":                          # same picture shape: SAR follows the new storage shape
            ctx["sar"] = ctx["sar"] * (w0 / h0) / (out.width / out.height)
    return out


def enabled(job):
    return [b for b in (job.get("chain") or []) if b.get("on", True) and b.get("type") in BLOCKS]


def build_chain(job, before=False):
    """the chain as a clip. before = the untouched DVD (only loading, field matching, crop and size, so Split lines up)"""
    v = _v()
    v.nnbackend.set_backend(v.opt(job, "backend"))
    ctx = chain_context(job, before)
    clip = None
    for b in enabled(job):
        T = BLOCKS[b["type"]]
        if before and not T["before"]:
            continue
        if b["type"] == "output":
            continue
        if clip is None and b["type"] != "source":
            clip = block(None, job, "source", {}, ctx)
        clip = block(clip, job, b["type"], b.get("p"), ctx)
    if clip is None:
        clip = block(None, job, "source", {}, ctx)
    clip = block(clip, job, "output", {}, ctx)
    return chain_finish(clip, ctx)


def block_label(b):
    """a block's name as the user sees it in the chain"""
    t, p = b.get("type"), b.get("p") or {}
    if t in ("vs", "jet"):
        return p.get("call") or BLOCKS[t]["name"]
    return BLOCKS[t]["name"] if t in BLOCKS else str(t)


def chain_steps(job):
    """the picture after every enabled block (for vsview: one tab per block). Each is the same clip the chain itself
    builds up to that point, only finished like the Output block so it can be shown - the graph is shared with the
    final chain, nothing is built twice"""
    v = _v()
    v.nnbackend.set_backend(v.opt(job, "backend"))
    ctx = chain_context(job)
    clip, steps = None, []
    for b in enabled(job):
        if b["type"] == "output":
            continue
        if clip is None and b["type"] != "source":
            clip = block(None, job, "source", {}, ctx)
        clip = block(clip, job, b["type"], b.get("p"), ctx)
        c = dict(ctx)
        steps.append((block_label(b), chain_finish(block(clip, job, "output", {}, c), c)))
    if clip is None:
        clip = block(None, job, "source", {}, ctx)
    final = chain_finish(block(clip, job, "output", {}, ctx), ctx)
    return steps, final


def chain_globals(chain):
    """the job keys the rest of the app reads outside the chain (VFR analysis, field order, auto detectors): taken from
    the chain's Source and Inverse telecine / PAL blocks. The VFR analysis must see exactly the clip the chain's IVTC
    block sees (else TFM refuses its own files: crc32)."""
    g = {"ai_model": "off", "stab": "off", "crawl": "off", "rainbow": "off", "composite": "off", "mode": "cfr", "raw_ivtc": False}
    for b in chain or []:
        if not b.get("on", True) or b.get("type") not in BLOCKS:
            continue
        t, p = b["type"], params(b["type"], b.get("p"))
        if t == "source":
            g.update(crawl=p["crawl"], rainbow=p["rainbow"], composite=p["composite"])
        elif t == "ivtc":
            g.update(mode=p["mode"], order=p["order"], comb=p["comb"], ivtc_repair=p["ivtc_repair"], chroma_unmix=p["chroma_unmix"],
                     deblend=p["blend"], aa="repair" if p["aa_repair"] else "off", raw_ivtc=bool(p["raw"]),
                     cthresh=int(p["cthresh"]), mi=int(p["mi"]), pp=int(p["pp"]), dupthresh=float(p["dupthresh"]),
                     vidthresh=float(p["vidthresh"]))
        elif t == "pal":
            g.update(pal_mode=p["pal_mode"], order=p["order"])
    return g


# ---------------------------------------------------------------- the old settings as a chain
def chain_from_job(job):
    """The fixed pipeline of the step panels (vsgraph.build) as an equivalent chain: what the user's current settings
    do, block by block (interdependencies resolved: an all-in-one model's dials replace the classic steps, an AI
    cleaner switches the classic luma filters off, ...)."""
    v = _v()
    o = lambda k: v.opt(job, k)
    pal = v.is_pal(job)
    ch = []
    B = lambda t, **p: ch.append({"type": t, "on": True, "p": p})
    stab_on = v._stablenet_on(job)
    ai_cl = v._ai_cleaner(job)
    cr, rb, comp = o("crawl"), o("rainbow"), o("composite")
    src_crawl = "off" if (stab_on or ai_cl) else cr        # xtalk_stage_a: no luma filter with an AI cleaner
    src_rb = "off" if (stab_on or comp in ("chroma", "full", "clean", "on")) else rb
    src_comp = "off" if stab_on else comp
    B("source", crawl=src_crawl if not pal else "off", rainbow=src_rb if not pal else "off", composite=src_comp)
    adv = {k: job[k] for k in ("cthresh", "mi", "pp", "dupthresh", "vidthresh") if k in job}
    if pal:
        B("pal", pal_mode=o("pal_mode"), order=o("order"))
    else:
        B("ivtc", mode=job.get("mode", "vfr"), order=o("order"), comb=o("comb"), ivtc_repair=v.repair_mode(job),
          chroma_unmix=o("chroma_unmix"), blend=o("deblend") if o("deblend") in ("dup", "interp") else "off",
          aa_repair=o("aa") == "repair", raw=bool(v.raw_ivtc(job)),
          **{k: v.DEFAULTS[k] if k not in adv else adv[k] for k in ("cthresh", "mi", "pp", "dupthresh", "vidthresh")})
    if o("deblend") == "ai":
        B("deblend_ai")
    if o("aa") == "all" and not (o("ai_model") == "field" and v.raw_ivtc(job)):
        B("aa", strength=1.0)
    uni = o("ai_model") in ("uni", "field") and ai_cl and (not pal or o("ai_model") == "field")
    ud = v.uni_dials(job) if uni else {}
    if not (uni and ud["mpeg"] > 0) and o("deblock") in v.DEBLOCK:
        B("deblock", level=o("deblock"))
    if not pal or uni:
        if stab_on or ai_cl:
            model = o("ai_model") if o("ai_model") in ("uni", "stable", "temporal", "precise") else "uni"
            B("cleanup", model=model, crawl=cr, rainbow=rb, rainbow_fill=o("rainbow_fill") != "off",
              **{f"uni_{k}": ud.get(kk, d) for k, kk, d in (("crawl", "crawl", 1.0), ("rainbow", "rainbow", 1.0), ("mpeg", "mpeg", 1.0),
                                                             ("halo", "halo", 1.0), ("chroma", "chroma", 1.0), ("grain", "grain_keep", 0.0))})
        elif cr != "off" or rb != "off":
            B("xtalk", crawl=cr, rainbow=rb)
    if o("ghost") != "off" and (job.get("halo_rx") or job.get("ghost_taps")) and not (uni and ud["halo"] > 0):
        B("halos", method=o("halo_method"), strength=float(o("ghost_strength")))
    chm = dict(job.get("chroma") or {})
    if uni and ud["chroma"] > 0:
        chm = dict(chm, method="off", shift_x=0.0, shift_y=0.0)
    # always: chroma_fix also brings the picture to 4:4:4 16-bit with the DVD's (MPEG-2) chroma siting
    B("chroma", shift_x=float(chm.get("shift_x", 0)), shift_y=float(chm.get("shift_y", 0)),
      bleed=f"{chm.get('method', 'off')}|{chm.get('strength', 'normal')}", warp=int(chm.get("warp", 7)))
    if chm.get("neutral", False):
        B("neutral")
    l, r, t, b = v.crop_values(job)
    if l or r or t or b:
        B("crop", left=l, right=r, top=t, bottom=b)
    if o("stab") != "off" and not (o("ai_model") == "field" and str(job.get("field_stab", "filters")) != "off"):
        B("stab", method=o("stab"), level=o("stab_level"), tfix_strength=float(job.get("tfix_strength", 2.0)))
    for s in v.stage_order(job):
        if s == "colour":
            B("colour")
        else:
            ow, oh = job.get("out_w") or 0, job.get("out_h") or 0
            if ow and oh:
                B("resize", mode="custom", width=ow, height=oh, kernel=o("kernel"),
                  kernel_b=float(job.get("kernel_b", 1 / 3)), kernel_c=float(job.get("kernel_c", 1 / 3)))
            sg = v.sharp_settings(job)
            if float(sg["lines"]) > 0:
                B("lines", lines=float(sg["lines"]))
            if float(sg["luma"]) > 0 or float(sg["chroma"]) > 0:
                B("sharpen", luma=float(sg["luma"]), chroma=float(sg["chroma"]), mode=sg["mode"], radius=sg["radius"])
    B("output")
    return ch


# ---------------------------------------------------------------- the chain as a script
def _lit(x):
    return json.dumps(x, ensure_ascii=False) if not isinstance(x, (dict, list)) else repr(x)


def chain_lines(job):
    """the chain's body as script lines: one dr.block(...) call per enabled block, settings written out"""
    L = ["ctx = dr.chain_context(job)                  # pixel shape carried through the chain (square-pixel resizing)"]
    first = True
    for b in enabled(job):
        t = b["type"]
        if t == "output":
            continue
        T = BLOCKS[t]
        p = dict(b.get("p") or {}) if t in ("vs", "jet") else params(t, b.get("p"))
        if first and t != "source":
            L.append("clip = dr.block(None, job, 'source', {}, ctx)")
        first = False
        what = p.get("call") if t in ("vs", "jet") else T["name"]
        if t == "code":
            L.append(f"# ---- {what}")
        L.append(f"clip = dr.block({'None' if t == 'source' else 'clip'}, job, {t!r}, {p!r}, ctx)   # {what}")
    if first:
        L.append("clip = dr.block(None, job, 'source', {}, ctx)")
    L.append("clip = dr.block(clip, job, 'output', {}, ctx)   # Output: encoder format, colour tags")
    L.append("clip = dr.chain_finish(clip, ctx)")
    return L


def chain_finish(clip, ctx):
    s = Fraction(ctx["sar"]).limit_denominator(1000)
    return core.std.SetFrameProps(clip, _SARNum=s.numerator, _SARDen=s.denominator)


# ---------------------------------------------------------------- the library for the GUI
def library(options):
    """the EVA toolbox for the GUI: block schemas (incl. your recipes), plugin and vs-jetpack functions with the
    catalog's curation and the compatibility scan's result. `options` = the server's option lists (for opts names)"""
    load_recipes()
    try:
        import eva_toolbox
        cat = eva_toolbox.catalog()
    except Exception:                                    # noqa: BLE001
        cat = {"hide": {}, "defaults": {}, "needs": {}, "adapt": {}, "notes": {}}
    out = []
    for T in BLOCKS.values():
        if T["id"] in ("vs", "jet"):
            continue
        ps = []
        for q in T["params"]:
            q = dict(q)
            if "opts" in q:
                name = q.pop("opts")
                if name == "stab_methods":
                    q["options"] = [o for o in options.get("stab", []) if o[0] != "off"]
                else:
                    src = options.get(name, [])
                    q["options"] = [(o if isinstance(o, list) else [o.get("key"), o.get("label")]) for o in src]
            ps.append(q)
        out.append({k: T[k] for k in ("id", "name", "cat", "desc", "auto", "before", "ntsc", "pinned")} | {"params": ps})
    vsl, jet = vs_catalogue(), jet_catalogue()
    comp, made = compat()
    hidden = []
    res = {}
    for kind, lst in (("vs", vsl), ("jet", jet)):
        keep = []
        for f in lst:
            c = f["call"]
            if c in cat["hide"]:
                hidden.append({"call": c, "kind": kind, "why": cat["hide"][c]})
                continue
            r = comp.get(f"{kind}:{c}")
            f["group"], f["name"] = eva_toolbox.label(c)  # what it is for + a plain name (names.json)
            f["kind"] = kind
            f["status"] = r["status"] if r else "untested"
            if r and r.get("error"):
                f["error"] = r["error"]
            if r and r.get("formats"):
                f["formats"] = list(r["formats"])
            if r and r.get("adapt"):
                f["adapt"] = r["adapt"]
            if c in cat["adapt"]:
                f["adapt"] = cat["adapt"][c]
            if c in cat["defaults"]:
                f["defaults"] = dict(cat["defaults"][c])
            if c in cat["needs"]:
                f["needs"] = cat["needs"][c]
                if f["status"] == "error":
                    f["status"] = "args"
            if c in cat["notes"]:
                f["note"] = cat["notes"][c]
            if c in cat.get("prep", {}):
                f["prep"] = cat["prep"][c]
            keep.append(f)
        res[kind] = keep
    return {"blocks": out, "vs": res["vs"], "jet": res["jet"], "groups": eva_toolbox.names()["groups"], "scanned": made, "hidden": hidden,
            "recipe_errors": dict(RECIPE_ERRORS), "catalog_error": cat.get("_error", "")}


def compat():
    """the compatibility scan's results (blocks_scan.py): {"vs:std.BoxBlur": {status, error}}, and when it was made.
    A scan made for other VapourSynth / vs-jetpack / plugin versions is not used."""
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocks_compat.json")
    try:
        d = json.load(open(path, encoding="utf-8"))
        import blocks_scan
        if d.get("versions") != blocks_scan.versions():
            return {}, ""
        return d.get("results", {}), d.get("made", "")
    except Exception:                                    # noqa: BLE001
        return {}, ""


load_recipes()
