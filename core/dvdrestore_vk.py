"""EVA toolkit's repair stages as VapourKit filters (the .vkfilter files in VapourKit's filter-templates call these).

Each function takes VapourKit's clip and returns it repaired, running exactly the stage EVA toolkit runs (blocks.py):
the picture is handed over the way EVA toolkit's own chain hands it (8-bit 4:2:0, DVD colour tags), so the result is the
same as in EVA toolkit; it comes back at 16-bit (VapourKit's working depth), 4:4:4 when the stage makes full-size colour.

    import sys; sys.path.insert(0, r"D:\\projects\\DVDRestore")
    import dvdrestore_vk as dr
    clip = dr.ai_cleanup(clip, model="uni", mpeg=1.0)

Order (as in EVA toolkit): Composite decode (optional, on the telecined frames) -> Inverse telecine -> AI cleanup / classic crawl filter -> MPEG repair -> Halos -> Colour
alignment -> Clean white/black -> Temporal stabilizer -> Still lock. The temporal stages (UniNet, StableNet, stabilizer,
still lock) remember earlier frames: judge them while playing forward, not by jumping around."""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import vapoursynth as vs                                 # noqa: E402

core = vs.core
WORK = os.path.join(tempfile.gettempdir(), "dvdrestore_vk")


def _job(clip, dar=None, backend="pytorch", **extra):
    """the few job settings the stages read (no source file: VapourKit loaded it)"""
    if dar is None:                                      # display shape from the clip: 720x480 / 720x576 DVD = 4:3
        dar = 4 / 3 if clip.width in (704, 720) and clip.height in (480, 486, 576) else clip.width / clip.height
    os.makedirs(WORK, exist_ok=True)
    return dict({"source": "", "dar": float(dar), "src_w": clip.width, "src_h": clip.height, "work": WORK, "mode": "cfr",
                 "backend": backend, "fps_num": clip.fps.numerator or 24000, "fps_den": clip.fps.denominator or 1001}, **extra)


def _to_dvd(clip):
    """VapourKit's 16-bit picture -> what EVA toolkit's stages are fed: YUV 4:2:0 8-bit, BT.601 tags (dithered)"""
    if clip.format is None:
        raise vs.Error("EVA toolkit: the clip changes format - add a format conversion before this filter")
    kw = {}
    if clip.format.color_family == vs.RGB:
        kw["matrix_s"] = "170m"
    elif clip.format.color_family == vs.GRAY:
        import blocks
        clip = blocks.to_yuv(clip)
    if clip.format.id != vs.YUV420P8:
        clip = core.resize.Bicubic(clip, format=vs.YUV420P8, dither_type="error_diffusion", **kw)
    props = {}
    f0 = clip.get_frame(0).props
    if f0.get("_Matrix", 2) in (2, 0):                   # unspecified: a DVD is BT.601
        props.update(_Matrix=6, _Primaries=6, _Transfer=6)
    if "_FieldBased" not in f0:
        props["_FieldBased"] = 0
    return core.std.SetFrameProps(clip, **props) if props else clip


def _back(out, depth=16):
    """the stage's result at VapourKit's working depth (subsampling as the stage made it)"""
    if out.format is None or out.format.color_family != vs.YUV:
        import blocks
        out = blocks.to_yuv(out)
    want = core.query_video_format(vs.YUV, vs.INTEGER, depth, out.format.subsampling_w, out.format.subsampling_h)
    return out if out.format.id == want.id else core.resize.Point(out, format=want.id)


def run(clip, stage, backend="pytorch", dar=None, **p):
    """any EVA toolkit stage by its block id (blocks.BLOCKS), settings as in EVA toolkit"""
    import blocks
    import nnbackend
    nnbackend.set_backend(backend)
    job = _job(clip, dar=dar, backend=backend)
    ctx = blocks.chain_context(job)
    return _back(blocks.block(_to_dvd(clip), job, stage, p, ctx))


# ---------------------------------------------------------------- the stages (EVA toolkit's names and defaults)
def composite_decode(clip, mode="full", order="tff", **kw):
    """Composite artifacts at the source: re-encodes the picture into the composite signal the master went through and
    decodes it again with a motion-adaptive 3D comb (vapoursynth-composite). Put it FIRST, on the telecined frames,
    before Inverse telecine. mode: "chroma" (colour re-decode, can shift colour), "full" (sharper look, slight halos),
    "clean" (full re-decode without sharpening / halos)."""
    import vsgraph
    job = _job(clip, composite=mode, order=order, **kw)
    out = vsgraph.composite_restore(_to_dvd(clip), job)
    return _back(out)


def ivtc(clip, order="tff", comb="nnedi3", repair="ai", blend="off", raw=False, **kw):
    """Inverse telecine to 23.976 (CFR): TIVTC field matching + decimation, repair of what cannot be matched"""
    return run(clip, "ivtc", mode="cfr", order=order, comb=comb, ivtc_repair=repair, blend=blend, raw=raw, **kw)


def ai_cleanup(clip, model="uni", crawl=1.0, rainbow=1.0, mpeg=1.0, halo=1.0, bleed=1.0, grain=0.0, stain_fill=False, **kw):
    """Composite artifacts: dot crawl + rainbows (UniNet also MPEG, halos and colour bleed). model: uni / stable /
    temporal / precise. UniNet dials 0..1 (1 = remove fully); other models take crawl/rainbow as off/light/normal/strong"""
    if model == "uni":
        return run(clip, "cleanup", model="uni", uni_crawl=float(crawl), uni_rainbow=float(rainbow), uni_mpeg=float(mpeg),
                   uni_halo=float(halo), uni_chroma=float(bleed), uni_grain=float(grain), rainbow_fill=bool(stain_fill), **kw)
    lv = lambda x: x if isinstance(x, str) else ("off" if x <= 0 else "light" if x < 0.5 else "normal" if x < 0.9 else "strong")
    return run(clip, "cleanup", model=model, crawl=lv(crawl), rainbow=lv(rainbow), rainbow_fill=bool(stain_fill), **kw)


def crawl_classic(clip, crawl="normal", rainbow="normal", **kw):
    """Classic motion-compensated dot crawl / rainbow filter (no neural network)"""
    return run(clip, "xtalk", crawl=crawl, rainbow=rainbow, **kw)


def mpeg_repair(clip, level="normal", **kw):
    """Compression: MPEG blocking, mosquito noise around lines, blocky colour (BlockNet). level: light / normal"""
    return run(clip, "deblock", level=level, **kw)


def halos(clip, method="ai", strength=1.0, **kw):
    """Halos, ringing and short echoes next to edges. method: ai / classic"""
    return run(clip, "halos", method=method, strength=float(strength), **kw)


def colour_alignment(clip, shift_x=0.0, shift_y=0.0, bleed="ai|normal", warp=7, **kw):
    """Colour back on the outlines + colour bleed removal; returns full-size colour (4:4:4). bleed: 'off|normal',
    'ai|light', 'ai|normal', 'ai|strong', 'awarp|normal' ..."""
    return run(clip, "chroma", shift_x=float(shift_x), shift_y=float(shift_y), bleed=bleed, warp=int(warp), **kw)


def clean_white_black(clip, **kw):
    """Removes the colour tint DVD colour leaves on thin white and black detail (text, eyes, outlines)"""
    return run(clip, "neutral", **kw)


def stabilizer(clip, method="ustab", level="normal", **kw):
    """Shimmer, flicker, boil and wobble left after cleaning. method: ustab (Universal - every kind of shot: fades,
    cuts, flashes, zooms, credits, drawings on twos; nothing dragged into the next frame), chain (Steady chain = MC +
    fine detail + still smoothing + still lock), plus, mc, ttemp, gtfix, tfix_ai; level: light / normal / strong"""
    return run(clip, "stab", method=method, level=level, **kw)


def still_lock(clip, keep=0.97, **kw):
    """Holds still detail completely steady (long memory, reset by any real change)"""
    return run(clip, "still_lock", keep=float(keep), **kw)


def blend_removal(clip, **kw):
    """Frames that are a mix of their neighbours rebuilt as their own moment (DeblendNet)"""
    return run(clip, "deblend_ai", **kw)


def anti_aliasing(clip, strength=1.0, **kw):
    """Stair-steps bad deinterlacing leaves on outlines (AANet)"""
    return run(clip, "aa", strength=float(strength), **kw)
