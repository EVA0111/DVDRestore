"""EVA toolkit backend for the Electron app.

Protocol: JSON lines over stdin / stdout.
    request   {"id": 1, "method": "probe", "params": {...}}
    response  {"id": 1, "result": ...}            or  {"id": 1, "error": "message"}
    event     {"event": "log", "msg": "..."}       {"event": "progress", "cur": n, "tot": m}
              {"event": "busy", "busy": true/false}

Long operations (DVD import, analysis, processing, VFR analysis) run in a worker thread; only one at a time.
Previews run on their own thread and only the newest request is rendered.
"""
import base64
import io
import json
import subprocess
import os
import sys
import threading
import time
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent          # core/ (the engine)
sys.path.insert(0, str(APP_DIR))
import paths  # noqa: E402
import runner  # noqa: E402

SETTINGS = Path(paths.SETTINGS)
_OLD_SETTINGS = Path(paths.ROOT) / "settings_electron.json"          # before 2026-11-23: moved once
if not SETTINGS.exists() and _OLD_SETTINGS.exists():
    try:
        _OLD_SETTINGS.replace(SETTINGS)
    except OSError:
        pass
_out_lock = threading.Lock()


def emit(obj):
    line = json.dumps(obj, ensure_ascii=False)
    with _out_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def fail(rid, what, e):
    """Report a failed request: the window gets the error TYPE, message and where it happened (a bare KeyError
    used to read "Preview: 'info'"); the full traceback goes to stderr, which the app writes to temp/evatoolkit.log."""
    tb = traceback.extract_tb(e.__traceback__)
    where = next((f"{os.path.basename(fr.filename)}:{fr.lineno}" for fr in reversed(tb) if "site-packages" not in fr.filename), "")
    msg = f"{what}{type(e).__name__}: {e}" + (f" ({where})" if where else "")
    print(f"{what or 'Request '}failed: {msg}\n" + "".join(traceback.format_exception(e))[-4000:], file=sys.stderr, flush=True)
    emit({"id": rid, "error": msg, "trace": "".join(traceback.format_exception(e))[-1500:]})


def log(msg, cur=None, tot=None):
    if msg is None:
        emit({"event": "progress", "cur": cur, "tot": tot})
    else:
        emit({"event": "log", "msg": str(msg)})


# ---------------------------------------------------------------- option lists (shown in the UI)

OPTIONS = {
    "sizes": [
        {"key": "keep", "label": "Keep 720x480 / 720x576 (anamorphic)", "mode": "keep", "h": 0},
        {"key": "sd", "label": "Square pixels SD (854x480 / 640x480)", "mode": "sq", "h": 480},
        {"key": "540", "label": "960x540 / 720x540", "mode": "sq", "h": 540},
        {"key": "720", "label": "1280x720 / 960x720", "mode": "sq", "h": 720},
        {"key": "1080", "label": "1920x1080 / 1440x1080", "mode": "sq", "h": 1080},
        {"key": "custom", "label": "Custom width x height", "mode": "custom", "h": 0},
    ],
    "kernels": [
        ["spline36", "Spline36 (default)"], ["spline64", "Spline64 (sharper)"], ["spline16", "Spline16 (softer)"],
        ["lanczos3", "Lanczos3"], ["lanczos4", "Lanczos4"], ["catmull-rom", "Bicubic Catmull-Rom"],
        ["mitchell", "Bicubic Mitchell (soft)"], ["hermite", "Hermite (very soft)"], ["bilinear", "Bilinear"],
        ["point", "Point (nearest)"], ["nnedi3", "NNEDI3 + Spline36 (best upscale)"],
        ["mks2021", "Magic Kernel Sharp 2021"], ["mks2013", "Magic Kernel Sharp 2013"],
        ["bicubic", "Bicubic - your own b / c"], ["bspline", "Bicubic B-spline (b 1, c 0, softest)"],
        ["bicubic_sharp", "Bicubic sharp (b 0, c 1)"], ["robidoux", "Robidoux"], ["robidoux_sharp", "Robidoux sharp"],
        ["robidoux_soft", "Robidoux soft"], ["lanczos2", "Lanczos2"],
        ["ewa_lanczossharp", "EWA Lanczos sharp (GPU)"], ["ewa_lanczos", "EWA Lanczos (GPU)"],
        ["ewa_robidouxsharp", "EWA Robidoux sharp (GPU)"],
    ],
    "aspects": [["auto", "From the disc"], ["4:3", "4:3"], ["16:9", "16:9"], ["custom", "Custom"]],
    "ivtc_repair": [["ai", "AI (CombNet) - only the combed areas (best)"], ["masked", "NNEDI3 + Bwdif - only the combed areas"],
                    ["frame", "Whole frame (old behaviour)"]],
    "deblend": [["off", "Off"], ["ai", "AI - rebuild the picture of that moment (smooth pans)"],
                ["dup", "Replace by the nearer clean frame (can stutter on pans)"], ["interp", "Replace by a motion-interpolated frame"]],
    "aa": [["off", "Off"], ["repair", "AI - rebuilt / deinterlaced areas only"], ["all", "AI - whole picture"]],
    "gamut": [["off", "Off"], ["compress", "Compress (colour stored beyond the screen range)"],
              ["ai", "AI rebuild (colour already clipped flat)"]],
    "backends": [["pytorch", "PyTorch"], ["tensorrt", "TensorRT"]],
    "engine_presets": [["balanced", "Balanced", "fp16, optimisation 3: VapourKit-like speed, ~1 min build"],
                       ["max_speed", "Max speed", "fp16, optimisation 5: longest build, sometimes a few % faster"],
                       ["quick_build", "Quick build", "fp16, optimisation 1: builds in seconds, a little slower"],
                       ["safe_speed", "Safe speed (bf16)", "bf16, optimisation 3: fp32's range, so no overflow; for models that go wrong in fp16"],
                       ["exact", "Exact (fp32)", "fp32 without TF32: identical to the model's training precision, 2-3x slower"]],
    "encoders": [[k, e["label"]] for k, e in __import__("encoding").ENC.items()],
    "enc_table": __import__("encoding").gui_table(),
    "audio_codecs": [[k, v] for k, v in __import__("encoding").AUDIO.items()],
    "levels": [["off", "Off"], ["light", "Light"], ["normal", "Normal"], ["strong", "Strong"]],
    "composite": [["off", "Off (recommended)"], ["chroma", "Chroma re-decode (can shift colour)"], ["full", "Full re-decode (sharper look, slight halos)"], ["clean", "Full re-decode - clean (no sharpening / halos)"]],
    "bleed": [["off|normal", "Off"], ["ai|normal", "AI ChromaNet (best)"], ["ai|strong", "AI ChromaNet - strong (2 passes)"], ["ai|light", "AI ChromaNet - light (60 %)"],
              ["rebuild|normal", "Rebuild filter - normal"], ["rebuild|strong", "Rebuild filter - strong"],
              ["awarp|normal", "AWarp (fast, mild)"]],
    "pal_modes": [["auto", "Auto-detect"], ["progressive", "Progressive (no change)"],
                  ["fieldmatch", "Field-shifted -> field match 25p"], ["deint50", "Interlaced -> 50p (QTGMC)"],
                  ["deint25", "Interlaced -> 25p"]],
    "ai_model": [["precise", "Precise single-frame - dots + rainbows"], ["temporal", "Precise temporal (5 frames) - dots + rainbows"],
                 ["off", "Off - classic filters only"]],
    "comb": [["nnedi3", "NNEDI3 repair (recommended)"], ["safe", "Bwdif repair (softer)"],
             ["strong", "Strong: more sensitive + Vinverse"], ["off", "Off"]],
    "stab": [["off", "Off"], ["ustab", "Universal - steadies every kind of shot (fades, cuts, flashes, zooms, credits) without dragging the picture"], ["plus", "Steady plus - Steady chain + motion-aware still lock (steadiest; best for cartoons)"], ["chain", "Steady chain - MC + fine detail + still smoothing + still lock (best for grainy film)"], ["ttemp", "TTempSmooth (best for crawl / flicker)"], ["gtfix", "Temporal fix (guided, line wobble / shimmer)"], ["tfix_ai", "TemporalFix AI (TensorRT, line wobble)"],
             ["tfix_classic", "TemporalFix classic (CPU)"], ["mc", "MC degrain (motion-compensated)"]],
    "stab_levels": [["light", "Light"], ["normal", "Normal"], ["strong", "Strong"]],
    "vsf_pos": [["source", "On the source (before IVTC)"], ["ivtc", "Instead of the app's IVTC (your own field matching)"],
                ["after_ivtc", "After IVTC (before the clean-up)"], ["repairs", "After the repairs (before crop / stabilizer)"],
                ["end", "At the end (after crop, stabilizer, colour, size)"]],
    "advanced": [
        ["cthresh", "Comb threshold (cthresh)", "9", "Per-pixel difference that counts as combing. Lower finds fainter combing."],
        ["mi", "Combed pixels per block (MI)", "80", "Raise (100-160) if clean frames get softened; lower (40-60) if combing slips through."],
        ["pp", "Combed-frame repair (PP)", "6", "0 never deinterlace, 1 only flag, 6 motion-adaptive (recommended)."],
        ["dupthresh", "Duplicate threshold %", "1.1", "Raise for noisy sources if telecine duplicates are kept."],
        ["vidthresh", "Video threshold % (VFR)", "1.1", "Raise (2-4) if film parts are wrongly kept at 30p (judder)."],
    ],
}
SIZE_BY_KEY = {s["key"]: s for s in OPTIONS["sizes"]}


# ---------------------------------------------------------------- job building (same rules as the Tk GUI)

def parse_ratio(text):
    """'16:9', '16/9', '1.85', '2,35' -> float, None if not a usable ratio."""
    try:
        t = str(text).strip().replace(",", ".").replace("/", ":").replace("x", ":")
        if ":" in t:
            a, b = t.split(":", 1)
            v = float(a) / float(b)
        else:
            v = float(t)
        return v if 0.5 <= v <= 4.0 else None
    except (ValueError, ZeroDivisionError):
        return None


def source_dar(st):
    """Display aspect ratio of the whole (uncropped) source frame: from the disc, or the user's choice."""
    asp = st.get("aspect", "auto")
    v = parse_ratio(st.get("aspect_custom", "")) if asp == "custom" else parse_ratio(asp) if asp != "auto" else None
    return v or st["info"]["dar"]


def make_job(st, k):
    # every episode can have its own picture settings (Repair, Colour, Crop & size)
    per = (st.get("per_episode") or {}).get(str(k)) or (st.get("per_episode") or {}).get(k)
    if per:
        st = {**st, **per}
    """Build an engine job for episode k from the UI state."""
    info = st["info"]
    a, b = st["episodes"][k]
    c = {key: max(0, int(st.get("crop", {}).get(key, 0) or 0)) // 2 * 2 for key in "lrtb"}
    cw, chh = info["width"] - c["l"] - c["r"], info["height"] - c["t"] - c["b"]
    dar = source_dar(st)
    dar_c = dar * (cw / info["width"]) / (chh / info["height"])
    size = SIZE_BY_KEY.get(st.get("size", "keep"), OPTIONS["sizes"][0])
    if size["mode"] == "sq":
        w, h, anam = int(round(size["h"] * dar_c / 2) * 2), size["h"], False
    elif size["mode"] == "custom":
        even = lambda v, lo, hi: int(max(lo, min(hi, round(float(v or 0) / 2) * 2)))
        h = even(st.get("custom_h", 480), 64, 4320)
        w = even(h * dar_c, 64, 7680) if st.get("custom_lock", True) else even(st.get("custom_w", 640), 64, 7680)
        anam = False
    else:
        w, h, anam = 0, 0, True
    src = Path(info["path"])
    out_dir = Path(st.get("out_dir") or src.parent / "restored")
    stem = src.stem
    try:
        name = stem if len(st["episodes"]) == 1 else (st.get("name") or "Episode_{n:02d}").format(n=k + 1, name=stem)
    except (KeyError, IndexError, ValueError):
        name = f"{stem}_{k + 1:02d}"
    name = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in name) or f"Episode_{k + 1:02d}"
    adv = {}
    for key, v in (st.get("adv") or {}).items():
        v = str(v).strip()
        if v:
            try:
                adv[key] = float(v) if key in ("dupthresh", "vidthresh") else int(float(v))
            except ValueError:
                pass
    method, strength = (st.get("bleed") or "ai|normal").split("|")
    job = {
        **adv,
        "source": info["path"], "start": a, "end": b, "mode": st.get("mode", "vfr"),
        "fps_num": info["fps_num"], "fps_den": info["fps_den"], "order": st.get("order", "tff"),
        "crawl": st.get("crawl", "off"), "rainbow": st.get("rainbow", "off"), "composite": st.get("composite", "off"),
        # halo removal switched on by hand without a measurement: DeHalo_alpha's standard radius 2
        "ghost": "auto" if st.get("ghost") == "auto" else "off",
        "ghost_taps": st.get("ghost_taps") or [], "halo_rx": (st.get("halo_rx") or 2.0) if st.get("ghost") == "auto" else 0,
        "ghost_strength": float(st.get("ghost_strength", 1.0)), "halo_method": st.get("halo_method", "ai"), "repairnet": st.get("repairnet", "on"), "vs_filters": [f for f in (st.get("vs_filters") or []) if isinstance(f, dict) and f.get("on", True)], "vs_filters_pos": st.get("vs_filters_pos") or "repairs", "vs_groups": [g for g in (st.get("vs_groups") or []) if isinstance(g, dict)],
        "deblock": st.get("deblock", "off"), "rainbow_fill": "on" if st.get("rainbow_fill") else "off",
        "ivtc_repair": st.get("ivtc_repair", "ai"), "chroma_unmix": st.get("chroma_unmix", "off"), "deblend": st.get("deblend", "off"), "aa": st.get("aa", "off"),
        "grade": dict(st.get("grade") or {}),
        "colour_ref": [str(x) for x in st.get("colour_ref") or []],      # Colour step: reference of the same show
        # adjustment layers that touch this episode (source frames of the whole title), bottom first
        "grade_layers": [{"start": int(l["start"]), "end": int(l["end"]), "opacity": float(l.get("opacity", 1)), "grade": l.get("grade") or {}}
                         for l in st.get("layers") or [] if l.get("on", True) and int(l["end"]) > a and int(l["start"]) < b],
        "pal_mode": st.get("pal_mode", "auto"), "comb": st.get("comb", "nnedi3"), "ai_model": st.get("ai_model", "temporal"),
        "stab": st.get("stab", "off"), "stab_level": st.get("stab_level", "normal"), "stab_pos": st.get("stab_pos") or "default", "repair_order": [str(x) for x in st.get("repair_order") or []], "removed_steps": [str(x) for x in st.get("removed_steps") or []],
        **{k: float(st.get(k, d)) for k, d in (("uni_crawl", 1), ("uni_rainbow", 1), ("uni_mpeg", 1), ("uni_halo", 1),
                                               ("uni_chroma", 1), ("uni_grain", 0), ("uni_detail", 1))},
        "tfix_strength": float(st.get("tfix_strength", 2.0)),
        "chroma": {"shift_x": float(st.get("sx", 0)), "shift_y": float(st.get("sy", 0)), "warp": int(st.get("warp", 7)),
                   "method": method, "strength": strength, "neutral": bool(st.get("neutral", True))},
        "crop": c, "kernel": st.get("kernel", "spline36"),
        "kernel_b": float(st.get("kernel_b", 1 / 3)), "kernel_c": float(st.get("kernel_c", 1 / 3)),
        "out_w": w, "out_h": h, "anamorphic": anam, "dar": dar, "dar_display": dar_c,
        "display": [int(round(chh * dar_c / 2) * 2), chh] if anam else None,
        "encoder": st.get("encoder", "x264"), "crf": float(st.get("crf", 16)),
        "enc_rate": st.get("enc_rate", "q"), "enc_bitrate": int(st.get("enc_bitrate") or 8000),
        "enc_preset": st.get("enc_preset", ""), "enc_tune": st.get("enc_tune", ""),
        "enc_depth": int(st.get("enc_depth") or 10), "enc_chroma": str(st.get("enc_chroma") or "420"),
        "enc_keyint": int(st.get("enc_keyint") or 0), "enc_extra": st.get("enc_extra", ""),
        "audio_codec": st.get("audio_codec", "copy"), "audio_bitrate": int(st.get("audio_bitrate") or 192),
        "backend": st.get("backend", "pytorch"),
        "sharp": dict(st.get("sharp") or {}),
        "stage_order": [s for s in (st.get("stage_order") or ["colour", "size"]) if s != "upscale"],
        "work": str(work_root(st, src) / f".work_{name}"), "output": str(out_dir / (f"{name}.png" if info.get("image") else f"{name}.mkv")),
        "image": bool(info.get("image")),
        "src_w": info.get("width"), "src_h": info.get("height"), "progressive24": bool(info.get("progressive24")) and progressive_choice(st, info),
        "progressive": progressive_choice(st, info),
    } | chain_job(st) | ({"mode": "cfr"} if progressive_choice(st, info) else {})   # progressive: nothing to inverse-telecine
    if st.get("ranges") or st.get("range_only"):
        job["ranges"] = job_ranges(st, k, job)
        job["range_only"] = {s: bool(v) for s, v in (st.get("range_only") or {}).items() if s in RANGE_OK}
    cuts = kept_cuts(st.get("cuts"), a, b)
    if cuts and not info.get("image"):
        job["cuts"] = cuts                               # Timeline: parts cut out (left out of the export; not the preview)
    return job


def kept_cuts(cuts, a, b):
    """the timeline's cut-out parts inside the episode [a, b): sorted, merged [[c0, c1], ...] (source frames)"""
    out = []
    for c in cuts or []:
        try:
            c0, c1 = max(a, int(c[0])), min(b, int(c[1]))
        except (TypeError, ValueError, IndexError):
            continue
        if c1 > c0:
            out.append([c0, c1])
    out.sort()
    merged = []
    for c0, c1 in out:
        if merged and c0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], c1)
        else:
            merged.append([c0, c1])
    return merged


def kept_parts(job):
    """the parts of the episode the export keeps: [(a, b), ...]"""
    a, b = job["start"], job["end"]
    parts, cur = [], a
    for c0, c1 in job.get("cuts") or []:
        if c0 > cur:
            parts.append((cur, c0))
        cur = max(cur, c1)
    if cur < b:
        parts.append((cur, b))
    return parts


RANGE_OK = {"composite", "compression", "halos", "colour", "stab", "sharpen"}


def job_ranges(st, k, job):
    """the episode's frame ranges for the engine: [{step, start, end, on, set}], "set" = the job keys the range's own
    settings change (its settings are app settings, turned into job keys by make_job like the episode's)"""
    out = []
    for r in st.get("ranges") or []:
        if not isinstance(r, dict) or r.get("step") not in RANGE_OK:
            continue
        if not (st.get("range_only") or {}).get(r["step"]):   # the step runs on the whole episode: its ranges are kept but not used
            continue
        try:
            a, b = int(r["start"]), int(r["end"])
        except (KeyError, TypeError, ValueError):
            continue
        sets = r.get("settings") or {}
        over = {}
        if sets:
            alt = make_job({**st, **sets, "ranges": [], "range_only": {}}, k)
            over = {key: v for key, v in alt.items() if key not in ("ranges", "range_only", "work", "output") and job.get(key) != v}
        out.append({"step": r["step"], "start": a, "end": b, "on": r.get("on", True) is not False, "set": over})
    return out


def progressive_choice(st, info):
    """the Repair step's Picture switch (scan: progressive / interlaced), else what the probe found"""
    sc = st.get("scan")
    if sc in ("progressive", "interlaced"):
        return sc == "progressive" or bool(info.get("image"))
    return bool(info.get("progressive"))


def chain_job(st):
    """Chain step: the blocks, and the job keys the VFR analysis / detectors read, taken from them (blocks.py)"""
    ch = st.get("chain")
    if not ch:
        return {}
    import blocks
    chain = [{"type": b["type"], "on": bool(b.get("on", True)), "p": dict(b.get("p") or {}), **({"uid": b["uid"]} if b.get("uid") else {})}
             for b in ch if isinstance(b, dict) and b.get("type") in blocks.BLOCKS]
    view = st.get("chain_view")                          # Chain mode's "View": the preview stops after this block
    return {"chain": chain, **blocks.chain_globals(chain), **({"chain_view": view} if view and any(b.get("uid") == view for b in chain) else {})}


def detect_job(st, tag):
    sel = st.get("selected") or [0]
    job = make_job(st, sel[0])
    job["work"] = str(Path(job["work"]).with_name(f".work_{tag}"))
    if st.get("auto_ai_model"):                          # Chain step: the cleaner of the chain (stabilizer suggestion)
        job["ai_model"] = st["auto_ai_model"]
    return job


# ---------------------------------------------------------------- background tasks

class Tasks:
    def __init__(self):
        self.runner = None
        self.thread = None

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, rid, fn):
        if self.busy():
            emit({"id": rid, "error": "Another task is running - wait or press Stop."})
            return
        self.runner = runner.JobRunner(log)

        def wrap():
            emit({"event": "busy", "busy": True})
            try:
                emit({"id": rid, "result": fn(self.runner)})
            except runner.Cancelled:
                log("Stopped.")
                emit({"id": rid, "error": "Stopped"})
            except Exception as e:                     # noqa: BLE001
                log(f"ERROR: {e}")
                fail(rid, "", e)
            finally:
                emit({"event": "busy", "busy": False})
        self.thread = threading.Thread(target=wrap, daemon=True)
        self.thread.start()

    def cancel(self):
        if self.runner:
            self.runner.cancel()


TASKS = Tasks()


# ---------------------------------------------------------------- preview (own thread, newest wins)

_CLIPS = {}                 # built preview chains: (job, view, script) -> clip; building one takes ~2.5 s
_CLIPS_LOCK = threading.Lock()
LAST_USE = [time.time()]    # the last request from the app (any): the memory keeper below looks at it
IDLE_TRIM_S = int(os.environ.get("DVDRESTORE_IDLE_TRIM_S", 60))


def trim_memory():
    """Give the engine's cached frames back: the cached preview chains go, and VapourSynth's pool of freed frame
    buffers (it keeps them for reuse, up to max_cache_size - 2-3 GB after some scrubbing and Play) is flushed by
    lowering the cap and putting it back. Measured: 5.4 GB -> 1.1 GB; the next preview rebuilds its chain in ~0.2 s."""
    import gc
    import psutil
    import vapoursynth as vs
    before = psutil.Process().memory_info().rss
    with _CLIPS_LOCK:
        _CLIPS.clear()
    gc.collect()
    cap = vs.core.max_cache_size
    vs.core.max_cache_size = 64
    time.sleep(0.3)
    vs.core.max_cache_size = cap
    after = psutil.Process().memory_info().rss
    return before / 2**20, after / 2**20


def _memory_keeper():
    done_at = 0.0
    while True:
        time.sleep(15)
        try:
            if LAST_USE[0] <= done_at or time.time() - LAST_USE[0] < IDLE_TRIM_S:
                continue
            if TASKS.busy() or (PLAYER.thread and PLAYER.thread.is_alive()) or (AUTOVFR.thread and AUTOVFR.thread.is_alive()):
                continue
            a, b = trim_memory()
            done_at = time.time()
            if a - b > 300:
                log(f"Engine idle for {IDLE_TRIM_S} s: cached frames released ({a:,.0f} MB -> {b:,.0f} MB). The next preview picks up where you were.")
        except Exception:                                # noqa: BLE001
            pass


def preview_clip(st, k, view):
    """The preview chain for this state / episode / view, built once and reused (as vsview does): rebuilding
    it for every frame made Play run at 1.8 fps instead of 7-10."""
    import vsgraph
    job = make_job(st, k)
    if job.get("mode") == "vfr" and not runner.job_is_pal(job) and runner.JobRunner.has_pass1(job):
        job["preview_vfr"] = True                      # the export's own frames and timing
    job.pop("cuts", None)                              # an export matter: editing cuts must not rebuild the chain
    script = st.get("script") if view == "after" else None
    upto = st.get("script_line") if script else None
    key = json.dumps([job, view, script, upto], sort_keys=True, default=str)
    with _CLIPS_LOCK:
        c = _CLIPS.pop(key, None)
        if c is not None:
            _CLIPS[key] = c                            # most recently used last: eviction drops the least used
    if c is None:
        c = script_preview(script, job, upto, st["info"].get("frames")) if script else vsgraph.preview(job, after=(view == "after"))
        with _CLIPS_LOCK:
            _CLIPS[key] = c
            while len(_CLIPS) > 6:
                _CLIPS.pop(next(iter(_CLIPS)))
    return c


def frame_image(f, fmt="PNG"):
    import numpy as np
    from PIL import Image
    img = Image.fromarray(np.dstack([np.asarray(f[p]) for p in range(3)]))
    buf = io.BytesIO()
    if fmt == "BMP":                                   # lossless, ~5x faster than PNG (the still preview)
        img.save(buf, "BMP")
    elif fmt == "PNG":
        img.save(buf, "PNG", compress_level=1)
    else:
        img.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii"), img.width, img.height


class Player:
    """Playback: frames are pulled from the cached chain with several in flight at once (all CPU threads and
    the GPU busy, like vsview), sent as JPEG events and paced to the clip's frame rate."""

    def __init__(self):
        self.stop_ev = threading.Event()
        self.thread = None

    def start(self, rid, st, k, start, views):
        self.stop()
        self.stop_ev = threading.Event()
        ev = self.stop_ev
        self.thread = threading.Thread(target=self.run, args=(rid, st, k, start, views, ev), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_ev.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def run(self, rid, st, k, start, views, ev):
        import time
        import vapoursynth as vs
        try:
            clips = [preview_clip(st, k, v) for v in views]
            total = clips[0].num_frames
            start = max(0, min(int(start), total - 1))
            fps = clips[0].fps.numerator / clips[0].fps.denominator if clips[0].fps.numerator else 24000 / 1001
            depth = max(4, vs.core.num_threads)
            its = [c[start:].frames(prefetch=depth, backlog=depth * 2) for c in clips]
            import collections
            t0, shown, clock = time.perf_counter(), 0, 0.0
            recent = collections.deque(maxlen=24)          # emit times, for the fps shown in the UI
            for i, fr in enumerate(zip(*its)):
                if ev.is_set():
                    break
                n = start + i
                msg = {"event": "play_frame", "rid": rid, "frame": n, "total": total, "src": fr[0].props.get("_SrcN")}
                if fr[0].props.get("_DurationNum"):
                    msg["rate"] = round(fr[0].props["_DurationDen"] / fr[0].props["_DurationNum"], 3)
                for v, f in zip(views, fr):
                    msg[v], msg["width"], msg["height"] = frame_image(f, "JPEG")
                p0 = fr[0].props
                dur = p0["_DurationNum"] / p0["_DurationDen"] if "_DurationNum" in p0 and p0["_DurationDen"] else 1 / fps
                due = t0 + clock                            # never faster than real time
                wait = due - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                elif wait < -1.0:                           # fell behind: restart the clock (no catch-up burst)
                    t0, shown, clock = time.perf_counter(), 0, 0.0
                shown += 1
                clock += dur
                recent.append(time.perf_counter())
                msg["fps"] = round((len(recent) - 1) / max(recent[-1] - recent[0], 1e-3), 1) if len(recent) > 1 else 0.0
                emit(msg)
            emit({"id": rid, "result": {"stopped": ev.is_set()}})
        except Exception as e:                              # noqa: BLE001
            fail(rid, "Play: ", e)


PLAYER = Player()


class Preview:
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = None
        self.event = threading.Event()
        threading.Thread(target=self.loop, daemon=True).start()

    def request(self, rid, st, k, frac, views, frame=None):
        with self.lock:
            if self.pending:                               # superseded: tell the UI so it can drop it
                emit({"id": self.pending[0], "error": "superseded"})
            self.pending = (rid, st, k, frac, views, frame)
        self.event.set()

    def loop(self):
        import numpy as np
        from PIL import Image
        import vsgraph
        while True:
            self.event.wait()
            with self.lock:
                req, self.pending = self.pending, None
                self.event.clear()
            if not req:
                continue
            rid, st, k, frac, views, frame = req
            try:
                out = {}
                req = []                                # "before" and/or "after", same frame: rendered at the same time
                for view in views:
                    clip = preview_clip(st, k, view)
                    if frame is not None:
                        n = min(clip.num_frames - 1, max(0, int(frame)))
                    else:
                        n = min(clip.num_frames - 1, max(0, int(frac * (clip.num_frames - 1))))
                    try:
                        fut = clip.get_frame_async(n)
                    except Exception:                      # noqa: BLE001
                        fut = None
                    req.append((view, clip, n, fut))
                for view, clip, n, fut in req:
                    f = fut.result() if fut is not None else clip.get_frame(n)
                    if "_SrcN" in f.props:
                        out["src"] = int(f.props["_SrcN"])
                    out[view], w, h = frame_image(f, "BMP")
                    out["mime"] = "image/bmp"
                    rate = f.props["_DurationDen"] / f.props["_DurationNum"] if f.props.get("_DurationNum") else 0
                    out.update(frame=n, total=clip.num_frames, width=w, height=h, rate=round(rate, 3))
                out["vfr"] = runner.JobRunner.has_pass1(make_job(st, k)) if st.get("mode") == "vfr" else False
                emit({"id": rid, "result": out})
            except Exception as e:                     # noqa: BLE001
                fail(rid, "Preview: ", e)


PREVIEW = None


def episode_of(job):
    return {k: job[k] for k in vsgraph_mod().EPISODE_KEYS if k in job}


def vsgraph_mod():
    import vsgraph
    return vsgraph


def script_upto(text, n):
    """Script mode's "preview to the cursor": the script up to line n (1-based), showing `clip` as it is there"""
    lines = text.split("\n")[: max(1, int(n))]
    return "\n".join(lines) + "\nclip.set_output()   # Script mode: preview to this line\n"


def is_external(text):
    """a script written outside the app: no EPISODE block for the app to fill in"""
    return "EPISODE = " not in text or "# ---- end of episode ----" not in text


def external_wrap(text, job, total):
    """an outside script runs as written (its own source and filters); the episode is its part of the output, taken by
    position: source frames [start, end) of the title's `total` -> the same share of the script's frames (exact for
    scripts that keep the length or change it by a fixed ratio, e.g. IVTC 5 -> 4)"""
    a, b = int(job["start"]), int(job["end"])
    total = max(1, int(total or b))
    if a <= 0 and b >= total:
        return text
    a, b, total = _decoder_frames(job, a, b, total)
    return text + f"""

# ---- EVA toolkit: this episode's part of the script's output (source frames {a} - {b} of {total}) ----
import vapoursynth as _vs
_o = _vs.get_output(0)
_o = _o.clip if hasattr(_o, "clip") else _o
_a, _b = round({a} * _o.num_frames / {total}), round({b} * _o.num_frames / {total})
_vs.clear_outputs()
_o[_a:max(_a + 1, _b)].set_output()
"""


_DEC = {}


def _decoder_frames(job, a, b, total):
    """the app's frame numbers -> a plain decoder's (bs.VideoSource): at a VOB clock jump the app holds the pictures the
    decoder drops (VTS_02_1: 13 at 47 s), an outside script loading the file itself does not - without this its
    episode came 13 frames (373 ms at 24p) early against the sound"""
    if job.get("image") or job.get("progressive"):
        return a, b, total
    try:
        key = job["source"]
        if key not in _DEC:
            import vsgraph
            import vapoursynth as vs
            n = vs.core.bs.VideoSource(key, rff=True).num_frames
            _DEC[key] = (n, vsgraph.timeline_holds(key, n, job["fps_num"] / job["fps_den"]))
        n, holds = _DEC[key]
    except Exception:                                    # noqa: BLE001
        return a, b, total
    if not holds:
        return a, b, total

    def dec(f):
        held = 0
        for at, k in sorted(holds):
            s = at + held                                # where this hold starts in the app's numbering
            if f >= s + k:
                held += k
            elif f >= s:
                return at
            else:
                break
        return f - held
    return dec(a), dec(b), n


def script_for(text, job, total, upto=None, preview=False):
    """the edited / loaded script for one episode: the app's own scripts get their EPISODE block filled in, an outside
    script is cut to the episode's part of its output. upto = only the lines up to there (Script mode)"""
    g = vsgraph_mod()
    if is_external(text):
        return external_wrap(script_upto(text, upto) if upto else text, job, total)
    t = g.set_episode(text, episode_of(job))
    if preview:                                          # VFR needs the pass-1 analysis, which runs on export
        t = t.replace("dr.ivtc_vfr(", "dr.ivtc_cfr(").replace("'mode': 'vfr'", "'mode': 'cfr'")
    return script_upto(t, upto) if upto else t


def script_sources(text, job):
    """video files an outside script loads that are not the file opened in the app (its sound would not match)"""
    import re
    exts = (".vob", ".mkv", ".mp4", ".m2ts", ".ts", ".avi", ".mov", ".mpg", ".mpeg", ".m2v", ".d2v", ".dgi", ".webm", ".wmv", ".iso")
    own = os.path.normcase(os.path.abspath(job["source"]))
    out = []
    for m in re.finditer(r"""[rRbBuU]?(["'])([^"'\n]+?)\1""", text):
        q = m.group(2)
        if q.lower().endswith(exts) and os.path.normcase(os.path.abspath(q.replace("\\\\", "\\"))) != own and q not in out:
            out.append(q)
    return out


def script_preview(text, job, upto=None, total=None):
    """Preview of an edited or loaded script (see script_for)."""
    g = vsgraph_mod()
    return g.to_display(g.run_script(script_for(text, job, total, upto, preview=True), "edited script"), job)


# ---------------------------------------------------------------- methods

DEFAULT_TEMP = Path(paths.TEMP)


def temp_root(st):
    """Folder for temporary files (VFR analysis, encode logs, scripts): Settings, else the "temp" folder in the app folder."""
    return Path(st.get("temp_dir") or DEFAULT_TEMP)


def work_root(st, src):
    """One sub-folder per source file (name + short hash of its path, so two discs with Episode_01 never mix)."""
    import hashlib
    key = hashlib.sha1(str(Path(src).resolve()).lower().encode("utf-8")).hexdigest()[:6]
    return temp_root(st) / f"{Path(src).stem}_{key}"


def m_temp_info(p):
    root = temp_root(p.get("state") or {})
    mb = 0.0
    if root.is_dir():
        mb = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e6
    return {"path": str(root), "default": str(DEFAULT_TEMP), "mb": round(mb, 1), "exists": root.is_dir()}


def m_temp_clear(p):
    """Delete the contents of the temporary folder (only folders this app made: <source>_<hash>/.work_*)."""
    import shutil
    if TASKS.busy():
        return {"error": "A task is running - wait until it is done."}
    AUTOVFR.stop()
    root = temp_root(p.get("state") or {})
    n = 0
    if root.is_dir():
        for d in root.iterdir():
            if d.is_dir() and any(c.name.startswith((".work_", ".report_tmp")) for c in d.iterdir()):
                shutil.rmtree(d, ignore_errors=True)
                n += 1
    return {"removed": n}


def env_info():
    """For Settings: which backends this machine can run."""
    import importlib.util as u
    info = {"tensorrt": bool(u.find_spec("tensorrt")), "cuda": False, "gpu": "", "torch": ""}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if info["cuda"] else ""
    except Exception:                                   # noqa: BLE001
        pass
    if info["tensorrt"]:
        try:
            import tensorrt
            info["tensorrt_version"] = tensorrt.__version__
        except Exception:                               # noqa: BLE001
            info["tensorrt"] = False
    eng = Path(paths.ENGINES)
    info["engines"] = len(list(eng.glob("*.engine"))) if eng.is_dir() else 0
    info["engines_mb"] = round(sum(f.stat().st_size for f in eng.glob("*.engine")) / 1e6) if eng.is_dir() else 0
    return info


def m_hello(p):
    s = {}
    try:
        s = json.loads(SETTINGS.read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        pass
    opts = dict(OPTIONS)
    if not Path(paths.model("deblendnet.pt")).is_file():          # AI blend removal only with its model
        opts["deblend"] = [o for o in OPTIONS["deblend"] if o[0] != "ai"]
        if s.get("deblend") == "ai":
            s["deblend"] = "off"
    return {"options": opts, "missing": runner.missing_tools(), "settings": s, "busy": TASKS.busy()}   # busy: a task is still running (the window was reloaded)


def m_make_report(p):
    """Help > Report a problem: ONE small zip of what is needed to find and fix a bug - what the user wrote, version and system,
    the log file (engine + window + errors), the last actions, the errors, the settings of the source, the script and a screenshot.
    User names in file paths can be hidden. Nothing is sent anywhere: the user sends the file."""
    import base64
    import re
    import time
    import zipfile
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = Path(paths.ROOT) / "temp" / "reports"
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / f"EVAtoolkit_problem_{stamp}.zip"
    hide = bool(p.get("hide_names", True))
    home = str(Path.home())

    def clean(t):
        t = str(t or "")
        if hide:
            for h in {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}:      # C:\Users\x, C:/Users/x and the JSON-escaped C:\\Users\\x
                t = t.replace(h, "~")
            t = re.sub(r"(?i)(users(?:\\{1,2}|/))[^\\/\s\"']+", r"\1<user>", t)
            name = Path(home).name
            if len(name) >= 3:
                t = re.sub(re.escape(name), "<user>", t, flags=re.I)                    # the name anywhere else (a share, a title)
        return t

    sysinfo = dict(p.get("sysinfo") or {})
    try:
        sysinfo["engine"] = env_info()
    except Exception as e:                               # noqa: BLE001
        sysinfo["engine"] = {"error": str(e)}
    try:
        import vapoursynth as _vsm
        sysinfo["vapoursynth"] = str(_vsm.core).splitlines()[1:3]
    except Exception:                                    # noqa: BLE001
        pass
    sysinfo["python"] = sys.version.split()[0]
    state = p.get("state")
    info = (state or {}).get("info") or {}
    where = p.get("page") or "?"
    md = [f"# EVA toolkit problem report", "", f"Made: {time.strftime('%Y-%m-%d %H:%M:%S')}", f"Version: {p.get('version') or sysinfo.get('app', '?')}",
          f"Page: {where}", "", "## What happened", "", (p.get("text") or "(nothing written)").strip(), "", "## System", ""]
    for k in ("platform", "release", "arch", "cpu", "ramGB", "freeGB", "electron", "chrome", "portable", "uptimeMin", "screens", "gpuDevices", "python"):
        if k in sysinfo:
            md.append(f"- {k}: {sysinfo[k]}")
    eng = sysinfo.get("engine") or {}
    md += [f"- engine: gpu={eng.get('gpu')} torch={eng.get('torch')} tensorrt={eng.get('tensorrt_version') or eng.get('tensorrt')} cuda={eng.get('cuda')}"]
    if info:
        md += ["", "## Source", "", f"- file: {Path(str(info.get('path', ''))).name}", f"- {info.get('width')}x{info.get('height')} {info.get('fps_num')}/{info.get('fps_den')} fps, "
               f"{info.get('frames')} frames, {'PAL' if info.get('pal') else 'NTSC'}, scan: {info.get('scan')}"]
    if p.get("errors"):
        md += ["", "## Errors noted by the app", "", "```", str(p["errors"])[:6000], "```"]
    files = []
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        def add(name, text):
            z.writestr(name, clean(text)); files.append(name)
        add("report.md", "\n".join(md) + "\n")
        add("system.json", json.dumps(sysinfo, indent=1, default=str))
        if p.get("with_logs", True):
            lf = Path(str(p.get("log_file") or Path(paths.ROOT) / "temp" / "evatoolkit.log"))
            for q in (lf.with_name(lf.name + ".1"), lf):          # the older copy, then the current one: the last ~600 KB
                if q.is_file():
                    data = q.read_bytes()[-300000:]
                    add(q.name, data.decode("utf-8", "replace"))
            add("actions.txt", p.get("actions") or "")
            add("window_log.txt", p.get("ui_log") or "")
        if p.get("errors"):
            add("errors.txt", p["errors"])
        if state is not None:
            slim = {k: (v if not (isinstance(v, list) and len(v) > 2000) else f"<{len(v)} items>") for k, v in state.items() if k != "info"}
            slim["info"] = {k: (v if not (isinstance(v, (list, dict)) and len(json.dumps(v, default=str)) > 4000) else "<large>") for k, v in info.items()}
            add("settings.json", json.dumps(slim, indent=1, default=str))
        if p.get("script"):
            add("script.vpy", p["script"])
        if p.get("with_source_info") and info.get("path") and Path(str(info["path"])).exists():
            try:
                r = runner._run([runner.TOOLS["ffprobe"], "-v", "error", "-show_streams", "-show_chapters", "-show_format", "-of", "json", str(info["path"])])
                add("ffprobe.json", r.stdout or r.stderr or "")
            except Exception as e:                       # noqa: BLE001
                add("ffprobe.json", f"ffprobe failed: {e}")
        if p.get("screenshot"):
            z.writestr("screenshot.png", base64.b64decode(p["screenshot"])); files.append("screenshot.png")
    return {"path": str(zpath), "mb": round(zpath.stat().st_size / 1e6, 2), "files": files}


PRESETS = Path(paths.ROOT) / "presets"                 # per-show presets (workflow files, by name)


def _preset_file(name):
    safe = "".join(ch for ch in str(name).strip() if ch not in '<>:"/\\|?*').strip(". ")[:80]
    if not safe:
        raise ValueError("a preset needs a name")
    return PRESETS / f"{safe}.json"


def m_presets(p):
    out = []
    for f in sorted(PRESETS.glob("*.json")) if PRESETS.is_dir() else []:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out.append({"name": d.get("name") or f.stem, "file": f.stem, "note": d.get("note", ""), "source": d.get("source", ""),
                        "created": d.get("created", ""), "keys": len(d.get("settings") or {})})
        except Exception:                                # noqa: BLE001
            continue
    return out


def m_preset_get(p):
    return json.loads(_preset_file(p["name"]).read_text(encoding="utf-8"))


def m_preset_save(p):
    wf = dict(p.get("workflow") or {})
    wf["name"] = str(p["name"]).strip()
    PRESETS.mkdir(parents=True, exist_ok=True)
    f = _preset_file(p["name"])
    f.write_text(json.dumps(wf, indent=1), encoding="utf-8")
    return {"file": f.stem}


def m_preset_delete(p):
    f = _preset_file(p["name"])
    if f.exists():
        f.unlink()
    return True


def m_save_settings(p):
    SETTINGS.write_text(json.dumps(p.get("settings", {}), indent=1), encoding="utf-8")
    return True


def m_probe(p):
    info = runner.probe(p["path"])
    _scan_in_background(p["path"], info)
    return {"info": info, "episodes": [list(e) for e in runner.detect_episodes(info)]}


def _scan_in_background(path, info):
    """the damaged-frame scan (vsgraph.scan_source, once per file, ~10 s per SD episode) starts as soon as a video is
    opened: preview, Analyze and the VFR analysis then skip damaged frames instead of stopping at them"""
    if info.get("image") or not os.path.isfile(path):
        return
    import threading

    def run():
        try:
            import vsgraph
            bad = vsgraph.scan_source(path, progress=False)
            if bad:
                log(f"{len(bad)} damaged frame(s) in the source (cannot be decoded): each is replaced by its neighbour")
        except Exception as e:                           # noqa: BLE001
            print(f"Damaged-frame scan skipped: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    threading.Thread(target=run, daemon=True).start()


def m_dvd_titles(p):
    return runner.disc_titles(p["path"])


def m_import(p, rid):
    TASKS.start(rid, lambda r: r.import_title(p["path"], int(p["title"]), float(p.get("duration") or 0)))


def m_analyze(p, rid):
    """Every detector in one go: PAL type, dot crawl / rainbows, ghost/halo, chroma drift + bleed, crop."""
    st = p["state"]

    def fn(r):
        job = detect_job(st, "measure")
        res = {}
        if runner.job_is_pal(job):
            log("Analyze 1/2: PAL type...")
            res["pal"] = r.pal_detect(job)
        log("Analyze 2/2: dot crawl, rainbows, MPEG blocking, ghost/halo, chroma drift and bleed...")
        res.update(r.measure(job))
        log("Analysis done.")
        return res
    TASKS.start(rid, fn)


def m_autodetect(p, rid):
    def fn(r):
        log("Auto-detect: dot crawl, rainbows, composite, ghost/halo, chroma drift/bleed...")
        return r.measure(detect_job(p["state"], "measure"))
    TASKS.start(rid, fn)


def m_detect(p, rid):
    kind = p["kind"]          # "ghost", "crop", "levels" (colour step) or auto_<group> (Adjust: frames composite block chroma stab)
    TASKS.start(rid, lambda r: r.detect(kind, detect_job(p["state"], kind)))


def m_filter_preview(p, rid):
    """the Add window: the frame shown (After) before and after one filter, as small JPEGs. The filter runs on the
    previewed picture (YUV 4:4:4 16-bit, converted to what it needs), so temporal filters see the neighbouring frames."""
    def run():
        try:
            import blocks
            import vapoursynth as vs
            core = vs.core
            st, k = p["state"], int(p.get("episode", 0))
            base = preview_clip(st, k, "after")
            n = max(0, min(base.num_frames - 1, int(p.get("frame") or 0)))
            job = make_job(st, k)
            yuv = core.resize.Bicubic(base, format=vs.YUV444P16, matrix_s="170m")
            out = blocks.block(yuv, job, p["type"], p.get("p") or {}, blocks.chain_context(job))
            if out.format.color_family == vs.RGB:
                rgb = core.resize.Bicubic(out, format=vs.RGB24, width=base.width, height=base.height)
            else:
                if out.format.color_family == vs.GRAY:         # a mask: shown as a grey picture
                    rgb = core.resize.Bicubic(core.std.ShufflePlanes([out, out, out], [0, 0, 0], vs.RGB), format=vs.RGB24,
                                              width=base.width, height=base.height)
                else:
                    rgb = core.resize.Bicubic(out, format=vs.RGB24, width=base.width, height=base.height, matrix_in_s="170m")
            m = min(n, rgb.num_frames - 1)
            scale = 480 / max(1, base.width)

            def small(c, i):
                c = core.resize.Bilinear(c, width=max(16, int(base.width * scale) // 2 * 2), height=max(16, int(base.height * scale) // 2 * 2))
                return frame_image(c.get_frame(i), "JPEG")[0]
            emit({"id": rid, "result": {"before": small(base, n), "after": small(rgb, m), "frame": n,
                                        "size": [out.width, out.height], "format": out.format.name}})
        except Exception as e:                            # noqa: BLE001
            emit({"id": rid, "result": {"error": f"{type(e).__name__}: {str(e)[:300]}"}})
    threading.Thread(target=run, daemon=True).start()


def m_scan_picture(p, rid):
    """Analyze, step 1: whole frames or fields? ffmpeg idet on 6 windows of 300 frames spread over the episode (the open
    check looks at 3 x 200 of the whole file). A file flagged progressive at a non-DVD rate, an image and soft telecine
    (pulldown flags: the engine applies it) are decided by their flags."""
    st = p["state"]
    info = st["info"]

    def run():
        try:
            if info.get("image"):
                res = {"progressive": True, "by": "image"}
            elif info.get("progressive") and (info.get("scan") or {}).get("by") == "flags":
                res = {"progressive": True, "by": "flags"}
            elif info.get("pulldown"):
                res = {"progressive": False, "by": "pulldown"}
            else:
                a, b = st["episodes"][int(p.get("episode", 0))]
                fps = info["fps_num"] / info["fps_den"]
                n = max(1, int(info.get("frames") or b))
                dur = float(info.get("duration") or n / fps)
                t0, t1 = dur * a / n, dur * b / n                 # the episode on the file's clock (close enough for windows)
                wins = [(t0 + (t1 - t0) * f) / dur for f in (0.08, 0.24, 0.4, 0.56, 0.72, 0.88)]
                sc = runner.picture_scan(info["path"], dur, windows=wins, frames=300)
                res = {"progressive": sc["frames"] >= 300 and sc["combed"] <= 0.05, "by": "picture", **sc}
            emit({"id": rid, "result": res})
        except Exception as e:                            # noqa: BLE001
            fail(rid, "Picture check: ", e)
    threading.Thread(target=run, daemon=True).start()


def m_report(p, rid):
    st = p["state"]
    job = detect_job(st, "report")
    dest = Path(job["output"]).parent
    TASKS.start(rid, lambda r: r.report(job, st, p.get("log", ""), dest))


def m_process(p, rid):
    st = dict(p["state"])
    st.pop("chain_view", None)                           # a Chain "View" is for the preview only
    AUTOVFR.stop()
    jobs = [make_job(st, k) for k in st.get("selected") or []]
    text = st.get("script")
    total = st["info"].get("frames")
    if text and not is_external(text):
        vsgraph_mod().set_episode(text, episode_of(jobs[0]))          # fail now if the block is broken

    def fn(r):
        done = []
        if text and is_external(text):
            log("Exporting with your own script (loaded from a file): it runs as written; each episode is its part of "
                "the output, with the sound and subtitles of the file opened in the app.")
            for q in script_sources(text, jobs[0]):
                log(f"WARNING: the script loads {q}, not the file opened in the app - the sound and subtitles come from "
                    f"{Path(jobs[0]['source']).name} and only match if the picture does")
        elif text:
            log("Exporting with the edited script (Script step).")
        for n, job in enumerate(jobs, 1):
            log(f"=== {Path(job['output']).name}  ({n}/{len(jobs)}) ===")
            done.append(run_edited(r, job, text, total))
        log("All done.")
        return done
    TASKS.start(rid, fn)


def run_edited(r, job, text=None, total=None):
    """export one episode; with parts cut out on the Timeline every kept part is exported on its own (picture, sound
    and subtitles each checked in sync) and the parts are joined (mkvmerge append), so nothing can drift at a cut"""
    sc = lambda j: script_for(text, j, total) if text else None
    if text and is_external(text):
        job = dict(job, external=True)                   # its own frames and rate (runner: like your own IVTC)
    parts = kept_parts(job) if job.get("cuts") else [(job["start"], job["end"])]
    if len(parts) == 1 and parts[0] == (job["start"], job["end"]):
        return r.run({k: v for k, v in job.items() if k != "cuts"}, script=sc(job))
    short = [(a, b) for a, b in parts if b - a < 12]
    parts = [(a, b) for a, b in parts if b - a >= 12]
    if short:
        log(f"Timeline: {len(short)} kept part(s) shorter than 12 frames left out too (too short to export on their own)")
    if not parts:
        raise RuntimeError("The Timeline cuts out the whole episode - nothing to export.")
    cut = sum(c1 - c0 for c0, c1 in job["cuts"])
    log(f"Timeline: {len(job['cuts'])} part(s) cut out ({cut} source frames) - the episode is exported in {len(parts)} parts, then joined")
    out = Path(job["output"])
    work = Path(job["work"]) / "parts"
    work.mkdir(parents=True, exist_ok=True)
    files = []
    for i, (a, b) in enumerate(parts):
        pj = {k: v for k, v in job.items() if k != "cuts"}
        pj.update(start=a, end=b, work=str(work / f"part_{i:02d}"), output=str(work / f"part_{i:02d}{out.suffix}"))
        log(f"--- part {i + 1} of {len(parts)}: source frames {a} - {b} ---")
        files.append(r.run(pj, script=sc(pj)))
    if len(files) == 1:
        os.replace(files[0], out)
    else:
        # mkvmerge appends a file at the end of the LONGEST track before it (the sound ran 12 ms past the picture:
        # every later part came 12.7 ms late) - each part is moved back onto the end of the picture before it
        args = [runner.TOOLS["mkvmerge"], "-q", "-o", str(out)]
        pic_end, all_end = 0.0, 0.0
        for i, f in enumerate(files):
            ends = _track_ends(f)
            if i:
                ms = round((pic_end - all_end) * 1000)
                args.append("+")
                for tid in sorted(ends):
                    args += ["--sync", f"{tid}:{ms}"]
            args.append(str(f))
            off = pic_end
            pic_end = off + ends.get(0, max(ends.values(), default=0.0))
            all_end = off + max(ends.values(), default=0.0)
        pr = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if pr.returncode not in (0, 1) or not out.exists():
            raise RuntimeError("Joining the parts failed (mkvmerge): " + (pr.stdout + pr.stderr)[-400:])
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
    log(f"Timeline: {len(files)} parts joined -> {out.name}")
    return str(out)


def _track_ends(path):
    """{track: end time (last packet + its duration)} of a part file (track 0 = the picture)"""
    pr = subprocess.run([runner.TOOLS["ffprobe"], "-v", "error", "-show_entries", "packet=stream_index,pts_time,duration_time",
                         "-of", "csv=p=0", str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                        stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    out = {}
    for line in (pr.stdout or "").splitlines():
        q = line.split(",")
        try:
            i, t = int(q[0]), float(q[1])
            d = float(q[2]) if len(q) > 2 and q[2] not in ("", "N/A") else 0.0
        except (ValueError, IndexError):
            continue
        out[i] = max(out.get(i, 0.0), t + d)
    return out


# ---------------------------------------------------------------- Timeline: filmstrip, waveform, subtitle timings
_THUMB = {}
_THUMB_LOCK = threading.Lock()


def m_thumbs(p, rid):
    """small pictures of source frames for the Timeline's video track (the untouched source, by source frame number)"""
    def run():
        try:
            import vsgraph
            import vapoursynth as vs
            st = p["state"]
            job = make_job(st, int(p.get("episode", 0)))
            h = max(16, int(p.get("h", 40)) // 2 * 2)
            key = (job["source"], job.get("order"), job.get("progressive"), h)
            with _THUMB_LOCK:
                c = _THUMB.get(key)
                if c is None:
                    src = vsgraph.source(dict(job, start=0, end=None), clean=False)
                    w = max(8, int(round(h * source_dar(st) / 2)) * 2)
                    c = vs.core.resize.Bilinear(src, width=w, height=h, format=vs.RGB24, matrix_in_s="170m")
                    _THUMB.clear()
                    _THUMB[key] = c
            out = {}
            for f in p.get("frames") or []:
                n = max(0, min(c.num_frames - 1, int(f)))
                out[str(f)] = frame_image(c.get_frame(n), "JPEG")[0]
            emit({"id": rid, "result": {"thumbs": out}})
        except Exception as e:                            # noqa: BLE001
            fail(rid, "Timeline pictures: ", e)
    threading.Thread(target=run, daemon=True).start()


_MEDIA = {}


def m_media_tracks(p, rid):
    """the sound's waveform (peaks, 50 per second) and every subtitle's time on screen, for the Timeline. Times are
    seconds from the first video frame on the file's own clock (as the app's frameTime)."""
    path = p["path"]

    def run():
        try:
            import hashlib
            if path in _MEDIA:
                emit({"id": rid, "result": _MEDIA[path]})
                return
            cache = Path(paths.TEMP) / "timeline" / (hashlib.sha1(path.lower().encode("utf-8")).hexdigest()[:12]
                                                     + f"_{int(os.path.getmtime(path))}.media.json")
            if cache.exists():
                res = json.loads(cache.read_text(encoding="utf-8"))
                _MEDIA[path] = res
                emit({"id": rid, "result": res})
                return
            ff, fp = runner.TOOLS["ffmpeg"], runner.TOOLS["ffprobe"]
            kw = dict(stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            pr = subprocess.run([fp, "-v", "error", "-show_entries", "stream=index,codec_type,start_time:stream_tags=language",
                                 "-of", "json", path], capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)
            streams = json.loads(pr.stdout or "{}").get("streams", [])
            num = lambda x: float(x) if x not in (None, "N/A", "") else None
            vst = next((num(q.get("start_time")) for q in streams if q.get("codec_type") == "video"), 0.0) or 0.0
            res = {"rate": 50, "audio": None, "subs": []}
            aud = next((q for q in streams if q.get("codec_type") == "audio"), None)
            if aud is not None:
                import numpy as np
                ast = num(aud.get("start_time"))
                ast = vst if ast is None else ast
                # async: holes in the sound (a VOB clock jump) become silence, so sample i is at start + i / rate
                pr2 = subprocess.run([ff, "-v", "error", "-i", path, "-map", f"0:{aud['index']}", "-vn", "-sn",
                                      "-af", "aresample=async=1", "-ac", "1", "-ar", "4000", "-f", "s16le", "-"],
                                     capture_output=True, **kw)
                a = np.frombuffer(pr2.stdout[: len(pr2.stdout) // 2 * 2], np.int16).astype(np.float32)
                n = len(a) // 80
                if n:
                    pk = np.abs(a[: n * 80]).reshape(n, 80).max(1)
                    pk = np.sqrt(pk / 32768.0)               # quiet speech stays visible next to loud music
                    res["audio"] = {"offset": round(ast - vst, 4),
                                    "peaks": base64.b64encode(np.clip(pk * 255, 0, 255).astype(np.uint8).tobytes()).decode("ascii")}
            for q in [q for q in streams if q.get("codec_type") == "subtitle"][:4]:
                pr3 = subprocess.run([fp, "-v", "error", "-select_streams", str(q["index"]), "-show_entries",
                                      "packet=pts_time,duration_time,size", "-of", "csv=p=0", path],
                                     capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)
                ev = []
                for line in (pr3.stdout or "").splitlines():
                    c = line.split(",")
                    t = num(c[0]) if c else None
                    if t is None:
                        continue
                    ev.append((t - vst, num(c[1]) if len(c) > 1 else None, int(c[2]) if len(c) > 2 and c[2].strip().isdigit() else 0))
                ev.sort()
                blocks = []
                for i, (t, d, size) in enumerate(ev):
                    nxt = ev[i + 1][0] if i + 1 < len(ev) else t + 4
                    if 0 < size < 40:                         # a DVD "clear" packet: only ends the one before
                        continue
                    blocks.append([round(t, 3), round(max(0.2, min(d or 4.0, nxt - t, 8.0)), 3)])
                res["subs"].append({"lang": (q.get("tags") or {}).get("language", "und"), "blocks": blocks})
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(res), encoding="utf-8")
            _MEDIA[path] = res
            emit({"id": rid, "result": res})
        except Exception as e:                            # noqa: BLE001
            fail(rid, "Timeline sound / subtitles: ", e)
    threading.Thread(target=run, daemon=True).start()


def _with_snaps(job, st, k):
    """A / B snapshots of the app as extra vsview tabs (each its own job)"""
    snaps = []
    for x in st.get("snap_states") or []:
        try:
            snaps.append({"name": str(x.get("name", "snapshot")), "job": make_job(x["state"], k)})
        except Exception:                                # noqa: BLE001
            pass
    if snaps:
        job["snaps"] = snaps
    return job


def m_vsview_update(p):
    """the open vsview follows every change (it reloads itself, keeping its tab and frame)"""
    st = p["state"]
    job = _with_snaps(make_job(st, (st.get("selected") or [0])[0]), st, (st.get("selected") or [0])[0])
    return {"updated": runner.update_vsview(job)}


def m_vsview(p, rid):
    st = p["state"]
    job = _with_snaps(make_job(st, (st.get("selected") or [0])[0]), st, (st.get("selected") or [0])[0])

    def fn(r):
        if job["mode"] == "vfr" and not runner.JobRunner.has_pass1(job) and not runner.job_is_pal(job):
            log("Running VFR analysis for the preview (reused when you process this episode)...")
            r.ensure_pass1(job)
        runner.open_in_vsview(job)
        log("vsview opened")
        return True
    TASKS.start(rid, fn)


def m_script(p):
    """The current settings as a VapourSynth script (Script step)."""
    st = p["state"]
    k = int(p.get("episode", (st.get("selected") or [0])[0]))
    job = make_job(st, k)
    return vsgraph_mod().script_text(job, Path(job["output"]).stem)


def m_script_api(p):
    """Script mode's completion: the app's functions (dr.*) with their settings and what they do"""
    import inspect
    g = vsgraph_mod()
    out = []
    for name, fn in inspect.getmembers(g, inspect.isfunction):
        if name.startswith("_") or fn.__module__ != g.__name__:
            continue
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig = "(...)"
        doc = (inspect.getdoc(fn) or "").strip().split("\n\n")[0].replace("\n", " ")
        out.append({"name": name, "sig": sig, "doc": doc[:400]})
    import blocks
    kinds = [{"id": k, "name": T["name"], "desc": T["desc"]} for k, T in blocks.BLOCKS.items()]
    return {"dr": out, "blocks": kinds}


def m_check_script(p):
    """Run the edited script far enough to know it works: returns frames / size / rate or the error."""
    st = p["state"]
    job = make_job(st, int(p.get("episode", 0)))
    g = vsgraph_mod()
    try:
        text = script_for(p["text"], job, p["state"]["info"].get("frames"), p.get("upto"), preview=True)
        c = g.run_script(text, "edited script")
        c.get_frame(min(10, c.num_frames - 1))
        return {"ok": True, "frames": c.num_frames, "width": c.width, "height": c.height,
                "format": c.format.name, "fps": f"{c.fps.numerator}/{c.fps.denominator}",
                "external": is_external(p["text"]), "other_sources": script_sources(p["text"], job) if is_external(p["text"]) else []}
    except Exception as e:                                # noqa: BLE001
        tb = traceback.format_exc()
        line = [l for l in tb.splitlines() if "edited script" in l]
        import re as _re
        ln = getattr(e, "lineno", None) if isinstance(e, SyntaxError) else None
        if ln is None and line:
            m = _re.search(r"line (\d+)", line[-1])
            ln = int(m.group(1)) if m else None
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "where": line[-1].strip() if line else "", "line": ln}


_LIBRARY = {}


def _scan_toolbox():
    """the EVA toolbox's compatibility scan (blocks_scan.py, ~10 s): every plugin / vs-jetpack function tried once"""
    p = subprocess.run([sys.executable, "-W", "ignore", str(APP_DIR / "blocks_scan.py")], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    last = [l for l in (p.stdout or "").splitlines() if l.startswith("done")]
    return last[-1] if last else (p.stderr or "").strip()[-300:]


def m_my_filter_add(p):
    """copy your .py filter files / .dll plugins into my_filters (plugins load at once; the library is read again)"""
    import shutil
    import blocks
    d = blocks.my_dir()
    done = []
    for src in p.get("paths") or []:
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(d, {".dll": "plugins", ".json": "nodes"}.get(ext, ""), os.path.basename(src))
        if ext not in (".py", ".dll", ".json") or not os.path.isfile(src):
            continue
        if ext == ".json":                               # a custom filter someone exported: check it is one
            try:
                n = json.loads(open(src, encoding="utf-8").read())
                if not (isinstance(n, dict) and n.get("name") and isinstance(n.get("code"), str)):
                    continue
            except Exception:                            # noqa: BLE001
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        done.append(os.path.basename(src))
    loaded = blocks.load_my_plugins()
    global _LIBRARY
    _LIBRARY = None                                      # read again (the compatibility scan tries a new plugin's functions)
    return {"added": done, "loaded": loaded, "dir": d}


# ---- your custom filters (VapourKit-style): a name, what it does and its code, saved from a filter card as
# my_filters\nodes\<name>.json; they appear in the library under "Your own filters" and are added as that code
def _nodes_dir():
    import blocks
    d = os.path.join(blocks.my_dir(), "nodes")
    os.makedirs(d, exist_ok=True)
    return d


def _node_file(name):
    import re
    slug = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name)).strip(" .") or "filter"
    return os.path.join(_nodes_dir(), slug[:80] + ".json")


def m_my_nodes(p):
    out = []
    d = _nodes_dir()
    for n in sorted(os.listdir(d), key=str.lower):
        if not n.lower().endswith(".json"):
            continue
        try:
            j = json.loads(open(os.path.join(d, n), encoding="utf-8").read())
            if j.get("name") and isinstance(j.get("code"), str):
                out.append({"name": str(j["name"]), "doc": str(j.get("doc") or ""), "code": j["code"], "file": os.path.join(d, n)})
        except Exception as e:                           # noqa: BLE001
            log(f"Custom filter {n} cannot be read: {e}")
    return {"nodes": out, "dir": d}


def m_my_node_save(p):
    name = str(p.get("name") or "").strip()
    if not name:
        return {"error": "Give the filter a name first"}
    if p.get("old") and p["old"] != name:                # renamed: the old file goes
        try:
            os.remove(_node_file(p["old"]))
        except OSError:
            pass
    f = _node_file(name)
    with open(f, "w", encoding="utf-8") as fh:
        json.dump({"name": name, "doc": str(p.get("doc") or ""), "code": str(p.get("code") or ""), "eva_filter": 1}, fh, indent=1, ensure_ascii=False)
    return {**m_my_nodes({}), "saved": f}


def m_my_node_delete(p):
    try:
        os.remove(_node_file(p.get("name") or ""))
    except OSError:
        pass
    return m_my_nodes({})


def m_ai_models(p):
    """the AI model files in your model folders (Repair -> VapourSynth filters -> AI upscale & restore models)"""
    import blocks
    shipped = os.path.join(str(paths.ROOT), "models", "upscale")            # the models that come with the app
    return {"models": blocks.ai_models([shipped] + list(p.get("dirs") or []), p.get("files") or [])}


def m_ai_engines(p):
    """the model library: the built TensorRT engines of one model (details, which one the node uses), see engines.py"""
    import blocks
    import engines
    try:
        return engines.listing(p.get("model"), p.get("engine"), p.get("w") or 720, p.get("h") or 480, lambda path: blocks._ai_preset(path, "auto"))
    except Exception as e:                                # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def m_ai_engine_pick(p):
    import engines
    return engines.pick(p.get("model"), p.get("file"))


def m_ai_engine_delete(p):
    import engines
    return engines.delete(p.get("files"))


def m_model_info(p):
    """what an AI model is: ONNX / PyTorch, scale, fixed input size, precision, video model"""
    import aiupscale
    try:
        return aiupscale.info(p["path"])
    except Exception as e:                                # noqa: BLE001
        return {"error": f"Cannot read this model: {type(e).__name__}: {e}"}


def m_build_ai_engine(p, rid):
    """the engine dialog's "Save & build now": the model's TensorRT engine for this video's picture size, now"""
    import blocks

    def fn(r):
        import nnbackend
        log(f"Building the TensorRT engine for {os.path.basename(p['model'])} ({p.get('w')}x{p.get('h')}) - once, up to a few minutes...")
        nnbackend.BUILD_HOOK[0] = lambda ev: (emit(ev), not r.cancelled)[1]          # progress to the window; Stop ends the build
        try:
            out = blocks.build_ai_engine(p["model"], p.get("engine"), p.get("w") or 720, p.get("h") or 480, p.get("backend"))
        finally:
            nnbackend.BUILD_HOOK[0] = None
        log(f"Engine ready: {os.path.basename(p['model'])} ({out['w']}x{out['h']} out)")
        return out
    TASKS.start(rid, fn)


def m_block_library(p):
    """Chain step: the EVA toolbox (the app's blocks, your recipes, vs-jetpack, VapourSynth plugins). After plugins or
    vs-jetpack changed (other versions than the last scan) the compatibility scan runs again first, by itself."""
    if not _LIBRARY:
        import blocks
        import blocks_scan
        try:
            d = json.loads((APP_DIR / "blocks_compat.json").read_text(encoding="utf-8"))
            stale = d.get("versions") != blocks_scan.versions()
        except Exception:                               # noqa: BLE001
            stale = True
        if stale:
            log("EVA toolbox: plugins / vs-jetpack changed since the last check - testing every filter again (~10 s)...")
            log("EVA toolbox: " + _scan_toolbox())
        _LIBRARY.update(blocks.library(OPTIONS))
    return _LIBRARY


def m_toolbox_info(p):
    """EVA toolbox: where things are, the filter packages and their versions, the runtime fixes"""
    import eva_toolbox
    from eva_toolbox import shims
    return {"catalog": eva_toolbox.CATALOG, "recipes": eva_toolbox.RECIPES, "packages": eva_toolbox.packages(),
            "shims": {k: str(v) for k, v in shims.apply().items()}, "python": sys.executable}


def m_save_recipe(p):
    """EVA toolbox: a Custom code block saved as your own block (eva_toolbox/recipes)"""
    import eva_toolbox
    name = eva_toolbox.save_recipe(p.get("name", ""), p.get("code", ""), p.get("desc", ""))
    _LIBRARY.clear()
    return {"file": name, "type": "recipe:" + name[:-3]}


def _pip(cmd, r):
    """run a pip command with the app's Python, every line to the log (EVA_PIP_TEST=1: a harmless pip --version)"""
    import os as _os
    if _os.environ.get("EVA_PIP_TEST"):
        cmd = [sys.executable, "-m", "pip", "--version"]
    log("EVA toolbox: " + " ".join(Path(c).name if i == 0 else c for i, c in enumerate(cmd)))
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                          stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    r.proc.append(pr)
    for line in pr.stdout:
        line = line.rstrip()
        if line and not line.startswith("  "):
            log("pip: " + line[:300])
        r._check()
    if pr.wait() != 0:
        raise RuntimeError("pip failed - see the log (internet connection? package name?)")


def _toolbox_change(rid, cmd_fn, what):
    import eva_toolbox

    def fn(r):
        before = dict(eva_toolbox.packages())
        _pip(cmd_fn(), r)
        after = dict(eva_toolbox.packages())
        changed = [f"{n} {before.get(n, 'new')} -> {v}" for n, v in after.items() if before.get(n) != v]
        log("EVA toolbox: " + (", ".join(changed) if changed else "nothing changed"))
        log("EVA toolbox: testing every filter again...")
        log("EVA toolbox: " + _scan_toolbox())
        _LIBRARY.clear()
        log(f"EVA toolbox: {what} done.")
        return {"changed": changed}
    TASKS.start(rid, fn)


def m_toolbox_update(p, rid):
    """EVA toolbox: upgrade vs-jetpack and every VapourSynth plugin package, then test every filter again"""
    import eva_toolbox
    _toolbox_change(rid, eva_toolbox.update_command, "update")


def m_toolbox_install(p, rid):
    """EVA toolbox: install a new filter package (e.g. vapoursynth-neo_f3kdb), then test every filter again"""
    import eva_toolbox
    name = p.get("name", "")
    _toolbox_change(rid, lambda: eva_toolbox.install_command(name), f"install {name}")


def m_toolbox_rescan(p, rid):
    """EVA toolbox: test every filter again (after editing catalog.json or adding plugins by hand)"""
    def fn(r):
        log("EVA toolbox: " + _scan_toolbox())
        _LIBRARY.clear()
        return True
    TASKS.start(rid, fn)


def m_chain_default(p):
    """Chain step: the current step-panel settings of an episode as an equivalent chain (frame-identical; tested)"""
    st = dict(p["state"])
    st.pop("chain", None)
    k = int(p.get("episode", (st.get("selected") or [0])[0]))
    import blocks
    return blocks.chain_from_job(make_job(st, k))


def m_lut_info(p):
    import lut
    try:
        return lut.info(p["path"])
    except Exception as e:                                # noqa: BLE001
        return {"error": f"Cannot read this LUT: {e}"}


def m_cancel(p):
    TASKS.cancel()
    return True


def m_preview(p, rid):
    views = p.get("views") or ["after"]
    PREVIEW.request(rid, p["state"], int(p.get("episode", 0)), float(p.get("pos", 0)), views, p.get("frame"))


def m_vfr_pass(p, rid):
    """Run the VFR analysis (pass 1) for the episode being previewed, so preview and Play show the export's
    frames and timing. The export reuses it."""
    st = p["state"]
    job = make_job(st, int(p.get("episode", 0)))

    AUTOVFR.stop()
    def fn(r):
        log("VFR analysis for the preview (reused when this episode is exported)...")
        r.ensure_pass1(job)
        log("VFR analysis done - preview and Play now use the export's frame rates.")
        return True
    TASKS.start(rid, fn)


def m_scopes(p, rid):
    """Colour step scopes of the frame being previewed (After): waveform / parade / vectorscope / histogram."""
    def run():
        try:
            import numpy as np
            import scopes
            clip = preview_clip(p["state"], int(p.get("episode", 0)), "after")
            n = max(0, min(clip.num_frames - 1, int(p.get("frame", 0))))
            f = clip.get_frame(n)
            rgb = np.dstack([np.asarray(f[q]) for q in range(3)])
            emit({"id": rid, "result": {"kind": p.get("kind", "waveform"), "png": base64.b64encode(scopes.draw(p.get("kind", "waveform"), rgb)).decode("ascii"), "frame": n}})
        except Exception as e:                            # noqa: BLE001
            fail(rid, "Scopes: ", e)
    threading.Thread(target=run, daemon=True).start()


def m_scenes(p, rid):
    """Scene detection for the Colour step (episode being previewed)."""
    st = p["state"]
    job = make_job(st, int(p.get("episode", 0)))

    def fn(r):
        import vsgraph
        log("Finding scene cuts...")
        cuts = vsgraph.detect_scenes(job, progress=lambda i, n: log(None, i, n),
                                     cancelled=lambda: r.cancelled)
        log(f"{len(cuts) + 1} scenes found.")
        return {"cuts": cuts, "start": job["start"], "end": job["end"]}
    TASKS.start(rid, fn)


def m_locate(p, rid):
    """Preview frame that shows a given source frame (seek to a scene)."""
    def run():
        try:
            import vsgraph
            job = make_job(p["state"], int(p.get("episode", 0)))
            if job.get("mode") == "vfr" and not runner.job_is_pal(job) and runner.JobRunner.has_pass1(job):
                job["preview_vfr"] = True
            emit({"id": rid, "result": {"frame": vsgraph.locate_source_frame(job, int(p["src"]))}})
        except Exception as e:                            # noqa: BLE001
            fail(rid, "Locate: ", e)
    threading.Thread(target=run, daemon=True).start()


class AutoVfr:
    """VFR analysis (pass 1) for the preview, started by itself as soon as an episode in VFR mode is previewed,
    in the background (own runner, does not block other tasks). The preview shows CFR until it is done, then
    the export's own frames and timing. Settings that change the analysis start it again."""

    def __init__(self):
        self.lock = threading.Lock()
        self.sig = None
        self.runner = None
        self.thread = None
        self.progress = 0.0

    def request(self, st, k, start=True):
        """start=False: only say how it stands (the analysis runs when the user presses Analyze, not by itself)"""
        job = make_job(st, k)
        if job.get("mode") != "vfr" or runner.job_is_pal(job):
            # CFR / 60p / PAL / progressive (or switched to it while the analysis ran: a new file opens in VFR and it
            # started at the first preview) - no VFR analysis: a running one is stopped instead of finishing for nothing
            self.stop()
            return {"state": "off"}
        if runner.JobRunner.has_pass1(job):
            return {"state": "ready"}
        sig = runner.JobRunner.pass1_signature(job) + job["work"]
        with self.lock:
            if self.sig == sig and self.thread and self.thread.is_alive():
                return {"state": "running", "progress": self.progress}
            if not start:
                return {"state": "idle"}
            self._stop()
            self.sig, self.progress, self.k = sig, 0.0, k
            self.runner = r = runner.JobRunner(self._log)
            self.thread = threading.Thread(target=self.run, args=(r, job, k, sig), daemon=True)
            self.thread.start()
        return {"state": "running", "progress": 0.0}

    def _log(self, msg, cur=None, tot=None):
        if msg is None and tot:
            p = cur / tot
            if p - self.progress >= 0.01 or p >= 1:
                self.progress = p
                emit({"event": "vfr_auto", "state": "running", "progress": round(p, 3), "episode": getattr(self, "k", 0)})

    def run(self, r, job, k, sig):
        try:
            r.ensure_pass1(job)
            emit({"event": "vfr_auto", "state": "ready", "episode": k})
        except runner.Cancelled:
            pass
        except Exception as e:                            # noqa: BLE001
            if not r.cancelled:
                log(f"VFR analysis for the preview failed: {e}")
                emit({"event": "vfr_auto", "state": "failed", "episode": k})

    def _stop(self):
        if self.runner:
            self.runner.cancel()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
        self.sig = None

    def stop(self):
        with self.lock:
            self._stop()


AUTOVFR = AutoVfr()


def m_vfr_auto(p):
    return AUTOVFR.request(p["state"], int(p.get("episode", 0)), bool(p.get("start", False)))


def m_play(p, rid):
    PLAYER.start(rid, p["state"], int(p.get("episode", 0)), int(p.get("frame", 0)), p.get("views") or ["after"])


def m_stop_play(p):
    PLAYER.stop()
    return True


SYNC = {"presets": m_presets, "preset_get": m_preset_get, "preset_save": m_preset_save, "preset_delete": m_preset_delete,
        "hello": m_hello, "save_settings": m_save_settings, "probe": m_probe, "dvd_titles": m_dvd_titles,
        "cancel": m_cancel, "stop_play": m_stop_play, "lut_info": m_lut_info, "vfr_auto": m_vfr_auto,
        "temp_info": m_temp_info, "temp_clear": m_temp_clear, "script": m_script, "block_library": m_block_library, "ai_models": m_ai_models, "my_filter_add": m_my_filter_add, "my_nodes": m_my_nodes, "my_node_save": m_my_node_save, "my_node_delete": m_my_node_delete, "model_info": m_model_info, "vsview_update": m_vsview_update, "chain_default": m_chain_default, "toolbox_info": m_toolbox_info, "save_recipe": m_save_recipe, "check_script": m_check_script, "script_api": m_script_api, "make_report": m_make_report, "ai_engines": m_ai_engines, "ai_engine_pick": m_ai_engine_pick, "ai_engine_delete": m_ai_engine_delete, "env": lambda p: env_info()}
ASYNC = {"build_ai_engine": m_build_ai_engine, "import": m_import, "analyze": m_analyze, "autodetect": m_autodetect, "detect": m_detect, "report": m_report, "process": m_process,
         "vsview": m_vsview, "preview": m_preview, "play": m_play, "vfr_pass": m_vfr_pass, "scopes": m_scopes, "thumbs": m_thumbs, "scan_picture": m_scan_picture, "filter_preview": m_filter_preview, "media_tracks": m_media_tracks,
         "scenes": m_scenes, "locate": m_locate, "toolbox_update": m_toolbox_update, "toolbox_install": m_toolbox_install,
         "toolbox_rescan": m_toolbox_rescan}


def main():
    global PREVIEW
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:                                   # noqa: BLE001
        pass
    PREVIEW = Preview()
    threading.Thread(target=_memory_keeper, daemon=True).start()
    try:                                                 # CUDA + TensorRT started here, on the main thread
        import nnbackend
        nnbackend.warmup()
    except Exception:                                    # noqa: BLE001
        pass
    emit({"event": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        rid, method, params = req.get("id"), req.get("method"), req.get("params") or {}
        LAST_USE[0] = time.time()
        try:
            if method in SYNC:
                emit({"id": rid, "result": SYNC[method](params)})
            elif method in ASYNC:
                ASYNC[method](params, rid)
            else:
                emit({"id": rid, "error": f"unknown method {method}"})
        except Exception as e:                         # noqa: BLE001
            fail(rid, "", e)
    # stdin closed = the app quit or crashed: end playback and any running job (vspipe / ffmpeg), then exit hard -
    # worker threads must not keep this process (and its RAM / GPU memory) alive in the background
    for stop in (lambda: PLAYER.stop(), lambda: TASKS.cancel()):
        try:
            stop()
        except Exception:                              # noqa: BLE001
            pass
    os._exit(0)


if __name__ == "__main__":
    main()
