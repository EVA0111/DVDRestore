"""DVD Restore backend for the Electron app.

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
    return {
        **adv,
        "source": info["path"], "start": a, "end": b, "mode": st.get("mode", "vfr"),
        "fps_num": info["fps_num"], "fps_den": info["fps_den"], "order": st.get("order", "tff"),
        "crawl": st.get("crawl", "off"), "rainbow": st.get("rainbow", "off"), "composite": st.get("composite", "off"),
        # halo removal switched on by hand without a measurement: DeHalo_alpha's standard radius 2
        "ghost": "auto" if st.get("ghost") == "auto" else "off",
        "ghost_taps": st.get("ghost_taps") or [], "halo_rx": (st.get("halo_rx") or 2.0) if st.get("ghost") == "auto" else 0,
        "ghost_strength": float(st.get("ghost_strength", 1.0)), "halo_method": st.get("halo_method", "ai"),
        "deblock": st.get("deblock", "off"), "rainbow_fill": "on" if st.get("rainbow_fill") else "off",
        "ivtc_repair": st.get("ivtc_repair", "ai"), "chroma_unmix": st.get("chroma_unmix", "off"), "deblend": st.get("deblend", "off"), "aa": st.get("aa", "off"),
        "grade": dict(st.get("grade") or {}),
        # adjustment layers that touch this episode (source frames of the whole title), bottom first
        "grade_layers": [{"start": int(l["start"]), "end": int(l["end"]), "opacity": float(l.get("opacity", 1)), "grade": l.get("grade") or {}}
                         for l in st.get("layers") or [] if l.get("on", True) and int(l["end"]) > a and int(l["start"]) < b],
        "pal_mode": st.get("pal_mode", "auto"), "comb": st.get("comb", "nnedi3"), "ai_model": st.get("ai_model", "temporal"),
        "stab": st.get("stab", "off"), "stab_level": st.get("stab_level", "normal"),
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
        "src_w": info.get("width"), "src_h": info.get("height"), "progressive24": bool(info.get("progressive24")),
        "progressive": bool(info.get("progressive")),
    } | chain_job(st)


def chain_job(st):
    """Chain step: the blocks, and the job keys the VFR analysis / detectors read, taken from them (blocks.py)"""
    ch = st.get("chain")
    if not ch:
        return {}
    import blocks
    chain = [{"type": b["type"], "on": bool(b.get("on", True)), "p": dict(b.get("p") or {})}
             for b in ch if isinstance(b, dict) and b.get("type") in blocks.BLOCKS]
    return {"chain": chain, **blocks.chain_globals(chain)}


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
                emit({"id": rid, "error": str(e), "trace": traceback.format_exc()[-1500:]})
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


def preview_clip(st, k, view):
    """The preview chain for this state / episode / view, built once and reused (as vsview does): rebuilding
    it for every frame made Play run at 1.8 fps instead of 7-10."""
    import vsgraph
    job = make_job(st, k)
    if job.get("mode") == "vfr" and not runner.job_is_pal(job) and runner.JobRunner.has_pass1(job):
        job["preview_vfr"] = True                      # the export's own frames and timing
    script = st.get("script") if view == "after" else None
    key = json.dumps([job, view, script], sort_keys=True, default=str)
    with _CLIPS_LOCK:
        c = _CLIPS.pop(key, None)
        if c is not None:
            _CLIPS[key] = c                            # most recently used last: eviction drops the least used
    if c is None:
        c = script_preview(script, job) if script else vsgraph.preview(job, after=(view == "after"))
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
    if fmt == "PNG":
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
                msg = {"event": "play_frame", "rid": rid, "frame": n, "total": total}
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
            emit({"id": rid, "error": f"Play: {e}"})


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
                for view in views:                          # "before" and/or "after", same frame
                    clip = preview_clip(st, k, view)
                    if frame is not None:
                        n = min(clip.num_frames - 1, max(0, int(frame)))
                    else:
                        n = min(clip.num_frames - 1, max(0, int(frac * (clip.num_frames - 1))))
                    f = clip.get_frame(n)
                    if "_SrcN" in f.props:
                        out["src"] = int(f.props["_SrcN"])
                    out[view], w, h = frame_image(f)
                    rate = f.props["_DurationDen"] / f.props["_DurationNum"] if f.props.get("_DurationNum") else 0
                    out.update(frame=n, total=clip.num_frames, width=w, height=h, rate=round(rate, 3))
                out["vfr"] = runner.JobRunner.has_pass1(make_job(st, k)) if st.get("mode") == "vfr" else False
                emit({"id": rid, "result": out})
            except Exception as e:                     # noqa: BLE001
                emit({"id": rid, "error": f"Preview: {e}"})


PREVIEW = None


def episode_of(job):
    return {k: job[k] for k in vsgraph_mod().EPISODE_KEYS if k in job}


def vsgraph_mod():
    import vsgraph
    return vsgraph


def script_preview(text, job):
    """Preview of an edited script: its EPISODE block set to the previewed episode; VFR IVTC is shown as CFR
    (VFR needs the pass-1 analysis, which runs on export)."""
    g = vsgraph_mod()
    text = g.set_episode(text, episode_of(job)).replace("dr.ivtc_vfr(", "dr.ivtc_cfr(").replace("'mode': 'vfr'", "'mode': 'cfr'")
    return g.to_display(g.run_script(text, "edited script"), job)


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
    return {"options": opts, "missing": runner.missing_tools(), "settings": s}


def m_save_settings(p):
    SETTINGS.write_text(json.dumps(p.get("settings", {}), indent=1), encoding="utf-8")
    return True


def m_probe(p):
    info = runner.probe(p["path"])
    return {"info": info, "episodes": [list(e) for e in runner.detect_episodes(info)]}


def m_dvd_titles(p):
    return runner.dvd_titles(p["path"])


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


def m_report(p, rid):
    st = p["state"]
    job = detect_job(st, "report")
    dest = Path(job["output"]).parent
    TASKS.start(rid, lambda r: r.report(job, st, p.get("log", ""), dest))


def m_process(p, rid):
    st = p["state"]
    AUTOVFR.stop()
    jobs = [make_job(st, k) for k in st.get("selected") or []]
    text = st.get("script")
    if text:
        vsgraph_mod().set_episode(text, episode_of(jobs[0]))          # fail now if the block is missing

    def fn(r):
        done = []
        if text:
            log("Exporting with the edited script (Script step).")
        for n, job in enumerate(jobs, 1):
            log(f"=== {Path(job['output']).name}  ({n}/{len(jobs)}) ===")
            done.append(r.run(job, script=vsgraph_mod().set_episode(text, episode_of(job)) if text else None))
        log("All done.")
        return done
    TASKS.start(rid, fn)


def m_vsview_update(p):
    """the open vsview follows every change (it reloads itself, keeping its tab and frame)"""
    st = p["state"]
    job = make_job(st, (st.get("selected") or [0])[0])
    return {"updated": runner.update_vsview(job)}


def m_vsview(p, rid):
    st = p["state"]
    job = make_job(st, (st.get("selected") or [0])[0])

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


def m_check_script(p):
    """Run the edited script far enough to know it works: returns frames / size / rate or the error."""
    st = p["state"]
    job = make_job(st, int(p.get("episode", 0)))
    g = vsgraph_mod()
    try:
        text = g.set_episode(p["text"], episode_of(job)).replace("dr.ivtc_vfr(", "dr.ivtc_cfr(").replace("'mode': 'vfr'", "'mode': 'cfr'")
        c = g.run_script(text, "edited script")
        c.get_frame(min(10, c.num_frames - 1))
        return {"ok": True, "frames": c.num_frames, "width": c.width, "height": c.height,
                "format": c.format.name, "fps": f"{c.fps.numerator}/{c.fps.denominator}"}
    except Exception as e:                                # noqa: BLE001
        tb = traceback.format_exc()
        line = [l for l in tb.splitlines() if "edited script" in l]
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "where": line[-1].strip() if line else ""}


_LIBRARY = {}


def _scan_toolbox():
    """the EVA toolbox's compatibility scan (blocks_scan.py, ~10 s): every plugin / vs-jetpack function tried once"""
    p = subprocess.run([sys.executable, "-W", "ignore", str(APP_DIR / "blocks_scan.py")], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    last = [l for l in (p.stdout or "").splitlines() if l.startswith("done")]
    return last[-1] if last else (p.stderr or "").strip()[-300:]


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
            emit({"id": rid, "error": f"Scopes: {e}"})
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
            emit({"id": rid, "error": f"Locate: {e}"})
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

    def request(self, st, k):
        job = make_job(st, k)
        if job.get("mode") != "vfr" or runner.job_is_pal(job):
            return {"state": "off"}
        if runner.JobRunner.has_pass1(job):
            return {"state": "ready"}
        sig = runner.JobRunner.pass1_signature(job) + job["work"]
        with self.lock:
            if self.sig == sig and self.thread and self.thread.is_alive():
                return {"state": "running", "progress": self.progress}
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
    return AUTOVFR.request(p["state"], int(p.get("episode", 0)))


def m_play(p, rid):
    PLAYER.start(rid, p["state"], int(p.get("episode", 0)), int(p.get("frame", 0)), p.get("views") or ["after"])


def m_stop_play(p):
    PLAYER.stop()
    return True


SYNC = {"hello": m_hello, "save_settings": m_save_settings, "probe": m_probe, "dvd_titles": m_dvd_titles,
        "cancel": m_cancel, "stop_play": m_stop_play, "lut_info": m_lut_info, "vfr_auto": m_vfr_auto,
        "temp_info": m_temp_info, "temp_clear": m_temp_clear, "script": m_script, "block_library": m_block_library, "vsview_update": m_vsview_update, "chain_default": m_chain_default, "toolbox_info": m_toolbox_info, "save_recipe": m_save_recipe, "check_script": m_check_script, "env": lambda p: env_info()}
ASYNC = {"import": m_import, "analyze": m_analyze, "autodetect": m_autodetect, "detect": m_detect, "report": m_report, "process": m_process,
         "vsview": m_vsview, "preview": m_preview, "play": m_play, "vfr_pass": m_vfr_pass, "scopes": m_scopes,
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
        try:
            if method in SYNC:
                emit({"id": rid, "result": SYNC[method](params)})
            elif method in ASYNC:
                ASYNC[method](params, rid)
            else:
                emit({"id": rid, "error": f"unknown method {method}"})
        except Exception as e:                         # noqa: BLE001
            emit({"id": rid, "error": str(e), "trace": traceback.format_exc()[-1500:]})
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
