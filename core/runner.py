"""Job runner for EVA toolkit: external tools, probing, episode detection, ripping, encoding, muxing."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent          # core/ (the engine)
sys.path.insert(0, str(APP_DIR))
import paths  # noqa: E402
NO_WINDOW = 0x08000000 if os.name == "nt" else 0
FPS = 30000 / 1001


# ---------------------------------------------------------------- tools

def _find(name, extra):
    for p in extra:
        if p and Path(p).is_file():
            return str(p)
    return shutil.which(name)


def python_exe():
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        exe = exe.with_name("python.exe")
    return str(exe)


# the bundled copies in runtime/ first (portable version), then the usual install places, then PATH
TOOLS = {
    "ffmpeg": _find("ffmpeg", [paths.tool("ffmpeg", "ffmpeg.exe"), r"C:\ffmpeg\bin\ffmpeg.exe"]),
    "ffprobe": _find("ffprobe", [paths.tool("ffmpeg", "ffprobe.exe"), r"C:\ffmpeg\bin\ffprobe.exe"]),
    # the real vspipe binary inside the VapourSynth package first: it finds its Python by itself, so it keeps working
    # when the folder is moved (Scripts\vspipe.exe is a pip launcher with this Python's full path written into it)
    "vspipe": _find("vspipe", [Path(python_exe()).parent / "Lib" / "site-packages" / "vapoursynth" / "vspipe.exe",
                               Path(python_exe()).parent / "Scripts" / "vspipe.exe", Path(python_exe()).parent / "vspipe.exe"]),
    "mkvmerge": _find("mkvmerge", [paths.tool("mkvtoolnix", "mkvmerge.exe"), r"C:\Program Files\MKVToolNix\mkvmerge.exe"]),
    "vsview": _find("vsview", [Path(python_exe()).parent / "Scripts" / "vsview.exe"]),
}
try:                                                                  # vsview runs as "pythonw -m vsview" anyway
    import importlib.util as _iu
    if not TOOLS["vsview"] and _iu.find_spec("vsview"):
        TOOLS["vsview"] = "-m vsview"
except Exception:
    pass


LIVE = {}                                                             # the job file the open vsview watches


def update_vsview(job):
    """rewrite the job file the open vsview watches (it reloads itself, same tab and frame). Only when it changed."""
    jp = LIVE.get("json")
    if not jp or not os.path.exists(jp):
        return False
    txt = json.dumps(job, indent=1)
    try:
        if open(jp, encoding="utf-8").read() == txt:
            return False
    except OSError:
        pass
    tmp = jp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(txt)
    os.replace(tmp, jp)                                  # atomic: vsview never reads half a file
    return True


def open_in_vsview(job):
    """Write a preview script for the job and open it in vsview (detached)."""
    work = Path(job["work"])
    work.mkdir(parents=True, exist_ok=True)
    jp = work / "preview.json"
    jp.write_text(json.dumps(job, indent=1), encoding="utf-8")
    LIVE["json"] = str(jp)
    vpy = work / "preview.vpy"
    vpy.write_text(
        "import sys, json\n"
        f"sys.path.insert(0, {str(APP_DIR)!r})\n"
        "import vsgraph\n"
        f"with open({str(jp)!r}, encoding='utf-8') as f:\n    job = json.load(f)\n"
        "vsgraph.preview_outputs(job)\n"
        "import vsview_live\n"                                        # reloads itself when EVA toolkit changes the job
        f"vsview_live.watch({str(jp)!r}, {str(vpy)!r})\n", encoding="utf-8")
    if not TOOLS["vsview"]:
        raise RuntimeError("vsview is not installed - add it in the Dependencies window (Settings)")
    pyw = Path(python_exe()).with_name("pythonw.exe")                 # no console window
    if pyw.exists():
        cmd = [str(pyw), "-m", "vsview", str(vpy)]
    elif TOOLS["vsview"] == "-m vsview":
        cmd = [python_exe(), "-m", "vsview", str(vpy)]
    else:
        cmd = [TOOLS["vsview"], str(vpy)]
    subprocess.Popen(cmd, cwd=str(work), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, creationflags=0x00000008 | 0x00000200)
    return str(vpy)


def missing_tools():
    return [k for k, v in TOOLS.items() if not v]


def _run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                          creationflags=NO_WINDOW, **kw)


# ---------------------------------------------------------------- probing

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
IMAGE_FRAMES = 24          # a still is processed as a short clip of identical frames (detectors / temporal stages work as usual)


def is_image(path):
    return str(path).lower().endswith(IMAGE_EXT)


def probe_image(path):
    r = _run([TOOLS["ffprobe"], "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,codec_name,pix_fmt",
              "-of", "json", path])
    v = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    if not v.get("width"):
        raise RuntimeError("Cannot read this image")
    w, h = int(v["width"]), int(v["height"])
    pf = str(v.get("pix_fmt") or "")
    bits = 16 if "16" in pf or "48" in pf or "64" in pf else (10 if "10" in pf else 8)
    colour = "greyscale" if pf.startswith(("gray", "ya", "mono")) else ("palette" if pf == "pal8" else "colour")
    fmt = {"mjpeg": "JPEG", "png": "PNG", "tiff": "TIFF", "bmp": "BMP", "webp": "WebP"}.get(str(v.get("codec_name")), str(v.get("codec_name") or "").upper())
    img = {"format": fmt, "bits": bits, "colour": colour, "alpha": "a" in pf.replace("gray", "").replace("pal", "").replace("yuva", "a") and
           any(x in pf for x in ("rgba", "bgra", "argb", "abgr", "ya", "yuva", "gbrap")), "pix_fmt": pf, "bytes": os.path.getsize(path)}
    return {"path": path, "width": w, "height": h, "dar": w / h, "image_info": img, "duration": IMAGE_FRAMES * 1001 / 30000,
            "fps_num": 30000, "fps_den": 1001, "pal": False, "frames": IMAGE_FRAMES, "audio": [], "subs": [], "chapters": [],
            "progressive24": False, "progressive": True, "image": True, "order": "tff"}


def probe(path):
    """Return dict with duration, frames, dar, audio streams, chapters (start seconds)."""
    if is_image(path):
        return probe_image(path)
    r = _run([TOOLS["ffprobe"], "-v", "error", "-show_streams", "-show_chapters", "-show_format",
              "-of", "json", path])
    j = json.loads(r.stdout or "{}")
    v = next((s for s in j.get("streams", []) if s["codec_type"] == "video"), None)
    if not v:
        raise RuntimeError("No video stream found")
    dar = 4 / 3
    if v.get("display_aspect_ratio") and ":" in v["display_aspect_ratio"]:
        a, b = (float(x) for x in v["display_aspect_ratio"].split(":"))
        if b:
            dar = 16 / 9 if a / b > 1.5 else 4 / 3
    dur = float(j.get("format", {}).get("duration", 0))
    audio = [{"index": s["index"], "codec": s.get("codec_name"), "lang": s.get("tags", {}).get("language", "und"),
              "channels": s.get("channels")} for s in j.get("streams", []) if s["codec_type"] == "audio"]
    subs = [{"index": s["index"], "codec": s.get("codec_name"), "lang": s.get("tags", {}).get("language", "und")}
            for s in j.get("streams", []) if s["codec_type"] == "subtitle"]
    dur0 = float(j.get("format", {}).get("duration", 0) or 0)
    # a chapter mark at the very end (DVDs often have one: "chapter 4 - 0:07:38, length 0:00") starts nothing
    chapters = [float(c["start_time"]) for c in j.get("chapters", []) if not (dur0 and float(c["start_time"]) > dur0 - 0.5)]
    num, den = (int(x) for x in v.get("r_frame_rate", "30000/1001").split("/"))
    if v["height"] <= 576 and den and num / den > 45:
        # raw MPEG-2 streams (.m2v): ffprobe gives the FIELD rate (60000/1001, 50) - a DVD has at most 30 frames/s
        if abs(num / den - 50) < 0.5:
            num, den = 25, 1
        elif abs(num / den - 59.94) < 0.1:
            num, den = 30000, 1001
    pal = abs(num / den - 25) < 0.5
    if pal:
        num, den = 25, 1
    # Progressive sources (never telecined / interlaced: 23.976p / 24p rips, 50p / 60p, HD files) must not be inverse-
    # telecined or PAL-processed: the first 300 frames carry no interlace or pulldown flags, and it is not a DVD-standard
    # SD 29.97 / 25 stream (DVDs flag telecined frames as progressive; those keep the cadence analysis, which already
    # keeps real 30p and progressive PAL as they are). 23.976 with pulldown flags = soft telecine (engine sees 29.97i).
    # all cores: lossless captures (FFV1) decode ~6x faster (2.1 s -> 0.35 s on open); the flags are the same
    r2 = _run([TOOLS["ffprobe"], "-v", "error", "-threads", str(min(16, os.cpu_count() or 4)), "-select_streams", "v:0", "-read_intervals", "%+#300",
               "-show_entries", "frame=interlaced_frame,top_field_first,repeat_pict", "-of", "csv=p=0", path])
    rows = [[c for c in x.strip(", ").split(",") if c != ""] for x in (r2.stdout or "").split()]
    rows = [r for r in rows if r]
    # csv order is ffprobe's own (interlaced_frame, top_field_first, repeat_pict)
    clean = bool(rows) and all(all(c == "0" for c in r) for r in rows)
    il = [r for r in rows if len(r) >= 2 and r[0] == "1"]
    pulldown = any(len(r) >= 3 and r[2] != "0" for r in rows)   # repeat_pict: soft telecine (the engine applies it)
    dvd_std = v["height"] <= 576 and (abs(num / den - 30000 / 1001) < 0.01 or pal)
    progressive = clean and not dvd_std
    scan = None
    if not progressive and not pulldown and dvd_std and dur0 > 0:
        # an SD 29.97 / 25 file is not progressive by its flags alone (DVD streams flag telecined frames as progressive,
        # and progressive rips are often flagged interlaced): the picture decides. ffmpeg's idet over 3 x 200 frames:
        # real progressive files read 0-1.6 % combed (Desert Punk ep01, rips), telecined / interlaced 33-100 %
        # (Blue Dragon, clip.mkv, Dominion, PAL Sample.mpg). Soft telecine never gets here (pulldown flags): ffmpeg
        # ignores the flags and its frames read 0 % combed, but the engine applies the pulldown
        scan = picture_scan(path, dur0)
        if scan["frames"] >= 150 and scan["combed"] <= 0.05:          # progressive anime reads 1.4-1.6 % (fine lines)
            progressive = True
    progressive24 = progressive and abs(num / den - 24000 / 1001) < 0.01
    std_rates = (24000 / 1001, 24, 25, 30000 / 1001, 30, 50, 60000 / 1001, 60)
    fo = v.get("field_order", "tt")
    if len(il) >= 10 and len(il) >= 0.8 * len(rows):
        # a hard-interlaced stream: the frames' own flags decide (what a player shows): ffmpeg's MPEG-2 encoder writes a stream field order "tb"
        # (= bottom displayed first) for top-field-first frames, and that turned the whole IVTC into BFF. Soft-telecined
        # discs (Courage: 10 % interlaced-flagged frames, their flags follow the pulldown) keep the stream's value
        fo = "tt" if sum(r[1] == "1" for r in il) * 2 >= len(il) else "bb"
    if dur <= 0:
        # no container duration (raw .m2v / .vob stream): count the frames the engine will see (BestSource,
        # pulldown applied - the index is built now instead of at the first preview)
        try:
            import vapoursynth as vs
            n = vs.core.bs.VideoSource(path, rff=True).num_frames
            dur = n * den / num
        except Exception as e:                           # noqa: BLE001
            print(f"frame count failed: {e}", file=sys.stderr)
    frame_ms = None
    if progressive and not any(abs(num / den - r) < 0.01 for r in std_rates):
        # an average rate (17732/665): 24p and 30p parts. The frames' own times, as ms steps (~120 KB for a 24-min
        # episode), so chapters, episodes and every time shown are where they really are (the average rate put a
        # chapter 30 s into a 24p part ~4 s away)
        try:
            times = source_frame_times(path)
            if variable_rate(times):
                ms = [round(t * 1000) for t in times]
                frame_ms = [ms[0]] + [b - a for a, b in zip(ms, ms[1:])]
        except Exception as e:                           # noqa: BLE001
            print(f"frame times not read: {e}", file=sys.stderr)
    return {"path": path, "width": v["width"], "height": v["height"], "dar": dar, "duration": dur,
            "fps_num": num, "fps_den": den, "pal": pal, "frames": len(frame_ms) if frame_ms else int(round(dur * num / den)),
            "frame_ms": frame_ms,
            "audio": audio, "subs": subs, "chapters": chapters, "progressive24": progressive24, "progressive": progressive,
            # how it was decided ("flags" / "picture" + the combed share) and a hint that frame times vary (an
            # average rate such as 17732/665: 24p and 30p parts in one file - the export keeps each frame's time)
            "scan": ({"by": "picture", **scan} if scan else {"by": "flags"}),
            "vfr_hint": bool(progressive and not any(abs(num / den - r) < 0.01 for r in std_rates)),
            # ffprobe: tt/bb = top/bottom first; tb = top coded, BOTTOM displayed first; bt = the reverse
            "order": "bff" if fo in ("bb", "tb") else "tff"}


def picture_scan(path, dur, windows=(0.15, 0.45, 0.75), frames=200):
    """ffmpeg idet on a few windows of the file (in parallel): {"frames": frames it could judge, "combed": share of
    them that look interlaced}"""
    import concurrent.futures as cf
    ts = [dur * f for f in windows] if dur > 30 else [0.0]

    def one(t):
        r = subprocess.run([TOOLS["ffmpeg"], "-hide_banner", "-nostats", "-threads", "8", "-ss", f"{t:.3f}", "-i", path,
                            "-map", "0:v:0", "-vf", "idet", "-frames:v", str(frames), "-an", "-sn", "-f", "null", "-"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                           creationflags=NO_WINDOW)
        m = re.findall(r"Multi frame detection: TFF: *(\d+) BFF: *(\d+) Progressive: *(\d+)", r.stderr or "")
        return tuple(map(int, m[-1])) if m else (0, 0, 0)
    try:
        with cf.ThreadPoolExecutor(len(ts)) as ex:
            res = list(ex.map(one, ts))
    except Exception:                                    # noqa: BLE001
        return {"frames": 0, "combed": 1.0}
    il, pr = sum(r[0] + r[1] for r in res), sum(r[2] for r in res)
    return {"frames": il + pr, "combed": round(il / max(1, il + pr), 4)}


def _frame_secs(info):
    """frame start times (s, from the first frame) of a variable-rate source, or None (constant rate)"""
    ms = info.get("frame_ms")
    if not ms:
        return None
    out, c = [], 0
    for d in ms:
        c += d
        out.append(c)
    return [(x - out[0]) / 1000 for x in out]


def time_to_frame(info, t):
    """the frame shown at time t (s): the first frame that starts at or after t (half a ms of rounding allowed)"""
    secs = _frame_secs(info)
    if not secs:
        return int(round(t * info.get("fps_num", 30000) / info.get("fps_den", 1001)))
    import bisect
    return min(len(secs), bisect.bisect_left(secs, t - 0.0005))


def frame_to_time(info, f):
    secs = _frame_secs(info)
    if not secs:
        return f * info.get("fps_den", 1001) / info.get("fps_num", 30000)
    if f < len(secs):
        return secs[max(0, int(f))]
    step = secs[-1] - secs[-2] if len(secs) > 1 else 0.04
    return secs[-1] + (f - len(secs) + 1) * step


def source_frame_times(path):
    """Presentation times (s) of every video frame, in order, from the container (no decoding; ~1 s per episode)."""
    r = _run([TOOLS["ffprobe"], "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
              "-of", "csv=p=0", path])
    t = []
    for x in (r.stdout or "").split():
        try:
            t.append(float(x.strip(",")))
        except ValueError:
            pass
    return sorted(t)


def variable_rate(times):
    """True when the frame spacing really varies (24p and 30p parts), not just timestamp rounding"""
    if len(times) < 50:
        return False
    d = sorted(b - a for a, b in zip(times, times[1:]))
    med = d[len(d) // 2]
    off = sum(1 for x in d if abs(x - med) > 0.1 * med)
    return off > 0.02 * len(d)


def detect_episodes(info):
    """Group chapters into episodes. An episode starts at a chapter whose length matches the first
    chapter (the opening), that is followed by a long chapter (not an ending followed by a short
    preview), at least 15 minutes after the previous episode start.
    Where both rules see the same boundary, an opening of the same length as the disc's first chapter wins:
    Idaten Jump has ending, 30 s corner, 30 s preview, 11 s card, then the 90 s opening - the "ending +
    short chapter" rule had started the next episode right after the corner."""
    ch, total = info["chapters"], info["duration"]
    if len(ch) < 2:
        return [(0, info["frames"])]
    lens = [b - a for a, b in zip(ch, ch[1:] + [total])]
    real = [i for i in range(len(ch)) if lens[i] >= 2.0]          # ignore tiny navigation chapters

    def after(i, k=1):                                            # k-th real chapter after chapter i
        nxt = [j for j in real if j > i]
        return nxt[k - 1] if len(nxt) >= k else None

    cands = []                                                    # (time, priority): 0 = same length as the first opening
    op0 = lens[0] if 55 <= lens[0] <= 160 else None
    for pos, i in enumerate(real):
        if i == 0:
            continue
        n1, n2 = after(i), after(i, 2)
        L = lambda j: lens[j] if j is not None else 0
        # rule A - opening: 1-2.5 min chapter followed by a long story chapter (maybe after a short title card)
        if 55 <= lens[i] <= 160 and (L(n1) > 180 or (L(n1) < 60 and L(n2) > 180)):
            cands.append((ch[i], 0 if op0 and abs(lens[i] - op0) <= 3 else 1))
        # rule B - after an ending (1-2.5 min) + short preview (< 45 s) a new episode starts
        if pos >= 2:
            ed, pv = real[pos - 2], real[pos - 1]
            if 55 <= lens[ed] <= 160 and lens[pv] < 45:
                cands.append((ch[real[pos - 1] + 1] if real[pos - 1] + 1 < len(ch) else ch[i], 2))
    starts, prio = [0.0], [-1]
    for t, pr in sorted(cands):
        if t - starts[-1] > 15 * 60 and total - t > 10 * 60:
            starts.append(t); prio.append(pr)
        elif 0 < t - starts[-1] < 4 * 60 and len(starts) > 1:     # same boundary seen by both rules
            if pr < prio[-1] or (pr == prio[-1] and t < starts[-1]):
                starts[-1], prio[-1] = t, pr                      # the opening match wins, else the earliest
    frames = [time_to_frame(info, t) for t in starts] + [info["frames"]]
    return [(a, b) for a, b in zip(frames, frames[1:]) if b - a > 0]


# ---------------------------------------------------------------- audio

def _pcm(path, t, dur, sr=48000):
    import numpy as np
    if t < 5:
        # near the file start: decode from 0 with the gap before the first audio sample filled (first_pts=0), then
        # cut - seeking before the input starts at the first REAL sample, so with audio starting 86 ms in (01-01)
        # a read at 0 and one at 0.07 s were 0-86 ms apart and a stream-copied track read "+68 ms out of sync"
        cmd = [TOOLS["ffmpeg"], "-v", "error", "-i", path, "-map", "0:a:0", "-af", "aresample=first_pts=0",
               "-ss", f"{max(t, 0):.6f}", "-t", f"{dur:.6f}", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    else:
        # a fast seek to 2 s before, then cut on the real audio timestamps (-copyts + atrim): seeking straight to t
        # landed on the 32 ms AC3 frame grid in one of the two files, and the sync check read up to -28 ms on an
        # export that was exactly in sync (0.0 ms at every point when measured sample-exactly)
        cmd = [TOOLS["ffmpeg"], "-v", "error", "-ss", f"{max(0.0, t - 2):.6f}", "-copyts", "-i", path, "-map", "0:a:0",
               "-af", f"atrim=start={t:.6f}:duration={dur:.6f},asetpts=PTS-STARTPTS",
               "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    r = subprocess.run(cmd,
                       capture_output=True, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    return np.frombuffer(r.stdout, dtype=np.float32)


def audio_lag(src, src_t, out, out_t, sr=48000, window=4.0, search=1.0):
    """Cross-correlate the output's audio at out_t with the source's audio at src_t.
    Returns (lag_ms, match): lag > 0 = output audio late. None if the window is silent."""
    import numpy as np
    a = _pcm(out, out_t, window, sr)
    if len(a) < sr or np.sqrt(np.mean(a ** 2)) < 0.003:
        return None
    # the source window starts `pre` seconds before src_t - never before 0: a short clip's first point (01-01, 3.5 s)
    # asked for 1 s before the start, got the start, and read "+1000 ms out of sync"
    b0 = max(0.0, src_t - search)
    pre = src_t - b0
    b = _pcm(src, b0, len(a) / sr + pre + search, sr)
    if len(b) < len(a):
        return None
    a = a - a.mean()
    n = len(a)
    N = 1 << (len(b) + n).bit_length()
    c = np.fft.irfft(np.fft.rfft(b - b.mean(), N) * np.conj(np.fft.rfft(a, N)), N)[:len(b) - n + 1]
    k = int(np.argmax(c))
    seg = b[k:k + n] - b[k:k + n].mean()
    match = float(c[k] / (np.linalg.norm(a) * np.linalg.norm(seg) + 1e-12))
    # ambiguity check: periodic sound (a steady tone, music drone) correlates almost as well at other offsets
    guard = int(0.005 * sr)
    others = np.concatenate([c[:max(0, k - guard)], c[k + guard + 1:]])
    if others.size and others.max() > 0.9 * c[k]:
        return None
    return (pre - k / sr) * 1000, match


def cut_audio(src, start_s, dur_s, out_path):
    """Stream-copy the episode's audio (bit-exact, no re-encode).
    -ss AFTER -i (output seeking): ffmpeg keeps the AC3 blocks from the episode start and gives the
    first block its true offset (0-32 ms), so the audio lands on the right millisecond when muxed.
    (-ss before -i jumps to an index point and was ~0.5 s early on DVD rips.)"""
    out_path = Path(out_path)
    out_path.unlink(missing_ok=True)
    # all audio tracks + all subtitle tracks, original codecs, original languages/titles
    r = _run([TOOLS["ffmpeg"], "-hide_banner", "-v", "error", "-y", "-i", src, "-ss", f"{start_s:.6f}",
              "-t", f"{dur_s:.6f}", "-map", "0:a?", "-map", "0:s?", "-c", "copy",
              str(out_path)])
    if r.returncode != 0:     # some subtitle formats refuse to be cut -> retry with audio only
        r = _run([TOOLS["ffmpeg"], "-hide_banner", "-v", "error", "-y", "-i", src, "-ss", f"{start_s:.6f}",
                  "-t", f"{dur_s:.6f}", "-map", "0:a?", "-c", "copy", str(out_path)])
    return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def verify_sync(src, start_s, out, dur_s):
    """Measure audio sync of the finished file against the source at several points."""
    res = []
    for frac in (0.02, 0.25, 0.5, 0.75, 0.97):
        t = frac * dur_s
        for dt in (0, 5, 10, 20, 40):                 # skip silent / ambiguous windows
            win = min(4.0, dur_s - (t + dt) - 0.05)    # the window must lie inside the clip (short clips)
            if win < 1.0:
                break
            m = audio_lag(src, start_s + t + dt, out, t + dt, window=win)
            if m and m[1] > 0.9:
                res.append(m[0])
                break
    return res


# ---------------------------------------------------------------- ISO

def dvd_root(path):
    """An .iso image, a VIDEO_TS folder or the disc folder that contains VIDEO_TS (read by ffmpeg's dvdvideo)."""
    p = Path(path)
    if p.is_dir() and p.name.upper() != "VIDEO_TS" and (p / "VIDEO_TS").is_dir():
        p = p / "VIDEO_TS"
    return str(p)


def is_dvd(path):
    p = Path(path)
    return p.suffix.lower() == ".iso" or (p.is_dir() and (p.name.upper() == "VIDEO_TS" or (p / "VIDEO_TS").is_dir()))


_DVDVIDEO = None


def _has_dvdvideo():
    """ffmpeg's DVD-Video demuxer is only in builds with libdvdnav / libdvdread (gyan.dev "full", not "essentials")"""
    global _DVDVIDEO
    if _DVDVIDEO is None:
        r = _run([TOOLS["ffprobe"], "-hide_banner", "-demuxers"])
        _DVDVIDEO = bool(re.search(r"^\s*D\S*\s+dvdvideo\b", r.stdout or "", re.M))
    return _DVDVIDEO


def dvd_titles(path, max_titles=99):
    """List the disc's titles: length, chapters, audio and subtitle languages. Read directly from the IFO
    files by ffmpeg's DVD-Video demuxer (libdvdnav) - no ripping software needed."""
    root, titles = dvd_root(path), []
    if not _has_dvdvideo():
        raise RuntimeError("This ffmpeg cannot read DVDs (it has no DVD-Video reader, libdvdnav). Open Dependencies "
                           "and install ffmpeg again, or use a 'full' ffmpeg build.")
    for t in range(1, max_titles + 1):
        r = _run([TOOLS["ffprobe"], "-v", "error", "-f", "dvdvideo", "-title", str(t), "-show_format", "-show_chapters",
                  "-show_streams", "-of", "json", "-i", root])
        if r.returncode:
            if "not found" in r.stderr or t > 1 and not titles:
                break
            continue                                   # a broken title: skip it, keep looking
        j = json.loads(r.stdout or "{}")
        st = j.get("streams", [])
        v = next((x for x in st if x["codec_type"] == "video"), {})
        lang = lambda x: x.get("tags", {}).get("language", "und")
        titles.append({"id": t, "duration": float(j.get("format", {}).get("duration", 0)),
                       "chapters": len(j.get("chapters", [])),
                       "video": f'{v.get("width", "?")}x{v.get("height", "?")} {v.get("r_frame_rate", "")}',
                       "audio": [f'{lang(x)} {x.get("codec_name", "")} {x.get("channels", "")}ch'
                                 for x in st if x["codec_type"] == "audio"],
                       "subs": [lang(x) for x in st if x["codec_type"] == "subtitle"]})
    return titles


def title_audio_args(root, title):
    """Codec options per audio track for the import. Everything is stream-copied, except DVD LPCM (pcm_dvd):
    Matroska cannot store it as it is ("No wav codec tag found for codec pcm_dvd"), so it becomes FLAC -
    lossless, every sample identical, and smaller."""
    r = _run([TOOLS["ffprobe"], "-v", "error", "-f", "dvdvideo", "-title", str(title), "-show_streams",
              "-of", "json", "-i", root])
    try:
        st = json.loads(r.stdout or "{}").get("streams", [])
    except ValueError:
        return [], 0
    aud = [x for x in st if x.get("codec_type") == "audio"]
    args = []
    for k, x in enumerate(aud):
        if x.get("codec_name") == "pcm_dvd":
            args += [f"-c:a:{k}", "flac"]
    return args, len(args) // 2


def import_dest(path, title):
    """<disc name>_import	itle_NN.mkv next to the disc."""
    p = Path(dvd_root(path))
    disc = p.parent if p.name.upper() == "VIDEO_TS" else p.with_suffix("")
    if disc.name.upper() in ("", "VIDEO_TS"):
        disc = p.parent
    return disc.parent / f"{disc.name}_import" / f"title_{int(title):02d}.mkv"


# ---------------------------------------------------------------- Blu-ray (BDMV folder)

def bd_root(path):
    """the BDMV folder of a disc folder, the BDMV folder itself, or the one a .mpls playlist sits in; None = not a Blu-ray"""
    p = Path(path)
    if p.suffix.lower() == ".mpls":
        p = p.parent.parent
    if p.is_dir() and p.name.upper() != "BDMV" and (p / "BDMV" / "PLAYLIST").is_dir():
        p = p / "BDMV"
    return str(p) if p.is_dir() and p.name.upper() == "BDMV" and (p / "PLAYLIST").is_dir() else None


def is_bluray(path):
    return bd_root(path) is not None


def _mpls_items(mp):
    """[(clip name, in, out seconds)] of a .mpls playlist, read directly (mkvmerge does not open every playlist:
    Angel's Egg's main feature 00100.mpls joins nine clips and is "not recognized")"""
    import struct
    b = Path(mp).read_bytes()
    if b[:4] != b"MPLS":
        return []
    pl = struct.unpack(">I", b[8:12])[0]
    n = struct.unpack(">H", b[pl + 6:pl + 8])[0]
    pos, out = pl + 10, []
    for _ in range(n):
        ln = struct.unpack(">H", b[pos:pos + 2])[0]
        name = b[pos + 2:pos + 7].decode("ascii", "replace")
        t_in, t_out = struct.unpack(">II", b[pos + 14:pos + 22])
        out.append((name, t_in / 45000, t_out / 45000))
        pos += 2 + ln
    return out


def bd_titles(path, min_seconds=30):
    """the disc's playlists (main feature, episodes, extras) with length, chapters, languages - read by mkvmerge.
    Playlists that play the same clips (the same title for other menus) and short ones (menus, logos) are left out."""
    root = Path(bd_root(path))
    titles, seen = [], set()
    for mp in sorted((root / "PLAYLIST").glob("*.mpls")):
        r = _run([TOOLS["mkvmerge"], "-J", str(mp)])
        try:
            j = json.loads(r.stdout or "{}")
        except ValueError:
            continue
        c = j.get("container", {})
        pr = c.get("properties", {})
        dur = float(pr.get("playlist_duration", 0)) / 1e9
        files = tuple(Path(f).name.lower() for f in pr.get("playlist_file", []))
        joined = False
        if not c.get("recognized"):                        # read the playlist ourselves; imported by joining its clips
            try:
                items = _mpls_items(mp)
            except Exception:                              # noqa: BLE001
                items = []
            files = tuple(f"{nm.lower()}.m2ts" for nm, _, _ in items)
            dur = sum(b - a for _, a, b in items)
            if not items or not all((root / "STREAM" / f).is_file() for f in files):
                continue
            r = _run([TOOLS["mkvmerge"], "-J", str(root / "STREAM" / files[0])])
            try:
                j = json.loads(r.stdout or "{}")
            except ValueError:
                j = {}
            pr, joined = {"playlist_chapters": len(files)}, True
        if dur < min_seconds or (files, round(dur)) in seen:
            continue
        seen.add((files, round(dur)))
        tr = j.get("tracks", [])
        v = next((t for t in tr if t.get("type") == "video"), {})
        lang = lambda t: t.get("properties", {}).get("language") or "und"
        titles.append({"id": int(mp.stem) if mp.stem.isdigit() else mp.stem, "playlist": str(mp), "duration": dur,
                       "chapters": int(pr.get("playlist_chapters", 0) or 0),
                       "video": f'{v.get("properties", {}).get("pixel_dimensions", "?")} {v.get("codec", "")}',
                       "audio": [f'{lang(t)} {t.get("codec", "")}' for t in tr if t.get("type") == "audio"],
                       "subs": [lang(t) for t in tr if t.get("type") == "subtitles"], "clips": len(files),
                       "joined": [str(root / "STREAM" / f) for f in files] if joined else None})
    titles.sort(key=lambda t: -t["duration"])
    return titles


def bd_import_dest(path, title):
    root = Path(bd_root(path))
    disc = root.parent
    return disc.parent / f"{disc.name}_import" / f"playlist_{title}.mkv"


def disc_titles(path):
    """DVD (ISO / VIDEO_TS) or Blu-ray (BDMV) titles"""
    return bd_titles(path) if is_bluray(path) else dvd_titles(path)


# ---------------------------------------------------------------- jobs

def job_is_pal(job):
    return abs(job.get("fps_num", 30000) / job.get("fps_den", 1001) - 25) < 0.5


class Cancelled(Exception):
    pass


class JobRunner:
    """Runs one episode job. log(msg) for text, log(None, cur, total) for progress."""

    def __init__(self, log):
        self.log = log
        self.proc = []
        self.cancelled = False

    def cancel(self):
        self.cancelled = True
        for p in self.proc:
            try:
                p.kill()
            except Exception:
                pass

    def _check(self):
        if self.cancelled:
            raise Cancelled()

    def _stream(self, p, pattern):
        """Read a process stream (split on \\r and \\n) and report progress lines."""
        buf = ""
        stream = p.stderr if pattern == "vspipe" else p.stdout
        while True:
            ch = stream.read(1)
            if not ch:
                break
            if ch in "\r\n":
                line, buf = buf, ""
                if pattern == "vspipe":
                    m = re.match(r"Frame: (\d+)/(\d+)", line)
                    if m:
                        self.log(None, int(m[1]), int(m[2]))
                    elif line.strip() and "API 3" not in line:
                        self.log(line.strip())
                else:
                    m = re.match(r"PROGRESS (\d+) (\d+)", line)
                    if m:
                        self.log(None, int(m[1]), int(m[2]))
                    elif line.startswith("RESULT "):
                        return json.loads(line[7:])
                    elif line.strip() and "API 3" not in line:
                        self.log(line.strip())
            else:
                buf += ch
        return None

    def _py(self, cmd, job_path):
        p = subprocess.Popen([python_exe(), "-I", "-W", "ignore", str(APP_DIR / "vsgraph.py"), cmd, job_path],
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
        self.proc.append(p)
        res = self._stream(p, "py")
        if p.wait() != 0:
            self._check()
            raise RuntimeError(f"vsgraph {cmd} failed")
        return res

    def pal_detect(self, job):
        work = Path(job["work"])
        work.mkdir(parents=True, exist_ok=True)
        jp = work / "paldetect.json"
        jp.write_text(json.dumps(job), encoding="utf-8")
        return self._py("paldetect", str(jp))

    def detect(self, cmd, job):
        """Run one detector on its own: 'ghost' or 'crop'."""
        work = Path(job["work"])
        work.mkdir(parents=True, exist_ok=True)
        jp = work / f"{cmd}.json"
        jp.write_text(json.dumps(job), encoding="utf-8")
        return self._py(cmd, str(jp))

    def report(self, job, state, ui_log, dest_dir):
        """Collect a diagnostics bundle (zip) for tuning: measurements, sample frames, probe, settings,
        app log and two short untouched clips of the source."""
        import shutil
        import time
        import zipfile
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"EVAtoolkit_report_{Path(job['source']).stem}_{stamp}"
        dest_dir = Path(dest_dir)
        tmp = Path(job["work"]).with_name(".report_tmp") / name
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        job = dict(job, report_dir=str(tmp))
        self.log("Report 1/3: measuring and saving sample frames...")
        jp = tmp / "job.json"
        jp.write_text(json.dumps(job, indent=1), encoding="utf-8")
        self._py("report", str(jp))
        self._check()
        self.log("Report 2/3: source info and sample clips...")
        r = _run([TOOLS["ffprobe"], "-v", "error", "-show_streams", "-show_chapters", "-show_format", "-of", "json",
                  job["source"]])
        (tmp / "ffprobe.json").write_text(r.stdout or r.stderr or "", encoding="utf-8")
        st = {k: v for k, v in state.items() if k != "adv"} | {"adv": state.get("adv")}
        (tmp / "ui_state.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
        (tmp / "log.txt").write_text(ui_log or "", encoding="utf-8")
        for f in Path(job["work"]).parent.glob(".work_*/ffmpeg.log"):
            shutil.copy(f, tmp / f"ffmpeg_{f.parent.name.lstrip('.')}.log")
        dur = float(state.get("info", {}).get("duration") or 0)
        info = state.get("info") or {}
        if not info.get("frame_ms") and info.get("vfr_hint"):
            info = probe(job["source"])                   # the app does not send the frame-time table along
        info = {**info, "fps_num": job["fps_num"], "fps_den": job["fps_den"]}
        start_s = frame_to_time(info, job["start"])
        end_s = frame_to_time(info, job["end"]) if job.get("end") else dur
        for i, frac in enumerate((0.3, 0.7), 1):        # untouched video, stream copy (no re-encode)
            self._check()
            t = start_s + (end_s - start_s) * frac
            _run([TOOLS["ffmpeg"], "-v", "error", "-y", "-ss", f"{max(0, t):.2f}", "-i", job["source"], "-t", "8",
                  "-map", "0:v:0", "-c", "copy", str(tmp / f"clip{i}_{int(t)}s.mkv")])
        self.log("Report 3/3: zipping...")
        dest_dir.mkdir(parents=True, exist_ok=True)
        zpath = dest_dir / f"{name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(tmp.iterdir()):
                if f.name != "job.json":
                    z.write(f, f"{name}/{f.name}")
        shutil.rmtree(tmp.parent, ignore_errors=True)
        mb = zpath.stat().st_size / 1e6
        self.log(f"Report saved: {zpath}  ({mb:.1f} MB) - send this file.")
        return {"path": str(zpath), "mb": round(mb, 1)}

    def import_bd(self, path, title, duration=0.0):
        """Remux one Blu-ray playlist to MKV by stream copy (mkvmerge joins its clips, keeps every track, language and
        the chapters). Reuses an existing complete import."""
        out = bd_import_dest(path, title)
        out.parent.mkdir(parents=True, exist_ok=True)
        done = out.with_suffix(".done")
        if out.exists() and done.exists():
            self.log(f"Using existing import {out}")
            return str(out)
        mp = Path(bd_root(path)) / "PLAYLIST" / f"{int(title):05d}.mpls" if str(title).isdigit() else Path(str(title))
        tmp = out.with_suffix(".part.mkv")
        src = [str(mp)]
        try:
            items = _mpls_items(mp)
            r = _run([TOOLS["mkvmerge"], "-J", str(mp)])
            if items and not json.loads(r.stdout or "{}").get("container", {}).get("recognized"):
                clips = [str(Path(bd_root(path)) / "STREAM" / f"{nm}.m2ts") for nm, _, _ in items]
                src = [x for c in clips for x in ("+", c)][1:]   # a + b + c: the clips one after the other
                self.log(f"Playlist {mp.name} joins {len(clips)} clips: they are appended in its order")
        except Exception:                                  # noqa: BLE001
            pass
        self.log(f"Importing Blu-ray playlist {mp.name} (stream copy, nothing is re-encoded)...")
        p = subprocess.Popen([TOOLS["mkvmerge"], "--gui-mode", "-o", str(tmp), *src], stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                             creationflags=NO_WINDOW)
        self.proc.append(p)
        last = []
        for line in p.stdout:
            m = re.match(r"#GUI#progress (\d+)%", line.strip())
            if m:
                self.log(None, int(m.group(1)), 100)
            elif line.strip():
                last.append(line.strip())
        if p.wait() not in (0, 1):                        # 1 = warnings only
            self._check()
            tmp.unlink(missing_ok=True)
            raise RuntimeError("Blu-ray import failed: " + (last[-1] if last else "?"))
        self._check()
        tmp.replace(out)
        done.write_text("ok", encoding="utf-8")
        self.log(f"Imported: {out}")
        return str(out)

    def import_title(self, path, title, duration=0.0):
        """Remux one DVD title to MKV, bit-exact (stream copy): video with its pulldown flags, all audio and
        subtitle tracks with languages, accurate chapters (-preindex). Reuses an existing complete import."""
        if is_bluray(path):
            return self.import_bd(path, title, duration)
        out = import_dest(path, title)
        out.parent.mkdir(parents=True, exist_ok=True)
        done = out.with_suffix(".done")
        if out.exists() and done.exists():
            self.log(f"Using existing import {out}")
            return str(out)
        tmp = out.with_suffix(".part.mkv")
        aargs, n_lpcm = title_audio_args(dvd_root(path), title)
        self.log(f"Importing title {title} (stream copy, nothing is re-encoded)..." if not n_lpcm else
                 f"Importing title {title}: video and subtitles stream copy; {n_lpcm} LPCM audio track"
                 f"{'s' if n_lpcm > 1 else ''} to FLAC (lossless - MKV cannot hold DVD LPCM as it is)...")
        p = subprocess.Popen([TOOLS["ffmpeg"], "-v", "error", "-y", "-f", "dvdvideo", "-preindex", "1", "-title", str(title),
                              "-i", dvd_root(path), "-map", "0", "-c", "copy", *aargs, "-progress", "pipe:1", "-nostats", str(tmp)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
        self.proc.append(p)
        tot = max(1, int(duration))
        for line in p.stdout:
            if line.startswith("out_time_us="):
                try:
                    self.log(None, min(tot, int(line.split("=")[1]) // 1_000_000), tot)
                except ValueError:
                    pass
        err = p.stderr.read()
        if p.wait() != 0:
            self._check()
            tmp.unlink(missing_ok=True)
            raise RuntimeError("DVD import failed: " + (err.strip().splitlines() or ["?"])[-1])
        self._check()
        tmp.replace(out)
        done.write_text("ok", encoding="utf-8")
        self.log(f"Imported: {out}")
        return str(out)

    def measure(self, job):
        work = Path(job["work"])
        work.mkdir(parents=True, exist_ok=True)
        jp = work / "measure.json"
        jp.write_text(json.dumps(job), encoding="utf-8")
        return self._py("measure", str(jp))

    @staticmethod
    def pass1_signature(job):
        import pass1sig                                 # shared with the preview (vsgraph.preview_outputs)
        return pass1sig.signature(job)

    @classmethod
    def has_pass1(cls, job):
        w = Path(job["work"])
        sig = w / "pass1.sig"
        return (all((w / f).exists() for f in ("tfm.txt", "tdec.txt")) and sig.exists()
                and sig.read_text(encoding="utf-8") == cls.pass1_signature(job))

    def ensure_pass1(self, job):
        """Run the VFR analysis pass unless a matching one already exists in the work folder."""
        work = Path(job["work"])
        work.mkdir(parents=True, exist_ok=True)
        if self.has_pass1(job):
            self.log("Pass 1/2: reusing existing VFR analysis")
            return
        self.log("Pass 1/2: analysing cadence (VFR)...")
        for f in ("tfm.txt", "tdec.txt", "timecodes.txt", "pass1.sig", "diff.txt", "blk.txt", "loc.txt", "tdec_ovr.txt"):
            (work / f).unlink(missing_ok=True)
        jp = work / "pass1.json"
        jp.write_text(json.dumps(job, indent=1), encoding="utf-8")
        self._py("pass1", str(jp))
        self._check()
        (work / "pass1.sig").write_text(self.pass1_signature(job), encoding="utf-8")

    def run(self, job, script=None):
        """script: an edited VapourSynth script (Script step) to encode instead of the generated one."""
        work = Path(job["work"])
        work.mkdir(parents=True, exist_ok=True)
        out = Path(job["output"])
        out.parent.mkdir(parents=True, exist_ok=True)
        jp = work / "job.json"
        jp.write_text(json.dumps(job, indent=1), encoding="utf-8")
        vpy = work / "job.vpy"
        vpy.write_text(
            "import sys, json\n"
            "sys.stdout = sys.stderr   # stdout is the video pipe: a stray print would corrupt the Y4M stream\n"
            f"sys.path.insert(0, {str(APP_DIR)!r})\n"
            "import vsgraph\n"
            f"with open({str(jp)!r}, encoding='utf-8') as f:\n    job = json.load(f)\n"
            "vsgraph.build(job).set_output()\n", encoding="utf-8")
        if script:
            vpy.write_text(script, encoding="utf-8")

        if job.get("chain"):
            # Chain step: the output size and pixel shape come from the blocks - measure the finished chain
            gp = work / "geometry.json"                   # CFR: the size does not depend on the VFR timing (no pass 1 yet)
            gp.write_text(json.dumps(dict(job, mode="cfr"), indent=1), encoding="utf-8")
            geo = self._py("geometry", str(gp))
            w, h, sar = int(geo["width"]), int(geo["height"]), float(geo.get("sar", 1.0))
            job["out_w"], job["out_h"] = w, h
            job["anamorphic"] = abs(sar - 1.0) > 0.005
            job["display"] = [int(round(w * sar / 2)) * 2, h] if job["anamorphic"] else None
            self.log(f"Chain output: {w}x{h}" + (f", shown as {job['display'][0]}x{h}" if job["anamorphic"] else ", square pixels"))
            jp.write_text(json.dumps(job, indent=1), encoding="utf-8")
        if job.get("image"):
            # a still: the middle of its identical frames, as a 16-bit PNG (vsgraph.py image)
            self.log("Image: repairing and saving as PNG...")
            out_png = out.with_suffix(".png")
            job["output"] = str(out_png)
            jp.write_text(json.dumps(job, indent=1), encoding="utf-8")
            self._py("image", str(jp))
            self._check()
            if not out_png.is_file() or out_png.stat().st_size == 0:
                raise RuntimeError("The image was not written")
            self.log(f"Done: {out_png}")
            return
        pal = job_is_pal(job)
        # the probe estimates the frame count from the container's duration; a VOB's audio can run past the last
        # picture (VTS_02_1: 66392 estimated, 66348 decoded = 1.47 s): an episode ending "at the end" is clamped to
        # the frames that exist, so the length check and the audio cut use the real length (was: export failed)
        try:
            import vsgraph as _vgc
            n_real = _vgc.source({"source": job["source"]}, clean=False).num_frames
            if 0 < n_real < job["end"]:
                self.log(f"The source has {n_real} frames ({job['end'] - n_real} fewer than its length suggested): episode end adjusted")
                job["end"] = n_real
                jp.write_text(json.dumps(job, indent=1), encoding="utf-8")
        except Exception as e:                           # noqa: BLE001
            print(f"frame count not checked: {e}", file=sys.stderr)
        if pal and not job.get("progressive"):
            job["mode"] = "cfr"
            if job.get("pal_mode", "auto") == "auto" and not job.get("pal_mode_resolved"):
                self.log("PAL: detecting progressive / field-shifted / interlaced...")
                det = self.pal_detect(job)
                job["pal_mode_resolved"] = det["mode"]
                job["order"] = det["order"]
                self.log(f"PAL: {det['mode']}, field order {det['order'].upper()} (combed raw "
                         f"{det['raw_combed']:.0%}, after field matching {det['matched_combed']:.0%})")
            jp.write_text(json.dumps(job, indent=1), encoding="utf-8")

        if (job.get("progressive24") or job.get("progressive")) and job["mode"] == "vfr":
            job["mode"] = "cfr"                         # progressive source: nothing to inverse-telecine
            self.log(f"Progressive source ({job['fps_num'] / job['fps_den']:.3f} fps): no inverse telecine, frame rate kept")
            jp.write_text(json.dumps(job, indent=1), encoding="utf-8")
        import vsgraph as _vg
        own_ivtc = bool(_vg.vsf_at(job, "ivtc"))     # a filter group set to do the IVTC itself
        if own_ivtc and job["mode"] == "vfr":
            job["mode"] = "cfr"                         # the app's VFR analysis needs its own IVTC
            self.log("Your VapourSynth filters do the IVTC: exported with a constant frame rate (theirs)")
            jp.write_text(json.dumps(job, indent=1), encoding="utf-8")
        # 1. VFR analysis
        if job["mode"] == "vfr":
            self.ensure_pass1(job)

        # 2. video encode
        self.log(("Pass 2/2: " if job["mode"] == "vfr" else "") + "filtering + encoding video...")
        video = work / "video.mkv"
        hd = (job.get("out_h") or 480) >= 720
        if hd:
            col = trc = csp = "bt709"
        elif pal:
            col, trc, csp = "bt470bg", "smpte170m", "bt470bg"
        else:
            col = trc = csp = "smpte170m"
        import encoding
        enc = encoding.video_args(job)
        self.log("Encoder: " + encoding.describe(job))
        vs_p = subprocess.Popen([TOOLS["vspipe"], "-c", "y4m", "-p", str(vpy), "-"],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
        ff_log = work / "ffmpeg.log"                      # to a file: a full stderr pipe could stall the encode
        with open(ff_log, "w", encoding="utf-8", errors="replace") as ff_err_f:
            ff = subprocess.Popen([TOOLS["ffmpeg"], "-hide_banner", "-v", "error", "-y", "-i", "-", *enc,
                                   "-color_primaries", col, "-color_trc", trc,
                                   "-colorspace", csp, "-color_range", "tv",
                                   "-an", str(video)],
                                  stdin=vs_p.stdout, stderr=ff_err_f, creationflags=NO_WINDOW)
            vs_p.stdout.close()
            self.proc += [vs_p, ff]
            import io
            vs_p.stderr = io.TextIOWrapper(vs_p.stderr, encoding="utf-8", errors="replace")
            self._stream(vs_p, "vspipe")
            rc_vs, rc_ff = vs_p.wait(), ff.wait()
        if rc_vs != 0 or rc_ff != 0:
            self._check()
            raise RuntimeError("Encoding failed: " + ff_log.read_text(encoding="utf-8", errors="replace")[-500:])
        self._check()

        # 3. audio
        self.log("Cutting audio...")
        audio = work / "audio.mka"
        fps = job.get("fps_num", 30000) / job.get("fps_den", 1001)
        start_s = job["start"] / fps
        dur_s = (job["end"] - job["start"]) / fps
        src_tc = None
        if job.get("progressive") and not job.get("image") and job["mode"] != "vfr":
            # progressive with 24p and 30p parts (a VFR rip): a constant average rate (26.66 fps) put the picture
            # seconds ahead of / behind the audio inside the episode - every frame keeps its own time instead, and the
            # audio is cut at the real times of the first and last frame
            try:
                times = source_frame_times(job["source"])
                n_src = _vg.source({"source": job["source"]}, clean=False).num_frames   # the whole file (not the episode)
                if variable_rate(times) and len(times) != n_src:
                    self.log(f"Variable frame rate source, but its {len(times)} frame times do not match the {n_src} decoded "
                             "frames: exported at a constant rate")
                elif variable_rate(times) and 0 <= job["start"] < job["end"] <= n_src:
                    a, b = job["start"], job["end"]
                    t0 = times[0]
                    end_t = times[b] if b < len(times) else times[-1] + (times[-1] - times[-2])
                    start_s, dur_s = times[a] - t0, end_t - times[a]
                    src_tc = work / "timecodes_src.txt"
                    src_tc.write_text("# timecode format v2\n" + "".join(f"{(times[i] - times[a]) * 1000:.3f}\n" for i in range(a, b)),
                                      encoding="utf-8")
                    self.log(f"Variable frame rate source: each frame keeps its own time ({b - a} frames, {dur_s:.2f} s)")
            except Exception as e:                       # noqa: BLE001
                self.log(f"Frame times not read ({e}): constant rate")
        has_audio = cut_audio(job["source"], start_s, dur_s, audio)
        import encoding
        aa = encoding.audio_args(job)
        if has_audio and aa:
            self.log("Re-encoding audio: " + " ".join(aa))
            re_out = work / "audio_re.mka"
            r = _run([TOOLS["ffmpeg"], "-hide_banner", "-v", "error", "-y", "-i", str(audio), "-map", "0", "-c", "copy", *aa, str(re_out)])
            if r.returncode == 0 and re_out.exists() and re_out.stat().st_size > 0:
                os.replace(re_out, audio)
            else:
                self.log("WARNING: audio re-encode failed, keeping the original audio: " + r.stdout[-300:])

        # 3b. the video must have exactly the frames the timing is made for (an edited script that adds or
        # drops frames would otherwise drift against the audio)
        r = _run([TOOLS["ffprobe"], "-v", "error", "-select_streams", "v:0", "-count_packets",
                  "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(video)])
        try:
            n_video = int((r.stdout or "").strip().splitlines()[0])
        except (ValueError, IndexError):
            n_video = None
        if not n_video:
            # an empty video must never be muxed (01-01 with the FDAT upscale model: TensorRT's log on the video pipe,
            # ffmpeg found no stream, the episode came out as audio + subtitles only)
            raise RuntimeError("The encoder wrote no video frames - nothing was muxed. ffmpeg: "
                               + (ff_log.read_text(encoding="utf-8", errors="replace")[-400:] or "(no message)"))
        if n_video is not None:
            if job["mode"] == "vfr":
                with open(work / "timecodes.txt", encoding="utf-8") as f:
                    n_tc = sum(1 for l in f if l.strip() and not l.startswith("#"))
                if n_tc != n_video:
                    raise RuntimeError(
                        f"The video has {n_video} frames but the VFR timing has {n_tc}: the audio would drift. "
                        "Something in the script adds or removes frames (frame-rate change, interpolation, trim) - "
                        "remove it, or export in CFR mode.")
            elif src_tc is not None:
                if n_video != job["end"] - job["start"]:
                    raise RuntimeError(f"The video has {n_video} frames but the source part has {job['end'] - job['start']}: "
                                       "something in the script adds or removes frames, so the frame times cannot be kept.")
            elif own_ivtc:
                # your own IVTC sets the rate: measured from the frames it gave for the episode's length, snapped to the
                # standard rate it is (23.976 / 29.97 / 25 / 24 / 30 / 50 / 59.94 / 48 / 47.952 / 15 ...)
                # the probe's frame count is estimated from the duration (can be 1 too many): the frames that exist
                try:
                    n_src = min(job["end"], _vg.source(job, clean=False).num_frames) - job["start"]
                except Exception:                     # noqa: BLE001
                    n_src = job["end"] - job["start"]
                exact = n_video / max(n_src / fps, 1e-6)
                std = [(24000, 1001), (30000, 1001), (25, 1), (24, 1), (30, 1), (50, 1), (60000, 1001), (60, 1), (48, 1),
                       (48000, 1001), (12000, 1001), (15000, 1001), (12, 1), (15, 1)]
                num, den = min(std, key=lambda r: abs(r[0] / r[1] - exact))
                if abs(num / den - exact) / exact > 0.006:
                    from fractions import Fraction
                    fr = Fraction(exact).limit_denominator(1001)
                    num, den = fr.numerator, fr.denominator
                job["out_rate"] = [num, den]
                self.log(f"Your IVTC gives {n_video} frames for {dur_s:.2f} s: {num / den:.3f} fps")
            else:
                rate = (job["fps_num"] / job["fps_den"] if job.get("progressive") else
                        50.0 if pal and job.get("pal_mode_resolved", job.get("pal_mode")) == "deint50" else
                        (25.0 if pal else 60000 / 1001 if job["mode"] == "bob" else 24000 / 1001))
                diff = n_video / rate - dur_s
                if abs(diff) > 0.5:
                    raise RuntimeError(
                        f"The video lasts {n_video / rate:.2f} s at {rate:.3f} fps but the episode is {dur_s:.2f} s "
                        f"({diff:+.2f} s): the audio would drift. Something in the script changes the number of frames.")
            self.log(f"Frame count check: {n_video} frames - OK")

        # 4. mux
        self.log("Muxing...")
        m = [TOOLS["mkvmerge"], "-q", "-o", str(out)]
        if job["mode"] == "vfr":
            m += ["--timestamps", f"0:{work / 'timecodes.txt'}"]
        elif src_tc is not None:
            m += ["--timestamps", f"0:{src_tc}"]                                     # the source's own frame times
        elif job.get("out_rate"):
            m += ["--default-duration", f"0:{job['out_rate'][0]}/{job['out_rate'][1]}p"]   # your own IVTC's rate
        elif job.get("progressive"):
            m += ["--default-duration", f"0:{job['fps_num']}/{job['fps_den']}p"]        # the source's own rate
        elif pal:
            m += ["--default-duration", "0:50p" if job.get("pal_mode_resolved", job.get("pal_mode")) == "deint50"
                  else "0:25p"]
        elif job["mode"] == "bob":
            m += ["--default-duration", "0:60000/1001p"]                             # NTSC deinterlaced: 59.94p
        else:
            m += ["--default-duration", "0:24000/1001p"]
        if job.get("anamorphic"):
            dw, dh = job.get("display") or (0, 0)
            if dw and dh:                                  # exact display size of the (cropped) picture
                m += ["--display-dimensions", f"0:{dw}x{dh}"]
            else:
                m += ["--aspect-ratio", "0:16/9" if job.get("dar", 4 / 3) > 1.5 else "0:4/3"]
        mc, pc, tc = ("1", "1", "1") if hd else (("5", "5", "6") if pal else ("6", "6", "6"))
        m += ["--colour-matrix-coefficients", f"0:{mc}", "--colour-primaries", f"0:{pc}",
              "--colour-transfer-characteristics", f"0:{tc}", "--colour-range", "0:1"]
        m += ["--language", "0:und", str(video)]
        if has_audio:
            m += [str(audio)]                                   # keeps each track's language/title
        # fonts and other attachments (e.g. for ASS subtitles), chapters are not copied (episode cut)
        m += ["-A", "-D", "-S", "-B", "-T", "--no-chapters", "--no-global-tags", job["source"]]
        r = _run(m)
        if r.returncode not in (0, 1):
            raise RuntimeError("mkvmerge failed: " + r.stdout[-500:])
        if has_audio:
            lags = verify_sync(job["source"], start_s, str(out), dur_s)
            if lags:
                worst = max(lags, key=abs)
                self.log(f"Audio sync check ({len(lags)} points): " + ", ".join(f"{x:+.1f}" for x in lags)
                         + " ms" + ("  OK" if abs(worst) <= 2 else "  WARNING: out of sync!"))
            else:
                self.log("Audio sync check: inconclusive (silent or repetitive audio) - not an error")
        self.log(f"Done: {out}")
        if not job.get("keep_work"):
            for f in ("video.mkv", "audio.mka"):
                (work / f).unlink(missing_ok=True)
        return str(out)
