"""Export encoder settings: one table drives the GUI (which options each encoder has) and the ffmpeg command.

Job keys (all optional, defaults = the old fixed settings):
  encoder      x264 x265 svtav1 nvenc_h264 nvenc_hevc nvenc_av1 prores ffv1 utvideo
  crf          quality value (CRF / CQ; meaning per encoder, see "q")
  enc_rate     "q" (constant quality) or "bitrate"          enc_bitrate  kbit/s (bitrate mode)
  enc_preset   speed preset (per encoder; ProRes: profile)   enc_tune     tune (x264 / x265 / NVENC)
  enc_depth    8 or 10 bit                                   enc_chroma   420 / 422 / 444
  enc_keyint   keyframe interval in frames (0 = encoder default)
  enc_extra    extra ffmpeg arguments, added as typed (e.g. -x264-params aq-strength=0.8)
  audio_codec  copy / flac / aac / opus                      audio_bitrate kbit/s for aac / opus
The picture leaves VapourSynth already in the chosen depth and chroma (dithered down to 8 bit), so ffmpeg
does no conversion of its own.
"""
import shlex

import vapoursynth as vs

X264_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "placebo"]
NV_PRESETS = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
ENC = {
    "x264": dict(label="x264 (H.264)", codec="libx264", presets=X264_PRESETS, preset="slow",
                 tunes=["none", "animation", "film", "grain", "stillimage"], tune="animation",
                 depths=[8, 10], depth=10, chroma=["420", "422", "444"], q=(0, 51, 16, "CRF"), bitrate=True),
    "x265": dict(label="x265 (HEVC)", codec="libx265", presets=X264_PRESETS, preset="slow",
                 tunes=["none", "animation", "grain", "psnr", "ssim"], tune="none",
                 depths=[8, 10], depth=10, chroma=["420", "422", "444"], q=(0, 51, 16, "CRF"), bitrate=True),
    "svtav1": dict(label="SVT-AV1", codec="libsvtav1", presets=[str(i) for i in range(0, 14)], preset="4",
                   tunes=[], tune="", depths=[8, 10], depth=10, chroma=["420"], q=(1, 63, 24, "CRF"), bitrate=True),
    "nvenc_h264": dict(label="NVENC H.264 (GPU)", codec="h264_nvenc", presets=NV_PRESETS, preset="p6",
                       tunes=["hq", "uhq"], tune="hq", depths=[8], depth=8, chroma=["420", "444"], q=(0, 51, 19, "CQ"), bitrate=True),
    "nvenc_hevc": dict(label="NVENC HEVC (GPU)", codec="hevc_nvenc", presets=NV_PRESETS, preset="p6",
                       tunes=["hq", "uhq"], tune="hq", depths=[8, 10], depth=10, chroma=["420", "444"], q=(0, 51, 19, "CQ"), bitrate=True),
    "nvenc_av1": dict(label="NVENC AV1 (GPU)", codec="av1_nvenc", presets=NV_PRESETS, preset="p6",
                      tunes=["hq", "uhq"], tune="hq", depths=[8, 10], depth=10, chroma=["420"], q=(0, 51, 24, "CQ"), bitrate=True),
    "prores": dict(label="ProRes (edit)", codec="prores_ks", presets=["proxy", "lt", "standard", "hq", "4444", "4444xq"], preset="hq",
                   tunes=[], tune="", depths=[10], depth=10, chroma=["422", "444"], q=None, bitrate=False),
    "ffv1": dict(label="FFV1 lossless", codec="ffv1", presets=[], preset="", tunes=[], tune="",
                 depths=[8, 10], depth=10, chroma=["420", "422", "444"], q=None, bitrate=False),
    "utvideo": dict(label="Ut Video lossless (fast)", codec="utvideo", presets=[], preset="", tunes=[], tune="",
                    depths=[8], depth=8, chroma=["420", "422", "444"], q=None, bitrate=False),
}
AUDIO = {"copy": "Copy (untouched)", "flac": "FLAC (lossless)", "aac": "AAC", "opus": "Opus"}


def settings(job):
    """Effective, validated settings of the job."""
    key = job.get("encoder", "x264")
    e = ENC.get(key, ENC["x264"])
    pick = lambda v, allowed, d: v if v in allowed else d
    depth = int(job.get("enc_depth") or e["depth"])
    depth = depth if depth in e["depths"] else e["depth"]
    chroma = pick(str(job.get("enc_chroma") or "420"), e["chroma"], e["chroma"][0])
    preset = pick(str(job.get("enc_preset") or e["preset"]), e["presets"], e["preset"])
    if key == "prores" and preset.startswith("4444"):
        chroma = "444"
    tune = pick(str(job.get("enc_tune") or e["tune"]), e["tunes"], e["tune"])
    rate = "bitrate" if job.get("enc_rate") == "bitrate" and e["bitrate"] else "q"
    q = e["q"]
    crf = float(job.get("crf", q[2] if q else 0)) if q else None
    if q:
        crf = max(q[0], min(q[1], crf))
    return dict(encoder=key if key in ENC else "x264", depth=depth, chroma=chroma, preset=preset, tune=tune, rate=rate, crf=crf,
                bitrate=int(job.get("enc_bitrate") or 8000), keyint=int(job.get("enc_keyint") or 0),
                extra=str(job.get("enc_extra") or "").strip(),
                audio=job.get("audio_codec") if job.get("audio_codec") in AUDIO else "copy",
                audio_bitrate=int(job.get("audio_bitrate") or 192))


def vs_format(job):
    """VapourSynth output format for the chosen depth / chroma."""
    s = settings(job)
    fam = {("420", 8): vs.YUV420P8, ("422", 8): vs.YUV422P8, ("444", 8): vs.YUV444P8,
           ("420", 10): vs.YUV420P10, ("422", 10): vs.YUV422P10, ("444", 10): vs.YUV444P10}
    return fam[(s["chroma"], s["depth"])]


def pix_fmt(job):
    s = settings(job)
    return f"yuv{s['chroma']}p" + ("10le" if s["depth"] == 10 else "")


def video_args(job):
    s = settings(job)
    e = ENC[s["encoder"]]
    a = ["-c:v", e["codec"]]
    k = s["encoder"]
    if k in ("x264", "x265"):
        a += ["-preset", s["preset"]]
        if s["tune"] != "none":
            a += ["-tune", s["tune"]]
        a += ["-crf", f"{s['crf']:g}"] if s["rate"] == "q" else ["-b:v", f"{s['bitrate']}k"]
        if k == "x265":
            a += ["-x265-params", "log-level=error:aq-mode=3:no-sao=1:deblock=-1,-1"]
    elif k == "svtav1":
        a += ["-preset", s["preset"]]
        a += ["-crf", f"{int(s['crf'])}"] if s["rate"] == "q" else ["-b:v", f"{s['bitrate']}k"]
        a += ["-svtav1-params", "tune=0"]
    elif k.startswith("nvenc"):
        a += ["-preset", s["preset"], "-tune", s["tune"], "-rc", "vbr"]
        a += ["-cq", f"{int(s['crf'])}", "-b:v", "0"] if s["rate"] == "q" else ["-b:v", f"{s['bitrate']}k", "-maxrate", f"{int(s['bitrate'] * 1.5)}k"]
        if k == "nvenc_h264" and s["chroma"] == "444":
            a += ["-profile:v", "high444p"]
    elif k == "prores":
        a += ["-profile:v", str(e["presets"].index(s["preset"])), "-vendor", "apl0"]
    elif k == "ffv1":
        a += ["-level", "3", "-g", "1", "-slices", "16", "-slicecrc", "1"]
    if s["keyint"] > 0 and k not in ("ffv1", "utvideo", "prores"):
        a += ["-g", str(s["keyint"])]
    a += ["-pix_fmt", pix_fmt(job)]
    if s["extra"]:
        a += shlex.split(s["extra"], posix=True)
    return a


def audio_args(job):
    """None = keep the audio as cut (stream copy); else ffmpeg arguments to re-encode every audio track."""
    s = settings(job)
    if s["audio"] == "copy":
        return None
    if s["audio"] == "flac":
        return ["-c:a", "flac"]
    if s["audio"] == "aac":
        return ["-c:a", "aac", "-b:a", f"{s['audio_bitrate']}k"]
    return ["-c:a", "libopus", "-b:a", f"{s['audio_bitrate']}k"]


def describe(job):
    s = settings(job)
    e = ENC[s["encoder"]]
    q = (f"{e['q'][3]} {s['crf']:g}" if s["rate"] == "q" else f"{s['bitrate']} kbit/s") if e["q"] else "lossless" if s["encoder"] in ("ffv1", "utvideo") else s["preset"]
    return f"{e['label']}, {q}, {s['depth']}-bit {s['chroma'][0]}:{s['chroma'][1]}:{s['chroma'][2]}"


def gui_table():
    return {k: {kk: v for kk, v in e.items() if kk != "codec"} for k, e in ENC.items()}
