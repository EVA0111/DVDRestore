"""Signature of a VFR analysis (pass 1: tfm.txt / tdec.txt / timecodes.txt in the work folder), shared by the runner
(which reruns pass 1 when it changed) and the preview (which must not use a pass 1 made for another clip: TFM refuses it
with "crc32 in input file does not match that of the current clip").
Covers everything that changes the clip TFM sees: source, range, field order, TFM thresholds, the pre-IVTC cleaning,
and the IVTC set-up (FieldNet's raw IVTC decides on a crawl-free copy, the other cleaners on the repaired source)."""
import json
from pathlib import Path

KEYS = ("order", "cthresh", "mi", "pp", "dupthresh", "vidthresh", "crawl", "rainbow", "composite", "comp_dims", "comp_refine",
        "ivtc_decide")


def signature(job):
    src = Path(job["source"])
    st = src.stat()
    raw = bool(job.get("raw_ivtc")) or job.get("ai_model") == "field"
    d = {"source": str(src.resolve()), "size": st.st_size, "mtime": int(st.st_mtime),
         "start": job["start"], "end": job["end"], **{k: job.get(k) for k in KEYS},
         "ivtc": "raw" if raw else "repair", "version": 9}   # 9: clock jumps closed (frame numbers follow the file clock without its jumps)
    try:                                                 # undecodable frames replaced by vsgraph.source change the clip
        from vsgraph import bad_frames                   # (only listed when there are any: other sources keep their pass 1)
        bad = bad_frames(job["source"])
        if bad:
            d["bad_frames"] = bad
        from vsgraph import _ai_cleaner, _stablenet_on   # the pre-IVTC crawl filter is left out when an AI cleaner
        if _ai_cleaner(job) or _stablenet_on(job):       # runs: TFM sees another clip (TDecimate "crc32 does not match")
            d["ai_pre"] = [bool(_ai_cleaner(job)), bool(_stablenet_on(job))]
        from vsgraph import vsf_at                       # your VapourSynth filters on the source change what TFM sees
        if vsf_at(job, "source"):
            d["vs_filters"] = vsf_at(job, "source")
    except Exception:                                    # noqa: BLE001
        pass
    return json.dumps(d, sort_keys=True)


def matches(job, work=None):
    """True when the work folder holds a pass 1 made for exactly this clip"""
    w = Path(work or job["work"])
    sig = w / "pass1.sig"
    try:
        return (all((w / f).exists() for f in ("tfm.txt", "tdec.txt")) and sig.exists()
                and sig.read_text(encoding="utf-8") == signature(job))
    except Exception:                                    # noqa: BLE001
        return False
