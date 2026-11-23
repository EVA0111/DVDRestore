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
    return json.dumps({"source": str(src.resolve()), "size": st.st_size, "mtime": int(st.st_mtime),
                       "start": job["start"], "end": job["end"], **{k: job.get(k) for k in KEYS},
                       "ivtc": "raw" if raw else "repair", "version": 4}, sort_keys=True)


def matches(job, work=None):
    """True when the work folder holds a pass 1 made for exactly this clip"""
    w = Path(work or job["work"])
    sig = w / "pass1.sig"
    try:
        return (all((w / f).exists() for f in ("tfm.txt", "tdec.txt")) and sig.exists()
                and sig.read_text(encoding="utf-8") == signature(job))
    except Exception:                                    # noqa: BLE001
        return False
