"""Compatibility scan for the Chain step's library: every installed VapourSynth plugin function and every vs-jetpack
function is run once with its defaults on a DVD-sized clip (YUV 4:2:0 8-bit, then YUV 4:4:4 16-bit), each in its own
process with a time limit (a filter that hangs or crashes cannot take the app down). Result per function:
  ok      runs with its defaults (and on which format)
  args    has required settings (no default): cannot be tried blind - fill them in
  error   fails on this system (version mismatch, missing model / plugin ...), with the message
Written to blocks_compat.json (with the VapourSynth / vs-jetpack versions it was made for); the library shows it.
    python blocks_scan.py            (whole scan, parallel)
    python blocks_scan.py one <kind> <call>   (internal: one function)"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "blocks_compat.json")
TIMEOUT = 90


def versions():
    import vapoursynth as vs
    try:
        from importlib.metadata import version
        jet = version("vsjetpack")
    except Exception:                                    # noqa: BLE001
        jet = ""
    return {"vapoursynth": str(vs.__version__), "vsjetpack": jet,
            "plugins": sorted(f"{p.namespace}:{p.version}" for p in vs.core.plugins())}


def one(kind, call, skip=()):
    """try the function on the DVD frame in every format the chain can hand it (and interlaced); record each that works
    - the block then converts only when the picture it gets is in none of them (Cnr4: integer YUV only)"""
    sys.path.insert(0, HERE)
    import vapoursynth as vs
    import blocks
    import eva_toolbox
    core = vs.core
    cat = eva_toolbox.catalog()
    args = dict(cat["defaults"].get(call, {}))           # the EVA toolbox's defaults and preparation, as the block gets them
    prep = cat.get("prep", {}).get(call)
    base = core.akarin.Expr(core.std.BlankClip(format=vs.YUV420P8, width=720, height=480, length=24, fpsnum=24000, fpsden=1001),
                            ["X Y + 2 % 64 * 64 +", "X 4 % 32 * 96 +", "Y 4 % 32 * 96 +"])
    base = core.std.SetFrameProps(core.std.SetFieldBased(base, 0), _Matrix=6, _Primaries=6, _Transfer=6, _Range=vs.RANGE_LIMITED,
                                  _ChromaLocation=0)
    ok, errs, outfmt = [], [], ""
    job = {"fps_num": 30000, "fps_den": 1001}
    for fmt in blocks.SCAN_FORMATS:
        if fmt in skip:                                  # crashed the process on this format last time
            continue
        print("TRY " + fmt, flush=True)
        c = base if fmt == "YUV420P8" else blocks.adapt_in(base, fmt)
        try:
            out = blocks.block(c, {"source": "", "dar": 4 / 3}, kind, {"call": call, "args": dict(args), "prep": prep, "native": True}, {"sar": 0.889})
            out.get_frame(5)
            blocks.run_output(out, job, {}, {}).get_frame(5)                # what it leaves must reach the Output block too
            ok.append(fmt)
            print("PASS " + fmt + " " + (out.format.name if out.format else ""), flush=True)
            outfmt = outfmt or (out.format.name if out.format else "")
        except Exception as e:                           # noqa: BLE001
            errs.append(f"{type(e).__name__}: {e}"[:300])
    if ok:
        return {"status": "ok", "formats": ok, "format": outfmt}
    try:                                                 # filters made for the telecined / interlaced stream
        c = core.std.SetFieldBased(core.std.AssumeFPS(base, fpsnum=30000, fpsden=1001), 2)   # 29.97 telecined / interlaced
        out = blocks.block(c, {"source": "", "dar": 4 / 3}, kind, {"call": call, "args": dict(args), "prep": prep, "native": True}, {"sar": 0.889})
        out.get_frame(5)
        blocks.run_output(out, job, {}, {}).get_frame(5)
        return {"status": "ok", "formats": ["YUV420P8"], "interlaced": True, "format": out.format.name if out.format else ""}
    except Exception as e:                               # noqa: BLE001
        errs.append(f"{type(e).__name__}: {e}"[:300])
    return {"status": "error", "error": errs[0], "error_last": errs[-1]}


def scan(new_only=False):
    """new_only: keep the results already made for these versions, try only what has none yet (or needed settings
    before: a filter the catalog has since given defaults to)"""
    sys.path.insert(0, HERE)
    import blocks
    import eva_toolbox
    cat = eva_toolbox.catalog()
    todo = []
    res = {}
    for kind, call in [("vs", f["call"]) for f in blocks.vs_catalogue()] + [("jet", f["call"]) for f in blocks.jet_catalogue()]:
        if call in cat["hide"]:
            res[f"{kind}:{call}"] = {"status": "hidden", "why": cat["hide"][call]}
        elif call in cat["needs"]:
            res[f"{kind}:{call}"] = {"status": "args", "need": cat["needs"][call]}
        elif call in cat.get("trust", {}):                # not tried here (e.g. builds a TensorRT engine for minutes)
            res[f"{kind}:{call}"] = {"status": "ok", "note": cat["trust"][call],
                                     **({"formats": [cat["adapt"][call]]} if cat["adapt"].get(call) else {})}
    for f in blocks.vs_catalogue():
        if "vs:" + f["call"] in res:
            continue
        need = [a["name"] for a in f["args"] if not a["opt"] and a["name"] not in cat["defaults"].get(f["call"], {})
                and not blocks.auto_fills(f['call'], a['name'])]
        if need:
            res["vs:" + f["call"]] = {"status": "args", "need": need}
        else:
            todo.append(("vs", f["call"]))
    for f in blocks.jet_catalogue():
        if "jet:" + f["call"] in res:
            continue
        need = [a["name"] for a in f["args"] if a["required"] and a["name"] not in cat["defaults"].get(f["call"], {})
                ]
        if need:
            res["jet:" + f["call"]] = {"status": "args", "need": need}
        else:
            todo.append(("jet", f["call"]))
    if new_only:
        try:
            old = json.load(open(OUT, encoding="utf-8"))
            old = old.get("results", {}) if old.get("versions") == versions() else {}
        except Exception:                                # noqa: BLE001
            old = {}
        keep = [it for it in todo if old.get(f"{it[0]}:{it[1]}", {}).get("status") in ("ok", "error")]
        for it in keep:
            res[f"{it[0]}:{it[1]}"] = old[f"{it[0]}:{it[1]}"]
        todo = [it for it in todo if it not in keep]
    t0 = time.time()
    print(f"{len(todo)} functions to try, {len(res)} need settings", flush=True)

    def run(item):
        kind, call = item
        skip, passed, crash = [], [], ""
        for _ in range(len(__import__("blocks").SCAN_FORMATS) + 1):
            r, out = run1(kind, call, skip)
            if r is not None:
                if passed and r.get("status") == "error":            # some formats passed before a crash
                    r = {"status": "ok", "formats": [f for f, _ in passed], "format": passed[0][1]}
                elif passed and r.get("status") == "ok":
                    r["formats"] = sorted(set(r["formats"]) | {f for f, _ in passed}, key=__import__("blocks").SCAN_FORMATS.index)
                return item, r
            tried = [l[4:].strip() for l in out.splitlines() if l.startswith("TRY ")]
            passed += [tuple((l[5:].split() + [""])[:2]) for l in out.splitlines() if l.startswith("PASS ")]
            if not tried:
                return item, {"status": "error", "error": crash or "crashed"}
            skip.append(tried[-1])                       # the format it died on
            crash = f"crashes (access violation) on {', '.join(skip)}"
        return item, {"status": "error", "error": crash}

    def run1(kind, call, skip):
        try:
            # no console window and no inherited input: started from the app (a process without a console) every child
            # would otherwise get a console of its own - 20x slower, and windows flashing on the screen
            p = subprocess.run([sys.executable, "-W", "ignore", os.path.abspath(__file__), "one", kind, call, ",".join(skip)],
                               capture_output=True, text=True, timeout=TIMEOUT, encoding="utf-8", errors="replace",
                               stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            if line:
                return json.loads(line[-1][7:]), p.stdout
            return None, p.stdout                        # the process died (native crash): retried without that format
        except subprocess.TimeoutExpired as e:           # hangs: keep the formats that passed before it, no retry
            out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            ok = [tuple((l[5:].split() + [""])[:2]) for l in out.splitlines() if l.startswith("PASS ")]
            if ok:
                return {"status": "ok", "formats": [f for f, _ in ok], "format": ok[0][1]}, out
            return {"status": "error", "error": f"no frame within {TIMEOUT} s (hangs or far too slow)"}, out

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for i, ((kind, call), r) in enumerate(ex.map(run, todo), 1):
            res[f"{kind}:{call}"] = r
            if i % 25 == 0:
                print(f"{i}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"versions": versions(), "made": time.strftime("%Y-%m-%d %H:%M"), "results": res}, open(OUT, "w", encoding="utf-8"), indent=1)
    n = {s: sum(1 for r in res.values() if r["status"] == s) for s in ("ok", "args", "hidden", "error")}
    print(f"done in {time.time() - t0:.0f}s: {n}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "one":
        print("RESULT " + json.dumps(one(sys.argv[2], sys.argv[3], [f for f in (sys.argv[4] if len(sys.argv) > 4 else "").split(",") if f])), flush=True)
    else:
        scan(new_only="new" in sys.argv[1:])
