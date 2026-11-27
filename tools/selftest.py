"""EVA toolkit self-test: checks that everything the app needs is present and that every stage runs.
Uses only test footage it makes itself (ffmpeg's test patterns), so it works on any PC - also the portable version.

    runtime\\python\\python.exe tools\\selftest.py        (portable)      python tools\\selftest.py   (installed Python)
    options:  --quick   skip the slower model / stabilizer passes

Prints one line per check and a summary; exit code 0 = all passed."""
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

QUICK = "--quick" in sys.argv
RESULTS = []


def check(name, fn):
    t = time.time()
    try:
        info = fn()
        RESULTS.append((name, True))
        print(f"  OK    {name:46s} {time.time() - t:6.1f} s  {info or ''}", flush=True)
        return True
    except Exception as e:                               # noqa: BLE001
        RESULTS.append((name, False))
        print(f"  FAIL  {name:46s} {type(e).__name__}: {str(e)[:200]}", flush=True)
        if "--trace" in sys.argv:
            traceback.print_exc()
        return False


def main():
    import paths
    work = os.path.join(paths.TEMP, "selftest")
    os.makedirs(work, exist_ok=True)
    print(f"EVA toolkit self-test - {ROOT}")
    print("1. Python packages and tools")
    mods = {}

    def imp(name):
        def f():
            m = __import__(name)
            mods[name] = m
            return getattr(m, "__version__", "")
        return f
    for name in ("numpy", "vapoursynth", "torch", "PIL", "scipy"):
        check(f"import {name}", imp(name))
    check("CUDA GPU", lambda: (mods["torch"].cuda.get_device_name(0) if mods["torch"].cuda.is_available()
                               else (_ for _ in ()).throw(RuntimeError("no CUDA GPU - neural stages will be very slow"))))
    check("TensorRT (optional)", imp("tensorrt"))
    import runner
    for t in ("ffmpeg", "ffprobe", "vspipe", "mkvmerge"):
        check(f"tool {t}", lambda t=t: runner.TOOLS[t] or (_ for _ in ()).throw(FileNotFoundError(t + " not found")))
    core = mods["vapoursynth"].core
    for ns in ("bs", "tivtc", "vivtc", "mv", "akarin", "awarp", "zsmooth", "resize", "resize2", "composite", "bwdif", "znedi3"):
        check(f"VapourSynth plugin {ns}", lambda ns=ns: hasattr(core, ns) or (_ for _ in ()).throw(ImportError(ns + " missing")))
    print("2. Model files")
    for m in ("dvdnet_precise.pt", "dvdnet_temporal.pt", "stablenet.pt", "uninet.pt", "fieldnet.pt", "halonet.pt", "blocknet.pt",
              "chromanet.pt", "aanet.pt", "combnet.pt", "clipnet.pt", "dotnet.pt", "repairnet.pt"):
        check(f"model {m}", lambda m=m: os.path.isfile(paths.model(m)) or (_ for _ in ()).throw(FileNotFoundError(m)))

    print("3. Test footage (made here)")
    ff = runner.TOOLS["ffmpeg"]
    import subprocess
    ntsc = os.path.join(work, "ntsc_test.mkv")
    prog = os.path.join(work, "progressive_test.mp4")
    png = os.path.join(work, "still_test.png")
    odd = os.path.join(work, "odd_size_test.png")          # an image whose size 4:2:0 cannot hold as it is

    def make(args, out):
        r = subprocess.run([ff, "-v", "error", "-y", *args, out], capture_output=True, text=True)
        if r.returncode or not os.path.isfile(out):
            raise RuntimeError(r.stderr[-300:])
        return os.path.basename(out)
    # 720x480 29.97 interlaced MPEG-2 with stereo audio, like a DVD title
    check("NTSC MPEG-2 clip", lambda: make(["-f", "lavfi", "-i", "testsrc2=size=720x480:rate=30000/1001:duration=4",
                                            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                                            # setfield=tff marks top field first (ffmpeg 9 dropped the "-top" option)
                                            "-vf", "setsar=8/9,setfield=tff,format=yuv420p", "-c:v", "mpeg2video", "-b:v", "6M",
                                            "-flags", "+ilme+ildct", "-c:a", "ac3", "-b:a", "192k"], ntsc))
    check("progressive H.264 clip", lambda: make(["-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=24000/1001:duration=3",
                                                  "-c:v", "libx264", "-pix_fmt", "yuv420p"], prog))
    check("still image", lambda: make(["-f", "lavfi", "-i", "testsrc2=size=720x480", "-frames:v", "1"], png))
    check("odd-sized image (708x531)", lambda: make(["-f", "lavfi", "-i", "testsrc2=size=708x531", "-frames:v", "1"], odd))

    print("4. Engine")
    import server
    import vsgraph

    def job_for(path, **st_extra):
        info = runner.probe(path)
        eps = runner.detect_episodes(info)
        st = {"info": info, "episodes": [list(e) for e in eps], "selected": [0], "out_dir": os.path.join(work, "out"),
              "mode": "cfr", "size": "keep", "encoder": "x264"}
        st.update(st_extra)
        return info, st
    info, st = job_for(ntsc)
    check("probe NTSC clip", lambda: f"{info['width']}x{info['height']} {info['frames']} frames, progressive={info['progressive']}")
    res = {}
    check("Analyze (auto-detect all)", lambda: res.update(vsgraph.auto_detect(server.detect_job(dict(st, ai_model="precise"), "st_auto")))
          or f"crawl {res.get('crawl')}, stab {res.get('stab')}")

    def frames(clip, n=6):
        for k in range(min(n, clip.num_frames)):
            clip.get_frame(k)
        return f"{clip.width}x{clip.height} {clip.format.name}"

    def build(**kw):
        return lambda: frames(vsgraph.build(server.make_job(dict(st, **kw), 0)))
    models = ["off", "precise"] if QUICK else ["off", "precise", "temporal", "stable", "uni", "field"]
    for m in models:
        check(f"cleanup model: {m}", build(ai_model=m, crawl="normal", rainbow="normal"))
    for s in (["ustab"] if QUICK else ["ustab", "chain", "plus", "gtfix", "ttemp", "mc"]):
        check(f"stabilizer: {s}", build(ai_model="precise", crawl="normal", rainbow="normal", stab=s, stab_level="normal"))
    check("halo removal (RepairNet)", build(ghost="auto", halo_rx=2.0, halo_method="ai"))
    check("halo removal (HaloNet, RepairNet off)", build(ghost="auto", halo_rx=2.0, halo_method="ai", repairnet="off"))
    check("MPEG repair (BlockNet)", build(deblock="normal"))
    check("colour alignment + bleed (RepairNet)", build(sx=-0.75, bleed="ai|normal"))
    vsf = lambda call, **a: {"type": "jet" if call.startswith("vs") and not call.startswith("vszip") else "vs", "on": True,
                             "p": {"call": call, "args": a}}
    check("VapourSynth filters: MVTools + clip settings", build(vs_filters_pos="repairs", vs_filters=[
        vsf("mv.Degrain2"), vsf("mvu.Degrain3"), vsf("vsrgtools.contrasharpening", src="group"),
        vsf("vszip.LimitFilter", src="@group"), vsf("std.MaskedMerge", clipb="@prev"),
        # the size tokens: descale where it is installed (an optional plugin, not in the app's own runtime), else the
        # built-in resizer - the portable copy's self-test failed here
        vsf("descale.Debicubic", width="@w", height="@h") if hasattr(core, "descale") else vsf("resize.Spline36", width="@w", height="@h")]))
    check("resize to 1080 lines, EWA (GPU)", build(size="1080", kernel="ewa_lanczossharp"))

    print("5. Export")

    def export(path, out_name, **kw):
        _, s2 = job_for(path, **kw)
        job = server.make_job(s2, 0)
        job["output"] = os.path.join(work, "out", out_name)
        log = []
        runner.JobRunner(lambda m, *a: log.append(m) if m else None).run(job)
        out = job["output"] if not out_name.endswith(".png") else os.path.splitext(job["output"])[0] + ".png"
        if not os.path.isfile(out) or os.path.getsize(out) < 1000:
            raise RuntimeError("no output: " + " | ".join(log[-3:]))
        sync = [l for l in log if l.startswith("Audio sync")]
        return f"{os.path.getsize(out) // 1024} KB" + (f", {sync[0]}" if sync else "")
    check("export NTSC -> MKV (cleanup + Universal)", lambda: export(ntsc, "ntsc_out.mkv", ai_model="precise", crawl="normal",
                                                                     rainbow="normal", stab="ustab"))
    check("export progressive (frame rate kept)", lambda: export(prog, "progressive_out.mkv", ai_model="off"))

    def own_ivtc():                                    # Repair -> VapourSynth filters "instead of the app's IVTC"
        r = export(ntsc, "own_ivtc_out.mkv", ai_model="off", mode="vfr", vs_filters_pos="ivtc",
                   vs_filters=[{"type": "vs", "on": True, "p": {"call": "vivtc.VFM", "args": {"order": "1"}}},
                               {"type": "vs", "on": True, "p": {"call": "vivtc.VDecimate", "args": {}}}])
        fr = runner._run([runner.TOOLS["ffprobe"], "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                          "-of", "csv=p=0", os.path.join(work, "out", "own_ivtc_out.mkv")]).stdout.strip()
        if fr not in ("24000/1001", "2997/125"):
            raise RuntimeError(f"frame rate {fr}, expected 23.976")
        return f"{r}, {fr} fps"
    check("export with your own IVTC (vivtc VFM + VDecimate)", own_ivtc)
    check("export image -> PNG", lambda: export(png, "still_out.png", ai_model="precise", crawl="normal", rainbow="normal"))
    check("export odd-sized image -> PNG", lambda: export(odd, "odd_out.png", ai_model="precise", crawl="normal", rainbow="normal"))

    bad = [n for n, ok in RESULTS if not ok]
    opt = [n for n in bad if "optional" in n]
    print(f"\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed" + (f" - failed: {', '.join(bad)}" if bad else ""))
    return 0 if len(bad) == len(opt) else 1


if __name__ == "__main__":
    sys.exit(main())
