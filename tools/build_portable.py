r"""Builds the shareable, portable EVA toolkit: one folder that runs from anywhere, nothing personal in it.

    python tools\build_portable.py [destination] [--full] [--zip]     (default: dist\EVAtoolkit-v0.25-alpha)

Result (about 0.7 GB):
    EVA toolkit.exe        starts the app (tools\launcher.cs)     Self-test.bat    checks this PC can run everything
    app\                   the window + the Dependencies window   core\            the engine (+ requirements.txt)
    models\                network weights (+ code)               runtime\electron\ Electron
    portable.json          marks the copy as portable: it uses only its own runtime\ and installs what is missing there
On first start the Dependencies window downloads Python 3.12, PyTorch, TensorRT, VapourSynth + plugins, ffmpeg and
MKVToolNix into runtime\ (about 5.5 GB). --full copies them from this PC instead (offline copy, about 8 GB).
--zip also writes <destination>.zip (for a GitHub release).
Not copied: your settings, work folders, built TensorRT engines (made again on first use), training scripts,
benchmarks, backups, the development README. A scan for personal paths runs at the end."""
import fnmatch
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FULL = "--full" in sys.argv
ZIP = "--zip" in sys.argv
DEST = ARGS[0] if ARGS else os.path.join(SRC, "dist", "EVAtoolkit-v0.25-alpha")
VERSION = "0.25.0-alpha"
PY = os.path.dirname(sys.executable)                      # the Python this runs with (3.12 with everything installed)
GITHUB = "https://github.com/EVA0111"

RUNTIME_MODELS = ["aanet.pt", "blocknet.pt", "chromanet.pt", "clipnet.pt", "combnet.pt", "dotnet.pt", "dvdnet_precise.pt",
                  "dvdnet_precise2.pt", "dvdnet_restore.pt", "dvdnet_temporal.pt", "fieldnet.pt", "halonet.pt", "stablenet.pt",
                  "uninet.pt", "deblendnet.pt", "xnet.pt", "tstab.pt", "fieldnet5.pt", "repairnet.pt", "colourprior.pt"]          # missing ones are skipped
MODEL_CODE = ["fieldnet_core.py", "fieldnet5.py", "tstab_core.py", "neuflow_v2.py", "repairnet_core.py"]
PACKAGES = ["torch", "torchvision", "numpy", "pillow", "scipy", "onnx", "tensorrt-cu13", "triton-windows", "vapoursynth",
            "vs_temporalfix", "vsjetpack", "vsview", "typing_extensions", "packaging"]
SKIP = ["__pycache__", "*.pyc"]
# personal traces that must not be in a shared copy: this PC's user name and folders, private drives, e-mail addresses
_USER = os.environ.get("USERNAME", "") or os.path.basename(os.path.expanduser("~"))
PRIVATE = ([re.escape(_USER)] if len(_USER) > 2 else []) + [r"C:\\Users\\(?!Public)",r"[HF]:\\", r"[HF]:/",
           re.escape(os.path.dirname(SRC)), re.escape(os.path.dirname(SRC).replace("\\", "/")), r"@gmail", r"@outlook", r"@hotmail"]


def log(*a):
    print(*a, flush=True)


def copytree(a, b, skip=SKIP):
    shutil.copytree(a, b, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*skip))


def copy(a, b):
    os.makedirs(os.path.dirname(b), exist_ok=True)
    shutil.copy2(a, b)


def norm(n):
    return n.lower().replace("_", "-")


def closure():
    from packaging.requirements import Requirement
    dists = {norm(d.metadata["Name"]): d for d in md.distributions()}
    seen, stack = set(), [norm(p) for p in PACKAGES]
    while stack:
        n = stack.pop()
        if n in seen or n not in dists:
            continue
        seen.add(n)
        for r in dists[n].requires or []:
            req = Requirement(r)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            stack.append(norm(req.name))
    return [dists[n] for n in sorted(seen)]


def python_runtime(dst):
    """Python 3.12: the interpreter + standard library + only the packages above (whole top-level folders, so plugins
    dropped into vapoursynth/plugins by hand come along), their .dist-info and their Scripts (vspipe)."""
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(PY):
        p = os.path.join(PY, f)
        if os.path.isfile(p) and (f.lower().endswith((".exe", ".dll")) or f.upper().startswith("LICENSE")):
            if f.lower() not in ("uninstall.exe",):
                copy(p, os.path.join(dst, f))
    copytree(os.path.join(PY, "DLLs"), os.path.join(dst, "DLLs"))
    copytree(os.path.join(PY, "Lib"), os.path.join(dst, "Lib"),
             skip=SKIP + ["site-packages", "test", "idlelib", "turtledemo", "ensurepip", "lib2to3", "tkinter"])
    sp_src, sp_dst = os.path.join(PY, "Lib", "site-packages"), os.path.join(dst, "Lib", "site-packages")
    os.makedirs(sp_dst, exist_ok=True)
    tops, scripts = set(), set()
    for d in closure():
        files = d.files or []
        for f in files:
            parts = str(f).replace("\\", "/").split("/")
            if parts[0] == "..":                          # ../../Scripts/vspipe.exe
                if "Scripts" in parts:
                    scripts.add(parts[-1])
                continue
            tops.add(parts[0])
        log(f"    {d.metadata['Name']} {d.version}")
    for t in sorted(tops):
        s = os.path.join(sp_src, t)
        if os.path.isdir(s):
            copytree(s, os.path.join(sp_dst, t))
        elif os.path.isfile(s):
            copy(s, os.path.join(sp_dst, t))
    # Scripts\*.exe are pip launchers with this PC's Python path written into them - not copied; the app runs
    # vapoursynth\vspipe.exe and "python -m vsview" instead
    for dead in ("bm3dcpu",):                                     # VapourSynth API 3 plugins R81 refuses to load
        shutil.rmtree(os.path.join(sp_dst, "vapoursynth", "plugins", dead), ignore_errors=True)
    # .pth files of the copied packages only (some packages add their folder this way), none that point elsewhere
    for f in os.listdir(sp_src):
        if f.endswith(".pth"):
            txt = open(os.path.join(sp_src, f), encoding="utf-8", errors="replace").read()
            if not re.search(r"[A-Za-z]:\\", txt) and not f.startswith("__editable__"):
                copy(os.path.join(sp_src, f), os.path.join(sp_dst, f))


def readme(dst):
    open(os.path.join(dst, "README.md"), "w", encoding="utf-8").write(rf"""# EVA toolkit v0.25 alpha (portable)

Restores NTSC and PAL DVD video, other video files and still images - anime, cartoons and live action:

* **Inverse telecine** (VFR / CFR) and PAL field matching or deinterlacing. Progressive sources (23.976p, 24p, 25p,
  50p, 60p, HD files) are recognised and left at their own frame rate.
* **Dot crawl and rainbows** (composite artifacts): neural clean-up (Precise or Precise temporal)
  or classic filters, optional composite re-decode.
* **MPEG blocking**, **halos / ringing** (HaloNet), **colour bleed and colour shift** (ChromaNet), clean white / black.
* **Temporal stabilizer "Universal"**: steadies flicker, boil and line wobble in every kind of shot - fades, cuts,
  flashes and zooms are never dragged into the next frame.
* Colour correction with adjustment layers, crop, resize (incl. EWA on the GPU), sharpening, line darkening.
* Encode (x264 / x265 / SVT-AV1 / NVENC / ProRes / FFV1 / UT Video) and mux with all audio and subtitle tracks and a
  sync check. A still image (PNG / JPEG / TIFF / BMP / WebP) is saved as a 16-bit PNG.
* Opens .iso images and VIDEO_TS folders directly.

Project page: {GITHUB}

## Run

1. Unpack the zip anywhere - another drive or a USB disk is fine (a path without special characters is safest).
2. Start **`EVA toolkit.exe`**.
3. **First start:** the *Dependencies* window lists what the engine still needs and installs it with one click into
   `runtime\` next to the app: Python 3.12, PyTorch (CUDA 13), TensorRT, VapourSynth with its plugins, ffmpeg and
   MKVToolNix - about **5.5 GB download, 9 GB on disk**, from their official download sites. Then press
   *Start EVA toolkit*. Nothing is installed outside the folder: no registry, no PATH, no admin rights.
4. Optional: **`Self-test.bat`** checks that this PC can run every stage (a few minutes; it makes its own test video).

The folder can be moved or copied at any time - everything is found relative to `EVA toolkit.exe`. On every start the
app checks its folders: deleted cache folders are made again, and a missing or outdated part (for example after an
update changed `core\requirements.txt`) opens the Dependencies window again. The *Dependencies* button in the
sidebar opens it any time (repair, or add the optional vsview preview window).

## Needs

* Windows 10 / 11, 64-bit, and an internet connection for the first start.
* An **NVIDIA GPU** with a current driver (RTX 20-series or newer recommended; the CUDA 13 runtime comes with PyTorch).
  Without a CUDA GPU the neural stages fall back to the CPU and are very slow.
* About 10 GB of disk space, more for the work folders of long episodes.
* The first use of a model at a new frame size builds a TensorRT engine (a minute or two, once; kept in `engines\`).
  PyTorch can be chosen instead in Settings.

## Folders

| Folder | What |
|---|---|
| `EVA toolkit.exe` | starts the app |
| `app\` | the window (Electron front-end) and the Dependencies window |
| `core\` | the engine (Python + VapourSynth), `requirements.txt` = the exact package versions |
| `models\` | the trained networks |
| `runtime\` | `electron\` (comes with the zip); `python\`, `ffmpeg\`, `mkvtoolnix\` (installed on first start) |
| `engines\`, `temp\` | made on first use (TensorRT engines, work files) - safe to delete while the app is closed |
| `settings.json` | your settings, made on first start |

## Licences

EVA toolkit itself: **MIT** (see `LICENSE`).

The tools it uses: ffmpeg (LGPL / GPL), MKVToolNix (GPL-2.0), VapourSynth (LGPL-2.1) and its plugins, PyTorch (BSD-3), NVIDIA TensorRT
(NVIDIA licence), Electron (MIT), NeuFlow v2 weights (see `models\neuflow\LICENSE`). Each keeps its own licence. Apart
from Electron they are not in the zip: the Dependencies window downloads them from their publishers.
""")


def launchers(dst):
    r"""EVA toolkit.exe (compiled from tools\launcher.cs with the C# compiler every Windows has), Self-test.bat"""
    csc = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "Microsoft.NET", "Framework64", "v4.0.30319", "csc.exe")
    r = subprocess.run([csc, "-nologo", "-target:winexe", "-optimize", "-win32icon:" + os.path.join(SRC, "app", "icon.ico"),
                        "-out:" + os.path.join(dst, "EVA toolkit.exe"), os.path.join(SRC, "tools", "launcher.cs")],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("launcher did not compile:\n" + r.stdout + r.stderr)
    open(os.path.join(dst, "Self-test.bat"), "w").write(
        '@echo off\r\ncd /d "%~dp0"\r\nif not exist "runtime\\python\\python.exe" (\r\n'
        '  echo Start EVA toolkit.exe first: its Dependencies window installs what the self-test needs.\r\n  pause\r\n  exit /b 1\r\n)\r\n'
        '"%~dp0runtime\\python\\python.exe" -I -W ignore tools\\selftest.py\r\npause\r\n')
    import json
    json.dump({"name": "EVA toolkit", "version": VERSION, "portable": True,
               "note": "This file makes the app use only its own runtime folder. Delete it to use installed tools."},
              open(os.path.join(dst, "portable.json"), "w"), indent=2)


def stamp(dst):
    """--full: mark the copied packages as installed, so the Dependencies window does not install them again"""
    sys.path.insert(0, os.path.join(SRC, "app"))
    import hashlib
    for req, name in (("requirements.txt", "dvdrestore-packages.txt"), ("requirements-vsview.txt", "dvdrestore-vsview.txt")):
        h = hashlib.sha256(open(os.path.join(SRC, "core", req), "rb").read()).hexdigest()[:16]
        open(os.path.join(dst, "runtime", "python", name), "w").write(h)
    # this PC's ffmpeg is a gyan.dev full build (it has the DVD reader) - mark it as the expected one
    ff = re.search(r'file: "(ffmpeg-[^"]+)"', open(os.path.join(SRC, "app", "deps.js"), encoding="utf-8").read()).group(1)
    open(os.path.join(dst, "runtime", "ffmpeg", "dvdrestore-ffmpeg.txt"), "w").write(ff)


def scan(dst):
    """personal paths / names in anything that is text (code, html, json, md, bat)"""
    hits = []
    for root, _, files in os.walk(dst):
        if os.sep + "runtime" + os.sep in root + os.sep:
            continue
        for f in files:
            if not f.lower().endswith((".py", ".js", ".html", ".css", ".json", ".md", ".bat", ".txt", ".toml")):
                continue
            p = os.path.join(root, f)
            txt = open(p, encoding="utf-8", errors="replace").read()
            for pat in PRIVATE:
                for m in re.finditer(pat, txt):
                    line = txt.count("\n", 0, m.start()) + 1
                    hits.append(f"{os.path.relpath(p, dst)}:{line}: {txt[max(0, m.start() - 30):m.end() + 40]!r}")
    return hits


def main():
    log(f"Building the portable EVA toolkit in {DEST}")
    if os.path.exists(DEST) and os.listdir(DEST):
        raise SystemExit(f"{DEST} is not empty - delete it or give another destination")
    os.makedirs(DEST, exist_ok=True)
    log("app/")
    for f in ("main.js", "preload.js", "deps.js", "update.js", "package.json", "icon.png", "icon.ico"):
        copy(os.path.join(SRC, "app", f), os.path.join(DEST, "app", f))
    copytree(os.path.join(SRC, "app", "renderer"), os.path.join(DEST, "app", "renderer"))
    log("core/")
    copytree(os.path.join(SRC, "core"), os.path.join(DEST, "core"))
    copy(os.path.join(SRC, "LICENSE"), os.path.join(DEST, "LICENSE"))       # EVA toolkit: MIT
    log("my_filters/ (the example of your own filters; an empty plugins folder)")
    os.makedirs(os.path.join(DEST, "my_filters", "plugins"), exist_ok=True)
    copy(os.path.join(SRC, "my_filters", "example_filters.py"), os.path.join(DEST, "my_filters", "example_filters.py"))
    log("models/")
    for m in RUNTIME_MODELS + MODEL_CODE:
        s = os.path.join(SRC, "models", m)
        if os.path.isfile(s):
            copy(s, os.path.join(DEST, "models", m))
    copytree(os.path.join(SRC, "models", "neuflow"), os.path.join(DEST, "models", "neuflow"))
    copy(os.path.join(SRC, "tools", "selftest.py"), os.path.join(DEST, "tools", "selftest.py"))
    log("runtime/electron")
    copytree(os.path.join(SRC, "app", "node_modules", "electron", "dist"), os.path.join(DEST, "runtime", "electron"))
    if FULL:
        log("runtime/ffmpeg, mkvtoolnix (--full)")
        import runner as _r                               # the tools this PC's EVA toolkit uses
        for t in ("ffmpeg", "ffprobe"):
            copy(_r.TOOLS[t], os.path.join(DEST, "runtime", "ffmpeg", t + ".exe"))
        mkv = _r.TOOLS["mkvmerge"]
        copy(mkv, os.path.join(DEST, "runtime", "mkvtoolnix", "mkvmerge.exe"))
        for lic in ("COPYING.txt", "COPYING", "doc/COPYING.txt"):
            p = os.path.join(os.path.dirname(mkv), lic)
            if os.path.isfile(p):
                copy(p, os.path.join(DEST, "runtime", "mkvtoolnix", "COPYING.txt"))
                break
        log("runtime/python (--full; this takes a while: PyTorch and TensorRT are most of it)")
        python_runtime(os.path.join(DEST, "runtime", "python"))
        stamp(DEST)
    readme(DEST)
    launchers(DEST)
    hits = scan(DEST)
    log(f"personal-path scan: {len(hits)} hit(s)")
    for h in hits[:40]:
        log("   ", h)
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(DEST) for f in fs)
    log(f"done: {size / 1e9:.2f} GB")
    if ZIP:
        base = DEST.rstrip("\\/")
        log(f"zip: {base}.zip")
        shutil.make_archive(base, "zip", os.path.dirname(base), os.path.basename(base))
        log(f"zip done: {os.path.getsize(base + '.zip') / 1e9:.2f} GB")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(SRC, "core"))
    main()
