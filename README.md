# DVD Restore v0.2 alpha

Restores NTSC and PAL DVD video, other video files and still images: anime, cartoons and live action.
A Windows desktop app (Electron front-end, Python + VapourSynth engine, PyTorch / TensorRT networks).

> **Alpha.** It works on the discs it was tuned on and passes its self-test, but expect rough edges.
> Bug reports with a sample are very welcome (Issues).
<img width="1711" height="985" alt="image" src="https://github.com/user-attachments/assets/1cf0e28a-0548-442b-a3ed-5d1411fdb0a8" />

## Download and run

1. Download **`DVDRestore-v0.2-alpha.zip`** from [Releases](../../releases) (about 0.3 GB).
2. Unpack it anywhere: another drive or a USB disk is fine.
3. Start **`DVD Restore.exe`**.
4. **First start:** the *Dependencies* window installs what the engine needs with one click, into `runtime\` next to
   the app: Python 3.12, PyTorch (CUDA 13), TensorRT, VapourSynth with its plugins, ffmpeg and MKVToolNix. That's
   about **5.5 GB download, 9 GB on disk**, from their official download sites. Nothing is installed outside the
   folder: no registry, no PATH, no admin rights. Then press *Start DVD Restore*.

The folder stays portable: move or copy it any time. On every start the app checks itself and reopens the
Dependencies window if a part is missing or needs an update. Optional: `Self-test.bat` checks every stage on test
footage it makes itself.

### Needs

* Windows 10 / 11, 64-bit, internet for the first start.
* An **NVIDIA GPU** with a current driver (RTX 20-series or newer recommended). Without one the neural stages run on
  the CPU and are very slow.
* About 10 GB of free disk space, more for the work folders of long episodes.

Full documentation: **[the wiki](../../wiki)** (installation, every step of the window, troubleshooting).

## What it does

* **Inverse telecine** (VFR / CFR) and PAL field matching or deinterlacing. Progressive sources (23.976p, 24p, 25p,
  50p, 60p, HD files) are recognised and keep their own frame rate.
* **Dot crawl and rainbows** (composite artifacts): neural clean-up (Precise or Precise temporal)
  or classic filters, optional composite re-decode.
* **MPEG blocking**, **halos / ringing** (HaloNet), **colour bleed and colour shift** (ChromaNet), clean white / black.
* **Temporal stabilizer "Universal"**: steadies flicker, boil and line wobble in every kind of shot. Fades, cuts,
  flashes and zooms are never dragged into the next frame.
* Colour correction with adjustment layers, crop, resize (incl. EWA on the GPU), sharpening, line darkening.
* Encode (x264 / x265 / SVT-AV1 / NVENC / ProRes / FFV1 / UT Video) and mux with all audio and subtitle tracks and a
  sync check. Still images (PNG / JPEG / TIFF / BMP / WebP) are saved as 16-bit PNG.
* Opens `.iso` images, `VIDEO_TS` folders and Blu-ray `BDMV` folders directly.

## New in v0.2

* **Colour page laid out like DaVinci Resolve**: the picture in the middle, Auto correction left, scopes right, and a
  collapsible panel underneath with the wheels, light, colour, hue / shadow colour and curves on one page.
* **RepairNet**: halos and colour bleed fixed in one neural pass.
* **VapourSynth filter groups** in Repair: any installed plugin or vs-jetpack function, picked from a searchable,
  categorised filter browser, in draggable / duplicable groups placed anywhere in the chain (even instead of the IVTC).
  Plus **your own filters** (Python functions, `.dll` plugins, custom code) and **your AI upscale / restore models**
  (`.onnx`, `.pth`, `.safetensors`) with a VapourKit-style TensorRT engine dialog.
* **Leftover combing** after the IVTC is found and repaired much better: single frames field matching could not fix,
  and credits / captions laid on at 60 fields per second over film.
* **Damaged discs**: undecodable frames are found in seconds (a fast scan on open) and replaced; the VFR analysis no
  longer takes an hour on such files.
* **Undo / redo** (Ctrl+Z / Ctrl+Y), **workflows** you can save and share, crop & size moved into Repair, image
  information for stills, render progress with ETA, Blu-ray import, a better IVTC and colour auto-correction.

## Source layout

| Folder | What |
|---|---|
| `app/` | Electron front-end (`main.js`, `deps.js` = the Dependencies installer, `renderer/`) |
| `core/` | the engine: `server.py` (backend), `runner.py` (probe / export / mux), `vsgraph.py` (processing chain), networks; `requirements.txt` = exact package versions |
| `models/` | trained network weights (+ the network code some of them need) |
| `tools/` | `selftest.py`, `build_portable.py` (makes the release folder + zip), `launcher.cs` (`DVD Restore.exe`), `make_icon.py` |

### Run from source

```powershell
cd app
npm install
cd ..
python -m pip install -r core\requirements.txt
app\node_modules\electron\dist\electron.exe app
```

Python 3.12 is needed; ffmpeg (a *full* build, with the DVD-Video reader) and `mkvmerge` must be on PATH or in
`runtime\ffmpeg\` and `runtime\mkvtoolnix\`.

### Build the release

```powershell
python tools\build_portable.py dist\DVDRestore-v0.2-alpha --zip
```

## Licence

DVD Restore is free software under the **MIT License** (see `LICENSE`).

The tools it downloads keep their own licences: ffmpeg (GPL), MKVToolNix (GPL-2.0), VapourSynth (LGPL-2.1) and its
plugins, PyTorch (BSD-3), NVIDIA TensorRT (NVIDIA licence), Electron (MIT). The NeuFlow v2 optical-flow weights keep
theirs (`models/neuflow/LICENSE`).
