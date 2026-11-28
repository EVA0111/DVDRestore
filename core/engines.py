"""Built TensorRT engines of the user's AI models (engines\\): list them with their details, say which one a node uses, turn an
engine back into node settings (so the node reuses it instead of building again), delete engines.

Two families of names, by what the model is:
 .onnx models (aiupscale._TRTOnnx._engine):
  user_<model name[:40]>_<sha10><hc|bc|f><conv>o<opt>[n]_<1x3xHxW | dyn_<min>-<opt>-<max>>_trt<version>_<GPU>.engine
  .pth / .safetensors models (nnbackend.TRTNet): user_<name[:40]>_<fp16|bf16|fp32>_o<opt>_<sha10>_<1x3xHxW><dtype>_trt<version>_<GPU>.engine
  next to it (builds since 2026-11-28) <engine>.json: when it was built, how long it took, the settings.
An engine of the other family (a model that was an .onnx and is a .safetensors now, or the reverse) cannot be reused: listed, deletable."""
import json
import os
import re
import time

import aiupscale
import nnbackend

ENGINE_DIR = nnbackend.ENGINE_DIR
NEW = re.compile(r"^user_(?P<stem>.+)_(?P<sha>[0-9a-f]{10})(?P<code>hc|bc|f)(?P<conv>\d*)o(?P<opt>\d)(?P<n>n?)_(?P<shp>dyn_[\dx\-]+|[\dx]+)"
                 r"_trt(?P<trt>[\d.]+)_(?P<gpu>.+)\.engine$")
OLD = re.compile(r"^user_(?P<stem>.+)_(?P<prec>fp16|bf16|fp32)_o(?P<opt>\d)_(?P<sha>[0-9a-f]{10})_(?P<shp>[\dx]+?)(?P<dt>float16|bfloat16|float32)?"
                 r"_trt(?P<trt>[\d.]+)_(?P<gpu>.+)\.engine$")
CODE = {"hc": "fp16", "bc": "bf16", "f": "fp32"}
_INFO = {}


def _env():
    """TensorRT version and GPU name as the file names carry them."""
    gpu, trt = "", ""
    try:
        import tensorrt
        trt = tensorrt.__version__
        import torch
        gpu = torch.cuda.get_device_name(0).replace(" ", "_") if torch.cuda.is_available() else ""
    except Exception:                                    # noqa: BLE001
        pass
    return trt, gpu


def _info(path):
    if path not in _INFO:
        try:
            _INFO[path] = aiupscale.info(path)
        except Exception as e:                           # noqa: BLE001
            _INFO[path] = {"error": f"{type(e).__name__}: {e}"}
    return _INFO[path]


def padded_shape(path, w, h):
    """what the model is given for a w x h picture (run_model: 16 px padding each side, multiples of 8; a fixed-size model its size)"""
    inf = _info(path)
    if inf.get("error") or inf.get("vsr"):
        return None
    if inf.get("fixed"):
        fw, fh = inf["fixed"]
        return (1, int(inf.get("channels") or 3), int(fh), int(fw))
    pw, ph = (int(w) + 32 + 7) // 8 * 8, (int(h) + 32 + 7) // 8 * 8
    return (1, int(inf.get("channels") or 3), ph, pw)


def _dims(s):
    return tuple(int(x) for x in s.split("x"))


def parse(name):
    """a file name -> its parts (None = not an engine of a user model)"""
    m = NEW.match(name)
    if m:
        d = m.groupdict()
        shp = d["shp"]
        dyn = None
        if shp.startswith("dyn_"):
            a, b, c = (_dims(x) for x in shp[4:].split("-"))
            dyn = {"min": a, "opt": b, "max": c}
        return {"stem": d["stem"], "sha": d["sha"], "precision": CODE[d["code"]], "opt_level": int(d["opt"]), "tf32": d["n"] != "n",
                "shape": None if dyn else _dims(shp), "dynamic": dyn, "trt": d["trt"], "gpu": d["gpu"], "family": "onnx"}
    m = OLD.match(name)
    if m:
        d = m.groupdict()
        return {"stem": d["stem"], "sha": d["sha"], "precision": d["prec"], "opt_level": int(d["opt"]), "tf32": True,
                "shape": _dims(d["shp"]), "dynamic": None, "trt": d["trt"], "gpu": d["gpu"], "family": "torch"}
    return None


def _args_for(parts, input_name="input"):
    """trtexec text for an engine's settings (the engine dialog shows and parses it)"""
    nm = input_name
    s = ""
    if parts["dynamic"]:
        d = parts["dynamic"]
        s = " ".join(f"--{k}Shapes={nm}:{'x'.join(map(str, d[k]))}" for k in ("min", "opt", "max"))
    s += f" --saveEngine=OUTPUT_PATH --builderOptimizationLevel={parts['opt_level']}"
    if parts["precision"] == "fp16":
        s += " --fp16"
    elif parts["precision"] == "bf16":
        s += " --bf16"
    return s.strip()


def settings_of(parts, path):
    """node settings (f.p.engine) that make the node's engine file exactly this one"""
    inf = _info(path)
    e = {"type": "image", "precision": parts["precision"], "opt_level": parts["opt_level"], "workspace": 4096,
         "shape": "dynamic" if parts["dynamic"] else "static", "frames": 0}
    if parts["precision"] == "fp32":
        e["tf32"] = parts["tf32"]
    e["args"] = _args_for(parts, inf.get("input_name") or "input") if parts["dynamic"] else ""
    if parts["family"] == "torch":                       # a PyTorch model: one engine per input size, no size range
        e["shape"], e["args"] = "static", ""
    return e


def _fits(parts, key):
    d = parts["dynamic"]
    return bool(d) and len(d["min"]) == len(key) and all(a <= s <= b for a, s, b in zip(d["min"], key, d["max"]))


def _is_onnx(path):
    return os.path.splitext(path)[1].lower() == ".onnx"


def expected(path, eng, key):
    """the engine file a node with these settings uses for input shape `key`: the name aiupscale builds, without building"""
    if key is None:
        return None
    trt, gpu = _env()
    if not _is_onnx(path):                                # PyTorch model: nnbackend.TRTNet, one engine per input shape
        prec = eng["precision"] if eng["precision"] in ("fp16", "bf16", "fp32") else "fp16"
        dt = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}[prec]
        return (f"user_{os.path.splitext(os.path.basename(path))[0][:40]}_{prec}_o{eng['opt_level']}_{aiupscale._sha(path)}_"
                f"{'x'.join(map(str, key))}{dt}_trt{trt}_{gpu}.engine")
    prec = eng["precision"] if eng["precision"] in ("fp16", "bf16", "fp32") else "fp16"
    fp16 = prec != "fp32"
    tag = aiupscale._sha(path) + {"fp16": "hc", "bf16": "bc", "fp32": "f"}[prec] + (f"{aiupscale.CONV_VERSION}" if fp16 else "") + \
        f"o{eng['opt_level']}" + ("" if fp16 or eng["tf32"] else "n")
    prof = eng.get("profile") if eng.get("shape") == "dynamic" else None
    if _info(path).get("fixed"):
        prof = None
    dyn = prof is not None and len(prof[0]) == len(key) and all(a <= s <= b for a, s, b in zip(prof[0], key, prof[2]))
    shp = ("dyn_" + "-".join("x".join(map(str, s)) for s in prof)) if dyn else "x".join(map(str, key))
    return f"user_{os.path.splitext(os.path.basename(path))[0][:40]}_{tag}_{shp}_trt{trt}_{gpu}.engine"


def _meta(file):
    try:
        with open(os.path.join(ENGINE_DIR, file + ".json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:                                    # noqa: BLE001
        return None


def write_meta(engine_path, seconds, eng, dyn_profile):
    """called by aiupscale after a build: how long it took and with what"""
    try:
        trt, gpu = _env()
        with open(engine_path + ".json", "w", encoding="utf-8") as f:
            json.dump({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(seconds, 1), "precision": eng.get("precision"),
                       "opt_level": eng.get("opt_level"), "workspace": eng.get("workspace"), "tf32": eng.get("tf32"),
                       "dynamic": [list(s) for s in dyn_profile] if dyn_profile else None, "trt": trt, "gpu": gpu}, f)
    except Exception:                                    # noqa: BLE001
        pass


def listing(model, engine, w, h, preset_default):
    """everything the model library shows for one model. engine: the node's engine settings or None; preset_default(path) -> the settings a
    node without any uses."""
    path = str(model or "").strip().strip('"')
    if not path or not os.path.isfile(path):
        return {"error": "model file not found"}
    inf = _info(path)
    eng = aiupscale.engine_settings({"engine": engine}) if engine else preset_default(path)
    trt, gpu = _env()
    key = padded_shape(path, w, h)
    exp = expected(path, eng, key)
    stem = os.path.splitext(os.path.basename(path))[0][:40]
    sha = aiupscale._sha(path)
    rows, total = [], 0
    try:
        names = sorted(os.listdir(ENGINE_DIR))
    except OSError:
        names = []
    for n in names:
        if not n.endswith(".engine") or not n.startswith("user_"):
            continue
        parts = parse(n)
        if not parts or parts["stem"] != stem:
            continue
        fp = os.path.join(ENGINE_DIR, n)
        st = os.stat(fp)
        total += st.st_size
        why = ""
        if parts["family"] != ("onnx" if _is_onnx(path) else "torch"):
            why = "made for this model as another kind of file (.onnx / .pth) - it cannot reuse it"
        elif parts["sha"] != sha:
            why = "built for another copy of this model file"
        elif parts["trt"] != trt:
            why = f"built with TensorRT {parts['trt']} (this is {trt})"
        elif parts["gpu"] != gpu:
            why = f"built for {parts['gpu'].replace('_', ' ')}"
        fits = bool(key) and ((parts["shape"] is not None and tuple(parts["shape"]) == tuple(key)) or _fits(parts, key))
        rows.append({"file": n, "mb": round(st.st_size / 1e6, 1), "built": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
                     "precision": parts["precision"], "opt_level": parts["opt_level"], "tf32": parts["tf32"],
                     "shape": list(parts["shape"]) if parts["shape"] else None,
                     "dynamic": {k: list(v) for k, v in parts["dynamic"].items()} if parts["dynamic"] else None,
                     "trt": parts["trt"], "gpu": parts["gpu"].replace("_", " "), "usable": not why, "why": why, "fits": fits and not why,
                     "current": n == exp, "meta": _meta(n)})
    rows.sort(key=lambda r: (not r["usable"], not r["current"], not r["fits"], r["built"]))
    return {"model": {"name": os.path.basename(path), "mb": round(os.path.getsize(path) / 1e6, 1), "info": inf}, "trt": trt, "gpu": gpu.replace("_", " "),
            "settings": {k: eng[k] for k in ("precision", "opt_level", "workspace", "shape", "tf32", "type", "frames", "args")},
            "profile": [list(s) for s in eng["profile"]] if eng.get("profile") else None,
            "input": list(key) if key else None, "expected": exp, "engines": rows, "total_mb": round(total / 1e6, 1),
            "folder_mb": round(sum(os.path.getsize(os.path.join(ENGINE_DIR, n)) for n in names if os.path.isfile(os.path.join(ENGINE_DIR, n))) / 1e6)}


def pick(model, file):
    """node settings that make `model` use engine `file`"""
    path = str(model or "").strip().strip('"')
    parts = parse(os.path.basename(file))
    if not parts or parts["family"] != ("onnx" if _is_onnx(path) else "torch"):
        return {"error": "this engine cannot be reused by this model"}
    return {"engine": settings_of(parts, path)}


def delete(files):
    """remove engines (and their .json); only files of this folder that look like engines"""
    done, gone_mb = [], 0.0
    for n in files or []:
        n = os.path.basename(str(n))
        fp = os.path.join(ENGINE_DIR, n)
        if not n.endswith(".engine") or not os.path.isfile(fp):
            continue
        try:
            gone_mb += os.path.getsize(fp) / 1e6
            os.remove(fp)
            done.append(n)
            for extra in (fp + ".json",):
                if os.path.isfile(extra):
                    os.remove(extra)
        except OSError:
            pass
    return {"deleted": done, "freed_mb": round(gone_mb, 1)}
