"""AI upscaling step: run your own models (the ones you use in VapourKit) inside EVA toolkit.

Formats
  .onnx                       VapourKit / chaiNNer exports. Fixed-size models (e.g. ..._static_720x540.onnx)
                              get the picture resized to exactly that size first (720x480 NTSC 4:3 -> 720x540
                              square pixels, as VapourKit does); size-free models run at the picture's size.
                              fp32, fp16 and bf16 models.
  .pth / .safetensors         ESRGAN, Compact, SPAN, FDAT, DAT, ... anything spandrel can load.
Where they run: the app's backend setting (Settings). TensorRT compiles each model once per size into
./engines; PyTorch runs .pth models directly and ONNX models through onnxruntime (CUDA). bf16 ONNX models
always use TensorRT (onnxruntime cannot feed bf16 from numpy).
Models are applied one after another (a 1x clean-up model, then a 2x upscaler, ...). Strength < 1 blends
the model's result with a plain Spline36 resize of its input.
"""
import hashlib
import os
import sys
import threading
import time

import numpy as np
import vapoursynth as vs

import nnbackend

core = vs.core
ENGINE_DIR = nnbackend.ENGINE_DIR
_LOCK = threading.Lock()
_CACHE = {}

# Engine build settings, chosen in the dialog that opens when a model is loaded (GUI: AI upscale step).
ENGINE = {
    "precision": "fp16",      # fp16 (fast, like VapourKit's default engines) / bf16 (as fast, fp32's range: no
                              # overflow, a little less fine) / fp32 (exact, slower)
    "opt_level": 3,           # TensorRT builder optimisation level 0-5: higher = longer build, sometimes faster engine
    "workspace": 4096,        # MB of GPU memory TensorRT may use while choosing kernels
    "tf32": True,             # fp32 only: allow TF32 tensor cores (faster, ~fp16-like rounding inside convolutions)
    "type": "image",          # image (one frame in) / vsr (video model: several frames in, the middle one out)
    "shape": "dynamic",       # dynamic (one engine for a range of sizes) / static (one engine per picture size)
    "frames": 0,              # vsr: frames the model takes (0 = from the model / the command)
    "args": "",               # TensorRT build parameters in trtexec syntax (the dialog writes them; they win)
}
PRESETS = {
    "balanced": {"precision": "fp16", "opt_level": 3, "workspace": 4096, "tf32": True},
    "max_speed": {"precision": "fp16", "opt_level": 5, "workspace": 8192, "tf32": True},
    "quick_build": {"precision": "fp16", "opt_level": 1, "workspace": 2048, "tf32": True},
    "safe_speed": {"precision": "bf16", "opt_level": 3, "workspace": 4096, "tf32": True},
    "exact": {"precision": "fp32", "opt_level": 3, "workspace": 4096, "tf32": False},
}


def parse_args(args):
    """trtexec-style build parameters -> settings. Understood: --minShapes / --optShapes / --maxShapes /
    --shapes=<input>:1x3xHxW (H / W may be letters = the picture size: static), --builderOptimizationLevel=N,
    --fp16 / --bf16 (none = fp32), --memPoolSize=workspace:NNNNM, --noTF32. --saveEngine is ignored (engines
    are kept in engines/). Unknown flags are reported, not fatal."""
    out, unknown = {}, []
    dims = lambda v: tuple(int(x) if x.isdigit() else None for x in v.split(":", 1)[-1].split("x"))
    prec = "fp32"
    for tok in str(args or "").split():
        k, _, v = tok.partition("=")
        if k in ("--minShapes", "--optShapes", "--maxShapes", "--shapes"):
            out[k[2:5].lower() if k != "--shapes" else "shapes"] = dims(v)
        elif k == "--builderOptimizationLevel":
            out["opt_level"] = int(v)
        elif k == "--fp16":
            prec = "fp16"
        elif k == "--bf16":
            prec = "bf16"
        elif k == "--memPoolSize" and v.startswith("workspace:"):
            n = v.split(":", 1)[1].upper().rstrip("B")
            out["workspace"] = int(float(n[:-1]) * (1024 if n.endswith("G") else 1)) if n[-1] in "MG" else int(n) >> 20
        elif k == "--noTF32":
            out["tf32"] = False
        elif k in ("--saveEngine", "--onnx"):
            pass
        elif tok:
            unknown.append(tok)
    out["precision"] = prec
    if "shapes" in out:
        out["shape"] = "static"
    elif all(x in out for x in ("min", "opt", "max")):
        out["shape"] = "dynamic"
        if None in out["min"] + out["opt"] + out["max"]:
            out["shape"] = "static"
    out["unknown"] = unknown
    return out


def engine_settings(m):
    e = dict(ENGINE)
    e.update({k: v for k, v in (m.get("engine") or {}).items() if k in ENGINE and v is not None})
    e["profile"] = None
    if e.get("args"):
        p = parse_args(e["args"])
        for k in ("precision", "opt_level", "workspace", "tf32", "shape"):
            if k in p:
                e[k] = p[k]
        if p.get("shape") == "dynamic":
            e["profile"] = (p["min"], p["opt"], p["max"])
            if len(p["opt"]) == 5:
                e["frames"] = p["opt"][1]
            elif len(p["opt"]) == 4 and p["opt"][1] > 3 and p["opt"][1] % 3 == 0:
                e["frames"] = p["opt"][1] // 3
        elif "shapes" in p:
            s = p["shapes"]
            if len(s) == 5 and s[1]:
                e["frames"] = s[1]
    e["opt_level"] = int(max(0, min(5, int(e["opt_level"]))))
    e["workspace"] = int(max(512, min(16384, int(e["workspace"]))))
    return e


def _sha(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def info(path):
    """What the model is: kind, scale (when known), fixed input size, precision. For the GUI."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".onnx":
        import onnx
        m = onnx.load(path, load_external_data=False)
        i = m.graph.input[0]
        dims = [d.dim_value or 0 for d in i.type.tensor_type.shape.dim]
        el = {1: "fp32", 10: "fp16", 16: "bf16"}.get(i.type.tensor_type.elem_type, "?")
        fixed = (dims[-1], dims[-2]) if len(dims) in (4, 5) and dims[-2] and dims[-1] else None
        # video models: [1, T, 3, H, W], or the frames stacked in the channels [1, 3T, H, W]
        frames, layout = None, "image"
        if len(dims) == 5:
            frames, layout = (dims[1] or None), "5d"
        elif len(dims) == 4 and dims[1] > 3 and dims[1] % 3 == 0:
            frames, layout = dims[1] // 3, "4d"
        scale = None
        name = os.path.basename(path).lower()
        for s in (1, 2, 3, 4, 8):
            if name.startswith(f"{s}x") or f"_{s}x" in name or f"{s}x_" in name[:4]:
                scale = s
                break
        return {"kind": "onnx", "fixed": fixed, "precision": el, "scale": scale, "channels": dims[1] if len(dims) > 1 else 3,
                "input_name": i.name, "layout": layout, "frames": frames, "vsr": layout != "image"}
    import spandrel
    d = spandrel.ModelLoader().load_from_file(path)
    return {"kind": "pytorch", "arch": d.architecture.name, "scale": d.scale, "fixed": None,
            "precision": "fp16" if d.supports_half else "fp32", "channels": d.input_channels}


# ---------------------------------------------------------------- runners: [1,3,H,W] torch cuda float -> [1,3,h,w] float
FP32_OPS = ["Softmax", "ReduceMean", "ReduceSum", "Pow", "Sqrt", "Reciprocal", "Exp", "Erf",
            "ReduceL2", "ReduceSumSquare"]     # L2 norms (FDAT's cosine attention): a sum of squares overflows fp16 (7.4 levels off)
# (normalisation layers cannot be kept fp32: TensorRT needs their input and scale in the same type)
CONV_VERSION = 3            # fp16 / bf16 conversion: 3 = Clip limits follow the data, L2 norms fp32 (2026-11-22)
FP16_MAX_DIFF = 0.75        # mean 8-bit levels: above this the fp16 engine is not used (see _fp16_check)


def _to_bf16(m):
    """fp16 graph (from the fp16 conversion) -> bf16: same layers in reduced precision, fp32's range."""
    from onnx import TensorProto, numpy_helper

    def bf16_tensor(arr, name):
        a = np.ascontiguousarray(arr.astype(np.float32))
        u = a.view(np.uint32)
        r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)          # round to nearest even
        t = TensorProto()
        t.name, t.data_type = name, TensorProto.BFLOAT16
        t.dims.extend(a.shape)
        t.raw_data = r.tobytes()
        return t
    for t in m.graph.initializer:
        if t.data_type == TensorProto.FLOAT16:
            t.CopyFrom(bf16_tensor(numpy_helper.to_array(t), t.name))
    for v in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
        if v.type.tensor_type.elem_type == TensorProto.FLOAT16:
            v.type.tensor_type.elem_type = TensorProto.BFLOAT16
    for n in m.graph.node:
        for a in n.attribute:
            if a.name == "to" and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.BFLOAT16
            if a.name == "value" and a.t.data_type == TensorProto.FLOAT16:
                a.t.CopyFrom(bf16_tensor(numpy_helper.to_array(a.t), a.t.name))
    return m


def _fix_clip_types(m):
    """After the fp16 conversion: a Clip whose min / max come from the model's own Cast(to=float32) (FDAT's attention
    clamps its L2 norm that way) got half data with a float limit - TensorRT refused the whole file ("MAX must have
    same input types: Half and Float", 1xGTv4.5s-FDAT: no bf16 / fp16 engine, fp32 only). Those casts now follow
    the data (fp16 here, bf16 after _to_bf16); float32 initializers / constants as limits are converted too."""
    from onnx import TensorProto, numpy_helper
    prod = {o: n for n in m.graph.node for o in n.output}
    init = {i.name: i for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type != "Clip":
            continue
        for name in list(n.input[1:]):
            if not name:
                continue
            if name in init and init[name].data_type == TensorProto.FLOAT:
                init[name].CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init[name]).astype(np.float16), name))
                continue
            p_ = prod.get(name)
            if p_ is None:
                continue
            for a in p_.attribute:
                if p_.op_type == "Cast" and a.name == "to" and a.i == TensorProto.FLOAT:
                    a.i = TensorProto.FLOAT16
                if p_.op_type == "Constant" and a.name == "value" and a.t.data_type == TensorProto.FLOAT:
                    a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float16), a.t.name))
    return m


def _fp32_island(m, keep):
    """Nodes for which keep(node) is true run in fp32 inside the fp16 graph: Cast(float32) on every computed input,
    their float16 weights back to float32 (copied if shared), Cast(float16) on every output. onnxconverter's own
    op_block_list keeps the weights fp32 but leaves the input half, which TensorRT refuses ("INormalizationLayer
    input and scale must have identical types", "IDeconvolutionLayer input and kernel_weights must have same type")."""
    from onnx import TensorProto, helper, numpy_helper
    init = {i.name: i for i in m.graph.initializer}
    uses = {}
    for n in m.graph.node:
        for x in n.input:
            uses[x] = uses.get(x, 0) + 1
    consts = {o: n for n in m.graph.node if n.op_type == "Constant" for o in n.output}
    nodes, k = [], 0
    for n in m.graph.node:
        if not keep(n):
            nodes.append(n)
            continue
        k += 1
        tag = f"_f32isl{k}"
        for idx, x in enumerate(list(n.input)):
            if not x:
                continue
            if x in init:
                if init[x].data_type == TensorProto.FLOAT16:
                    a = numpy_helper.to_array(init[x]).astype(np.float32)
                    if uses.get(x, 0) > 1:
                        nm = x + tag
                        m.graph.initializer.append(numpy_helper.from_array(a, nm))
                        n.input[idx] = nm
                    else:
                        init[x].CopyFrom(numpy_helper.from_array(a, x))
                continue
            if x in consts and n.op_type in ("Reshape", "Expand", "Slice", "Unsqueeze", "Squeeze", "Tile"):
                continue                                 # shape inputs (int64) stay as they are
            c = x + tag + f"_{idx}"
            nodes.append(helper.make_node("Cast", [x], [c], to=TensorProto.FLOAT, name=c + "_cast"))
            n.input[idx] = c
        outs = list(n.output)
        for idx, o in enumerate(outs):
            o32 = o + tag + "_o"
            n.output[idx] = o32
        nodes.append(n)
        for idx, o in enumerate(outs):
            nodes.append(helper.make_node("Cast", [o + tag + "_o"], [o], to=TensorProto.FLOAT16, name=o + tag + "_ocast"))
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    return m


def _prepared_onnx(path, fp16=True):
    """TensorRT 11 builds strongly typed networks: an fp32 model runs in fp32 (slow), and bf16 models fail on
    some layers (no bf16 ConvTranspose). Like VapourKit's fp16 engines, the model is converted once to fp16
    inside (inputs / outputs stay as they were for fp32 models, bf16 becomes fp16) and cached in ./engines."""
    import onnx
    from onnx import TensorProto, numpy_helper
    m = onnx.load(path)
    has_bf16 = any(i.data_type == TensorProto.BFLOAT16 for i in m.graph.initializer) or \
        any(v.type.tensor_type.elem_type == TensorProto.BFLOAT16 for v in list(m.graph.input) + list(m.graph.output))
    prec = fp16 if isinstance(fp16, str) else ("fp16" if fp16 else "fp32")
    fp16 = prec in ("fp16", "bf16")
    if not fp16 and not has_bf16:
        return path
    n16 = sum(1 for i in m.graph.initializer if i.data_type == TensorProto.FLOAT16)
    if prec == "fp16" and not has_bf16 and n16 > sum(1 for i in m.graph.initializer if i.data_type == TensorProto.FLOAT):
        return path                                      # already an fp16 model (VapourKit's _fp16 exports): as it is
    out = os.path.join(ENGINE_DIR, f"user_{os.path.splitext(os.path.basename(path))[0][:40]}_{_sha(path)}_{ {'fp16': 'fp16c3', 'bf16': 'bf16c5', 'fp32': 'fp32'}[prec]}.onnx")
    if os.path.isfile(out):
        return out
    os.makedirs(ENGINE_DIR, exist_ok=True)
    if has_bf16:                                         # bf16 -> fp32 everywhere first
        for t in m.graph.initializer:
            if t.data_type == TensorProto.BFLOAT16:
                a = numpy_helper.to_array(t).astype(np.float32)
                t.CopyFrom(numpy_helper.from_array(a, t.name))
        for v in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
            if v.type.tensor_type.elem_type == TensorProto.BFLOAT16:
                v.type.tensor_type.elem_type = TensorProto.FLOAT
        for n in m.graph.node:
            for a in n.attribute:
                if a.name == "to" and a.i == TensorProto.BFLOAT16:
                    a.i = TensorProto.FLOAT
                if a.name == "value" and a.t.data_type == TensorProto.BFLOAT16:
                    a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32), a.t.name))
    if fp16:
        from onnxconverter_common import float16
        # layers that overflow or lose too much in fp16 stay fp32 (attention softmax, normalisation, powers):
        # without this, transformer models (DAT / FDAT-style attention) went wrong by up to 144 levels
        # bf16: TensorRT has no bf16 ConvTranspose (FDAT's upsampler: "Could not find any implementation"), so that
        # layer stays fp32; bf16 keeps fp32's range, so attention models stay accurate (fp16 was 7.4 levels off)
        m = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True, op_block_list=FP32_OPS)
        m = _fix_clip_types(m)
        if prec == "bf16":
            m = _to_bf16(m)
    onnx.save(m, out + ".part.onnx")
    os.replace(out + ".part.onnx", out)
    return out


class _TRTOnnx:
    """TensorRT engine from the ONNX file (strongly typed; converted to fp16 inside first, see above)."""

    def __init__(self, path, eng=None):
        self.eng = eng or dict(ENGINE)
        prec = self.eng["precision"] if self.eng["precision"] in ("fp16", "bf16", "fp32") else "fp16"
        fp16 = prec != "fp32"
        self.orig = path
        self.path = _prepared_onnx(path, prec)
        # CONV_VERSION in the tag: engines and fp16 verdicts are cached by it, so a changed conversion is used (and
        # re-checked) instead of an engine built from the old one
        self.tag = _sha(path) + {"fp16": "hc", "bf16": "bc", "fp32": "f"}[prec] + (f"{CONV_VERSION}" if fp16 else "") +             f"o{self.eng['opt_level']}" + ("" if fp16 or self.eng["tf32"] else "n")
        self.profile = self.eng.get("profile") if self.eng.get("shape") == "dynamic" else None
        try:
            if info(path)["fixed"]:                       # fixed-size model: one size only, no range
                self.profile = None
        except Exception:                                # noqa: BLE001
            pass
        self.ctx = {}

    def _fits(self, shape):
        p = self.profile
        return p is not None and len(p[0]) == len(shape) and all(a <= s <= b for a, s, b in zip(p[0], shape, p[2]))

    def _engine(self, shape, dtype):
        import tensorrt as trt
        import torch
        key = tuple(shape)
        dyn = self._fits(key)
        if dyn and "dyn" in self.ctx:
            c = self.ctx["dyn"]
            if c[6] != key:                                # same engine, another size: set it, read the output size
                c[0].set_input_shape(c[1], key)
                self.ctx["dyn"] = c = c[:5] + (tuple(c[0].get_tensor_shape(c[2])), key)
            return c[:6]
        if key in self.ctx:
            return self.ctx[key]
        gpu = torch.cuda.get_device_name(0).replace(" ", "_")
        p = self.profile
        shp = ("dyn_" + "-".join("x".join(map(str, s)) for s in p)) if dyn else "x".join(map(str, shape))
        ep = os.path.join(ENGINE_DIR, f"user_{os.path.splitext(os.path.basename(self.orig))[0][:40]}_{self.tag}_"
                                      f"{shp}_trt{trt.__version__}_{gpu}.engine")
        if self.profile is not None and not dyn:
            print(f"{os.path.basename(self.orig)}: {'x'.join(map(str, shape))} is outside the engine's size range "
                  f"({'x'.join(map(str, p[0]))} - {'x'.join(map(str, p[2]))}) - building a static engine for it",
                  file=sys.stderr, flush=True)
        logger = __import__('nnbackend').trt_logger(trt)
        if not (os.path.isfile(ep) and os.path.getsize(ep) > 512):
            what = (f"sizes {p[0][-1]}x{p[0][-2]} to {p[2][-1]}x{p[2][-2]}" if dyn else f"{shape[-1]}x{shape[-2]}")
            print(f"TensorRT: building an engine for {os.path.basename(self.path)}, {what} "
                  f"(once, up to a few minutes)...", file=sys.stderr, flush=True)
            os.makedirs(ENGINE_DIR, exist_ok=True)
            b = trt.Builder(logger)
            net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
            parser = trt.OnnxParser(net, logger)
            if not parser.parse_from_file(self.path):
                raise RuntimeError("TensorRT cannot read this ONNX file: " +
                                   "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            cfg = b.create_builder_config()
            cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(self.eng["workspace"]) << 20)
            cfg.builder_optimization_level = int(self.eng["opt_level"])
            _mon = nnbackend.progress_monitor(trt, os.path.basename(self.orig))
            if _mon is not None:
                cfg.progress_monitor = _mon
            if not self.eng["tf32"]:
                cfg.clear_flag(trt.BuilderFlag.TF32)
            prof = b.create_optimization_profile()
            nm = net.get_input(0).name
            if dyn:
                prof.set_shape(nm, p[0], p[1], p[2])
            else:
                prof.set_shape(nm, key, key, key)
            cfg.add_optimization_profile(prof)
            _t_build = time.time()
            blob = b.build_serialized_network(net, cfg)
            if blob is None and dyn:
                # a size range the model cannot be built for (memory of attention layers at the largest size,
                # layers without a kernel for some sizes): one engine per picture size instead
                print(f"{os.path.basename(self.orig)}: TensorRT cannot build one engine for sizes "
                      f"{p[0][-1]}x{p[0][-2]} to {p[2][-1]}x{p[2][-2]} - using a static engine per picture size "
                      f"(a smaller --maxShapes may work)", file=sys.stderr, flush=True)
                self.profile = None
                return self._engine(shape, dtype)
            if blob is None:
                raise RuntimeError("TensorRT engine build failed")
            nnbackend.save_engine(ep, blob)
            try:                                          # how long it took and with what (the model library lists it)
                import engines
                engines.write_meta(ep, time.time() - _t_build, self.eng, p if dyn else None)
            except Exception:                            # noqa: BLE001
                pass
        rt = nnbackend.runtime()                          # process-wide, outlives the engines (see nnbackend)
        with open(ep, "rb") as f:
            eng = rt.deserialize_cuda_engine(f.read())
        self._engines = getattr(self, "_engines", []) + [eng]
        ctx = eng.create_execution_context()
        names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
        iname = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
        oname = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
        ctx.set_input_shape(iname, key)
        dt = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32, trt.DataType.BF16: torch.bfloat16}
        idt, odt = dt[eng.get_tensor_dtype(iname)], dt[eng.get_tensor_dtype(oname)]
        oshape = tuple(ctx.get_tensor_shape(oname))
        if dyn:
            self.ctx["dyn"] = (ctx, iname, oname, idt, odt, oshape, key)
            return self.ctx["dyn"][:6]
        self.ctx[key] = (ctx, iname, oname, idt, odt, oshape)
        return self.ctx[key]

    def __call__(self, x):
        with nnbackend.TRT_LOCK:                          # see nnbackend.TRT_LOCK
            return self._call(x)

    def _call(self, x):
        import torch
        if getattr(self, "fallback", None) is not None:
            return self.fallback(x)
        if self.eng["precision"] in ("fp16", "bf16") and not getattr(self, "checked", False):
            nxt = "fp16" if self.eng["precision"] == "bf16" else "fp32"
            # a precision that could not be built for this model and size is remembered next to the engines: the
            # next run goes straight on (1xGTv4.5s-FDAT: bf16 failed - and printed TensorRT's dump - on every run)
            mark = os.path.join(ENGINE_DIR, f"user_{self.tag}_{'x'.join(map(str, x.shape))}.nobuild")
            if os.path.isfile(mark):
                print(f"{os.path.basename(self.orig)}: no {self.eng['precision']} engine for this model (found before) - "
                      f"using {nxt}", file=sys.stderr, flush=True)
                self.checked = True
                self.fallback = _TRTOnnx(self.orig, dict(self.eng, precision=nxt))
                return self.fallback(x)
            try:
                self._check(x)
            except RuntimeError as e:                    # e.g. a layer TensorRT has no bf16 version of
                print(f"{os.path.basename(self.orig)}: {self.eng['precision']} engine cannot be built ({str(e)[:120]}) - "
                      f"using {nxt}", file=sys.stderr, flush=True)
                try:
                    with open(mark, "w", encoding="utf-8") as f:
                        f.write(str(e)[:500])
                except OSError:
                    pass
                self.fallback = _TRTOnnx(self.orig, dict(self.eng, precision=nxt))
            if getattr(self, "fallback", None) is not None:
                return self.fallback(x)
        ctx, iname, oname, idt, odt, oshape = self._engine(tuple(x.shape), x.dtype)
        xi = x.to(idt).contiguous()
        out = torch.empty(oshape, dtype=odt, device=x.device)
        ctx.set_tensor_address(iname, xi.data_ptr())
        ctx.set_tensor_address(oname, out.data_ptr())
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()
        return out.float()


def _fp16_check(self, x):
    """First frame of an fp16 engine: compare with the fp32 engine on that frame. Some models (attention,
    large activations) go wrong in fp16 - one of the user's models by up to 144 levels; then fp32 is used.
    The verdict is kept next to the engines, so the check (and the extra fp32 build) happens once."""
    import json
    import torch
    self.checked = True
    key = os.path.join(ENGINE_DIR, f"user_{self.tag}_{'x'.join(map(str, x.shape))}.fp16check.json")
    ref = None
    try:
        v = json.load(open(key))
    except Exception:                                    # noqa: BLE001
        ref = _TRTOnnx(self.orig, dict(self.eng, precision="fp32", tf32=False))
        with torch.no_grad():
            a = ref(x).clamp(0, 1)
            b = self(x).clamp(0, 1)
        d = float((a - b).abs().mean()) * 255
        v = {"mean_diff": round(d, 3), "ok": d <= FP16_MAX_DIFF}
        json.dump(v, open(key, "w"))
    self.fp16_check = v
    if not v["ok"]:
        print(f"{os.path.basename(self.orig)}: {self.eng['precision']} changes the picture by {v['mean_diff']:.1f} levels on average - "
              f"using fp32 for this model", file=sys.stderr, flush=True)
        self.fallback = ref or _TRTOnnx(self.orig, dict(self.eng, precision="fp32", tf32=False))


_TRTOnnx._check = _fp16_check


class _ORT:
    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        self.s = ort.InferenceSession(path, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.i = self.s.get_inputs()[0]
        self.fp16 = "float16" in self.i.type

    def __call__(self, x):
        import torch
        a = x.float().cpu().numpy().astype(np.float16 if self.fp16 else np.float32)
        o = self.s.run(None, {self.i.name: a})[0]
        return torch.from_numpy(o.astype(np.float32)).to(x.device)


class _Torch:
    def __init__(self, path, eng=None):
        import spandrel
        import torch
        self.eng = eng or dict(ENGINE)
        d = spandrel.ModelLoader().load_from_file(path)
        p = self.eng["precision"]
        if p == "bf16" and not d.supports_bfloat16:
            p = "fp16"
        if p == "fp16" and not d.supports_half:
            p = "fp32"
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[p]
        self.prec = p
        self.half = p == "fp16"
        self.net = d.model.eval().cuda().to(self.dtype)
        self.path = path

    def __call__(self, x):
        e = self.eng
        net = nnbackend.wrap("user_" + os.path.splitext(os.path.basename(self.path))[0][:40] + f"_{self.prec}"
                             + f"_o{e['opt_level']}", self.net, self.path, level=e["opt_level"], workspace=e["workspace"])
        if net is not self.net and not getattr(self, "checked", False):
            # TensorRT engine vs the same model in PyTorch, first frame: one of the user's models (2x CelPrime V3)
            # came out 4 levels off in a TensorRT fp16 engine while PyTorch fp16 was exact to 0.04
            self.checked = True
            import torch
            with torch.no_grad():
                a = self.net(x.to(self.dtype)).float().clamp(0, 1)
                b = net(x.to(self.dtype)).float().clamp(0, 1)
            d = float((a - b).abs().mean()) * 255
            if d > FP16_MAX_DIFF:
                print(f"{os.path.basename(self.path)}: the TensorRT engine differs from PyTorch by {d:.1f} levels on "
                      f"average - running this model on PyTorch", file=sys.stderr, flush=True)
                self.no_trt = True
        if getattr(self, "no_trt", False):
            net = self.net
        return net(x.to(self.dtype)).float()


def _runner(path, eng=None):
    eng = eng or dict(ENGINE)
    key = (path, nnbackend.backend(), tuple(sorted(eng.items())))
    if key not in _CACHE:
        ext = os.path.splitext(path)[1].lower()
        if ext != ".onnx":
            _CACHE[key] = _Torch(path, eng)
        elif nnbackend.tensorrt_available():
            # onnxruntime's CUDA provider needs CUDA 12 libraries that a CUDA 13 PyTorch setup does not
            # have (it falls back to the CPU: 0.2 fps), so ONNX models always run on TensorRT when possible
            _CACHE[key] = _TRTOnnx(path, eng)
        else:
            _CACHE[key] = _ORT(path)
    return _CACHE[key]


def _vsr_input(frames, layout):
    """frames: list of [1,3,H,W] tensors (oldest first) -> the model's input."""
    import torch
    return torch.stack(frames, 1) if layout == "5d" else torch.cat(frames, 1)


def _vsr_output(o, T):
    """The middle frame of a video model's output ([1,T,3,h,w] or [1,3T,h,w]); [1,3,h,w] as it is."""
    if o.dim() == 5:
        return o[:, o.shape[1] // 2]
    if o.dim() == 4 and o.shape[1] == 3 * T and T > 1:
        return o[:, 3 * (T // 2):3 * (T // 2) + 3]
    return o


def _probe_scale(path, w, h, eng=None, vsr=None):
    """Output size of the model for a w x h input (one dummy run; builds the TensorRT engine if needed)."""
    import torch
    r = _runner(path, eng)
    with _LOCK, torch.no_grad():
        if vsr:
            T, layout = vsr
            o = _vsr_output(r(_vsr_input([torch.zeros(1, 3, h, w, device="cuda")] * T, layout)), T)
        else:
            o = r(torch.zeros(1, 3, h, w, device="cuda"))
    return o.shape[-1], o.shape[-2]


def run_model(rgb, path, strength=1.0, eng=None):
    """rgb: RGBS clip. Returns RGBS at the model's output size."""
    import torch
    inf = info(path)
    if inf["fixed"]:
        fw, fh = inf["fixed"]
        if (rgb.width, rgb.height) != (fw, fh):
            rgb = core.resize.Spline36(rgb, fw, fh)
    w, h = rgb.width, rgb.height
    h_ = h
    pad = 0 if inf["fixed"] else 16
    pw, ph = (w + pad * 2 + 7) // 8 * 8 if pad else w, (h + pad * 2 + 7) // 8 * 8 if pad else h
    es = eng or {}
    vsr = None
    if es.get("type") == "vsr" or inf.get("vsr"):
        T = int(inf.get("frames") or es.get("frames") or 5)
        vsr = (T, inf.get("layout") if inf.get("layout") in ("5d", "4d") else "5d")
    ow, oh = _probe_scale(path, pw, ph, eng, vsr)
    sx, sy = ow / pw, oh / ph
    out_w, out_h = int(round(w * sx)), int(round(h * sy))
    base = core.std.BlankClip(rgb, width=out_w, height=out_h, format=vs.RGBS)
    plain = core.resize.Spline36(rgb, out_w, out_h) if strength < 1 else None

    T = vsr[0] if vsr else 1
    r = T // 2
    n_fr = rgb.num_frames
    def shifted(k):                                     # clip whose frame n is rgb's frame n+k (ends repeated)
        if k < 0:
            return rgb[0] * (-k) + rgb[:n_fr + k]
        if k > 0:
            return rgb[k:] + rgb[-1] * k
        return rgb
    srcs = [shifted(k) for k in range(-r, r + 1)] if vsr else [rgb]

    # Only the GPU work holds the lock: the frames are put together and the result copied into the output frame by
    # the calling thread (VapourSynth runs several at once), so the GPU is not idle while the CPU copies. Video models
    # keep the frames already on the GPU (each source frame is needed by T output frames). 2026-11-26: same
    # pictures, 2x AnimeJaNai UltraCompact 8.5 -> 7.5 ms/frame.
    import collections
    gpu_cache = collections.OrderedDict()
    local = threading.local()

    def host(fr):
        return np.stack([np.asarray(fr[p]) for p in range(3)])

    def to_gpu(h, key):
        if key is not None and key in gpu_cache:
            gpu_cache.move_to_end(key)
            return gpu_cache[key]
        t = torch.from_numpy(h)[None].cuda(non_blocking=False).clamp_(0, 1)
        t = torch.nn.functional.pad(t, (pad, pw - w - pad, pad, ph - h_ - pad), mode="reflect") if pad else t
        if key is not None:
            gpu_cache[key] = t
            while len(gpu_cache) > 2 * T + 2:
                gpu_cache.popitem(last=False)
        return t

    def pinned(shape):
        b = getattr(local, "buf", None)
        if b is None or tuple(b.shape) != tuple(shape):
            b = local.buf = torch.empty(shape, dtype=torch.float32, pin_memory=True)
        return b

    def proc(n, f):
        hs = [host(x) for x in f[:T]] if vsr else [host(f[0])]
        keys = [min(max(n + k, 0), n_fr - 1) for k in range(-r, r + 1)] if vsr else [None]
        with _LOCK, torch.no_grad():
            ts = [to_gpu(hh, k) for hh, k in zip(hs, keys)]
            if vsr:
                o = _vsr_output(_runner(path, eng)(_vsr_input(ts, vsr[1])), T)
            else:
                o = _runner(path, eng)(ts[0])
            if pad:
                o = o[..., int(pad * sy):int(pad * sy) + out_h, int(pad * sx):int(pad * sx) + out_w]
            o = o.clamp(0, 1)[0]
            buf = pinned(o.shape)
            buf.copy_(o)
        a = buf.numpy()
        dst = f[-1].copy()
        for p in range(3):
            np.copyto(np.asarray(dst[p]), a[p])
        return dst

    out = core.std.ModifyFrame(base, clips=srcs + [base], selector=proc)
    if plain is not None:
        out = core.std.Merge(plain, out, weight=float(strength))
    return out


def detect_borders(clip, threshold=24, samples=40):
    """Black borders of a YUV clip (like VapourKit's auto crop): a row / column is border when its brightest
    typical value stays at or below `threshold` (8-bit) over the sampled frames; dark fades are skipped.
    Returns l, r, t, b (even)."""
    import numpy as np
    y = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    sc = 255.0 / ((1 << y.format.bits_per_sample) - 1) if y.format.sample_type == vs.INTEGER else 255.0
    n = y.num_frames
    acc = []
    for t in np.linspace(n * 0.05, n * 0.95, samples).astype(int):
        a = np.asarray(y.get_frame(int(t))[0], dtype=np.float32) * sc
        if a.mean() >= 30:
            acc.append(a)
    if not acc:
        return 0, 0, 0, 0
    A = np.stack(acc)
    col, row = np.percentile(A, 98, axis=(0, 1)), np.percentile(A, 98, axis=(0, 2))

    def run(v):
        k = 0
        while k < len(v) // 3 and v[k] <= threshold:
            k += 1
        return k
    up = lambda x: (x + 1) // 2 * 2
    return up(run(col)), up(run(col[::-1])), up(run(row)), up(run(row[::-1]))


def modulus(rgb, mod, mode="pad"):
    """Width and height to a multiple of `mod` (many models need 2 / 4 / 8 / 16): pad with black or crop,
    centred."""
    mod = max(2, int(mod))
    w, h = rgb.width, rgb.height
    if mode == "crop":
        nw, nh = w // mod * mod, h // mod * mod
        if (nw, nh) == (w, h):
            return rgb
        l, t = (w - nw) // 2, (h - nh) // 2
        return core.std.CropAbs(rgb, nw, nh, l, t)
    nw, nh = -(-w // mod) * mod, -(-h // mod) * mod
    if (nw, nh) == (w, h):
        return rgb
    l, t = (nw - w) // 2, (nh - h) // 2
    return core.std.AddBorders(rgb, l, nw - w - l, t, nh - h - t, color=[0, 0, 0])


def _wavelet_low(c, levels):
    """Low frequencies of a wavelet decomposition (a trous: [1 2 1]/4 blur, dilation 1, 2, 4, ... per level)."""
    for i in range(levels):
        d = 1 << i
        c = core.akarin.Expr(c, f"x[-{d},0] x 2 * + x[{d},0] + 4 /")
        c = core.akarin.Expr(c, f"x[0,-{d}] x 2 * + x[0,{d}] + 4 /")
    return c


def wavelet_colour_fix(rgb, ref, levels=5, strength=1.0):
    """Wavelet colour fix (as in StableSR / chaiNNer): the upscaled picture keeps its detail (high
    frequencies) and takes its colour and brightness at large scale (low frequencies) from the picture
    before the model, so a model that shifts colour, brightness or contrast is put back. ref is resized to
    the model's output size first. strength 0..1 blends the corrected result with the model's."""
    levels = max(1, min(8, levels))
    if (ref.width, ref.height) != (rgb.width, rgb.height):
        ref = core.resize.Bicubic(ref, rgb.width, rgb.height)
    lo_out, lo_ref = _wavelet_low(rgb, levels), _wavelet_low(ref, levels)
    s = max(0.0, min(1.0, strength))
    return core.akarin.Expr([rgb, lo_out, lo_ref], f"x z y - {s} * +")


def apply(clip, job, models, matrix):
    """clip: YUV (any). models: the AI upscale chain in order, each {"type": "model" (path, strength, engine) |
    "modulus" (mod, mode) | "autocrop" (threshold)}. Returns YUV444P16."""
    nnbackend.set_backend(job.get("backend", "pytorch"))
    props = clip
    rgb = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in=matrix, range_in=0)
    ref = rgb                                       # picture before the latest model (wavelet colour fix)
    pads = []                                       # padding Modulus added: (right, bottom, padded w, padded h)
    for m in models:
        kind = m.get("type", "model")
        if kind == "wavelet":
            rgb = wavelet_colour_fix(rgb, ref, int(m.get("levels", 5)), float(m.get("strength", 1.0)))
            continue
        if kind == "modulus":
            # as vs_tiletools.mod (VapourKit's Modulus): pad right / bottom to the next multiple - mirrored by default,
            # so the model sees picture, not a black edge it smears - or crop to the lower multiple ("discard").
            # The reference picture of the wavelet colour fix gets the same geometry (scaled), so the two stay aligned.
            import vs_tiletools
            mod = max(1, int(m.get("mod", 8)))
            mode = {"pad": "mirror", "crop": "discard"}.get(m.get("mode", "mirror"), m.get("mode", "mirror"))
            w, h = rgb.width, rgb.height
            sx, sy = ref.width / w, ref.height / h
            if mode == "discard":
                cr, cb = w % mod, h % mod
                if cr or cb:
                    same = ref is rgb
                    rgb = core.std.Crop(rgb, right=cr, bottom=cb)
                    ref = rgb if same else core.std.Crop(ref, right=int(round(cr * sx)), bottom=int(round(cb * sy)))
                continue
            pw, ph = (-w) % mod, (-h) % mod
            if pw or ph:
                same = ref is rgb
                rgb = vs_tiletools.pad(rgb, right=pw, bottom=ph, mode=mode)
                ref = rgb if same else vs_tiletools.pad(ref, right=int(round(pw * sx)), bottom=int(round(ph * sy)), mode=mode)
            pads.append((pw, ph, w + pw, h + ph))
            continue
        if kind == "tfix":
            # TemporalFix AI at this point of the chain (e.g. after an upscaler: line wobble the model adds)
            st = round(max(0.0, min(3.0, float(m.get("strength", 2.0)))), 2)
            if st > 0:
                import trt_temporalfix
                if trt_temporalfix.available():
                    out_t = trt_temporalfix.temporalfix(rgb, strength=st, log=lambda s: print(s, file=sys.stderr, flush=True))
                else:
                    import vs_temporalfix
                    out_t = vs_temporalfix.model(core.std.Expr(rgb, "x 0 max 1 min"), strength=st, backend="cuda")
                rgb = core.resize.Point(out_t, format=vs.RGBS)
            continue
        if kind == "gtfix":
            # guided temporal fix (gtfix.py): steadies what the model above added, judged on that model's input
            st = max(0.0, float(m.get("strength", 6.0)))
            if st > 0:
                import gtfix
                yo = core.resize.Bicubic(rgb, format=vs.YUV444P16, matrix_s="709")
                yg = core.resize.Bicubic(ref, format=vs.YUV444P16, matrix_s="709") if ref is not rgb else None
                fixed = gtfix.gtfix(yo, yg, radius=int(m.get("radius", 3)), strength=st)
                rgb = core.resize.Bicubic(fixed, format=vs.RGBS, matrix_in_s="709")
            continue
        if kind == "autocrop":
            if pads:
                # as vs_tiletools.crop (VapourKit's Crop (Auto)): removes exactly the padding the last Modulus added,
                # scaled to the size the picture has now (after a 2x model: twice the padding)
                pw, ph, PW, PH = pads.pop()
                cr, cb = int(round(pw * rgb.width / PW)), int(round(ph * rgb.height / PH))
                rr, rb = int(round(pw * ref.width / PW)), int(round(ph * ref.height / PH))
                same = ref is rgb
                if cr or cb:
                    rgb = core.std.Crop(rgb, right=cr, bottom=cb)
                ref = rgb if same else (core.std.Crop(ref, right=rr, bottom=rb) if rr or rb else ref)
                continue
            # no Modulus before it: black borders found once on the chain's input, scaled to the size now
            l, r, t, b = detect_borders(clip, int(m.get("threshold", 24)))
            sx, sy = rgb.width / clip.width, rgb.height / clip.height
            ev = lambda v, s: int(round(v * s / 2)) * 2
            l, r, t, b = ev(l, sx), ev(r, sx), ev(t, sy), ev(b, sy)
            if l + r < rgb.width - 16 and t + b < rgb.height - 16 and (l or r or t or b):
                same = ref is rgb
                kx, ky = ref.width / rgb.width, ref.height / rgb.height
                rgb = core.std.Crop(rgb, l, r, t, b)
                ref = rgb if same else core.std.Crop(ref, *(int(round(v * k)) for v, k in ((l, kx), (r, kx), (t, ky), (b, ky))))
            continue
        ref = rgb
        rgb = run_model(rgb, m["path"], float(m.get("strength", 1.0)), engine_settings(m))
    out = core.resize.Bicubic(rgb, format=vs.YUV444P16, matrix=matrix, range=0)
    return core.std.CopyFrameProps(out, props)
