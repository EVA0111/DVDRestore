"""TemporalFix AI on TensorRT without the vs-mlrt plugin.

vs-mlrt's vstrt plugin uses VapourSynth API 3, which VapourSynth R81 no longer loads. This runs the same
TemporalFix ONNX models with the TensorRT Python package instead: the engine is built once per strength and
frame size (cached in ./engines), inference runs on GPU buffers owned by PyTorch, and frames are prepared
exactly like vs_temporalfix does (clamped RGBH, 7 frames: n-3 .. n+3, concatenated to 21 channels).
"""
import os
from collections import OrderedDict

import vapoursynth as vs
import paths  # noqa: E402  (core/paths.py: every folder in one place)

core = vs.core
ENGINE_DIR = paths.ENGINES
MODELS = {0: "temporalfix_s0_v1_op18_fp16.onnx", 1: "temporalfix_s1_v1.1_op18_fp16.onnx",
          2: "temporalfix_s2_v1_op18_fp16.onnx", 3: "temporalfix_s3_v1_op18_fp16.onnx"}


def available():
    """True when TensorRT and a usable CUDA PyTorch are installed."""
    try:
        import tensorrt  # noqa: F401
        import torch
        if not torch.cuda.is_available():
            return False
        (torch.ones(8, device="cuda") * 2).sum().item()
        return True
    except Exception:                                    # noqa: BLE001
        return False


def _onnx_for(strength, log=print):
    """ONNX file for a strength 0-3. Whole numbers are the trained models; in between, the weights of the two
    neighbouring models are blended linearly (as vs_temporalfix does for its own engines)."""
    import math
    import vs_temporalfix
    mdir = os.path.join(os.path.dirname(vs_temporalfix.__file__), "models")
    lo, hi = math.floor(strength), math.ceil(strength)
    if lo == hi:
        return os.path.join(mdir, MODELS[int(lo)])
    out = os.path.join(ENGINE_DIR, f"temporalfix_s{strength:.2f}_blend.onnx")
    if os.path.isfile(out):
        return out
    import numpy as np
    import onnx
    from onnx import numpy_helper
    log(f"TemporalFix AI: blending models {lo} and {hi} for strength {strength:.2f}...")
    a, b = onnx.load(os.path.join(mdir, MODELS[int(lo)])), onnx.load(os.path.join(mdir, MODELS[int(hi)]))
    wb = {t.name: t for t in b.graph.initializer}
    wgt = strength - lo
    for t in a.graph.initializer:
        x = numpy_helper.to_array(t)
        if x.dtype.kind == "f" and t.name in wb:
            y = numpy_helper.to_array(wb[t.name])
            if y.shape == x.shape:
                t.CopyFrom(numpy_helper.from_array((x.astype(np.float32) * (1 - wgt) + y.astype(np.float32) * wgt).astype(x.dtype), t.name))
    os.makedirs(ENGINE_DIR, exist_ok=True)
    onnx.save(a, out + ".part.onnx")
    os.replace(out + ".part.onnx", out)
    return out


def _engine_path(strength, w, h):
    import tensorrt as trt
    import torch
    gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    tag = str(int(strength)) if float(strength).is_integer() else f"{strength:.2f}"
    return os.path.join(ENGINE_DIR, f"temporalfix_s{tag}_w{w}_h{h}_trt{trt.__version__}_{gpu}.engine")


def build_engine(strength, w, h, log=print):
    """Build (or reuse) the TensorRT engine; same settings as vs_temporalfix's own builder."""
    import tensorrt as trt
    import vs_temporalfix
    path = _engine_path(strength, w, h)
    if os.path.isfile(path) and os.path.getsize(path) > 512:
        return path
    os.makedirs(ENGINE_DIR, exist_ok=True)
    onnx = _onnx_for(strength, log)
    log(f"TemporalFix TensorRT: building engine for strength {strength}, {w}x{h} (once, about a minute)...")
    logger = __import__('nnbackend').trt_logger(trt)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx):
        raise RuntimeError("TensorRT could not parse " + onnx)
    config = builder.create_builder_config()
    network.get_input(0).allowed_formats = network.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 << 20)
    config.builder_optimization_level = 3
    profile = builder.create_optimization_profile()
    shape = (1, 21, h, w)
    profile.set_shape(network.get_input(0).name, shape, shape, shape)
    config.add_optimization_profile(profile)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT engine build failed")
    return __import__("nnbackend").save_engine(path, blob)


def temporalfix(clip, strength=2, log=print):
    """clip: RGB (any float/int). Returns RGBH with TemporalFix AI applied (strength 1-3, integer)."""
    import numpy as np
    import tensorrt as trt
    import torch
    from vs_temporalfix.utils import gen_shifts

    strength = round(max(0.0, min(3.0, float(strength))), 2)       # 0-3, fractions blend two models
    if strength <= 0:
        return clip
    w, h = clip.width, clip.height
    path = build_engine(strength, w, h, log)

    runtime = trt.Runtime(__import__('nnbackend').trt_logger(trt))
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:                                   # stale engine (driver / TensorRT update): rebuild
        os.remove(path)
        with open(build_engine(strength, w, h, log), "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    ctx.set_input_shape("input", (1, 21, h, w))
    dev = torch.device("cuda", 0)
    inp = torch.empty((1, 21, h, w), dtype=torch.float16, device=dev)
    out = torch.empty((1, 3, h, w), dtype=torch.float16, device=dev)
    ctx.set_tensor_address("input", inp.data_ptr())
    ctx.set_tensor_address("output", out.data_ptr())
    stream = torch.cuda.Stream(device=dev)
    import nnbackend
    lock = nnbackend.TRT_LOCK                            # one lock for every TensorRT engine in the process
    cache = OrderedDict()                                # frame index -> [3, h, w] half tensor on the GPU

    def to_gpu(frame, idx):
        t = cache.pop(idx, None)
        if t is None:
            a = np.empty((3, h, w), dtype=np.float16)
            for p in range(3):
                a[p] = np.asarray(frame[p])
            t = torch.from_numpy(a).to(dev, non_blocking=True)
        cache[idx] = t
        while len(cache) > 12:
            cache.popitem(last=False)
        return t

    def infer(n, f):
        dst = f[3].copy()
        with lock, torch.cuda.stream(stream):
            for i, idx in enumerate(range(n - 3, n + 4)):
                inp[0, 3 * i:3 * i + 3] = to_gpu(f[i], idx)
            ctx.execute_async_v3(stream.cuda_stream)
            res = out[0].cpu().numpy()                   # synchronises the stream
        for p in range(3):
            np.copyto(np.asarray(dst[p]), res[p])
        return dst

    clip = core.std.Expr(core.resize.Point(clip, format=vs.RGBS), "x 0 max 1 min")
    clip = core.resize.Point(clip, format=vs.RGBH)
    shifts = gen_shifts(clip, radius=3)                  # n-3 .. n+3 (edges repeat)
    return core.std.ModifyFrame(clip, clips=shifts, selector=infer)
