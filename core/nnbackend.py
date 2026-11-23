"""Where the neural models run: PyTorch or TensorRT.

Every model module (dvdnet, dotnet, chromanet, halonet, blocknet, clipnet, aanet) builds its PyTorch network
as before and passes it through wrap(). With the backend set to "tensorrt" the network is exported to ONNX
and compiled into a TensorRT engine the first time it sees a frame size; the engine is cached in ./engines
(one file per model, weights, input shape, TensorRT version and GPU), so later runs start at once.

The engine is built from the very network PyTorch runs (half precision, first layer in float32 where the
model does that), as a strongly typed network: TensorRT keeps exactly those precisions, so the output matches
PyTorch to rounding. Inputs and outputs stay on the GPU in PyTorch tensors, nothing is copied extra.

If TensorRT is missing, the build fails or the input is not on the GPU, the model quietly runs on PyTorch
and a note goes to stderr once.
"""
import hashlib
import os
import sys
import threading
import paths  # noqa: E402  (core/paths.py: every folder in one place)

ENGINE_DIR = paths.ENGINES
BACKENDS = ("pytorch", "tensorrt")
_backend = os.environ.get("DVDRESTORE_BACKEND", "pytorch")
_warned = set()
_build_lock = threading.Lock()
# Every TensorRT call (engine load, context creation, launch) goes through this one lock: two models loading /
# launching engines at the same moment on different VapourSynth threads crashed TensorRT (access violation,
# seen when Play fetched frames in parallel). The GPU still works while the next frame is prepared.
TRT_LOCK = threading.RLock()


_RT = None


def warmup():
    """Start CUDA (PyTorch) and one TensorRT runtime on the calling thread, once. If the first CUDA / TensorRT
    use happens on two threads at once (a VapourSynth worker loading an engine while the chain is still
    being built and checks CUDA), TensorRT crashed with an access violation. Call from the main thread
    before frames are requested (server start, set_backend)."""
    global _RT
    if _RT is not None:
        return _RT
    try:
        import torch
        if torch.cuda.is_available():
            torch.ones(1, device="cuda").sum().item()
        import tensorrt as trt
        _RT = (trt_logger(trt),)
        _RT = (_RT[0], trt.Runtime(_RT[0]))
    except Exception:                                    # noqa: BLE001
        _RT = False
    return _RT


def runtime():
    """The process-wide TensorRT runtime (lives as long as the process, as engines need)."""
    r = warmup()
    if not r:
        import tensorrt as trt
        lg = trt_logger(trt)
        return trt.Runtime(lg)
    return r[1]


def set_backend(name):
    """'pytorch' or 'tensorrt'. Takes effect for networks wrapped from now on (model caches are per backend)."""
    global _backend
    _backend = name if name in BACKENDS else "pytorch"
    if _backend == "tensorrt":
        warmup()


def backend():
    return _backend


def tensorrt_available():
    try:
        import tensorrt  # noqa: F401
        import torch
        return torch.cuda.is_available()
    except Exception:                                    # noqa: BLE001
        return False


def _note(key, msg):
    if key not in _warned:
        _warned.add(key)
        print(msg, file=sys.stderr, flush=True)


_wrapped = {}


def wrap(name, net, weights, level=3, workspace=4096):
    """net: the PyTorch module, already on its device and in its precision. Returns something callable
    like the module: the module itself (PyTorch) or a TensorRT runner. Cheap to call for every frame."""
    if _backend != "tensorrt":
        return net
    key = (name, id(net))
    w = _wrapped.get(key)
    if w is None:
        if not tensorrt_available():
            _note("no-trt", "TensorRT is not installed or no CUDA GPU: running the models on PyTorch")
            return net
        w = _wrapped[key] = TRTNet(name, net, weights, level, workspace)
    return w


def _weights_tag(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


MAX_ENGINES = 3                                          # frame sizes kept per model (crop changes make new ones)


class TRTNet:
    """Callable like the module. One engine + execution context per input shape set."""

    def __init__(self, name, net, weights, level=3, workspace=4096):
        self.name, self.net, self.weights = name, net, weights
        self.level, self.workspace = level, workspace
        self.tag = _weights_tag(weights)
        self.engines = {}                                # shapes -> (engine, context, output specs) or None
        self.failed = False

    def __getattr__(self, k):                            # e.g. .head, .t used by some modules
        return getattr(self.__dict__["net"], k)

    def _path(self, key):
        import tensorrt as trt
        import torch
        gpu = torch.cuda.get_device_name(0).replace(" ", "_")
        shp = "_".join("x".join(map(str, s)) + d for s, d in key)
        return os.path.join(ENGINE_DIR, f"{self.name}_{self.tag}_{shp}_trt{trt.__version__}_{gpu}.engine")

    def _build(self, args, path):
        import tensorrt as trt
        import torch
        os.makedirs(ENGINE_DIR, exist_ok=True)
        onnx_path = path[:-7] + ".onnx"
        n_in = len(args)
        with torch.no_grad():
            outs = self.net(*args)
        n_out = len(outs) if isinstance(outs, (tuple, list)) else 1
        print(f"TensorRT: building the {self.name} engine for {'x'.join(map(str, args[0].shape[-2:]))} "
              f"(once per frame size, up to a few minutes)...", file=sys.stderr, flush=True)
        torch.onnx.export(self.net, tuple(args), onnx_path, opset_version=18, dynamo=False,
                          input_names=[f"in{i}" for i in range(n_in)], output_names=[f"out{i}" for i in range(n_out)])
        logger = trt_logger(trt)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError("ONNX parse failed: " + errs)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(self.workspace) << 20)
        config.builder_optimization_level = int(self.level)
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise RuntimeError("engine build failed")
        save_engine(path, blob)
        try:
            os.remove(onnx_path)
        except OSError:
            pass

    def _load(self, args):
        import tensorrt as trt
        import torch
        key = tuple((tuple(a.shape), str(a.dtype).replace("torch.", "")) for a in args)
        if key in self.engines:
            self.engines[key] = self.engines.pop(key)    # most recently used last
            return self.engines[key]
        while len(self.engines) >= MAX_ENGINES:          # each engine + context holds GPU memory: drop the least used
            old_key = next(iter(self.engines))
            ent = self.engines.pop(old_key)
            del ent
            torch.cuda.empty_cache()
        path = self._path(key)
        with _build_lock:
            if not (os.path.isfile(path) and os.path.getsize(path) > 512):
                self._build(args, path)
            # the runtime (and its logger) must live as long as the engine: freed early it crashed TensorRT
            # (access violation) as soon as frames were fetched in parallel (Play)
            runtime = globals()["runtime"]()
            with open(path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:                           # TensorRT / driver changed: build again
                os.remove(path)
                self._build(args, path)
                with open(path, "rb") as f:
                    engine = runtime.deserialize_cuda_engine(f.read())
        ctx = engine.create_execution_context()
        dt = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32, trt.DataType.BF16: torch.bfloat16}
        outs = []
        for i in range(engine.num_io_tensors):
            nm = engine.get_tensor_name(i)
            if engine.get_tensor_mode(nm) == trt.TensorIOMode.OUTPUT:
                outs.append((nm, tuple(engine.get_tensor_shape(nm)), dt[engine.get_tensor_dtype(nm)]))
        outs.sort(key=lambda o: int(o[0][3:]))
        self.engines[key] = (engine, ctx, outs)
        return self.engines[key]

    def __call__(self, *args):
        import torch
        if self.failed or not all(a.is_cuda for a in args):
            return self.net(*args)
        with TRT_LOCK:
            try:
                engine, ctx, outs = self._load(args)
            except Exception as e:                       # noqa: BLE001
                self.failed = True
                _note("fail-" + self.name, f"TensorRT could not build {self.name} ({e}); running it on PyTorch")
                return self.net(*args)
            args = [a.contiguous() for a in args]
            for i, a in enumerate(args):
                ctx.set_tensor_address(f"in{i}", a.data_ptr())
            res = [torch.empty(s, dtype=d, device=args[0].device) for _, s, d in outs]
            for (nm, _, _), r in zip(outs, res):
                ctx.set_tensor_address(nm, r.data_ptr())
            stream = torch.cuda.current_stream(args[0].device)
            ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
        return res[0] if len(res) == 1 else tuple(res)


_TRT_LOGGER = None


def trt_logger(trt=None):
    """One TensorRT logger for the whole process that writes to STDERR. TensorRT's own trt.Logger prints to stdout -
    and in an export stdout is the video pipe (vspipe -> ffmpeg): a failed engine build (FDAT upscale model) put its
    "[TRT] [E] ..." lines in front of the Y4M header, ffmpeg found no video stream and wrote an empty file, and the
    episode was muxed without picture ("Audio sync check ... out of sync")."""
    global _TRT_LOGGER
    if _TRT_LOGGER is None:
        if trt is None:
            import tensorrt as trt

        class _Err(trt.ILogger):
            def __init__(self):
                trt.ILogger.__init__(self)

            def log(self, severity, msg):
                if severity <= trt.ILogger.Severity.ERROR:
                    # one short line: a failed build (with a fallback after it) dumped whole tensor listings
                    line = str(msg).strip().splitlines()[0] if str(msg).strip() else ""
                    if line:
                        print(f"[TRT] {line[:200]}", file=sys.stderr, flush=True)
        _TRT_LOGGER = _Err()
    return _TRT_LOGGER


def save_engine(path, blob):
    """Write a built engine atomically. The preview and the export can build the SAME engine at once (TemporalFix
    1416x1080: "[WinError 32] ... .engine.part -> .engine" killed the preview): each process writes its own temporary
    file, and if the other finished first (the target exists / is open) its engine is kept."""
    import os
    tmp = f"{path}.{os.getpid()}.part"
    with open(tmp, "wb") as f:
        f.write(blob)
    try:
        os.replace(tmp, path)
    except OSError:
        if not os.path.exists(path):
            raise
        try:
            os.remove(tmp)
        except OSError:
            pass
    return path
