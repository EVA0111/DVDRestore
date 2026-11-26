"""RepairNet in the app: halos / ringing / echoes + colour bleed in ONE pass, in place of HaloNet + ChromaNet.
(models/repairnet_core.py, trained by models/repairnet_train.py on models/repairnet_gen.py data.)

The network also repairs MPEG damage, but on real discs BlockNet does that better (Shin Hakkenden grid 1.18 vs 1.34),
so the chain is: BlockNet (MPEG repair step) -> ... -> colour shift -> RepairNet (halos + bleed, dials).
Measured on real frames (Simpsons S1, Shin Hakkenden, Bleach 190) against BlockNet + HaloNet + ChromaNet: rims beside
ink lines 1.5 -> 1.0 levels, line darkness kept (116 = DVD), block grid 1.13 -> 1.08 / 1.18 -> 1.15, one pass fewer.
Its luma goes through halonet._guard (ink-line cores keep their darkness, small highlights their brightness).

Dials: halo 0..1 (0 = leave halos), bleed 0..1 (0 = leave the colour smear). Input: any YUV clip; output YUV444P16
(what the colour step hands on)."""
import os
import sys
import threading
from collections import OrderedDict

import numpy as np
import vapoursynth as vs

import nnbackend
import paths  # noqa: E402

core = vs.core
MODEL = paths.model("repairnet.pt")
_NET = None
_LOCK = threading.Lock()
# All GPU work on ONE thread: PyTorch makes a cuDNN handle per thread on its first GPU call (0.3-1.3 s each), and
# VapourSynth hands frames to up to 24 threads - the warm-up alone made the first ~30 frames run at 2.6 fps.
_GPU = None


def _gpu():
    global _GPU
    if _GPU is None:
        from concurrent.futures import ThreadPoolExecutor
        _GPU = ThreadPoolExecutor(1, thread_name_prefix="repairnet-gpu")
    return _GPU


def available():
    if not os.path.isfile(MODEL) or not os.path.isfile(os.path.join(paths.MODELS, "repairnet_core.py")):
        return False
    try:
        import torch  # noqa: F401
        return True
    except Exception:                                    # noqa: BLE001
        return False


def _net():
    global _NET
    if _NET is None:
        import torch
        if paths.MODELS not in sys.path:
            sys.path.insert(0, paths.MODELS)
        from repairnet_core import RepairNet
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = RepairNet()
        net.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))
        net = net.to(dev).eval()
        if dev.type == "cuda":
            net = net.half()
        _NET = (net, dev)
    return _NET


def repairnet(clip, halo=1.0, bleed=1.0):
    import torch
    import halonet
    halo, bleed = float(max(0.0, min(1.0, halo))), float(max(0.0, min(1.0, bleed)))
    c = core.resize.Point(clip, format=vs.YUV444P16) if clip.format.id != vs.YUV444P16 else clip
    W, H = c.width, c.height
    Hp, Wp = -(-H // 16) * 16, -(-W // 16) * 16                  # the U-Net needs multiples of 16
    yf = core.std.ShufflePlanes(c, 0, vs.GRAY)
    # 4:2:0 colour for the network (the DVD's own resolution; training averaged 2x2)
    uf = core.resize.Bilinear(core.std.ShufflePlanes(c, 1, vs.GRAY), W // 2, H // 2)
    vf = core.resize.Bilinear(core.std.ShufflePlanes(c, 2, vs.GRAY), W // 2, H // 2)
    cache, clock, busy = OrderedDict(), threading.Lock(), {}

    def run(n, fy, fu, fv):
        # Y, U and V are asked for in parallel: the first request runs the network, the others wait for its result
        with clock:
            if n in cache:
                return cache[n]
            ev = busy.get(n)
            mine = ev is None
            if mine:
                ev = busy[n] = threading.Event()
        if not mine:
            ev.wait()
            with clock:
                if n in cache:
                    return cache[n]
            return run(n, fy, fu, fv)
        try:
            ay, au, av = (np.asarray(fr[0]).astype(np.int32) for fr in (fy, fu, fv))

            def work():
                net, dev = _net()
                net_ = nnbackend.wrap("repairnet", net, MODEL)
                dt = torch.float16 if dev.type == "cuda" else torch.float32
                g = lambda a: torch.from_numpy(a).to(dev).float() / 256
                with torch.no_grad():
                    y = (g(ay) - 16) / 219
                    uv = (torch.stack([g(au), g(av)]) - 128) / 224
                    if (Hp, Wp) != (H, W):
                        y = torch.nn.functional.pad(y[None, None], (0, Wp - W, 0, Hp - H), mode="reflect")[0, 0]
                        uv = torch.nn.functional.pad(uv[None], (0, (Wp - W) // 2, 0, (Hp - H) // 2), mode="reflect")[0]
                    ty = y[None, None]
                    oy, oc = net_(ty.to(dt), uv[None].to(dt), torch.tensor([[halo, bleed]], device=dev, dtype=dt))
                    oy = halonet._guard(ty, oy.float().clamp(-0.1, 1.1))[0, 0, :H, :W] * 219 + 16
                    oc = oc.float()[0, :, :H // 2, :W // 2] * 224 + 128
                    q = lambda a: (a * 256).round().clamp(0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)
                    return q(oy), q(oc[0]), q(oc[1])
            res = _gpu().submit(work).result()
            with clock:
                cache[n] = res
                while len(cache) > 24:
                    cache.popitem(last=False)
            return res
        finally:
            with clock:
                busy.pop(n, None)
            ev.set()

    def plane(i, base):
        def sel(n, f):                                     # f: [base frame, Y, U, V] (base = the plane's own size)
            out = f[0].copy()
            np.copyto(np.asarray(out[0]), run(n, f[1], f[2], f[3])[i])
            return out
        return core.std.ModifyFrame(base, clips=[base, yf, uf, vf], selector=sel)

    yo = plane(0, yf)
    uo = core.resize.Spline36(plane(1, uf), W, H)                 # back to 4:4:4 for the rest of the chain
    vo = core.resize.Spline36(plane(2, vf), W, H)
    out = core.std.ShufflePlanes([yo, uo, vo], [0, 0, 0], vs.YUV)
    return core.std.CopyFrameProps(out, c)
