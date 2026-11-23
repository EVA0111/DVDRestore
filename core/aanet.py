"""AANet: neural anti-aliasing - removes the stair-steps / jagged lines that deinterlacing (one field rebuilt
from the other) and resizing without a filter leave on outlines. Luma only, single frame.

Trained (models/aanet_train.py) on clean HD anime / cartoon frames at DVD size whose second field was thrown
away and rebuilt the way deinterlacers do it (line doubling, line average, cubic, edge-directed ELA; whole
frame or only inside a soft mask), or which were resized without a low-pass. Target = the clean picture.
"""
import os
import threading

import numpy as np
import vapoursynth as vs

import nnbackend
import paths  # noqa: E402  (core/paths.py: every folder in one place)

core = vs.core
MODEL = paths.model("aanet.pt")
_NET = None
PAD = 16
_LOCK = threading.Lock()


def _build(sd):
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, ch, d):
            super().__init__()
            self.a = nn.Conv2d(ch, ch, 3, padding=d, dilation=d)
            self.b = nn.Conv2d(ch, ch, 3, padding=1)

        def forward(self, x):
            return x + self.b(F.relu(self.a(F.relu(x))))

    class AANet(nn.Module):
        def __init__(self, ch, dil):
            super().__init__()
            self.head = nn.Conv2d(4, ch, 3, padding=1)
            self.blocks = nn.Sequential(*[Block(ch, d) for d in dil])
            self.tail = nn.Conv2d(ch, 4, 3, padding=1)

        def forward(self, x):
            h = self.head(F.pixel_unshuffle(x.float(), 2)).to(x.dtype)     # first layer in float32, as in training
            return x + F.pixel_shuffle(self.tail(F.relu(self.blocks(h))), 2)

    n = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    dil = (1, 2, 4, 2, 1, 1, 1, 1)
    dil = dil[:n] if n <= len(dil) else dil + (1,) * (n - len(dil))
    return AANet(sd["head.weight"].shape[0], dil)


def _net():
    global _NET
    if _NET is None:
        import torch
        dev = torch.device("cpu")
        try:
            if torch.cuda.is_available():
                (torch.ones(4, device="cuda") * 2).sum().item()
                dev = torch.device("cuda")
        except Exception:                                # noqa: BLE001
            pass
        sd = torch.load(MODEL, map_location="cpu", weights_only=True)
        net = _build(sd)
        net.load_state_dict(sd)
        net = net.to(dev).eval()
        if dev.type == "cuda":
            net = net.half()
            net.head.float()
        _NET = (net, dev)
    return _NET


def available():
    if not os.path.isfile(MODEL):
        return False
    try:
        import torch  # noqa: F401
        return True
    except Exception:                                    # noqa: BLE001
        return False


def aanet(clip, strength=1.0):
    """clip: YUV, limited range, any subsampling / depth. Returns the same format with anti-aliased luma."""
    import torch
    if strength <= 0:
        return clip
    y = core.resize.Point(core.std.ShufflePlanes(clip, 0, vs.GRAY), format=vs.GRAYS, range_in=0, range=1)

    def proc(n, f):
        net, dev = _net(); net = nnbackend.wrap("aanet", net, MODEL)
        x = np.asarray(f[0])[None, None]
        with _LOCK, torch.no_grad():
            t0 = torch.from_numpy(np.array(x)).to(dev)
            H, W = t0.shape[-2:]
            t = torch.nn.functional.pad(t0.clamp(0, 1), (PAD, PAD + W % 2, PAD, PAD + H % 2), mode="reflect")
            t = t.half() if dev.type == "cuda" else t
            o = net(t).float()[..., PAD:PAD + H, PAD:PAD + W]
            out = (t0 + strength * (o - t0.clamp(0, 1)))[0, 0].cpu().numpy()
        dst = f.copy()
        np.copyto(np.asarray(dst[0]), out)
        return dst

    out = core.std.ModifyFrame(y, clips=y, selector=proc)
    out = core.resize.Point(out, format=clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id, range_in=1, range=0)
    out = core.std.ShufflePlanes([out, clip], [0, 1, 2], vs.YUV)
    return core.std.CopyFrameProps(out, clip)
