"""ChromaNet: neural colour-bleed repair. Colour planes only, guided by luma; luma is passed through untouched.

DVD colour is quarter resolution and, on analog masters, smeared sideways, so colour spills over the outlines.
Trained (models/chromanet_train.py) on clean HD anime / cartoon frames whose colour was degraded the way a DVD
does it, from mild to extreme (4:2:0 only ... 8 px smear with long tails, interlaced-chroma combing,
leftover shift up to 2 px, colour noise). Target = the original colour, only as sharp as
the luma. No remaster pairs, so no restyling is learned.
"""
import os
import threading

import numpy as np
import vapoursynth as vs

import nnbackend
import paths  # noqa: E402  (core/paths.py: every folder in one place)

core = vs.core
MODEL = paths.model("chromanet.pt")
_NET = None
_LOCK = threading.Lock()


def _build():
    import torch.nn as nn
    import torch.nn.functional as F

    class ChromaNet(nn.Module):
        """Works at half resolution inside (pixel-unshuffle): ~110 px view, enough for 8 px smears."""
        def __init__(self, ch=128, dil=(1, 1, 2, 4, 8, 4, 2, 1, 1, 1)):
            super().__init__()
            layers = [nn.Conv2d(12, ch, 3, padding=1), nn.ReLU(inplace=True)]
            for d in dil:
                layers += [nn.Conv2d(ch, ch, 3, padding=d, dilation=d), nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(ch, 8, 3, padding=1)]
            self.body = nn.Sequential(*layers)

        def forward(self, x):
            return x[:, 1:] + F.pixel_shuffle(self.body(F.pixel_unshuffle(x, 2)), 2)
    return ChromaNet()


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
        net = _build()
        net.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))
        net = net.to(dev).eval()
        if dev.type == "cuda":
            net = net.half()
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


def chromanet(clip444, strength=1.0):
    """clip444: YUV 4:4:4 limited range (any depth). Returns the same format with repaired colour planes."""
    import torch
    full = core.resize.Point(clip444, format=vs.YUV444PS, range_in=0, range=1)

    def proc(n, f):
        net, dev = _net(); net = nnbackend.wrap("chromanet", net, MODEL)
        x = np.stack([np.asarray(f[p]) for p in range(3)])[None]
        with _LOCK, torch.no_grad():
            t = torch.from_numpy(x).to(dev)
            t = t.half() if dev.type == "cuda" else t
            H, W = t.shape[2:]
            if H % 2 or W % 2:                               # the half-resolution core needs even sizes
                t = torch.nn.functional.pad(t, (0, W % 2, 0, H % 2), mode="replicate")
            y = net(t).float()[:, :, :H, :W]
            uv = t[:, 1:, :H, :W].float()
            out = (uv + strength * (y - uv))[0].clamp(-0.5, 0.5).cpu().numpy()
        dst = f.copy()
        for p in (1, 2):
            np.copyto(np.asarray(dst[p]), out[p - 1])
        return dst

    out = core.std.ModifyFrame(full, clips=full, selector=proc)
    out = core.resize.Point(out, format=clip444.format.id, range_in=1, range=0)
    out = core.std.ShufflePlanes([clip444, out, out], [0, 1, 2], vs.YUV)      # luma: the original samples
    return core.std.CopyFrameProps(out, clip444)
