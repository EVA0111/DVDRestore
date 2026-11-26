"""Colour prior (auto colour without a reference): models/colourprior.pt, trained by models/colourprior_train.py.

A small CNN that looks at a frame (after auto colour's own levels and cast wheels) and says how far its colour
strength, midtone brightness and hue are from what well-transferred animation looks like. Asked per sampled frame;
the median over the frames is the source's answer. It is a learned prior, not a measurement: on shows it never saw
it recognises fades, midtone and hue faults partly (see README "COLOUR PRIOR"), so autocolour.py only acts on what
is clearly beyond what it says about good transfers, and only part of the way.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "models", "colourprior.pt")
_NET = {}


def available():
    if not os.path.isfile(MODEL):
        return False
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _net():
    if "net" not in _NET:
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, w=48):
                super().__init__()

                def blk(i, o, s):
                    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(),
                                         nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU())
                self.f = nn.Sequential(blk(3, w, 2), blk(w, w * 2, 2), blk(w * 2, w * 4, 2), blk(w * 4, w * 4, 2),
                                       blk(w * 4, w * 8, 2))
                self.h = nn.Sequential(nn.Linear(w * 8 * 2, 256), nn.SiLU(), nn.Linear(256, 3))

            def forward(self, x):
                z = self.f(x * 2 - 1)
                z = torch.cat([z.mean((2, 3)), z.amax((2, 3))], 1)
                return self.h(z)
        ck = torch.load(MODEL, map_location="cpu", weights_only=True)
        net = Net(ck.get("w", 48))
        net.load_state_dict(ck["state"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _NET["net"], _NET["dev"] = net.to(dev).eval(), dev
    return _NET["net"], _NET["dev"]


def predict(frames_rgb):
    """frames_rgb: N x 240 x 320 x 3 uint8 (what the disc shows after auto colour's levels / casts).
    Returns (colour scale, gamma, hue degrees) of the FAULT the prior sees - median over the frames - and the
    per-frame spread (median absolute deviation) of each."""
    import torch
    net, dev = _net()
    x = torch.from_numpy(np.ascontiguousarray(frames_rgb)).to(dev).permute(0, 3, 1, 2).float() / 255
    x = x[:, :, 16:224, 32:288]
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 32):
            xb = x[i:i + 32]
            p = (net(xb) + net(xb.flip(3))) / 2          # left-right flip: the answer must not depend on framing
            out.append(p.float().cpu())
    p = torch.cat(out).numpy()
    p[:, 2] *= 20
    med = np.median(p, 0)
    mad = np.median(np.abs(p - med), 0)
    return float(np.exp(med[0])), float(np.exp(med[1])), float(med[2]), mad
