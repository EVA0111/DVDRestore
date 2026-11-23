"""ClipNet: rebuilds colour and tone that were clipped flat before the disc / capture was made.

A capture card or a broadcast legaliser cuts every colour channel off at its limit: a strong red becomes
R = maximum, G = B = 0 over a whole area and the gradation inside it is gone. The stored-out-of-range colour
of a DVD is a different case (the data is still there, see vsgraph.gamut_compress).
Trained (models/clipnet_train.py) on clean HD anime / cartoon frames whose saturation, channel gains,
highlights and blacks were pushed past the range and cut off. Target = the same pushed picture squeezed into
range with a soft knee instead of cut off. The model sees RGB, works at half resolution inside (view of
about 150 px) and estimates the lost gradation from the channels that did not clip and from the
surroundings. Where all three channels are flat over a large area there is nothing to go on.
"""
import os
import threading

import numpy as np
import vapoursynth as vs

import nnbackend
import paths  # noqa: E402  (core/paths.py: every folder in one place)

core = vs.core
MODEL = paths.model("clipnet.pt")
_NET = None
PAD = 32
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

    class ClipNet(nn.Module):
        def __init__(self, ch, dil):
            super().__init__()
            self.head = nn.Conv2d(12, ch, 3, padding=1)
            self.blocks = nn.Sequential(*[Block(ch, d) for d in dil])
            self.tail = nn.Conv2d(ch, 12, 3, padding=1)

        def forward(self, x):
            h = self.head(F.pixel_unshuffle(x.float(), 2)).to(x.dtype)     # first layer in float32, as in training
            return x + F.pixel_shuffle(self.tail(F.relu(self.blocks(h))), 2)

    n = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    dil = (1, 2, 4, 8, 16, 8, 4, 2, 1, 1)
    dil = dil[:n] if n <= len(dil) else dil + (1,) * (n - len(dil))
    return ClipNet(sd["head.weight"].shape[0], dil)


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


def clipnet(rgb, strength=1.0, guard=True):
    """rgb: RGBS (float, 0..1 = the screen's range; values outside are cut off first, as a screen would).
    Returns RGBS inside 0..1."""
    import torch
    import torch.nn.functional as F

    def proc(n, f):
        net, dev = _net(); net = nnbackend.wrap("clipnet", net, MODEL)
        x = np.stack([np.asarray(f[p]) for p in range(3)])[None]
        with _LOCK, torch.no_grad():
            t0 = torch.from_numpy(np.array(x)).to(dev).clamp(0, 1)
            H, W = t0.shape[-2:]
            # even size for the 2x2 folding, mirrored margin so the frame border is not an edge
            t = torch.nn.functional.pad(t0, (PAD, PAD + W % 2, PAD, PAD + H % 2), mode="reflect")
            t = t.half() if dev.type == "cuda" else t
            o = net(t).float()[..., PAD:PAD + H, PAD:PAD + W]
            w = strength
            if guard:
                # only where a colour sits at the limit (and a few pixels around it); everything else is left
                # exactly as it is. Without this the model shifts clean parts of the picture a little.
                mx, mn = t0.amax(1, keepdim=True), t0.amin(1, keepdim=True)
                lim = (((mx >= 0.985) | (mn <= 0.015)) & (mx - mn > 0.12)).float()
                lim = F.max_pool2d(lim, 13, 1, 6)
                w = strength * F.avg_pool2d(F.avg_pool2d(lim, 9, 1, 4, count_include_pad=False), 9, 1, 4, count_include_pad=False)
            out = (t0 + w * (o - t0)).clamp(0, 1)[0].cpu().numpy()
        dst = f.copy()
        for p in range(3):
            np.copyto(np.asarray(dst[p]), out[p])
        return dst

    return core.std.ModifyFrame(rgb, clips=rgb, selector=proc)
