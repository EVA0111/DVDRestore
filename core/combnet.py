"""CombNet: multi-frame repair of frames that had to be rebuilt from one field (comb repair).

Field matching cannot clean every frame (30i titles over film, orphan fields at cuts). Those are rebuilt from
the field that is there: NNEDI3 for the missing lines, Bwdif borrowing them from the neighbouring frames
where nothing moves. Where things move, half the vertical detail is gone. CombNet looks at the rebuilt frame
and the frames before and after it and corrects the rebuilt lines; the kept field passes through unchanged.
Trained (models/combnet_train.py) on real DVD film frames from seven discs: a video-rate field stream where
every field is a different moment, so the truth of every rebuilt line is known.
Held-out frames, outline pixels of the rebuilt lines: moving areas +2.3 dB, still areas +1.0 dB over
NNEDI3 + Bwdif (see README). Luma only.
"""
import os
import threading

import numpy as np
import vapoursynth as vs

import nnbackend
import paths  # noqa: E402  (core/paths.py: every folder in one place)

core = vs.core
MODEL = paths.model("combnet.pt")
_NET = None
_LOCK = threading.Lock()
PAD = 32


def _build(sd):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, ch, d):
            super().__init__()
            self.a = nn.Conv2d(ch, ch, 3, padding=d, dilation=d)
            self.b = nn.Conv2d(ch, ch, 3, padding=1)

        def forward(self, x):
            return x + self.b(F.relu(self.a(F.relu(x))))

    class CombNet(nn.Module):
        def __init__(self, ch, dil=(1, 2, 4, 8, 16, 8, 4, 2, 1, 1)):
            super().__init__()
            self.head = nn.Conv2d(12, ch, 3, padding=1)
            self.blocks = nn.Sequential(*[Block(ch, d) for d in dil])
            self.tail = nn.Conv2d(ch, 2, 3, padding=1)

        def forward(self, x):
            u = F.pixel_unshuffle(x, 2)
            h = self.head(u.float()).to(x.dtype)
            r = self.tail(F.relu(self.blocks(h)))
            cur = u[:, 4:8]
            return F.pixel_shuffle(torch.cat([cur[:, 0:2], cur[:, 2:4] + r], 1), 2)
    return CombNet(sd["head.weight"].shape[0])


def _net():
    global _NET
    if _NET is None:
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def combnet(rebuilt, kept_top):
    """rebuilt: the comb repair's output (frames rebuilt from one field, any YUV / GRAY). kept_top: int or
    callable(n) -> True when frame n kept its top field (even lines), False for the bottom field.
    Returns the clip with the luma of every frame corrected (chroma unchanged)."""
    import torch
    y = core.std.ShufflePlanes(rebuilt, 0, vs.GRAY)
    fmt = y.format
    ys = core.resize.Point(y, format=vs.GRAYS) if fmt.sample_type != vs.FLOAT else y
    prv, nxt = ys[:1] + ys[:-1], ys[1:] + ys[-1:]

    def proc(n, f):
        net, dev = _net()
        top = kept_top(n) if callable(kept_top) else bool(kept_top)
        x = np.stack([np.asarray(fr[0]) for fr in (f[0], f[1], f[2])])[None]
        with _LOCK, torch.no_grad():
            t = torch.from_numpy(x).to(dev)
            H, W = t.shape[-2:]
            off = 0 if top else 1                                   # kept lines must sit on even rows
            ph = PAD + off
            t = torch.nn.functional.pad(t, (PAD, PAD + W % 2, ph, PAD + (H + ph + PAD) % 2), mode="reflect")
            t = t.half() if dev.type == "cuda" else t
            mdl = nnbackend.wrap("combnet", net, MODEL)
            o = mdl(t).float()[0, 0, ph:ph + H, PAD:PAD + W].clamp(0, 1).cpu().numpy()
        dst = f[1].copy()
        np.copyto(np.asarray(dst[0]), o)
        return dst

    out = core.std.ModifyFrame(ys, clips=[prv, ys, nxt], selector=proc)
    out = core.resize.Point(out, format=fmt.id) if fmt.sample_type != vs.FLOAT else out
    if rebuilt.format.num_planes == 1:
        return out
    return core.std.ShufflePlanes([out, rebuilt], [0, 1, 2], rebuilt.format.color_family)
