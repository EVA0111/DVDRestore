"""HaloNet: neural repair of halos (edge enhancement), ringing and short ghosts / echoes. Luma only.

Trained (models/halonet_train.py) on clean HD anime / cartoon frames degraded with simulated echoes (2-16 px),
damped ringing after edges and unsharp-mask halos. Target: ghosts and ringing gone, halo over/undershoot cut
off while the edge keeps its steepness. Input and target carry the same noise, so grain is kept, and the
colour planes are passed through untouched. Echoes further than ~16 px stay with the measured cancellation.
"""
import os
import threading

import numpy as np
import vapoursynth as vs

import nnbackend
import paths  # noqa: E402  (core/paths.py: every folder in one place)

core = vs.core
MODEL = paths.model("halonet.pt")
_NET = None
PAD = 32
_LOCK = threading.Lock()


def _build(sd=None):
    import torch
    import torch.nn as nn
    if sd is not None and "head.weight" in sd:
        # HaloNet v3 (models/halonet3_train.py): 128-ch residual, 8 dilated blocks - recognised by its weights
        class Block(nn.Module):
            def __init__(self, ch, d):
                super().__init__()
                self.a = nn.Conv2d(ch, ch, 3, padding=d, dilation=d)
                self.b = nn.Conv2d(ch, ch, 3, padding=1)

            def forward(self, x):
                return x + self.b(torch.relu(self.a(torch.relu(x))))

        class HaloNet3(nn.Module):
            def __init__(self, ch=128, dil=(1, 2, 4, 8, 4, 2, 1, 1)):
                super().__init__()
                self.head = nn.Conv2d(1, ch, 3, padding=1)
                self.body = nn.Sequential(*[Block(ch, d) for d in dil])
                self.tail = nn.Conv2d(ch, 1, 3, padding=1)

            def forward(self, x):
                h = self.head(x.float()).to(x.dtype)          # first layer in float32, as in training
                return x + self.tail(torch.relu(self.body(h))).to(x.dtype)
        return HaloNet3()

    class HaloNet(nn.Module):
        def __init__(self, ch=96, dil=(1, 1, 2, 4, 8, 4, 2, 1, 1, 1, 1, 1)):
            super().__init__()
            layers = [nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(inplace=True)]
            for d in dil:
                layers += [nn.Conv2d(ch, ch, 3, padding=d, dilation=d), nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(ch, 1, 3, padding=1)]
            self.body = nn.Sequential(*layers)

        def forward(self, x):
            return x + self.body[1:](self.body[0](x.float()).to(x.dtype))
    return HaloNet()


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
            (net.head if hasattr(net, "head") else net.body[0]).float()   # first layer in float32, as in training
        _NET = (net, dev)
    return _NET


def _borders(y, maxw=24):
    """Width of the dark border on each side (left, right, top, bottom) of one luma frame, 0 if none.
    A side counts as border up to the first column / row that reaches 85 % of the picture level just
    inside. In a dark scene nothing is found, and nothing needs hiding."""
    def width(m):                                        # m: mean per column / row, from the frame edge inwards
        ref = m[maxw:2 * maxw].median()
        if ref < 0.08 or m[0] > 0.5 * ref:
            return 0
        ok = (m[:maxw] >= 0.85 * ref).nonzero()
        return int(ok[0]) if len(ok) else maxw
    cm, rm = y.mean(0), y.mean(1)
    return width(cm), width(cm.flip(0)), width(rm), width(rm.flip(0))


# Line / highlight guard (2026-11-23). On The Simpsons S1 (JPN NTSC DVD) the model took out the bright rims beside the
# outlines (6.5 -> 0.5 levels) but also lifted the ink lines themselves (line depth 122 -> 106: "washed out") and dimmed
# small whites (teeth 219 -> 213): it read part of every line as dark overshoot. A halo rim is LOW: a few to ~20 levels
# off the fill next to it. An ink line or a highlight is DEEP / HIGH. So:
#   depth  = closing(x) - x   (closing fills thin dark features: how far a pixel sits below the fill around it)
#   height = x - opening(x)   (opening removes thin bright features: how far it sticks out above it)
# brightening is kept where depth < LIFT[0] and fades to none at LIFT[1] (line cores stay as dark as on the disc);
# darkening is kept where height < DIM[0] and fades to none at DIM[1] (teeth, eye whites, sparkles stay).
GUARD = os.environ.get("DVDRESTORE_HALO_GUARD", "1") != "0"      # 0 = the model's raw output (benchmarks)
LIFT = (12 / 255, 35 / 255)
DIM = (20 / 255, 50 / 255)
GUARD_R = 2               # closing 5x5: ink lines on SD are 1-4 px wide
GUARD_RH = 4              # opening 9x9: a rim (1-3 px) lies against a wide fill and is removed by it; teeth, pupils'
                          # glints and small whites up to ~8 px stand out against dark on all sides and are kept


def _guard(x, out):
    """x, out: [1, 1, H, W] float luma 0..1 (model input / output). Returns out with line cores and highlights kept."""
    import torch
    import torch.nn.functional as F
    dil = lambda a, r: F.max_pool2d(F.pad(a, (r,) * 4, mode="replicate"), 2 * r + 1, 1)
    ero = lambda a, r: -F.max_pool2d(F.pad(-a, (r,) * 4, mode="replicate"), 2 * r + 1, 1)
    closing = ero(dil(x, GUARD_R), GUARD_R)
    opening = dil(ero(x, GUARD_RH), GUARD_RH)
    depth = (closing - x).clamp(min=0)
    height = (x - opening).clamp(min=0)
    wl = ((LIFT[1] - depth) / (LIFT[1] - LIFT[0])).clamp(0, 1)
    wd = ((DIM[1] - height) / (DIM[1] - DIM[0])).clamp(0, 1)
    d = out - x
    return (x + torch.where(d > 0, d * wl, d * wd)).clamp(0, 1)


def available():
    if not os.path.isfile(MODEL):
        return False
    try:
        import torch  # noqa: F401
        return True
    except Exception:                                    # noqa: BLE001
        return False


def halonet(clip, strength=1.0):
    """clip: YUV, limited range, any subsampling / depth. Returns the same format with repaired luma."""
    import torch
    if strength <= 0:
        return clip
    y = core.resize.Point(core.std.ShufflePlanes(clip, 0, vs.GRAY), format=vs.GRAYS, range_in=0, range=1)

    def proc(n, f):
        net, dev = _net(); net = nnbackend.wrap("halonet", net, MODEL)
        x = np.asarray(f[0])[None, None]
        with _LOCK, torch.no_grad():
            t0 = torch.from_numpy(np.array(x)).to(dev)
            # Black borders (blanking at the sides, letterbox bars) are a full-height step far larger than
            # anything in training; the model treated the 30 px beside them as echoes of that step and drew
            # stripes. Hide the borders from the model (continue the picture into them) and leave them as
            # they are in the output.
            l, r, tp, bt = _borders(t0[0, 0])
            t = t0.clone()
            H, W = t.shape[-2:]
            if l: t[..., :l] = t[..., l:l + 1]
            if r: t[..., W - r:] = t[..., W - r - 1:W - r]
            if tp: t[..., :tp, :] = t[..., tp:tp + 1, :]
            if bt: t[..., H - bt:, :] = t[..., H - bt - 1:H - bt, :]
            t = t.clamp(0, 1)
            # The network was trained on the inside of patches only: within its view (29 px) of the frame
            # border its own zero padding looks like a huge edge and it drew stripes there. Mirror the
            # picture outwards and cut the margin off again.
            t = torch.nn.functional.pad(t, (PAD, PAD, PAD, PAD), mode="reflect")
            t = t.half() if dev.type == "cuda" else t
            o = net(t).float()[..., PAD:-PAD, PAD:-PAD]
            t = t[..., PAD:-PAD, PAD:-PAD]
            out = (t.float() + strength * (o - t.float())).clamp(0, 1)
            if GUARD:
                out = _guard(t.float(), out)
            keep = torch.ones_like(out, dtype=torch.bool)
            keep[..., tp:H - bt, l:W - r] = False
            out = torch.where(keep, t0, out)[0, 0].cpu().numpy()
        dst = f.copy()
        np.copyto(np.asarray(dst[0]), out)
        return dst

    out = core.std.ModifyFrame(y, clips=y, selector=proc)
    out = core.resize.Point(out, format=clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id, range_in=1, range=0)
    out = core.std.ShufflePlanes([out, clip], [0, 1, 2], vs.YUV)
    return core.std.CopyFrameProps(out, clip)
