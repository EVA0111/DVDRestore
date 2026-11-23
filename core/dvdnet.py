"""DVDNet: neural removal of NTSC dot crawl (luma) and rainbows / cross-colour (chroma) in one pass.

Training (models/dvdnet_train.py), every batch half and half:
  * simulated: clean HD anime / cartoon frames and cel scans through an NTSC composite chain (real subcarrier
    grid, notch / 2-line / 3-line comb decoders). Dots = chroma left in luma, rainbows = luma detail
    demodulated as colour. Target = the same decoder without either leak (no sharpening is learned).
  * real: aligned DVD -> remaster crops (One Piece, Naruto, Cowboy Bebop, Lodoss), chroma pre-aligned so no
    colour shift is learned (the app's Colour shift step stays in charge of that).
Input / output: Y, U, V at full resolution (full range, UV centred); output = input + correction.
"""
import os
import threading

import numpy as np
import vapoursynth as vs

import nnbackend
import paths  # noqa: E402  (core/paths.py: every folder in one place)

core = vs.core
MODELS = {
    # real DVD -> remaster pairs + simulation: strongest on real discs, also crisper / thinner lines (+6-8 % edge strength)
    "restore": paths.model("dvdnet_restore.pt"),
    # simulation only: removes dots + rainbows and nothing else (clean Blu-ray: 59-63 dB, edges +-0.3 %)
    "precise": paths.model("dvdnet_precise.pt"),
    # Precise v2 candidate (models/precise2_train.py): res128 single frame, field-level simulation + real-shot target
    "precise2": paths.model("dvdnet_precise2.pt"),
    # temporal (5 frames in, centre frame out): field-level NTSC simulation + real static shots whose 8-frame
    # mean gives a crawl-free target; the pattern flips between frames, real detail does not (models/dvdnet_train_t.py)
    "temporal": paths.model("dvdnet_temporal.pt"),
    # XNet (models/xnet_train.py): temporal, luma correction limited to the crawl band, trained with film grain
    # and live action - keeps detail and grain (bench/: end-to-end NTSC benchmark)
    "xnet": paths.model("xnet.pt"),
}
TEMPORAL = ("temporal", "xnet")
T_FRAMES = 5
_NET = {}
_LOCK = threading.Lock()


def _build(mode="restore", sd=None):
    import torch.nn as nn
    if sd is not None and "head.weight" in sd:
        # (also single-frame: Precise v2 = the same network with 3 input channels, t read from the weights)
        # larger residual network (2.4M parameters): 128 channels, 8 residual blocks. Recognised by its weights.
        import torch

        class Block(nn.Module):
            def __init__(self, ch, d):
                super().__init__()
                self.a = nn.Conv2d(ch, ch, 3, padding=d, dilation=d)
                self.b = nn.Conv2d(ch, ch, 3, padding=1)

            def forward(self, x):
                return x + self.b(torch.relu(self.a(torch.relu(x))))

        class BigTemporalDVDNet(nn.Module):
            def __init__(self, ch, dil=(1, 2, 4, 8, 4, 2, 1, 1), t=T_FRAMES):
                super().__init__()
                self.t = t
                self.head = nn.Conv2d(3 * t, ch, 3, padding=1)
                self.blocks = nn.Sequential(*[Block(ch, d) for d in dil])
                self.tail = nn.Conv2d(ch, 3, 3, padding=1)

            def forward(self, x):
                c = 3 * (self.t // 2)
                h = self.head(x.float()).to(x.dtype)             # first layer in float32, as in training
                r = self.tail(torch.relu(self.blocks(h)))
                if hasattr(self, "bp"):                          # XNet: luma correction limited to the crawl band
                    import torch.nn.functional as F
                    ry = F.conv2d(F.pad(r[:, :1].float(), (15, 15, 0, 0), mode="reflect"), self.bp.float())
                    r = torch.cat([ry.to(r.dtype), r[:, 1:]], 1)
                return x[:, c:c + 3] + r
        net = BigTemporalDVDNet(sd["head.weight"].shape[0], t=sd["head.weight"].shape[1] // 3)
        if "bp" in sd:
            net.register_buffer("bp", torch.zeros_like(sd["bp"]))
        return net
    if mode == "temporal":
        class TemporalDVDNet(nn.Module):
            def __init__(self, ch=96, dil=(1, 1, 2, 4, 8, 4, 2, 1, 1, 1, 1, 1), t=T_FRAMES):
                super().__init__()
                self.t = t
                layers = [nn.Conv2d(3 * t, ch, 3, padding=1), nn.ReLU(inplace=True)]
                for d in dil:
                    layers += [nn.Conv2d(ch, ch, 3, padding=d, dilation=d), nn.ReLU(inplace=True)]
                layers += [nn.Conv2d(ch, 3, 3, padding=1)]
                self.body = nn.Sequential(*layers)

            def forward(self, x):
                c = 3 * (self.t // 2)
                return x[:, c:c + 3] + self.body(x)
        return TemporalDVDNet()

    class DVDNet(nn.Module):
        def __init__(self, ch=64, dil=(1, 1, 2, 4, 8, 4, 2, 1, 1, 1)):
            super().__init__()
            layers = [nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(inplace=True)]
            for d in dil:
                layers += [nn.Conv2d(ch, ch, 3, padding=d, dilation=d), nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(ch, 3, 3, padding=1)]
            self.body = nn.Sequential(*layers)

        def forward(self, x):
            return x + self.body(x)
    return DVDNet()


def _net(mode):
    if mode not in _NET:
        import torch
        dev = torch.device("cpu")
        try:
            if torch.cuda.is_available():
                (torch.ones(4, device="cuda") * 2).sum().item()
                dev = torch.device("cuda")
        except Exception:                                # noqa: BLE001
            pass
        sd = torch.load(MODELS[mode], map_location="cpu", weights_only=True)
        net = _build(mode, sd)
        net.load_state_dict(sd)
        net = net.to(dev).eval()
        if dev.type == "cuda":
            net = net.half()
            if hasattr(net, "head"):
                net.head.float()
        _NET[mode] = (net, dev)
    return _NET[mode]


def available(mode="restore"):
    return mode in MODELS and os.path.isfile(MODELS[mode])


def dvdnet(clip, strength=1.0, mode="restore"):
    """Clean dots + rainbows in a YUV clip (limited range, any subsampling). Returns the clip's format."""
    import torch
    if strength <= 0:
        return clip
    full = core.resize.Bicubic(clip, format=vs.YUV444PS, range_in=0, range=1)
    if mode in TEMPORAL:
        return _temporal(clip, full, strength, mode)

    def proc(n, f):
        net, dev = _net(mode); net = nnbackend.wrap("dvdnet_" + mode, net, MODELS[mode])
        x = np.stack([np.asarray(f[p]) for p in range(3)])[None]
        with _LOCK, torch.no_grad():
            t = torch.from_numpy(x).to(dev)
            t = t.half() if dev.type == "cuda" else t
            y = net(t).float()
            xt = t.float()
            out = (xt + strength * (y - xt))[0].cpu().numpy()
        dst = f.copy()
        for p in range(3):
            np.copyto(np.asarray(dst[p]), out[p])
        return dst

    out = core.std.ModifyFrame(full, clips=full, selector=proc)
    out = core.resize.Bicubic(out, format=clip.format.id, range_in=1, range=0)
    return core.std.CopyFrameProps(out, clip)


def _temporal(clip, full, strength, mode="temporal"):
    """5-frame window n-2..n+2 (ends repeat the edge frame), one network call per output frame."""
    import torch
    r = T_FRAMES // 2
    n = full.num_frames
    def at(k):                                                # clip whose frame n is full's frame n+k
        if k < 0:
            return full[0] * (-k) + full[:n + k]
        if k > 0:
            return full[k:] + full[-1] * k
        return full
    shifted = [at(k) for k in range(-r, r + 1)]

    def proc(n, f):
        net, dev = _net(mode); net = nnbackend.wrap("dvdnet_" + mode, net, MODELS[mode])
        x = np.concatenate([np.stack([np.asarray(fr[p]) for p in range(3)]) for fr in f])[None]   # [1, 15, H, W]
        with _LOCK, torch.no_grad():
            t = torch.from_numpy(x).to(dev)
            t = t.half() if dev.type == "cuda" else t
            y = net(t).float()
            mid = t[:, 3 * r:3 * r + 3].float()
            out = (mid + strength * (y - mid))[0].cpu().numpy()
        dst = f[r].copy()
        for p in range(3):
            np.copyto(np.asarray(dst[p]), out[p])
        return dst

    out = core.std.ModifyFrame(full, clips=shifted, selector=proc)
    out = core.resize.Bicubic(out, format=clip.format.id, range_in=1, range=0)
    return core.std.CopyFrameProps(out, clip)
