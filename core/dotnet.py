"""DotNet: neural dot-crawl / hanging-dot remover for NTSC DVDs.

Trained on clean HD anime and cartoon frames run through a simulated NTSC chain: composite encode with the
real subcarrier grid (0.245-0.27 cyc/px, phase flips each field line), decoded with 1990s-style notch,
2-line and 3-line comb decoders. Target = the same chain without the chroma leak, so the network only
learns to remove the leaked carrier (it never sharpens or invents detail). Input: luma + chroma, because
dots only occur where the colour changes - that is what lets it tell static dots from static detail.

Measured: clean Blu-ray frames change only 50-57 dB PSNR (the old spatial comb filter: 39.8 dB), while the
difference maps on Lodoss / Courage contain only the dot pattern along colour edges.
"""
import os
import threading

import numpy as np
import vapoursynth as vs

import nnbackend
import paths  # noqa: E402  (core/paths.py: every folder in one place)

core = vs.core
WEIGHTS = paths.model("dotnet.pt")
_NET = {}
_LOCK = threading.Lock()
GATE = 0.01                      # chroma-gradient level below which corrections fade out


def _build():
    import torch.nn as nn

    class DotNet(nn.Module):
        def __init__(self, ch=48, n=8):
            super().__init__()
            layers = [nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(inplace=True)]
            for i in range(n - 2):
                d = 2 if i in (2, 4) else 1
                layers += [nn.Conv2d(ch, ch, 3, padding=d, dilation=d), nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(ch, 1, 3, padding=1)]
            self.body = nn.Sequential(*layers)

        def forward(self, x):
            return x[:, :1] - self.body(x)
    return DotNet()


def _net():
    if "net" not in _NET:
        import torch
        dev = torch.device("cpu")
        try:
            if torch.cuda.is_available():
                (torch.ones(4, device="cuda") * 2).sum().item()
                dev = torch.device("cuda")
        except Exception:                                # noqa: BLE001
            pass
        net = _build()
        net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
        net = net.to(dev).eval()
        if dev.type == "cuda":
            net = net.half()
        _NET.update(net=net, dev=dev)
    return _NET["net"], _NET["dev"]


def available():
    return os.path.isfile(WEIGHTS)


def dotnet(clip, strength=1.0):
    """Remove dot crawl from the luma of a YUV clip (any bit depth, limited range). strength scales the
    removed pattern (0.6 light, 1.0 normal, 1.3 strong). Chroma is passed through untouched."""
    import torch
    if strength <= 0:
        return clip
    full = core.resize.Bicubic(clip, format=vs.YUV444PS, range_in=0, range=1)
    luma = core.std.ShufflePlanes(full, 0, vs.GRAY)

    def proc(n, f):
        net, dev = _net(); net = nnbackend.wrap("dotnet", net, WEIGHTS)
        src = f[1]
        x = np.stack([np.asarray(src[p]) for p in range(3)])[None]
        with _LOCK, torch.no_grad():
            t = torch.from_numpy(x).to(dev)
            t = t.half() if dev.type == "cuda" else t
            y = net(t)[0, 0].float()
            # gate: dots only exist near colour edges; block corrections where chroma is flat
            # (clean Batman Blu-ray: 26 % of changes blocked; Lodoss / Courage keep 96-99 % of the removal)
            uv = t[0, 1:].float()
            gx = (uv[:, :, 2:] - uv[:, :, :-2]).abs().sum(0) / 2
            gy = (uv[:, 2:, :] - uv[:, :-2, :]).abs().sum(0) / 2
            gch = torch.nn.functional.pad(gx, (1, 1))[None, None] + torch.nn.functional.pad(gy, (0, 0, 1, 1))[None, None]
            gch = torch.nn.functional.max_pool2d(gch, 7, 1, 3)[0, 0]
            gate = (gch / GATE).clamp(0, 1)
            lum = t[0, 0].float()
            out = (lum - strength * gate * (lum - y)).cpu().numpy()
        dst = f[0].copy()
        np.copyto(np.asarray(dst[0]), out)
        return dst

    y = core.std.ModifyFrame(luma, clips=[luma, full], selector=proc)
    y = core.resize.Point(y, format=clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id,
                          range_in=1, range=0)
    return core.std.ShufflePlanes([y, clip], [0, 1, 2], vs.YUV)
