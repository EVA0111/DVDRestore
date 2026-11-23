"""DeblendNet: AI blend removal (Adjust -> Frames -> Blend removal -> AI).

A blended frame (a mix of the frame before and after, from standards conversion or a blending
deinterlacer) is turned into the picture of its own moment, from the frames before and after it. Nothing is
duplicated, so pans and zooms stay smooth - the old "nearer clean frame" mode repeated a frame there.
Every frame goes through the network; it was trained to leave clean frames, pans, fades and dissolves
alone, and its result is only used where it really changes the picture (frame gate + soft pixel mask),
so untouched frames stay bit-identical.
Trained (models/deblendnet_train.py) on clean 5-frame anime / cartoon clips: blends of the frames around a
known middle frame, blend chains, fades and dissolves.
"""
import os
import threading

import numpy as np
import vapoursynth as vs

import nnbackend
import paths  # noqa: E402  (core/paths.py: every folder in one place)

core = vs.core
MODEL = paths.model("deblendnet.pt")
_NET = None
_LOCK = threading.Lock()
PAD = 32
FRAME_GATE = 0.35          # mean change (8-bit levels) below which the frame is passed through untouched
PIX_LO, PIX_HI = 1.5, 4.0  # soft pixel mask: blurred change in levels from where the result starts / is fully used


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

    class DeblendNet(nn.Module):
        def __init__(self, ch, dil=(1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1, 1)):
            super().__init__()
            self.head = nn.Conv2d(36, ch, 3, padding=1)
            self.blocks = nn.Sequential(*[Block(ch, d) for d in dil])
            self.tail = nn.Conv2d(ch, 12, 3, padding=1)

        def forward(self, x):
            u = F.pixel_unshuffle(x, 2)
            h = self.head(u.float()).to(x.dtype)
            return x[:, 3:6] + F.pixel_shuffle(self.tail(F.relu(self.blocks(h))), 2)
    return DeblendNet(sd["head.weight"].shape[0])


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


def deblendnet(clip, matrix):
    """clip: YUV (progressive). Returns the same format: blended frames rebuilt (prop _Deblended = 1), every
    other frame bit-identical."""
    import torch
    rgb = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in=matrix, range_in=0)
    prv, nxt = rgb[:1] + rgb[:-1], rgb[1:] + rgb[-1:]

    def proc(n, f):
        net, dev = _net()
        x = np.concatenate([np.stack([np.asarray(fr[p]) for p in range(3)]) for fr in f])[None]
        with _LOCK, torch.no_grad():
            t = torch.from_numpy(x).to(dev).clamp(0, 1)
            H, W = t.shape[-2:]
            t = torch.nn.functional.pad(t, (PAD, PAD + W % 2, PAD, PAD + H % 2), mode="reflect")
            t = t.half() if dev.type == "cuda" else t
            o = nnbackend.wrap("deblendnet", net, MODEL)(t).float()[0, :, PAD:PAD + H, PAD:PAD + W].clamp(0, 1)
            change = float((o - t[0, 3:6, PAD:PAD + H, PAD:PAD + W].float()).abs().mean()) * 255
            o = o.cpu().numpy() if change >= FRAME_GATE else None
        dst = f[1].copy()
        if o is not None:
            for p in range(3):
                np.copyto(np.asarray(dst[p]), o[p])
            dst.props["_Deblended"] = 1
            dst.props["DeblendChange"] = change
        return dst

    res = core.std.ModifyFrame(rgb, clips=[prv, rgb, nxt], selector=proc)
    # soft mask of where the result differs (0 on untouched frames, so they come back bit-identical)
    d = core.std.Expr([res, rgb], "x y - abs 255 *")
    d = core.std.Expr(core.std.SplitPlanes(d), "x y max z max")
    d = core.std.BoxBlur(d, hradius=4, vradius=4)
    peak = (1 << clip.format.bits_per_sample) - 1 if clip.format.sample_type == vs.INTEGER else 1
    m = core.std.Expr(d, f"x {PIX_LO} - {PIX_HI - PIX_LO} / 0 max 1 min {peak} *")
    if clip.format.sample_type == vs.INTEGER:
        m = core.resize.Point(m, format=core.query_video_format(vs.GRAY, vs.INTEGER, clip.format.bits_per_sample, 0, 0).id)
    back = core.resize.Bicubic(res, format=clip.format.id, matrix=matrix, range=0)
    out = core.std.MaskedMerge(clip, back, m, first_plane=True)
    return core.std.CopyFrameProps(out, res)
