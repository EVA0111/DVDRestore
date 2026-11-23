"""BlockNet: neural repair of MPEG-1 / MPEG-2 compression damage - blocking, mosquito noise around lines,
blocky and smeared colour. Luma and both colour planes, single frame, YUV 4:2:0.

Trained (models/blocknet_train.py, data from models/blocknet_gen.py) on clean HD anime / cartoon clips that
went through a real MPEG encoder (ffmpeg mpeg2video / mpeg1video, quantiser 1.5-28, frame and field DCT,
I/P/B frames, random quantiser matrices, a share as VCD 352x240). Target = the picture the encoder was
given, grain included, so this is a de-compressor and not a denoiser. The block grid is found by the model
itself (random patch positions and resized pictures in training), so resized captures work too.
"""
import os
import threading

import numpy as np
import vapoursynth as vs

import nnbackend
import paths  # noqa: E402  (core/paths.py: every folder in one place)

core = vs.core
MODEL = paths.model("blocknet.pt")
_NET = None
PAD = 16
_LOCK = threading.Lock()
SCALE = 255.0 * 256.0                                    # 16-bit code value -> the 8-bit / 255 scale of training


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

    class BlockNet(nn.Module):
        def __init__(self, ch, dil):
            super().__init__()
            self.head = nn.Conv2d(6, ch, 3, padding=1)
            self.blocks = nn.Sequential(*[Block(ch, d) for d in dil])
            self.tail = nn.Conv2d(ch, 6, 3, padding=1)

        def forward(self, y, uv):
            x = torch.cat([F.pixel_unshuffle(y, 2), uv], 1)
            h = self.head(x.float()).to(y.dtype)         # first layer in float32, as in training
            r = self.tail(F.relu(self.blocks(h)))
            return y + F.pixel_shuffle(r[:, :4], 2), uv + r[:, 4:]

    n = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    dil = (1, 2, 4, 8, 4, 2, 1, 1, 1, 1)[:n] if n <= 10 else (1, 2, 4, 8, 4, 2) + (1,) * (n - 6)
    return BlockNet(sd["head.weight"].shape[0], dil)


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


def blocknet(clip, strength=1.0):
    """clip: YUV 4:2:0, limited range, any depth, even width and height. Returns the same format."""
    import torch
    if strength <= 0 or clip.format.subsampling_w != 1 or clip.format.subsampling_h != 1 or clip.width % 2 or clip.height % 2:
        return clip
    c16 = clip if clip.format.id == vs.YUV420P16 else core.resize.Point(clip, format=vs.YUV420P16)

    def proc(n, f):
        net, dev = _net(); net = nnbackend.wrap("blocknet", net, MODEL)
        pl = [np.asarray(f[p]).astype(np.float32) for p in range(3)]
        with _LOCK, torch.no_grad():
            y0 = torch.from_numpy(pl[0]).to(dev)[None, None] / SCALE
            c0 = torch.from_numpy(np.stack(pl[1:])).to(dev)[None] / SCALE
            # mirror the picture outwards so the frame border is not an edge the model never trained on
            y = torch.nn.functional.pad(y0, (PAD, PAD, PAD, PAD), mode="reflect")
            c = torch.nn.functional.pad(c0, (PAD // 2,) * 4, mode="reflect")
            if dev.type == "cuda":
                y, c = y.half(), c.half()
            oy, oc = net(y, c)
            oy = oy.float()[..., PAD:-PAD, PAD:-PAD]
            oc = oc.float()[..., PAD // 2:-PAD // 2, PAD // 2:-PAD // 2]
            oy = ((y0 + strength * (oy - y0)) * SCALE).round().clamp(0, 65535)[0, 0].cpu().numpy()
            oc = ((c0 + strength * (oc - c0)) * SCALE).round().clamp(0, 65535)[0].cpu().numpy()
        dst = f.copy()
        np.copyto(np.asarray(dst[0]), oy.astype(np.uint16))
        np.copyto(np.asarray(dst[1]), oc[0].astype(np.uint16))
        np.copyto(np.asarray(dst[2]), oc[1].astype(np.uint16))
        return dst

    out = core.std.ModifyFrame(c16, clips=c16, selector=proc)
    if clip.format.id != vs.YUV420P16:
        out = core.resize.Point(out, format=clip.format.id, dither_type="none")
    return out
