"""StableNet: dot crawl + rainbows + MPEG flicker + noise, temporally stable by itself (models/stablenet_train.py).

Recurrent: frame n is computed from input frames n-2..n+2 and the model's own output for frame n-1, so where nothing
moves the steady answer is carried over instead of being decided afresh. VapourSynth may ask for frames in parallel
and out of order: frame n waits (briefly) for frame n-1's output; after a seek / jump, or if n-1 is not coming, the
recurrence restarts from the input frame itself (as at a scene cut in training).
Input / output: YUV, limited range, any subsampling; returns the clip's format.
FieldNet (models/fieldnet_train.py, 42 input channels) also needs the raw telecined source: pass raw= (the source
clip before IVTC, frame 0 = source frame raw_start); for each frame the window raw[k-2..k+2] around its source frame
k (_SrcN) and its field match (TFMMatch) go in as well.
FieldNet v2 (models/fieldnet2_train.py, weights with "enc.*"): raw frames k-3..k+3 split into their fields (native
interlaced 4:2:0), through its field encoder; match one-hot over p / c / n / b / u."""
import os
import threading

import numpy as np
import vapoursynth as vs

import nnbackend
import paths  # noqa: E402  (core/paths.py: every folder in one place)

core = vs.core
MODEL = paths.model("stablenet.pt")
UNI_MODEL = paths.model("uninet.pt")
FIELD_MODEL = paths.model("fieldnet.pt")
_NET = None                        # (path, net, dev) of the model loaded last
STATS = {"frames": 0, "restarts": 0}  # how often the recurrence had no previous output (seek / out-of-order start)
DIALS = ("crawl", "rainbow", "mpeg", "halo", "chroma", "grain_keep")   # UniNet (all-in-one) dials, 0..1
_LOCK = threading.Lock()
T = 5
WAIT_S = 20.0                      # how long frame n waits for frame n-1 before restarting the recurrence
WARMUP = 6                         # frames run first when a frame has no history (paused preview, seek)
CUT = 0.08                         # mean luma change (0..1, 1/8 size) between two frames that counts as a scene cut


def available(model=None):
    return os.path.isfile(model or MODEL)


def _build(sd):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, ch, d):
            super().__init__()
            self.a = nn.Conv2d(ch, ch, 3, padding=d, dilation=d)
            self.b = nn.Conv2d(ch, ch, 3, padding=1)

        def forward(self, x):
            return x + self.b(torch.relu(self.a(torch.relu(x))))

    class StableNet(nn.Module):
        """StableNet (18 input channels), UniNet (+ 6 dials) or FieldNet (+ 6 dials + 15 raw + 3 match): from the weights."""
        def __init__(self, ch, inch, dil=(1, 2, 4, 8, 4, 2, 1, 1)):
            super().__init__()
            self.field = inch == 3 * T + 3 + 6 + 3 * T + 3
            self.ndial = 6 if self.field else inch - (3 * T + 3)
            self.head = nn.Conv2d(inch, ch, 3, padding=1)
            self.blocks = nn.Sequential(*[Block(ch, d) for d in dil])
            self.tail = nn.Conv2d(ch, 3, 3, padding=1)

        def forward(self, frames, prev, dials=None, raw=None, match=None):
            c = 3 * (T // 2)
            parts = [frames, prev]
            if self.ndial:
                parts.append(dials[:, :, None, None].expand(-1, -1, *frames.shape[-2:]).to(frames.dtype))
            if self.field:
                parts += [raw, match[:, :, None, None].expand(-1, -1, *frames.shape[-2:]).to(frames.dtype)]
            h = self.head(torch.cat(parts, 1).float()).to(frames.dtype)
            return frames[:, c:c + 3] + self.tail(torch.relu(self.blocks(h)))
    class FieldNet2(nn.Module):
        """FieldNet v2: fields of raw k-3..k+3 -> field encoder (half height, folded to full) -> head with the rest"""
        def __init__(self, ch, inch, nfc, enc_ch, dil):
            super().__init__()
            self.field, self.ndial, self.nfc = 2, 6, nfc
            self.nx = inch - (3 * T + 3 + 6 + nfc + 5)          # v3: phase top / bottom + pal
            self.enc = nn.Module()
            self.enc.a = nn.Conv2d(7 * 6, enc_ch, 3, padding=1)
            self.enc.b = nn.Conv2d(enc_ch, enc_ch, 3, padding=2, dilation=2)
            self.enc.c = nn.Conv2d(enc_ch, enc_ch, 3, padding=1)
            self.enc.o = nn.Conv2d(enc_ch, 2 * nfc, 3, padding=1)
            self.head = nn.Conv2d(inch, ch, 3, padding=1)
            self.blocks = nn.Sequential(*[Block(ch, d) for d in dil])
            self.tail = nn.Conv2d(ch, 3, 3, padding=1)

        def forward(self, frames, prev, dials, fields, match, extra=None):
            c = 3 * (T // 2)
            sp = frames.shape[-2:]
            e = self.enc
            f = fields.float()
            h = torch.relu(e.a(f))
            h = h + torch.relu(e.c(torch.relu(e.b(h))))
            o = e.o(h)
            B, _, hh, w = o.shape
            fe = o.view(B, 2, self.nfc, hh, w).permute(0, 2, 3, 1, 4).reshape(B, self.nfc, 2 * hh, w)
            dm = dials[:, :, None, None].expand(-1, -1, *sp).float()
            mm = match[:, :, None, None].expand(-1, -1, *sp).float()
            parts = [frames.float(), prev.float(), dm, fe, mm]
            if self.nx:
                parts.append(extra[:, :, None, None].expand(-1, -1, *sp).float())
            h = self.head(torch.cat(parts, 1)).to(frames.dtype)
            return frames[:, c:c + 3] + self.tail(torch.relu(self.blocks(h)))
    n = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    if "enc.a.weight" in sd:
        return FieldNet2(sd["head.weight"].shape[0], sd["head.weight"].shape[1], sd["enc.o.weight"].shape[0] // 2,
                         sd["enc.a.weight"].shape[0], (1, 2, 4, 8, 4, 2, 1, 1)[:n])
    return StableNet(sd["head.weight"].shape[0], sd["head.weight"].shape[1], (1, 2, 4, 8, 4, 2, 1, 1)[:n])


def _net(path=None):
    global _NET
    path = path or MODEL
    if _NET is None or _NET[0] != path:
        import torch
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        net = _build(sd)
        net.load_state_dict(sd)
        net = net.to(dev).eval()
        if dev.type == "cuda":
            net = net.half()
            net.head.float()
            if hasattr(net, "enc"):
                net.enc.float()
        _NET = (path, net, dev)
    return _NET[1], _NET[2]


_V2 = {}


def _is_v2(model):
    if model not in _V2:
        import torch
        _V2[model] = "enc.a.weight" in torch.load(model, map_location="cpu", weights_only=True)
    return _V2[model]


def stablenet(clip, strength=1.0, model=None, dials=None, raw=None, raw_start=0, pal=False):
    """dials: dict of DIALS (0..1) for a UniNet / FieldNet model; ignored by StableNet.
    raw / raw_start: the raw telecined source for FieldNet (see the module note)."""
    import torch
    if strength <= 0:
        return clip
    model = model or MODEL
    dial_vec = [float((dials or {}).get(k, 0.0 if k == "grain_keep" else 1.0)) for k in DIALS]
    full = core.resize.Bicubic(clip, format=vs.YUV444PS, range_in=0, range=1)
    n_all = full.num_frames
    r = T // 2

    def at(k):
        if k < 0:
            return full[0] * (-k) + full[:n_all + k]
        if k > 0:
            return full[k:] + full[-1] * k
        return full
    K = WARMUP
    shifted = [at(k) for k in range(-(r + K), r + 1)]       # frames n-K-2 .. n+2: the window of n and K frames before
    nsh = len(shifted)
    v2 = raw is not None and _is_v2(model)
    rr = 3 if v2 else r                                      # raw frames k-rr .. k+rr
    if raw is not None:
        # raw window of every output frame, stacked vertically: raw k-rr .. k+rr, k = _SrcN - raw_start
        # (v1: YUV444 float frames; v2: the native interlaced 4:2:0 frames, split into fields in proc)
        rf = core.resize.Point(raw, format=vs.YUV420P8) if v2 else core.resize.Bicubic(raw, format=vs.YUV444PS, range_in=0, range=1)
        nr = rf.num_frames

        def pick(d):
            def sel(n, f):
                k = int(f.props.get("_SrcN", raw_start + n)) - raw_start + d
                return rf[min(max(k, 0), nr - 1)]
            return core.std.FrameEval(core.std.BlankClip(rf, length=n_all), sel, prop_src=full)
        rv = core.std.StackVertical([pick(d) for d in range(-rr, rr + 1)])

        def rat(k):
            return rv[0] * (-k) + rv[:n_all + k] if k < 0 else rv
        shifted += [rat(k) for k in range(-K, 1)]            # raw windows of frames n-K .. n
    state = {"out": {}, "busy": set()}
    cond = threading.Condition()

    def low(fr):                                             # small luma for cut detection
        return np.asarray(fr[0])[::8, ::8]

    def proc(n, f):
        net, dev = _net(model)
        rawf = f[nsh:]                                       # raw windows of frames n-K .. n (FieldNet)
        f = f[:nsh]
        planes = [np.stack([np.asarray(fr[p]) for p in range(3)]) for fr in f]     # f[j] = frame n - K - r + j
        pos = lambda t: t - n + r + K                        # list position of frame t
        win = lambda t: np.concatenate(planes[pos(t) - r:pos(t) + r + 1])[None]    # [1, 15, H, W] around frame t
        cut_at = lambda t: t > 0 and float(np.mean(np.abs(low(f[pos(t)]) - low(f[pos(t) - 1])))) > CUT
        with cond:
            state["busy"].add(n)
            running = any(k in state["out"] or k in state["busy"] for k in (n - 1, n - 2))
            if running and n - 1 not in state["out"] and not cut_at(n):
                cond.wait_for(lambda: n - 1 in state["out"], timeout=WAIT_S)
            prev = None if cut_at(n) else state["out"].get(n - 1)
            STATS["frames"] += 1
            STATS["restarts"] += prev is None
        with _LOCK, torch.no_grad():
            dv = None
            def run(t, p):
                nonlocal dv
                x = torch.from_numpy(win(t)).to(dev)
                x = x.half() if dev.type == "cuda" else x
                if dv is None:
                    dv = torch.tensor([dial_vec], device=dev, dtype=x.dtype)
                mid_ = x[:, 3 * r:3 * r + 3]
                if not net.field:
                    return net(x, mid_ if p is None else p, dv), mid_
                if net.field == 2:
                    if rawf:
                        fr = rawf[t - n + K]
                        nf = 2 * rr + 1
                        Y = np.asarray(fr[0]); hh, ww = Y.shape[0] // nf, Y.shape[1]
                        Yt = (torch.from_numpy(np.ascontiguousarray(Y.reshape(nf, hh, ww))).to(dev).float() - 16) / 219
                        UV = np.stack([np.asarray(fr[q]).reshape(nf, hh // 2, ww // 2) for q in (1, 2)], 1)
                        UVt = (torch.from_numpy(np.ascontiguousarray(UV)).to(dev).float() - 128) / 224
                        parts = []
                        for par in (0, 1):
                            parts += [Yt[:, None, par::2], torch.nn.functional.interpolate(UVt[:, :, par::2], size=(hh // 2, ww), mode="bilinear", align_corners=False)]
                        fw = torch.cat(parts, 1).reshape(1, nf * 6, hh // 2, ww)
                        m = int(f[pos(t)].props.get("TFMMatch", 1))
                    else:                                    # no raw source: fields of the IVTC window, as progressive
                        xf = x.float().reshape(T, 3, *x.shape[-2:])
                        xf = torch.cat([xf[:, :, 0::2], xf[:, :, 1::2]], 1)                    # [5, 6, h/2, w]
                        xf = torch.cat([xf[:1], xf, xf[-1:]], 0)                                # 7 frames
                        fw, m = xf.reshape(1, 7 * 6, *xf.shape[-2:]), 1
                    mt = torch.tensor([[float(m == i) for i in range(5)]], device=dev)
                    ex = None
                    if net.nx:                               # v3: field phase of a bobbed PAL frame, PAL flag
                        props = f[pos(t)].props
                        ph = int(props["_Phase"]) if "_Phase" in props else -1
                        ex = torch.tensor([[float(ph == 0), float(ph == 1), float(pal)]], device=dev)
                    return net(x, mid_ if p is None else p, dv, fw, mt, ex), mid_
                if rawf:
                    fr = rawf[t - n + K]
                    hh = fr.height // T
                    rw = np.stack([np.asarray(fr[q]) for q in range(3)])                 # [3, 5H, W]
                    rw = rw.reshape(3, T, hh, -1).transpose(1, 0, 2, 3).reshape(1, 3 * T, hh, -1)
                    rw = torch.from_numpy(np.ascontiguousarray(rw)).to(dev).to(x.dtype)
                    m = int(f[pos(t)].props.get("TFMMatch", 1))
                else:                                        # no raw source given: as a progressive source
                    rw, m = x, 1
                mt = torch.tensor([[float(m == i) for i in range(3)]], device=dev, dtype=x.dtype)
                return net(x, mid_ if p is None else p, dv, rw, mt), mid_
            if prev is None and not cut_at(n):
                # no history (paused preview, seek, first frame): warm up over the frames before n, from the last cut
                t0 = max(0, n - K)
                for t in range(n - 1, t0, -1):
                    if cut_at(t):
                        t0 = t
                        break
                p = None
                for t in range(t0, n):
                    p, _ = run(t, p)
                prev = p
            y, mid = run(n, prev)
            res = y.float()
            out = (mid.float() + strength * (res - mid.float())) if strength != 1.0 else res
            keep = y.detach()
        with cond:
            state["out"][n] = keep
            for k in [k for k in state["out"] if k < n - 4]:
                del state["out"][k]
            state["busy"].discard(n)
            cond.notify_all()
        dst = f[pos(n)].copy()
        o = out[0].cpu().numpy()
        for p_ in range(3):
            np.copyto(np.asarray(dst[p_]), o[p_])
        return dst

    out = core.std.ModifyFrame(full, clips=shifted, selector=proc)
    out = core.resize.Bicubic(out, format=clip.format.id, range_in=1, range=0)
    return core.std.CopyFrameProps(out, clip)
