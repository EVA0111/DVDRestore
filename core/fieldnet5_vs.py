"""FieldNet v5 in the app (models/fieldnet5.py, trained by models/fieldnet5_train.py): v4 trunk + backward pass +
NeuFlow v2 motion + TStab cascade, then the StillLock.

Frames are computed in order (frame n waits briefly for n-1, like fieldnet.py). Per scene (cuts detected as in
fieldnet.py, nothing crosses a cut):
  backward   in blocks of BLK frames: the trunk runs backwards in time from LA frames past the block end (the scene end
             when it is closer) down to the block start, leaving each frame its future-aware output + memory
  forward    the trunk runs forwards with its own memory (aligned by NeuFlow) plus the backward state of the same frame
  TStab      on the forward outputs n-9..n+9 (motion / detail / still stages, NeuFlow alignment), cached per frame
  StillLock  long memory for still areas; reset at cuts, warmed up after a seek
Input / output: YUV, limited range, any subsampling; returns the clip's format."""
import os
import sys
import threading

import numpy as np
import vapoursynth as vs

import fieldnet as f3
import paths  # noqa: E402  (core/paths.py: every folder in one place)

core = vs.core
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = paths.model("fieldnet5.pt")
T, K, RR = 5, 6, 3
TSR = 9                            # TStab reach (tstab_core.REACH)
BLK, LA = 16, 16                   # backward blocks and their look-ahead
WARM = 12                          # still-lock warm-up after a seek
WAIT_S = 20.0
CUT = f3.CUT
SWAP_PN = f3.SWAP_PN
DIALS = f3.DIALS
STATS = {"frames": 0, "seek_warmups": 0, "bwd_blocks": 0}
_NET = None
_LOCK = threading.Lock()


def available(model=None):
    return os.path.isfile(model or MODEL)


def is_v5(model):
    import torch
    try:
        sd = torch.load(model, map_location="cpu", weights_only=True)
        return int(sd.get("version", torch.tensor(0))) == 5
    except Exception:                                     # noqa: BLE001
        return False


def _net(path):
    global _NET
    if _NET is None or _NET[0] != path:
        import torch
        sys.path.insert(0, paths.MODELS)
        import fieldnet5 as m5
        import fieldnet_core as fc
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        net = m5.build(path, dev).eval()
        if dev.type == "cuda" and not os.environ.get("FIELDNET_NO_CL"):
            net.trunk = net.trunk.to(memory_format=torch.channels_last)
        _NET = (path, net, dev, fc, m5.ts)
    return _NET[1:]


def fieldnet5(clip, model=None, dials=None, raw=None, raw_start=0, pal=False, lock=True, lock_keep=0.97):
    import torch
    model = model or MODEL
    dial_vec = [float((dials or {}).get(k, 0.0 if k == "grain_keep" else 1.0)) for k in DIALS]
    full = core.resize.Bicubic(clip, format=vs.YUV444PS, range_in=0, range=1)
    n_all = full.num_frames
    r = T // 2
    EB = TSR + K + r + (WARM if lock else 0) + 2               # frames before n the view must hold (seek warm-ups)
    EF = TSR + BLK + LA + r + 1                                # frames after n (TStab reach + backward block + look-ahead)

    def at(c, k, n_):
        if k < 0:
            return c[0] * (-k) + c[:n_ + k]
        if k > 0:
            return c[k:] + c[-1] * k
        return c
    shifted = [at(full, k, n_all) for k in range(-EB, EF + 1)]
    nsh = len(shifted)
    have_raw = raw is not None
    if have_raw:
        rf = core.resize.Point(raw, format=vs.YUV420P8)
        nr = rf.num_frames

        def pick(d):
            def sel(n, f):
                kk = int(f.props.get("_SrcN", raw_start + n)) - raw_start + d
                return rf[min(max(kk, 0), nr - 1)]
            return core.std.FrameEval(core.std.BlankClip(rf, length=n_all), sel, prop_src=full)
        rv = core.std.StackVertical([pick(d) for d in range(-RR, RR + 1)])
        shifted += [at(rv, k, n_all) for k in range(-EB, EF + 1)]   # raw window of every frame in the view
    st = {"fwd": {}, "bwd": {}, "s1": {}, "s2": {}, "fl": {}, "lock": None, "last": None, "scene": {}}
    cond = threading.Condition()

    def proc(n, f):
        net, dev, fc, ts = _net(model)
        F_, R_ = f[:nsh], f[nsh:]
        pos = lambda t: t - n + EB
        ok = lambda t: 0 <= t < n_all and 0 <= pos(t) < nsh
        low, planes = {}, {}

        def lowy(t):
            if t not in low:
                low[t] = np.asarray(F_[pos(t)][0])[::8, ::8]
            return low[t]

        def cut_at(t):
            return t > 0 and ok(t) and ok(t - 1) and float(np.mean(np.abs(lowy(t) - lowy(t - 1)))) > CUT

        def scene(t):
            if t in st["scene"]:
                return st["scene"][t]
            a = t
            while ok(a - 1) and not cut_at(a):
                a -= 1
            b = t
            while ok(b + 1) and not cut_at(b + 1):
                b += 1
            if (a == 0 or ok(a - 1)) and (b == n_all - 1 or ok(b + 1)):
                st["scene"][t] = (a, b)
            return a, b

        def plane3(t):
            if t not in planes:
                planes[t] = torch.from_numpy(np.stack([np.asarray(F_[pos(t)][p]) for p in range(3)])).to(dev, non_blocking=True)
            return planes[t]

        def kof(t):
            return int(F_[pos(t)].props.get("_SrcN", raw_start + t)) - raw_start

        def inputs(t, rev, sa, sb):
            """window, fields, match one-hot, extra for frame t (fieldnet.py's, with the raw window of t's own view slot)"""
            idx = [min(max(t + j, sa), sb) for j in range(-r, r + 1)]
            if rev:
                idx = idx[::-1]
            win = torch.cat([plane3(i) for i in idx])[None]
            props = F_[pos(t)].props
            m = int(props.get("TFMMatch", 1)) if have_raw else 1
            if rev:
                m = SWAP_PN[m] if 0 <= m < 5 else m
            mt = torch.tensor([[float(m == i) for i in range(5)]], device=dev)
            ph = int(props["_Phase"]) if "_Phase" in props else -1
            ex = torch.tensor([[float(ph == 0), float(ph == 1), float(pal)]], device=dev)
            if have_raw:
                fr = R_[pos(t)]
                nf = 2 * RR + 1
                Y = np.asarray(fr[0]); hh, ww = Y.shape[0] // nf, Y.shape[1]
                kt, klo, khi = kof(t), kof(sa), kof(sb)
                slots = [min(max(kt + d, klo), khi) - kt + RR for d in range(-RR, RR + 1)]
                slots = [min(max(s, 0), nf - 1) for s in slots]
                if rev:
                    slots = slots[::-1]
                Yt = (torch.from_numpy(np.ascontiguousarray(Y.reshape(nf, hh, ww))).to(dev).float() - 16) / 219
                UV = np.stack([np.asarray(fr[q]).reshape(nf, hh // 2, ww // 2) for q in (1, 2)], 1)
                UVt = (torch.from_numpy(np.ascontiguousarray(UV)).to(dev).float() - 128) / 224
                fw = fc.split_fields(Yt[slots], UVt[slots]).reshape(1, nf * 6, hh // 2, ww)
            else:
                xf = win.float().reshape(T, 3, *win.shape[-2:])
                xf = torch.cat([xf[:, :, 0::2], xf[:, :, 1::2]], 1)
                xf = torch.cat([xf[:1], xf, xf[-1:]], 0)
                fw = xf.reshape(1, 7 * 6, *xf.shape[-2:])
            return win, fw, mt, ex

        dv = torch.tensor([dial_vec[:net.trunk.ndial]], device=dev)

        def motion(cur, prv):                                   # as FieldNet5.motion: NeuFlow on lightly blurred frames
            fl = net.flow(cur, prv, blur=1)
            err = (fc._blur(cur[:, :1], 2) - fc._blur(fc.warp(prv[:, :1], fl), 2)).abs()
            return fl, (err * 30).clamp(0, 1)

        def step(t, prev, rev, sa, sb, hadd=None):
            """one trunk step on frame t; prev = (out, hid, mid frame) of its neighbour in run order, or None"""
            win, fw, mt, ex = inputs(t, rev, sa, sb)
            mid = plane3(t)[None].float()
            if prev is None:
                pw, hw, occ = mid, torch.zeros(1, fc.NH, *mid.shape[-2:], device=dev), torch.ones(1, 1, *mid.shape[-2:], device=dev)
            else:
                fl, occ = motion(mid, prev[2])
                pw, hw = fc.warp(prev[0], fl), fc.warp(prev[1], fl)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o, h = net.trunk(win, pw, hw, occ, dv, fw, mt, ex, hadd)
            return o.float(), h.float(), mid

        def bwd(t, sa, sb):
            """backward state of frame t, computing its whole block once"""
            if t not in st["bwd"]:
                b0 = sa + ((t - sa) // BLK) * BLK
                b1 = min(sb, b0 + BLK - 1)
                end = min(sb, b1 + LA)
                prev = None
                for u in range(end, b0 - 1, -1):
                    o, h, mid = step(u, prev, True, sa, sb)
                    if u <= b1:
                        st["bwd"][u] = (o, h)
                    prev = (o, h, mid)
                STATS["bwd_blocks"] += 1
            return st["bwd"][t]

        def fwd(t, sa, sb):
            """forward output of frame t (computed in order from the last cached frame of its scene)"""
            if t in st["fwd"]:
                return st["fwd"][t][0]
            t0 = t
            while t0 > sa and (t0 - 1) not in st["fwd"]:
                t0 -= 1
            if t0 > sa and t - t0 > K + TSR:                      # nothing cached near: warm up only K frames back
                t0 = max(sa, t - K - TSR)
            for u in range(t0, t + 1):
                ob, hb = bwd(u, sa, sb)
                hadd = net.bwd(torch.cat([ob, hb], 1).float())
                prev = None if u == sa else (st["fwd"][u - 1] if (u - 1) in st["fwd"] else None)
                o, h, mid = step(u, prev, False, sa, sb, hadd)
                st["fwd"][u] = (o, h, mid)
            return st["fwd"][t][0]

        def flow(a, b, sa, sb):                                 # TStab alignment between forward outputs
            if a == b:
                return torch.zeros(1, 2, *plane3(n).shape[-2:], device=dev)
            k = (a, b)
            if k not in st["fl"]:
                st["fl"][k] = net.flow(fwd(a, sa, sb), fwd(b, sa, sb))
            return st["fl"][k]

        def window(t, src, sa, sb):
            R = ts.R
            ids = [min(max(t + j, sa), sb) for j in range(-R, R + 1)]
            seq = torch.cat([src(i) for i in ids])[None]
            H, W = seq.shape[-2:]
            fl = torch.zeros(1, 7, len(ts.OFFS), 2, H, W, device=dev)
            for oi, o_ in enumerate(ts.OFFS):
                fl[0, R, oi] = flow(t, ids[R + o_], sa, sb)[0]
                fl[0, R + o_, ts.OFFS.index(-o_)] = flow(ids[R + o_], t, sa, sb)[0]
            return seq, fl

        def s1(t, sa, sb):
            if t not in st["s1"]:
                seq, fl = window(t, lambda i: fwd(i, sa, sb), sa, sb)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    st["s1"][t] = net.ts.s1(seq, fl, ts.R, 1)[:, 0].float()
            return st["s1"][t]

        def s2(t, sa, sb):
            if t not in st["s2"]:
                seq, fl = window(t, lambda i: s1(i, sa, sb), sa, sb)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    st["s2"][t] = net.ts.s2(seq, fl, ts.R, 1)[:, 0].float()
            return st["s2"][t]

        def s3(t, sa, sb):
            seq, fl = window(t, lambda i: s2(i, sa, sb), sa, sb)
            mot = fl[:, ts.R].norm(dim=2).amax(1, keepdim=True)
            mot = torch.nn.functional.max_pool2d(mot, 5, 1, 2)[:, None]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                return net.ts.s3(seq, ts.R, 1, mot)[:, 0].float()

        def prune(t):
            lo, hi = t - EB, t + EF
            for d in (st["fwd"], st["bwd"], st["s1"], st["s2"], st["scene"]):
                for k in [k for k in d if k < lo or k > hi]:
                    del d[k]
            for k in [k for k in st["fl"] if min(k) < lo or max(k) > hi]:
                del st["fl"][k]

        cut_n = cut_at(n)
        with cond:
            running = st["last"] is not None and st["last"] in (n - 1, n - 2)
            if running and st["last"] != n - 1 and not cut_n:
                cond.wait_for(lambda: st["last"] == n - 1, timeout=WAIT_S)
            STATS["frames"] += 1
        with _LOCK, torch.no_grad():
            sa, sb = scene(n)
            o = s3(n, sa, sb)
            if lock:
                lk = st["lock"]
                if cut_n or n == sa or lk is None or st["last"] != n - 1:
                    lk = ts.StillLock(keep=lock_keep)
                    if not cut_n and n != sa:                   # seek: warm the memory up over the frames before n
                        STATS["seek_warmups"] += 1
                        for t in range(max(sa, n - WARM), n):
                            lk(s3(t, sa, sb))
                    st["lock"] = lk
                o = lk(o)
            prune(n)
        with cond:
            st["last"] = n
            cond.notify_all()
        dst = F_[pos(n)].copy()
        oc = o[0].cpu().numpy()
        for p_ in range(3):
            np.copyto(np.asarray(dst[p_]), oc[p_])
        return dst

    out = core.std.ModifyFrame(full, clips=shifted, selector=proc)
    out = core.resize.Bicubic(out, format=clip.format.id, range_in=1, range=0)
    return core.std.CopyFrameProps(out, clip)
