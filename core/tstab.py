"""TStab v2 + StillLock in the app: the learned temporal stabilizer that runs right after FieldNet
(models/tstab_core.py, trained by models/tstab_train.py on top of the frozen FieldNet).
  TStab      cascade of three averaging stages over FieldNet's own outputs: motion-aligned (mc-like), fine detail
             (gtfix-like), still areas (ttemp-like); every stage only re-uses real content; windows never cross a
             scene cut. Flows and stage outputs are cached per frame, so each new frame costs one pass of each stage.
  StillLock  long memory for still areas (fine detail averaged over many frames where nothing really changes; any real
             change resets it); stateful: frames are computed in order (n waits briefly for n-1), cuts reset it, a seek
             warms it up over the frames before it
Input / output: YUV, limited range, any subsampling; returns the clip's format."""
import os
import sys
import threading

import numpy as np
import vapoursynth as vs
import paths  # noqa: E402  (core/paths.py: every folder in one place)

core = vs.core
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = paths.model("tstab.pt")
WARM = 12                          # still-lock warm-up after a seek
CUT = 0.08                         # same scene-cut test as fieldnet.py
WAIT_S = 20.0
_NET = None
_LOCK = threading.Lock()


def available(model=None):
    return os.path.isfile(model or MODEL)


def _net(path):
    global _NET
    if _NET is None or _NET[0] != path:
        import torch
        sys.path.insert(0, paths.MODELS)
        import tstab_core as ts
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        net = ts.TStab()
        if path and os.path.isfile(path):
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        _NET = (path, net.to(dev).eval(), dev, ts)
    return _NET[1:]


def _at(c, k, n_):
    if k < 0:
        return c[0] * (-k) + c[:n_ + k]
    if k > 0:
        return c[k:] + c[-1] * k
    return c


def still_lock(clip, keep=0.97, thr=1.0, s=1.5, motion=None, anchor=False, fade=False, chroma=False):
    """the still-area lock on its own (after FieldNet + mc + gtfix + ttemp in the app's FieldNet pipeline): luma fine
    detail averaged over a long memory where the picture has not really changed, reset by any real change. Frames are
    computed in order (n waits briefly for n-1); cuts reset it; a seek warms it up over the WARM frames before.
    Real Courage disc (v4 + mc + gtfix + ttemp + lock): door / rug 0.11, WELCOME letters shimmer 0.04, wobble 0.38."""
    import torch
    full = core.resize.Bicubic(clip, format=vs.YUV444PS, range_in=0, range=1)
    n_all = full.num_frames
    shifted = [_at(full, k, n_all) for k in range(-WARM, 1)]
    if motion is not None:                                     # GRAY float clip, > 0.5 where something moved
        mo = core.resize.Point(motion, format=vs.GRAYS)
        shifted += [_at(mo, k, n_all) for k in range(-WARM, 1)]
    state = {"lock": None, "last": None}
    cond = threading.Condition()

    def proc(n, f):
        sys.path.insert(0, paths.MODELS)
        import tstab_core as ts
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        pos = lambda t: t - n + WARM
        ok = lambda t: 0 <= t < n_all and 0 <= pos(t) <= WARM

        def lowy(t):
            return np.asarray(f[pos(t)][0])[::8, ::8]

        def cut_at(t):
            return t > 0 and ok(t) and ok(t - 1) and float(np.mean(np.abs(lowy(t) - lowy(t - 1)))) > CUT

        def frame(t):
            return torch.from_numpy(np.stack([np.asarray(f[pos(t)][p]) for p in range(3)])).to(dev)[None]

        def mov(t):
            if motion is None:
                return None
            return torch.from_numpy(np.asarray(f[WARM + 1 + pos(t)][0]).copy()).to(dev)[None, None]

        cut_n = cut_at(n)
        with cond:
            running = state["last"] is not None and state["last"] in (n - 1, n - 2)
            if running and state["last"] != n - 1 and not cut_n:
                cond.wait_for(lambda: state["last"] == n - 1, timeout=WAIT_S)
        with _LOCK, torch.no_grad():
            lk = state["lock"]
            if cut_n or lk is None or state["last"] != n - 1:
                lk = ts.StillLock(s=s, thr=thr / 255, keep=keep, anchor=bool(anchor), fade=bool(fade), chroma=bool(chroma))
                if not cut_n:                                  # seek / start: warm up over the frames before n (same scene)
                    a = n
                    while ok(a - 1) and not cut_at(a):
                        a -= 1
                    for t in range(a, n):
                        lk(frame(t), mov(t))
                state["lock"] = lk
            o = lk(frame(n), mov(n))
        with cond:
            state["last"] = n
            cond.notify_all()
        dst = f[WARM].copy()
        oc = o[0].cpu().numpy()
        for p_ in range(3):
            np.copyto(np.asarray(dst[p_]), oc[p_])
        return dst

    out = core.std.ModifyFrame(full, clips=shifted, selector=proc)
    out = core.resize.Bicubic(out, format=clip.format.id, range_in=1, range=0)
    return core.std.CopyFrameProps(out, clip)


def tstab(clip, model=None, lock=True, lock_keep=0.97):
    """model None = trained weights if installed, else the untrained cascade (the filter rules, already useful)"""
    import torch
    model = model if model is not None else (MODEL if available() else "")
    full = core.resize.Bicubic(clip, format=vs.YUV444PS, range_in=0, range=1)
    n_all = full.num_frames
    RCH = 9                                                    # tstab_core.REACH
    EB = RCH + (WARM if lock else 0)                           # frames n-EB .. n+RCH
    shifted = [_at(full, k, n_all) for k in range(-EB, RCH + 1)]
    nsh = len(shifted)
    state = {"lock": None, "last": None, "fl": {}, "s1": {}, "s2": {}, "y": {}, "scene": {}}
    cond = threading.Condition()

    def proc(n, f):
        net, dev, ts = _net(model)
        R = ts.R
        pos = lambda t: t - n + EB
        ok = lambda t: 0 <= t < n_all and 0 <= pos(t) < nsh
        low, planes = {}, {}

        def lowy(t):
            if t not in low:
                low[t] = np.asarray(f[pos(t)][0])[::8, ::8]
            return low[t]

        def cut_at(t):
            return t > 0 and ok(t) and ok(t - 1) and float(np.mean(np.abs(lowy(t) - lowy(t - 1)))) > CUT

        def scene(t):
            if t in state["scene"]:
                return state["scene"][t]
            a = t
            while ok(a - 1) and not cut_at(a):
                a -= 1
            b = t
            while ok(b + 1) and not cut_at(b + 1):
                b += 1
            if ok(a - 1) or a == 0:                                # both ends found inside the view: cacheable
                if ok(b + 1) or b == n_all - 1:
                    state["scene"][t] = (a, b)
            return a, b

        def frame(t):
            if t not in planes:
                planes[t] = torch.from_numpy(np.stack([np.asarray(f[pos(t)][p]) for p in range(3)])).to(dev)[None]
            return planes[t]

        def flow(a, b):                                        # frame_b(x + fl) ~ frame_a(x), FieldNet outputs
            if a == b:
                return torch.zeros(1, 2, *frame(n).shape[-2:], device=dev)
            k = (a, b)
            if k not in state["fl"]:
                state["fl"][k] = ts.fc.est_flow(frame(a)[:, :1], frame(b)[:, :1], True)
            return state["fl"][k]

        def window(t, src):
            """seq [1,7,3,H,W] around t (clamped to t's scene) from src(frame index), and the flows the stage reads"""
            sa, sb = scene(t)
            ids = [min(max(t + j, sa), sb) for j in range(-R, R + 1)]
            seq = torch.cat([src(i) for i in ids])[None]
            H, W = seq.shape[-2:]
            fl = torch.zeros(1, 7, len(ts.OFFS), 2, H, W, device=dev)
            for oi, o in enumerate(ts.OFFS):
                fl[0, R, oi] = flow(t, ids[R + o])[0]
                fl[0, R + o, ts.OFFS.index(-o)] = flow(ids[R + o], t)[0]
            return seq, fl, ids

        def s1(t):
            if t not in state["s1"]:
                seq, fl, _ = window(t, frame)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    state["s1"][t] = net.s1(seq, fl, R, 1)[:, 0].float()
            return state["s1"][t]

        def s2(t):
            if t not in state["s2"]:
                seq, fl, _ = window(t, s1)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    state["s2"][t] = net.s2(seq, fl, R, 1)[:, 0].float()
            return state["s2"][t]

        def s3(t):
            seq, fl, ids = window(t, s2)
            mot = fl[:, R].norm(dim=2).amax(1, keepdim=True)
            mot = torch.nn.functional.max_pool2d(mot, 5, 1, 2)[:, None]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                return net.s3(seq, R, 1, mot)[:, 0].float()

        def prune(t):
            for d in (state["s1"], state["s2"], state["scene"]):
                for k in [k for k in d if k < t - 2 * EB or k > t + 2 * EB]:
                    del d[k]
            for k in [k for k in state["fl"] if min(k) < t - 2 * EB or max(k) > t + 2 * EB]:
                del state["fl"][k]

        cut_n = cut_at(n)
        with cond:
            running = state["last"] is not None and state["last"] in (n - 1, n - 2)
            if lock and running and state["last"] != n - 1 and not cut_n:
                cond.wait_for(lambda: state["last"] == n - 1, timeout=WAIT_S)
        with _LOCK, torch.no_grad():
            sa, sb = scene(n)
            o = s3(n)
            if lock:
                lk = state["lock"]
                if cut_n or n == sa or lk is None or state["last"] != n - 1:
                    lk = ts.StillLock(keep=lock_keep)
                    if not cut_n and n != sa:                   # seek: warm the memory up over the frames before n
                        for t in range(max(sa, n - WARM), n):
                            lk(s3(t))
                    state["lock"] = lk
                o = lk(o)
            prune(n)
        with cond:
            state["last"] = n
            cond.notify_all()
        dst = f[pos(n)].copy()
        oc = o[0].cpu().numpy()
        for p_ in range(3):
            np.copyto(np.asarray(dst[p_]), oc[p_])
        return dst

    out = core.std.ModifyFrame(full, clips=shifted, selector=proc)
    out = core.resize.Bicubic(out, format=clip.format.id, range_in=1, range=0)
    return core.std.CopyFrameProps(out, clip)
