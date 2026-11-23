"""FieldNet v3 in the app: the all-in-one recurrent model with the IVTC inside it and motion-aligned memory
(models/fieldnet_core.py, trained by models/fieldnet3_train.py).

Per output frame n it reads the IVTC (or PAL processing) frames n-2..n+2, the raw telecined frames k-3..k+3 around its
source frame k (_SrcN), the field match (TFMMatch), the PAL bob field (_Phase) and the six dials, plus its own output
and memory of frame n-1 warped along the motion.
Steadiness where the old models flickered:
  scene cuts   a cut is detected from the frames themselves; no frame of the other scene enters the 5-frame window
               or the raw window, and the memory of the new scene is warmed up BACKWARDS over the next frames of the
               same scene (the model is trained on time-reversed runs), so the first frame after a cut is not decided
               cold
  seeks        a frame with no history (paused preview, jump) is warmed up forward over the frames before it, from
               the last cut, so it matches what playback shows
  motion       history follows pans and moving patterns (warped), new content is marked and decided fresh
Frames are computed in order (frame n waits briefly for n-1), like stablenet.py.
Input / output: YUV, limited range, any subsampling; returns the clip's format."""
import os
import sys
import threading

import numpy as np
import vapoursynth as vs
import paths  # noqa: E402  (core/paths.py: every folder in one place)

core = vs.core
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = paths.model("fieldnet.pt")
T, K, RR = 5, 6, 3                 # IVTC window, warm-up length, raw frames each side
WAIT_S = 20.0
CUT = 0.08                         # mean luma change (0..1, 1/8 size) that counts as a scene cut
SWAP_PN = [2, 1, 0, 3, 4]
DIALS = ("crawl", "rainbow", "mpeg", "halo", "chroma", "grain_keep", "detail")    # 'detail': v4 only
STATS = {"frames": 0, "cold": 0, "cut_warmups": 0, "seek_warmups": 0}
_NET = None
_LOCK = threading.Lock()


def available(model=None):
    return os.path.isfile(model or MODEL)


_RAW = {}


def wants_raw_ivtc(model):
    """True when the weights were trained on raw IVTC input (no comb repair before the model)"""
    model = model or MODEL
    if model not in _RAW:
        import torch
        try:
            sd = torch.load(model, map_location="cpu", weights_only=True)
            # every FieldNet v3 was trained on (practically) unrepaired IVTC frames: the simulated clips almost never
            # trigger a comb repair, while on real discs it fires on fine patterns and costs detail (Courage's door: a
            # smudged frame every 4; raw IVTC: flicker 1.35 -> 1.20, no smudge). So raw IVTC unless a checkpoint says 0.
            _RAW[model] = bool(int(sd.get("raw_ivtc", torch.tensor(1))))
        except Exception:                                 # noqa: BLE001
            _RAW[model] = False
    return _RAW[model]


def is_v3(model):
    import torch
    return "version" in torch.load(model, map_location="cpu", weights_only=True)


def _net(path):
    global _NET
    if _NET is None or _NET[0] != path:
        import torch
        sys.path.insert(0, paths.MODELS)
        import fieldnet_core as fc
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        net = fc.build(sd)                                  # v3 (6 dials) or v4 (7 dials, wider, sub-pixel motion)
        net.load_state_dict(sd, strict=False)
        net = net.to(dev).eval()
        if dev.type == "cuda":
            if not os.environ.get("FIELDNET_NO_CL"):
                net = net.to(memory_format=torch.channels_last)  # ~18 % faster convolutions (measured, full frames)
            _fast_flow(fc)
        _NET = (path, net, dev, fc)
    return _NET[1:]


def _fast_flow(fc):
    """motion estimation compiled with Triton (triton-windows) when available: ~8x faster, same results; falls back
    to the plain version if the compiler is missing or fails"""
    if getattr(fc, "_fast", False):
        return
    fc._fast = True
    if os.environ.get("FIELDNET_NO_COMPILE"):
        return
    try:
        import torch
        import triton                                       # noqa: F401
        plain = fc.est_flow
        comp = torch.compile(plain, dynamic=False)
        state = {"ok": True}

        def est_flow(cur, prv, subpix=False):
            if state["ok"]:
                try:
                    return comp(cur, prv, subpix)
                except Exception as e:                      # noqa: BLE001
                    state["ok"] = False
                    print(f"FieldNet: compiled motion estimation failed ({e}); using the plain one", file=sys.stderr)
            return plain(cur, prv, subpix)
        fc.est_flow = est_flow
    except Exception:                                       # noqa: BLE001
        pass


def fieldnet(clip, model=None, dials=None, raw=None, raw_start=0, pal=False):
    import torch
    model = model or MODEL
    dial_vec = [float((dials or {}).get(k, 0.0 if k == "grain_keep" else 1.0)) for k in DIALS]     # trimmed per net
    full = core.resize.Bicubic(clip, format=vs.YUV444PS, range_in=0, range=1)
    n_all = full.num_frames
    r = T // 2
    E = K + r                                                 # IVTC frames n-E .. n+E

    def at(c, k, n_):
        if k < 0:
            return c[0] * (-k) + c[:n_ + k]
        if k > 0:
            return c[k:] + c[-1] * k
        return c
    shifted = [at(full, k, n_all) for k in range(-E, E + 1)]
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
        shifted += [at(rv, k, n_all) for k in range(-K, K + 1)]   # raw windows of frames n-K .. n+K
    state = {"out": {}, "busy": set()}
    gpu = {}                                                 # frame number -> its planes on the GPU (reused by windows)
    cond = threading.Condition()

    def proc(n, f):
        net, dev, fc = _net(model)
        F_, R_ = f[:nsh], f[nsh:]
        pos = lambda t: t - n + E                                # list position of IVTC frame t
        ok = lambda t: 0 <= t < n_all and 0 <= pos(t) < nsh
        low = {}

        def lowy(t):
            if t not in low:
                low[t] = np.asarray(F_[pos(t)][0])[::8, ::8]
            return low[t]

        def cut_at(t):                                           # t starts a new scene
            return t > 0 and ok(t) and ok(t - 1) and float(np.mean(np.abs(lowy(t) - lowy(t - 1)))) > CUT

        def scene(t):                                            # [first, last] frame of t's scene inside the view
            a = t
            while ok(a - 1) and not cut_at(a):
                a -= 1
            b = t
            while ok(b + 1) and not cut_at(b + 1):
                b += 1
            return a, b

        def kof(t):
            return int(F_[pos(t)].props.get("_SrcN", raw_start + t)) - raw_start

        def plane3(fr, t=None):
            if t is not None and t in gpu:
                return gpu[t]
            x = torch.from_numpy(np.stack([np.asarray(fr[p]) for p in range(3)])).to(dev, non_blocking=True)
            if t is not None:
                gpu[t] = x
                for k_ in [k_ for k_ in gpu if k_ < t - 2 * E or k_ > t + 2 * E]:
                    del gpu[k_]
            return x

        def inputs(t, rev, sa, sb):
            """window, fields, match one-hot, extra for frame t; rev = time reversed; sa..sb = its scene"""
            idx = [min(max(t + j, sa), sb) for j in range(-r, r + 1)]
            if rev:
                idx = idx[::-1]
            win = torch.cat([plane3(F_[pos(i)], i) for i in idx])[None]
            props = F_[pos(t)].props
            m = int(props.get("TFMMatch", 1)) if have_raw else 1
            if rev:
                m = SWAP_PN[m] if 0 <= m < 5 else m
            mt = torch.tensor([[float(m == i) for i in range(5)]], device=dev)
            ph = int(props["_Phase"]) if "_Phase" in props else -1
            ex = torch.tensor([[float(ph == 0), float(ph == 1), float(pal)]], device=dev)
            if have_raw:
                fr = R_[t - n + K]
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
                fl = fc.split_fields(Yt[slots], UVt[slots])                        # [7, 6, hh/2, ww]
                fw = fl.reshape(1, nf * 6, hh // 2, ww)
            else:
                xf = win.float().reshape(T, 3, *win.shape[-2:])
                xf = torch.cat([xf[:, :, 0::2], xf[:, :, 1::2]], 1)
                xf = torch.cat([xf[:1], xf, xf[-1:]], 0)
                fw = xf.reshape(1, 7 * 6, *xf.shape[-2:])
            return win, fw, mt, ex

        dv = torch.tensor([dial_vec[:net.ndial]], device=dev)

        def step(t, st, rev, sa, sb):
            """one recurrent step on frame t from state st = (out, hid, y) of its neighbour in run order (or None)"""
            win, fw, mt, ex = inputs(t, rev, sa, sb)
            mid = win[:, 3 * r:3 * r + 3].float()
            cy = mid[:, :1]
            if st is None:
                pw, hw, occ = fc.align_memory(cy, None, None, None)
                pw = mid
            else:
                pw, hw, occ = fc.align_memory(cy, st[2], st[0].float(), st[1].float(), net.subpix)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o, h = net(win, pw, hw, occ, dv, fw, mt, ex)
            return o.float(), h.float(), cy

        def scene_start_state(c, sa, sb):
            """memory for the first frame c of a scene: built backwards over the next frames of the same scene"""
            st = None
            for t in range(min(sb, c + K), c, -1):
                st = step(t, st, True, sa, sb)
            return st

        cut_n = cut_at(n)
        with cond:
            state["busy"].add(n)
            running = any(k in state["out"] or k in state["busy"] for k in (n - 1, n - 2))
            if running and n - 1 not in state["out"] and not cut_n:
                cond.wait_for(lambda: n - 1 in state["out"], timeout=WAIT_S)
            prev = None if cut_n else state["out"].get(n - 1)
            STATS["frames"] += 1
        with _LOCK, torch.no_grad():
            sa, sb = scene(n)
            if prev is not None:
                o, h, cy = step(n, prev, False, sa, sb)
            elif n == sa:
                # first frame of a scene: warm the memory up backwards over the next frames of the same scene
                st = scene_start_state(n, sa, sb)
                STATS["cut_warmups" if min(sb, n + K) > n else "cold"] += 1
                o, h, cy = step(n, st, False, sa, sb)
            else:
                # no history (seek / paused preview): warm up forward from the last cut or K frames back; when that
                # reaches the scene's first frame, start it exactly as playback does (backward warm-up first), so a
                # paused or jumped-to frame shows what playback shows
                t0 = max(sa, n - K)
                st = None
                if t0 == sa:
                    st = step(sa, scene_start_state(sa, sa, sb), False, sa, sb)
                    t0 = sa + 1
                for t in range(t0, n):
                    st = step(t, st, False, sa, sb)
                STATS["seek_warmups"] += 1
                o, h, cy = step(n, st, False, sa, sb)
        with cond:
            state["out"][n] = (o, h, cy)
            for k in [k for k in state["out"] if k < n - 4]:
                del state["out"][k]
            state["busy"].discard(n)
            cond.notify_all()
        dst = F_[pos(n)].copy()
        oc = o[0].cpu().numpy()
        for p_ in range(3):
            np.copyto(np.asarray(dst[p_]), oc[p_])
        return dst

    out = core.std.ModifyFrame(full, clips=shifted, selector=proc)
    out = core.resize.Bicubic(out, format=clip.format.id, range_in=1, range=0)
    return core.std.CopyFrameProps(out, clip)
