"""LUTs for the Colour step: .cube files (Resolve / Adobe format), 3D (trilinear) and 1D.
Applied on the GPU with PyTorch when there is one (grid_sample = trilinear), else with numpy.
The LUT expects the picture's RGB 0..1 (DOMAIN_MIN / DOMAIN_MAX respected); `mix` blends it with the input."""
import os
import threading

import numpy as np
import vapoursynth as vs

core = vs.core
_CACHE = {}
_LOCK = threading.Lock()


def parse_cube(path):
    """-> dict(kind="3d"|"1d", size, table, dmin, dmax, title). 3D table: [b, g, r, 3] (red changes fastest)."""
    st = os.stat(path)
    key = (path, st.st_mtime, st.st_size)
    if key in _CACHE:
        return _CACHE[key]
    size3 = size1 = None
    dmin, dmax, title, rows = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], "", []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            k = t.split()
            u = k[0].upper()
            if u == "TITLE":
                title = t[5:].strip().strip('"')
            elif u == "LUT_3D_SIZE":
                size3 = int(k[1])
            elif u == "LUT_1D_SIZE":
                size1 = int(k[1])
            elif u == "DOMAIN_MIN":
                dmin = [float(v) for v in k[1:4]]
            elif u == "DOMAIN_MAX":
                dmax = [float(v) for v in k[1:4]]
            elif u[0].isdigit() or u[0] in "-.+":
                rows.append([float(v) for v in k[:3]])
    a = np.asarray(rows, dtype=np.float32)
    if size3:
        if len(a) != size3 ** 3:
            raise ValueError(f"{os.path.basename(path)}: {len(a)} entries, expected {size3 ** 3}")
        res = dict(kind="3d", size=size3, table=a.reshape(size3, size3, size3, 3))
    elif size1:
        if len(a) != size1:
            raise ValueError(f"{os.path.basename(path)}: {len(a)} entries, expected {size1}")
        res = dict(kind="1d", size=size1, table=a)
    else:
        raise ValueError(f"{os.path.basename(path)}: no LUT_3D_SIZE / LUT_1D_SIZE")
    res.update(dmin=np.array(dmin, np.float32), dmax=np.array(dmax, np.float32), title=title or os.path.basename(path))
    _CACHE[key] = res
    return res


def info(path):
    L = parse_cube(path)
    return {"kind": L["kind"], "size": L["size"], "title": L["title"]}


def apply(rgb, path, mix=1.0):
    """rgb: RGBS clip (0..1). Returns RGBS."""
    L = parse_cube(path)
    mix = float(max(0.0, min(1.0, mix)))
    if mix <= 0:
        return rgb
    try:
        import torch
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    except Exception:                                    # noqa: BLE001
        torch, dev = None, None
    if torch is not None and L["kind"] == "3d":
        vol = torch.from_numpy(np.ascontiguousarray(L["table"].transpose(3, 0, 1, 2)))[None].to(dev)   # [1,3,B,G,R]
    dmin, dmax = L["dmin"], L["dmax"]

    def proc(n, f):
        x = np.stack([np.asarray(f[p]) for p in range(3)])                 # [3,H,W]
        H, W = x.shape[1:]
        xn = np.clip((x - dmin[:, None, None]) / np.maximum(dmax - dmin, 1e-6)[:, None, None], 0, 1)
        if L["kind"] == "1d":
            N = L["size"]
            out = np.stack([np.interp(xn[c], np.linspace(0, 1, N), L["table"][:, c]) for c in range(3)]).astype(np.float32)
        elif torch is not None:
            with _LOCK, torch.no_grad():
                t = torch.from_numpy(xn).to(dev)
                grid = (t.permute(1, 2, 0)[None, None] * 2 - 1)                 # [1,1,H,W,3] = (r, g, b) -> (x, y, z)
                o = torch.nn.functional.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)
                out = o[0, :, 0].cpu().numpy()
        else:                                                                # numpy trilinear
            N = L["size"]; T = L["table"]
            p = xn * (N - 1); i0 = np.floor(p).astype(np.int64).clip(0, N - 2); fr = p - i0
            r0, g0, b0 = i0; fr_, fg, fb = fr
            out = np.zeros((H, W, 3), np.float32)
            for db in (0, 1):
                for dg in (0, 1):
                    for dr in (0, 1):
                        w = (fb if db else 1 - fb) * (fg if dg else 1 - fg) * (fr_ if dr else 1 - fr_)
                        out += w[..., None] * T[b0 + db, g0 + dg, r0 + dr]
            out = out.transpose(2, 0, 1)
        res = x + mix * (out - x) if mix < 1 else out
        dst = f.copy()
        for p in range(3):
            np.copyto(np.asarray(dst[p]), res[p].astype(np.float32))
        return dst

    return core.std.ModifyFrame(rgb, rgb, proc)
