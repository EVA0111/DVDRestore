"""Elliptical (EWA / polar) resampling on the GPU with PyTorch - the libplacebo / mpv filters without Vulkan.

Why: vs-placebo opens its own Vulkan device per filter, and on this system (RTX 5070 Ti, driver 621) only ONE can exist
per process. A second one (the preview keeps the previous graph alive, two resizes in a chain ...) fails in
vkCreateDevice and libplacebo then crashes the whole process (segfault, no Python error) - the backend died with
"Failed creating vulkan context". Same kernels, computed directly:
  ewa_lanczos       jinc windowed by jinc, radius 3.2383 (third zero of jinc)
  ewa_lanczossharp  the same, blur 0.98125 (slightly sharper)
  ewa_robidouxsharp polar Mitchell-Netravali cubic B 0.2620 / C 0.3690, radius 2
Every output pixel is the normalised weighted sum of the source pixels within the radius (distance scaled by the
downscale factor per axis, so downscaling is antialiased). Pixel centres aligned like zimg / libplacebo.
"""
import math

import numpy as np
import vapoursynth as vs

core = vs.core

R_JINC = 3.2383154841662362         # third zero of jinc (j1 zero / pi)
R1_JINC = 1.2196698912665045        # first zero
FILTERS = {
    "ewa_lanczos": ("jinc", R_JINC, 1.0),
    "ewa_lanczossharp": ("jinc", R_JINC, 0.9812505644269356),
    "ewa_robidouxsharp": ("cubic", 2.0, 1.0),
}
_B, _C = 0.2620145123990142, 0.3689927438004929


def _dev():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _kernel(r, kind, radius):
    import torch
    if kind == "jinc":
        def jinc(x):
            px = math.pi * x
            out = 2 * torch.special.bessel_j1(px) / px.clamp(min=1e-12)
            return torch.where(x.abs() < 1e-8, torch.ones_like(x), out)
        w = jinc(r) * jinc(r * (R1_JINC / radius))
    else:
        B, C = _B, _C
        a = r
        w = torch.where(a < 1, ((12 - 9 * B - 6 * C) * a ** 3 + (-18 + 12 * B + 6 * C) * a ** 2 + (6 - 2 * B)) / 6,
                        ((-B - 6 * C) * a ** 3 + (6 * B + 30 * C) * a ** 2 + (-12 * B - 48 * C) * a + (8 * B + 24 * C)) / 6)
    return torch.where(r < radius, w, torch.zeros_like(w))


def _axis(n_in, n_out, radius, blur, dev):
    """source taps per output position along one axis: indices (clamped), signed distances (scaled), scale"""
    import torch
    s = max(1.0, n_in / n_out) / blur                   # kernel stretch: downscale factor, blur
    c = (torch.arange(n_out, device=dev, dtype=torch.float64) + 0.5) * (n_in / n_out) - 0.5
    k = int(math.ceil(radius * s))
    base = torch.floor(c).long()
    off = torch.arange(-k + 1, k + 1, device=dev)
    idx = base[:, None] + off[None, :]
    d = (idx.double() - c[:, None]) / s
    return idx.clamp(0, n_in - 1), d.float()


def resample_planes(x, w_out, h_out, name, rows=64):
    """x: torch float [P, H, W] -> [P, h_out, w_out] (the weights are computed once for all planes)"""
    import torch
    kind, radius, blur = FILTERS[name]
    P_, H, W = x.shape
    ix, dx = _axis(W, w_out, radius, blur, x.device)
    iy, dy = _axis(H, h_out, radius, blur, x.device)
    out = torch.empty(P_, h_out, w_out, device=x.device, dtype=x.dtype)
    cols = x[:, :, ix]                                  # [P, H, w_out, Kx]
    for y0 in range(0, h_out, rows):
        y1 = min(h_out, y0 + rows)
        r = torch.sqrt(dy[y0:y1, None, :, None] ** 2 + dx[None, :, None, :] ** 2)   # [rows, w_out, Ky, Kx]
        wt = _kernel(r, kind, radius)
        wt = wt / wt.sum(dim=(2, 3), keepdim=True).clamp(min=1e-12)
        p = cols[:, iy[y0:y1]]                          # [P, rows, Ky, w_out, Kx]
        out[:, y0:y1] = (wt[None] * p.permute(0, 1, 3, 2, 4)).sum(dim=(3, 4))
    return out


def resample(clip, w, h, name="ewa_lanczossharp"):
    """VapourSynth clip (any planar format, 4:4:4 is what the app gives it) -> w x h, same format."""
    import torch
    if name not in FILTERS:
        raise ValueError(f"unknown EWA filter {name}")
    if clip.format.subsampling_w or clip.format.subsampling_h:
        raise ValueError("ewa.resample needs 4:4:4 or RGB")
    dev = _dev()
    fmt = clip.format
    integer = fmt.sample_type == vs.INTEGER
    peak = (1 << fmt.bits_per_sample) - 1
    blank = core.std.BlankClip(clip, width=w, height=h)

    def run(n, f):
        src = f[1]
        dst = f[0].copy()
        with torch.no_grad():
            arrs = [np.asarray(src[p]) for p in range(fmt.num_planes)]
            t = torch.from_numpy(np.stack(arrs).astype(np.float32)).to(dev)
            o = resample_planes(t, w, h, name)
            if integer:
                o = o.round_().clamp_(0, peak)
            o = o.cpu().numpy()
            for p in range(fmt.num_planes):
                np.copyto(np.asarray(dst[p]), o[p].astype(arrs[p].dtype))
        return dst
    out = core.std.ModifyFrame(blank, [blank, clip], run)
    return core.std.CopyFrameProps(out, clip)
