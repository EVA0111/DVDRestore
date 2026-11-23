"""FieldNet v3 core, shared by training (models/fieldnet3_train.py) and the app (fieldnet.py).

Recurrent with MOTION-ALIGNED memory. For every frame n the network gets
  frames   IVTC output n-2..n+2 (15 ch)
  prev     its own output for n-1, warped onto frame n along the motion (3 ch)
  hidden   its own 16-channel memory from n-1, warped the same way (16 ch)
  occ      where the warp cannot be trusted (new content, occlusion, scene start): 1 = no usable history (1 ch)
  dials    crawl rainbow mpeg halo chroma grain_keep (6 ch)
  fields   raw telecined frames k-3..k+3 split into their 14 fields -> field encoder (32 ch)
  match    TFM match one-hot p c n b u (5 ch)
  extra    PAL bob phase top / bottom, PAL flag (3 ch)
and returns the picture (3 ch, residual on frame n) and the new memory (16 ch, tanh).
Motion: hierarchical block matching on blurred luma of the INPUT frames (blur: dot crawl must not steer it), coarse
+-16 px at 1/4 size, refined +-2 px at full size, with a small bias towards zero motion (still areas stay exactly
still). The same code runs in training and in the app.
v4 (version buffer 4): wider (192 channels), a 7th dial 'detail' (0 = the clean picture an old notch / comb decoder
gives, 1 = the full detail of the signal, as a perfect decoder would: the band the old decoder cut out is recovered
from the raw fields), and SUB-PIXEL motion (parabolic refinement of the block match, lighter flow smoothing) so the
detail carried in the memory is not blurred a little at every frame.
v4 also has a CONTEXT branch (the head features at 1/2 and 1/4 size through their own blocks, added back before the
main blocks: a several times wider view for flat areas and long smears) and LEARNED ALIGNMENT (a deformable
convolution corrects the motion-warped memory per pixel where block matching is wrong: occlusions, rotation, zoom).
Both start with a zero output, so a v4 widened from v3 starts out exactly as v3."""
import torch
import torch.nn as nn
import torch.nn.functional as F

T = 5
NDIAL = 6
NRAW = 7
NFC = 32
NM = 5
NX = 3
NH = 16
NIN = 3 * T + 3 + NH + 1 + NDIAL + NFC + NM + NX


def n_in(ndial):
    return 3 * T + 3 + NH + 1 + ndial + NFC + NM + NX


def _blur(y, r):
    return F.avg_pool2d(F.pad(y, (r, r, r, r), mode="replicate"), 2 * r + 1, 1)


def _shift(b, dx, dy, pad):
    """b padded by pad -> view shifted by (dx, dy): out(x) = b(x + d)"""
    H, W = b.shape[-2] - 2 * pad, b.shape[-1] - 2 * pad
    return b[..., pad + dy:pad + dy + H, pad + dx:pad + dx + W]


def est_flow(cur, prv, subpix=False):
    """cur, prv: luma [B,1,H,W] (0..1). -> flow [B,2,H,W] (dx, dy, pixels) with prv(x + flow) ~ cur(x).
    subpix: refine to fractions of a pixel (parabola through the match costs around the best shift)."""
    B, _, H, W = cur.shape
    a, b = _blur(cur, 1), _blur(prv, 1)
    # coarse: 1/4 size, +-4 (= +-16 px)
    s = 4
    Hc, Wc = H // s, W // s
    ac = F.adaptive_avg_pool2d(a, (Hc, Wc)); bc = F.adaptive_avg_pool2d(b, (Hc, Wc))
    R = 4
    bp = F.pad(bc, (R, R, R, R), mode="replicate")
    best = torch.full((B, 1, Hc, Wc), 1e9, device=cur.device, dtype=cur.dtype)
    fx = torch.zeros_like(best); fy = torch.zeros_like(best)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            c = _blur((ac - _shift(bp, dx, dy, R)).abs(), 1) + 0.0015 * (abs(dx) + abs(dy))
            m = c < best
            best = torch.where(m, c, best); fx = torch.where(m, torch.full_like(fx, dx), fx); fy = torch.where(m, torch.full_like(fy, dy), fy)
    flow = torch.cat([fx, fy], 1) * s
    flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
    # refine at full size: +-2 around the coarse motion
    bw = warp(b, flow)
    R2 = 2
    bp = F.pad(bw, (R2, R2, R2, R2), mode="replicate")
    best = torch.full((B, 1, H, W), 1e9, device=cur.device, dtype=cur.dtype)
    rx = torch.zeros_like(best); ry = torch.zeros_like(best)
    costs = {}
    for dy in range(-R2, R2 + 1):
        for dx in range(-R2, R2 + 1):
            c = _blur((a - _shift(bp, dx, dy, R2)).abs(), 3) + 0.0008 * (abs(dx) + abs(dy))
            if subpix:
                costs[(dx, dy)] = c
            m = c < best
            best = torch.where(m, c, best); rx = torch.where(m, torch.full_like(rx, dx), rx); ry = torch.where(m, torch.full_like(ry, dy), ry)
    if subpix:                                              # parabola through the costs left / right and above / below
        cs = torch.stack([costs[(dx, dy)] for dy in range(-R2, R2 + 1) for dx in range(-R2, R2 + 1)], 0)   # [25, B,1,H,W]
        n = 2 * R2 + 1
        ix = (ry + R2) * n + (rx + R2)

        def at(ddx, ddy):
            j = ((ry + ddy).clamp(-R2, R2) + R2) * n + ((rx + ddx).clamp(-R2, R2) + R2)
            return cs.gather(0, j.long()[None])[0]
        c0 = cs.gather(0, ix.long()[None])[0]
        sx = (at(-1, 0) - at(1, 0)) / (2 * (at(-1, 0) - 2 * c0 + at(1, 0))).clamp(min=1e-6)
        sy = (at(0, -1) - at(0, 1)) / (2 * (at(0, -1) - 2 * c0 + at(0, 1))).clamp(min=1e-6)
        rx = rx + sx.clamp(-0.5, 0.5); ry = ry + sy.clamp(-0.5, 0.5)
    flow = flow + torch.cat([rx, ry], 1)
    return _blur(flow, 1 if subpix else 2)


def warp(x, flow):
    """x [B,C,H,W] sampled at (pixel + flow), bilinear, edges clamped"""
    B, _, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device, dtype=x.dtype), torch.arange(W, device=x.device, dtype=x.dtype), indexing="ij")
    gx = (xx[None] + flow[:, 0]) * (2 / max(W - 1, 1)) - 1
    gy = (yy[None] + flow[:, 1]) * (2 / max(H - 1, 1)) - 1
    return F.grid_sample(x, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=True)


def align_memory(cur_y, prv_y, prev_out, prev_hid, subpix=False):
    """Motion-aligned history for frame n: (prev_out warped, hidden warped, occlusion map). cur_y / prv_y: luma of
    the input frames n and n-1. prev_out None = no history (scene start / seek)."""
    B, _, H, W = cur_y.shape
    if prev_out is None:
        return None, torch.zeros(B, NH, H, W, device=cur_y.device, dtype=cur_y.dtype), torch.ones(B, 1, H, W, device=cur_y.device, dtype=cur_y.dtype)
    fl = est_flow(cur_y, prv_y, subpix)
    pw, hw = warp(prev_out, fl), warp(prev_hid, fl)
    err = (_blur(cur_y, 2) - _blur(warp(prv_y, fl), 2)).abs()
    occ = (err * 30).clamp(0, 1)
    return pw, hw, occ


class Block(nn.Module):
    def __init__(self, ch, d):
        super().__init__()
        self.a = nn.Conv2d(ch, ch, 3, padding=d, dilation=d)
        self.b = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.b(torch.relu(self.a(torch.relu(x))))


class FieldEnc(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.a = nn.Conv2d(NRAW * 6, ch, 3, padding=1)
        self.b = nn.Conv2d(ch, ch, 3, padding=2, dilation=2)
        self.c = nn.Conv2d(ch, ch, 3, padding=1)
        self.o = nn.Conv2d(ch, 2 * NFC, 3, padding=1)

    def forward(self, f):
        h = torch.relu(self.a(f))
        h = h + torch.relu(self.c(torch.relu(self.b(h))))
        o = self.o(h)
        B, _, hh, w = o.shape
        return o.view(B, 2, NFC, hh, w).permute(0, 2, 3, 1, 4).reshape(B, NFC, 2 * hh, w)


class Context(nn.Module):
    """features at 1/2 and 1/4 size, through their own blocks, back to full size (zero at the start)"""
    def __init__(self, ch):
        super().__init__()
        self.d2 = nn.Sequential(Block(ch, 1), Block(ch, 2))
        self.d4 = nn.Sequential(Block(ch, 1), Block(ch, 2), Block(ch, 4))
        self.out = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, h):
        H, W = h.shape[-2:]
        h2 = self.d2(F.avg_pool2d(h, 2, ceil_mode=True))
        h4 = self.d4(F.avg_pool2d(h2, 2, ceil_mode=True))
        h2 = h2 + F.interpolate(h4, size=h2.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(F.interpolate(h2, size=(H, W), mode="bilinear", align_corners=False))


class DAlign(nn.Module):
    """learned correction of the motion-warped memory (previous output + hidden state): offsets and a modulation mask
    from the memory and the current frame, then a deformable 3x3 convolution of the memory, added as a residual
    (zero at the start: the plain warp)"""
    def __init__(self, cm=3 + NH, ch=64, groups=1):
        super().__init__()
        self.g = groups
        self.feat = nn.Sequential(nn.Conv2d(cm + 3 + 1, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU())
        self.off = nn.Conv2d(ch, 3 * 9 * groups, 3, padding=1)
        nn.init.zeros_(self.off.weight); nn.init.zeros_(self.off.bias)
        self.weight = nn.Parameter(torch.zeros(cm, cm // groups, 3, 3))
        self.bias = nn.Parameter(torch.zeros(cm))

    def forward(self, mem, mid, occ):
        from torchvision.ops import deform_conv2d
        f = self.feat(torch.cat([mem, mid, occ], 1))
        o = self.off(f)
        k = 2 * 9 * self.g
        off, msk = 4.0 * torch.tanh(o[:, :k]), torch.sigmoid(o[:, k:])          # offsets within +-4 px of the warp
        with torch.autocast("cuda", enabled=False):
            d = deform_conv2d(mem.float(), off.float(), self.weight.float(), self.bias.float(), padding=1, mask=msk.float())
        return mem + d


class FieldNet3(nn.Module):
    VERSION = 3

    def __init__(self, ch=128, dil=(1, 2, 4, 8, 4, 2, 1, 1), ndial=NDIAL, version=3):
        super().__init__()
        self.ndial = ndial
        self.subpix = version >= 4
        self.enc = FieldEnc()
        self.head = nn.Conv2d(n_in(ndial), ch, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(ch, d) for d in dil])
        self.tail = nn.Conv2d(ch, 3 + NH, 3, padding=1)
        self.ctx = Context(ch) if version >= 4 else None
        self.dalign = DAlign() if version >= 4 else None
        self.register_buffer("version", torch.tensor(version))
        # 1 = for RAW IVTC input (field matching + decimation only, no comb repair / post-processing / safety net):
        # the model does every repair itself from the raw fields; 0 = repaired IVTC input
        self.register_buffer("raw_ivtc", torch.tensor(1))
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)
        with torch.no_grad():                                     # memory channels start small, not dead
            self.tail.weight[3:].normal_(0, 1e-3)

    def forward(self, frames, prevw, hidw, occ, dials, fields, match, extra, hadd=None):
        """hadd: extra head features added before the main blocks (v5: the backward pass of the same frame)"""
        c = 3 * (T // 2)
        sp = frames.shape[-2:]
        ex = lambda v: v[:, :, None, None].expand(-1, -1, *sp).float()
        mid = frames[:, c:c + 3]
        if prevw is None:
            prevw = mid
        if self.dalign is not None:                          # v4: learned correction of the warped memory
            mem = self.dalign(torch.cat([prevw.float(), hidw.float()], 1), mid.float(), occ.float())
            prevw, hidw = mem[:, :3], mem[:, 3:]
        with torch.autocast("cuda", enabled=False):
            fe = self.enc(fields.float())
            h = self.head(torch.cat([frames.float(), prevw.float(), hidw.float(), occ.float(), ex(dials), fe, ex(match), ex(extra)], 1))
        if hadd is not None:
            h = h + hadd
        if self.ctx is not None:                             # v4: wider view (1/2, 1/4 size)
            h = h + self.ctx(h)
        o = self.tail(torch.relu(self.blocks(h)))
        return mid + o[:, :3], torch.tanh(o[:, 3:])


def build(sd):
    """the right FieldNet for a checkpoint (v3: 128 ch, 6 dials, integer motion; v4: wider, 7 dials, sub-pixel)"""
    ch = sd["head.weight"].shape[0]
    ndial = sd["head.weight"].shape[1] - (n_in(0))
    return FieldNet3(ch=ch, ndial=ndial, version=int(sd.get("version", torch.tensor(3))))


def widen_from(net, sd):
    """v4 from a v3 checkpoint: the v3 network is copied into the first channels (and dials) of the wider one, the
    new weights start tiny, so step 0 behaves like v3 and every new channel can still learn"""
    own = net.state_dict()
    o3 = 3 * T + 3 + NH + 1                                  # head input: frames, prev, memory, occlusion | dials | rest
    for k, v in sd.items():
        if k not in own or k in ("version", "raw_ivtc"):
            continue
        w = own[k].clone()
        if v.shape == w.shape:
            own[k] = v
            continue
        with torch.no_grad():
            if w.dim() > 1:
                w.normal_(0, 1e-3)
            else:
                w.zero_()
            if k == "head.weight":
                nd_old = v.shape[1] - n_in(0)
                w[:v.shape[0], :o3] = v[:, :o3]
                w[:v.shape[0], o3:o3 + nd_old] = v[:, o3:o3 + nd_old]
                w[:v.shape[0], o3 + net.ndial:] = v[:, o3 + nd_old:]
            elif w.dim() == 4:
                w[:v.shape[0], :v.shape[1]] = v
            else:
                w[:v.shape[0]] = v
        own[k] = w
    net.load_state_dict(own)


def init_from(net, sd):
    """from FieldNet v1 / UniNet / StableNet (or v2) weights: IVTC window, previous output and dial head channels,
    blocks, tail picture channels (and the v2 field encoder) are copied; everything new starts at zero."""
    own = net.state_dict()
    for k, v in sd.items():
        if k == "head.weight":
            w = own[k].clone(); w.zero_()
            w[:, :3 * T + 3] = v[:, :3 * T + 3]                                     # frames + prev
            if v.shape[1] >= 3 * T + 3 + NDIAL:
                o = 3 * T + 3 + NH + 1
                w[:, o:o + NDIAL] = v[:, 3 * T + 3:3 * T + 3 + NDIAL]               # dials
            with torch.no_grad():
                w[:, 3 * T + 3:3 * T + 3 + NH].normal_(0, 1e-3)                      # memory in: small
            own[k] = w
        elif k == "tail.weight":
            w = own[k].clone(); w[:3] = v[:3]; own[k] = w
        elif k == "tail.bias":
            b = own[k].clone(); b[:3] = v[:3]; own[k] = b
        elif k in own and own[k].shape == v.shape:
            own[k] = v
    net.load_state_dict(own)


def split_fields(Y, UV):
    """Y [..., h, w], UV [..., 2, h/2, w/2] (interlaced 4:2:0) -> [..., 6, h/2, w]: Y U V top field, Y U V bottom"""
    sh = Y.shape[:-2]
    h, w = Y.shape[-2:]
    n = 1
    for s in sh:
        n *= s
    out = []
    for par in (0, 1):
        y = Y[..., par::2, :].reshape(n, 1, h // 2, w)
        c = UV[..., par::2, :].reshape(n, 2, h // 4, w // 2)
        out += [y, F.interpolate(c, size=(h // 2, w), mode="bilinear", align_corners=False)]
    return torch.cat(out, 1).reshape(*sh, 6, h // 2, w)
