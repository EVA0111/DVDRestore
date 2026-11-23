"""TStab v2: the learned temporal stabilizer that runs AFTER FieldNet, shared by training (models/tstab_train.py) and
the app (tstab.py).

Goal: zero boiling / shimmer / wobble / flicker without inventing anything: every stage outputs a weighted average of
real picture content (no hallucinated texture, real colours).

A CASCADE of three trainable stages, each initialised as an exact copy of a filter rule that measured well on the real
Courage disc after FieldNet v4 (mc + gtfix + ttemp: door 1.12 -> 0.20, rug 1.03 -> 0.19, letters shimmer 0.67 -> 0.11,
sharpness unchanged), so training starts from the earlier best and can only improve it:
  1 motion   (mc-like)    frames n-3..n+3 aligned onto n (block matching + learned refinement), per-donor confidence
                          from alignment cues, separate trust for coarse picture and fine detail, conservative average
  2 detail   (gtfix-like) a second aligned pass on stage 1's output, fine band only, weights from the coarse mismatch
                          after the warp (gtfix unguided: w = (1 - max(lowpassed mismatch / t1, local |mismatch| / t2))^2)
  3 still    (ttemp-like) unaligned, threshold-gated: a neighbour counts while it stays within the threshold of the
                          centre (and stops at the first one that does not, per direction)
Stacking reaches ~+-9 frames, the reason the filter chain beat any single pass. Then StillLock (deterministic long
memory for still areas) in the app.
Ideas taken from:
  - vs_temporalfix (pifroggi): pairwise alignment of 6 neighbours to the centre with learned flow refinement, a learned
    per-pixel confidence, the "conservative temporal average" (centre + gated confident supports)
  - RDUAL (Rift73): alignment cues per donor (photometric error after the warp, forward-backward flow inconsistency,
    out-of-frame sampling), fusion that starts as a known state, no donors across shot cuts
Every learned part is a residual on its stage's rule in logit space with a zero-initialised last layer.
Frames: [B,3,H,W] YUV, Y 0..1, UV centred (FieldNet's output format). Flow: fieldnet_core.est_flow on luma."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import fieldnet_core as fc
except ImportError:                                       # imported from the app root
    from models import fieldnet_core as fc

OFFS = (-3, -2, -1, 1, 2, 3)
R = 3                                                     # per stage, each side
NSTAGE_ALIGNED = 2
REACH = 3 * R                                             # frames each side the whole cascade reads
NCUE = 6                                                  # pe_lo, pe_hi, inconsistency, out-of-frame, |flow|, local |mismatch|


def _gauss(s, dev, dt):
    r = int(math.ceil(3 * s))
    x = torch.arange(-r, r + 1, device=dev, dtype=dt)
    k = torch.exp(-x * x / (2 * s * s)); return k / k.sum(), r


def blur(x, s=1.5):
    """separable Gaussian, reflect padding, per channel"""
    k, r = _gauss(s, x.device, x.dtype)
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)


def box(x, r):
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), 2 * r + 1, 1)


def bands(x, s=1.5):
    lo = blur(x, s)
    return lo, x - lo


def out_of_frame(fl):
    B, _, H, W = fl.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=fl.device, dtype=fl.dtype), torch.arange(W, device=fl.device, dtype=fl.dtype), indexing="ij")
    px, py = xx[None] + fl[:, 0], yy[None] + fl[:, 1]
    return ((px < 0) | (px > W - 1) | (py < 0) | (py > H - 1)).to(fl.dtype)[:, None]


def _chw(x):                                               # one mismatch map from Y + half of each colour plane
    return x[:, :1] + 0.5 * (x[:, 1:2] + x[:, 2:3])


def cues(center, warped, fl, rfl):
    """alignment cues of one donor on the centre grid [N,NCUE,H,W]. pe_*: levels / 4; inc, |flow|/8: pixels;
    lmis: gtfix's local mean |mismatch| of the coarse picture, levels"""
    lc, hc = bands(center); lw, hw = bands(warped)
    pe_lo = (lw - lc).abs().mean(1, keepdim=True) * 255 / 4
    pe_hi = (hw - hc).abs().mean(1, keepdim=True) * 255 / 4
    inc = (fl + fc.warp(rfl, fl)).norm(dim=1, keepdim=True).clamp(max=8)
    lmis = box(_chw((box(box(warped, 2), 2) - box(box(center, 2), 2)).abs()), 2) * 255
    return torch.cat([pe_lo, pe_hi, inc, out_of_frame(fl), fl.norm(dim=1, keepdim=True).clamp(max=32) / 8, lmis / 4], 1)


def run_flows(seq, subpix=True, flow_fn=None):
    """seq [B,L,3,H,W] -> fl [B,L,6,2,H,W]: flow from frame t to frame t+o (frame_{t+o}(x + fl) ~ frame_t(x)) for
    every t and o in OFFS (clamped at the run ends: a repeated frame gets its own exact flow 0).
    flow_fn(cur, prv) on full [N,3,H,W] frames replaces block matching (v5: NeuFlow v2)"""
    B, L = seq.shape[:2]
    y = seq.float() if flow_fn is not None else seq[:, :, :1].float()
    out = torch.zeros(B, L, len(OFFS), 2, *seq.shape[-2:], device=seq.device)
    for oi, o in enumerate(OFFS):
        ts_ = [t for t in range(L) if 0 <= t + o < L]
        if not ts_:
            continue
        c = y[:, ts_].flatten(0, 1); d = y[:, [t + o for t in ts_]].flatten(0, 1)
        f = flow_fn(c, d) if flow_fn is not None else fc.est_flow(c, d, subpix)
        out[:, ts_, oi] = f.view(B, len(ts_), 2, *seq.shape[-2:])
    return out


def _gather(seq, fl, t0, n):
    """windows for centres t0..t0+n-1 of seq: donors [B,n,6,C,H,W], flows / reverse flows [B,n,6,2,H,W], valid [B,n,6]"""
    B, L = seq.shape[:2]
    idx = torch.tensor([[min(max(t + o, 0), L - 1) for o in OFFS] for t in range(t0, t0 + n)], device=seq.device)
    don = seq[:, idx]                                                                     # [B,n,6,C,H,W]
    f = fl[:, t0:t0 + n]
    inv = [OFFS.index(-o) for o in OFFS]
    r = torch.stack([fl[:, idx[:, oi], inv[oi]] for oi in range(len(OFFS))], 2)            # flow back from each donor
    valid = torch.tensor([[0 <= t + o < L for o in OFFS] for t in range(t0, t0 + n)], device=seq.device, dtype=seq.dtype)
    return don, f, r, valid[None].expand(B, -1, -1)


class FlowRefine(nn.Module):
    """learned correction of the block-matching flow at 1/2 size (zero at the start), +-2 px (vs_temporalfix-style)"""
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3 + 3 + NCUE + 2 + 1, ch, 3, padding=1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.LeakyReLU(0.2),
                                 nn.Conv2d(ch, ch, 3, padding=1), nn.LeakyReLU(0.2), nn.Conv2d(ch, 2, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, center, warped, cue, fl, dt):
        H, W = center.shape[-2:]
        x = F.avg_pool2d(torch.cat([center, warped, cue, fl / 8, dt], 1), 2, ceil_mode=True)
        return fl + F.interpolate(2.0 * torch.tanh(self.net(x)), size=(H, W), mode="bilinear", align_corners=False)


class ConfNet(nn.Module):
    """learned logit residual per donor and band (zero at the start); sees centre, donor, cues, flow, time distance,
    consensus (mean aligned donor minus centre: is the centre itself the outlier?)"""
    def __init__(self, cin, ch=48):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, ch, 3, padding=1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.LeakyReLU(0.2),
                                 nn.Conv2d(ch, ch, 3, padding=1), nn.LeakyReLU(0.2), nn.Conv2d(ch, 2, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def _logit(p):
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p) - torch.log1p(-p)


class AlignedStage(nn.Module):
    """one motion-aligned averaging pass. rule 'mc': MDegrain-like prior on both bands + vs_temporalfix's gate;
    rule 'gtfix': gtfix-unguided prior on the fine band only (coarse band untouched), plain weights"""
    def __init__(self, rule, refine=True):
        super().__init__()
        self.rule = rule
        self.refine = FlowRefine() if refine else None
        self.conf = ConfNet(3 + 3 + NCUE + 2 + 1 + 3)
        if rule == "mc":
            # prior logit per band (lo, hi) = a - b_lo * pe_lo - b_hi * pe_hi - c * inc - 12 * oof
            self.a = nn.Parameter(torch.tensor([2.0, 3.0]))
            self.b_lo = nn.Parameter(torch.tensor([1.2, 0.6]))
            self.b_hi = nn.Parameter(torch.tensor([0.3, 0.8]))
            self.c = nn.Parameter(torch.tensor([1.0, 0.7]))
        else:
            # gtfix strong (strength 1.6): t1 = 1.2 * 1.6 levels (low-passed mismatch), t2 = 7 * 1.6 (local |mismatch|),
            # rguard = 4 * 1.6 levels (local |compensated fine - fine|)
            self.lt1 = nn.Parameter(torch.tensor(math.log(1.2 * 1.6)))
            self.lt2 = nn.Parameter(torch.tensor(math.log(7.0 * 1.6)))
            self.lt3 = nn.Parameter(torch.tensor(math.log(4.0 * 1.6)))

    def prior(self, cue, cen, wp):
        pe_lo, pe_hi, inc, oof, lmis = cue[:, 0:1], cue[:, 1:2], cue[:, 2:3], cue[:, 3:4], cue[:, 5:6] * 4
        if self.rule == "mc":
            v = lambda p: p.view(1, 2, 1, 1)
            return v(self.a) - v(self.b_lo) * pe_lo - v(self.b_hi) * pe_hi - v(self.c) * inc - 12 * oof   # logits
        lp = _chw((box(box(wp - cen, 2), 2)).abs()) * 255            # low-passed mismatch of the compensated picture
        w = (1 - torch.maximum(lp / self.lt1.exp(), lmis / self.lt2.exp())).clamp(min=0) ** 2
        rg = box((bands(wp)[1][:, :1] - bands(cen)[1][:, :1]).abs(), 2) * 255
        w = w * (1 - rg / self.lt3.exp()).clamp(min=0) ** 2
        w = w * (1 - oof)
        return torch.cat([torch.full_like(w, -9.0), _logit(w)], 1)  # coarse band: no averaging

    def forward(self, seq, fl, t0, n):
        """centres t0..t0+n-1 of seq [B,L,C,H,W] (fl: run_flows of the same positions) -> [B,n,C,H,W]"""
        B, L, C, H, W = seq.shape
        S = len(OFFS)
        don, f_, r_, valid = _gather(seq, fl, t0, n)
        cen = seq[:, t0:t0 + n]
        N = B * n
        cenr = cen[:, :, None].expand(-1, -1, S, -1, -1, -1).reshape(N * S, C, H, W)
        don, f_, r_ = don.reshape(N * S, C, H, W).float(), f_.reshape(N * S, 2, H, W).float(), r_.reshape(N * S, 2, H, W).float()
        dt = torch.tensor(OFFS, device=seq.device, dtype=torch.float32).abs().div(3).view(1, S, 1, 1, 1).expand(N, -1, 1, H, W).reshape(N * S, 1, H, W)
        wp = fc.warp(don, f_)
        cue = cues(cenr, wp, f_, r_)
        if self.refine is not None:
            f_ = self.refine(cenr, wp, cue, f_, dt)
            wp = fc.warp(don, f_)
            cue = cues(cenr, wp, f_, r_)
        cons = wp.view(N, S, C, H, W).mean(1, keepdim=True).expand(-1, S, -1, -1, -1).reshape(N * S, C, H, W) - cenr
        lg = self.prior(cue, cenr, wp) + self.conf(torch.cat([cenr, wp, cue, f_ / 8, dt, cons], 1))
        cf = torch.sigmoid(lg).view(N, S, 2, H, W)
        # shot gating (RDUAL): a donor whose coarse error is large over the whole picture is another shot, and so is
        # every donor further away on that side; donors past the run end do not exist
        cut = (cue[:, 0].reshape(N, S, -1).mean(-1) > 6.0).float()
        cut_l = torch.flip(torch.cummax(torch.flip(cut[:, :R], [1]), 1)[0], [1])
        cut_r = torch.cummax(cut[:, R:], 1)[0]
        keep = (1 - torch.cat([cut_l, cut_r], 1)) * valid.reshape(N, S)
        cf = cf * keep.view(N, S, 1, 1, 1)
        if self.rule == "mc":                                          # vs_temporalfix's conservative gate
            cf = cf * torch.sigmoid((cf - 0.5) * 12.0)
        center = cen.reshape(N, C, H, W).float()
        lc, hc = bands(center); lw, hw = bands(wp)
        lw, hw = lw.view(N, S, C, H, W), hw.view(N, S, C, H, W)
        wl, wh = cf[:, :, 0:1], cf[:, :, 1:2]
        lo = (lc + (wl * lw).sum(1)) / (1 + wl.sum(1))
        hi = (hc + (wh * hw).sum(1)) / (1 + wh.sum(1))
        return (lo + hi).view(B, n, C, H, W)


class StillStage(nn.Module):
    """ttemp-like: unaligned neighbours n-3..n+3 count while they stay within a threshold of the centre (per pixel and
    channel, in levels), the first one beyond it stops that direction. Weight 1 up to mdiff, falling to 0 at thresh
    (TTempSmooth strong: 12 / 5). Thresholds learned; a learned logit residual (zero at the start) on top."""
    def __init__(self):
        super().__init__()
        self.lth = nn.Parameter(torch.log(torch.tensor([12.0, 12.0, 12.0])))
        self.lmd = nn.Parameter(torch.log(torch.tensor([5.0, 5.0, 5.0])))
        self.conf = ConfNet(3 + 3 + 3 + 1, ch=24)

    def forward(self, seq, t0, n, mot=None):
        """mot [B,n,1,H,W]: largest motion (px) to the neighbours of each centre; averaging only where it is still
        (< ~0.5 px): ttemp is a still-area filter, unaligned averaging of anything that moves would smear it"""
        B, L, C, H, W = seq.shape
        cen = seq[:, t0:t0 + n].float()
        still = torch.ones_like(cen[:, :, :1]) if mot is None else (1 - ((mot - 0.25) / 0.5).clamp(0, 1))
        th, md = self.lth.exp().view(1, 1, C, 1, 1), self.lmd.exp().view(1, 1, C, 1, 1)
        md = torch.minimum(md, th * 0.95)
        num, den = cen.clone(), torch.ones_like(cen)
        for side in (-1, 1):
            alive = torch.ones_like(cen)
            for k in range(1, R + 1):
                idx = [min(max(t + side * k, 0), L - 1) for t in range(t0, t0 + n)]
                nb = seq[:, idx].float()
                d = (nb - cen).abs() * 255
                w = ((th - d) / (th - md)).clamp(0, 1)
                alive = alive * (d <= th).float()                     # stop at the first neighbour beyond the threshold
                lg = _logit(w) + self.conf(torch.cat([cen, nb, d / 16, torch.full_like(d[:, :, :1], k / 3)], 2).flatten(0, 1)).mean(1, keepdim=True).view(B, n, 1, H, W)
                w = torch.sigmoid(lg) * alive * still
                valid = torch.tensor([0 <= t + side * k < L for t in range(t0, t0 + n)], device=seq.device, dtype=cen.dtype).view(1, n, 1, 1, 1)
                w = w * valid
                num, den = num + w * nb, den + w
        return num / den


class TStab(nn.Module):
    VERSION = 2

    def __init__(self, refine=True):
        super().__init__()
        self.s1 = AlignedStage("mc", refine)
        self.s2 = AlignedStage("gtfix", refine)
        self.s3 = StillStage()
        self.register_buffer("version", torch.tensor(self.VERSION))

    def forward_run(self, seq, fl=None, stages=3):
        """seq [B,L,3,H,W] FieldNet output, fl = run_flows(seq) -> stabilized frames REACH..L-1-REACH, [B,L-2*REACH,3,H,W]
        (stages < 3: fewer passes, for tests; the output then covers more frames)"""
        B, L = seq.shape[:2]
        if fl is None:
            with torch.no_grad():
                fl = run_flows(seq)
        seq = seq.float()
        x, off = seq, 0
        x = self.s1(x, fl, R, L - 2 * R); off = R                    # frames R..L-1-R
        if stages >= 2:
            x = self.s2(x, fl[:, off:off + x.shape[1]], R, x.shape[1] - 2 * R); off += R
        if stages >= 3:
            n3 = x.shape[1] - 2 * R
            mot = fl[:, off + R:off + R + n3].norm(dim=3).amax(2, keepdim=True)      # [B,n3,1,H,W] px
            mot = F.max_pool2d(mot.flatten(0, 1), 5, 1, 2).view_as(mot)
            x = self.s3(x, R, n3, mot); off += R
        return x

    def forward(self, win, fl=None):
        """win [B,2*REACH+1,3,H,W] (centre = REACH) -> [B,3,H,W]"""
        return self.forward_run(win, fl)[:, 0]


class StillLock:
    """long memory for still areas (the prototype that won the Courage rug comparison): the picture is split into a
    blurred base and fine detail; where the base has not really changed, the detail is a running mean (then a long
    EMA, keep) of its history; any real change nearby resets it. Stateful, one clip / shot at a time; call reset() at
    cuts and seeks. Works on [B,3,H,W] (all channels; the change test uses every channel)."""
    def __init__(self, s=1.5, thr=1.0 / 255, keep=0.97, anchor=False, fade=False, chroma=False):
        # anchor: the change test compares with the running MEAN of the locked area (not a fast-following copy), so a
        # slow drift (camera creeping < 0.5 px per frame) adds up and resets the lock, while zero-mean shimmer does not
        # fade: a fade in / out scales the picture; the memory is scaled by the local brightness ratio first, so the
        # lock holds through fades instead of resetting everywhere (the letters boiled during Courage's fade from black)
        self.s, self.thr, self.keep, self.anchor, self.fade = s, thr, keep, anchor, fade
        self.chroma = chroma          # also lock the colour planes (leftover rainbows), still judged by the luma test
        self.reset()

    def reset(self):
        self.base = self.det = self.age = None

    @torch.no_grad()
    def __call__(self, f, moving=None):
        b = blur(f, self.s); d = f - b
        if self.base is None or self.base.shape != b.shape:
            self.base, self.det, self.age = b.clone(), d.clone(), torch.zeros_like(b[:, :1])
            return f
        # change test on LUMA only, lock applied to luma only (as the prototype that won the comparison): testing the
        # colour planes too let colour noise reset the memory everywhere (real disc: letters shimmer 0.41 vs 0.04)
        if self.fade:
            big = lambda t: blur(t, 12.0)
            g = ((big(b[:, :1]) + 0.02) / (big(self.base[:, :1]) + 0.02)).clamp(0.5, 2.0)
            self.base = self.base * g                  # colour (centred on 0) scales with a fade like luma
            self.det = self.det * g
        ch = (b[:, :1] - self.base[:, :1]).abs()
        ch = F.max_pool2d(ch, 5, 1, 2) > self.thr
        if moving is not None:                         # motion tracking says this moved: never lock it
            ch = ch | (F.max_pool2d(moving, 5, 1, 2) > 0.5)
        self.age = torch.where(ch, torch.zeros_like(self.age), self.age + 1)
        a = torch.where(ch, torch.ones_like(self.age), torch.clamp(1 / (self.age + 1), min=1 - self.keep))
        self.det = self.det + a * (d - self.det)
        if self.anchor:
            self.base = self.base + torch.where(ch, torch.ones_like(self.age), torch.clamp(1 / (self.age + 1), min=1 - self.keep)) * (b - self.base)
        else:
            self.base = self.base + torch.where(ch, torch.ones_like(self.age), torch.full_like(self.age, 0.5)) * (b - self.base)
        out = f.clone()
        out[:, :1] = b[:, :1] + self.det[:, :1]
        if self.chroma:                                # colour: the long-run mean of the whole colour (rainbows are broad)
            out[:, 1:] = self.base[:, 1:] + self.det[:, 1:]
        return out
