"""FieldNet v5: one network = FieldNet v4 trunk + backward pass + NeuFlow v2 motion + TStab cascade (+ StillLock in
the app). Shared by training (models/fieldnet5_train.py) and the app.

  motion      NeuFlow v2 (models/neuflow_v2.py, frozen, pretrained) replaces block matching everywhere: aligning the
              recurrent memory between input frames (lightly blurred: dot crawl must not steer it) and aligning
              neighbour outputs inside TStab. Synthetic tests: sub-pixel pans 0.05 px error vs 0.48 px, a repeating
              mesh (Courage's door) 0.5-1.9 px vs 13-19 px.
  backward    (idea from RDUAL) a reverse sweep first: the same trunk runs over the run backwards in time (v4 was
              trained on time-reversed runs, so it already does this well) and leaves, for every frame, its output
              and memory having seen the FUTURE frames. The forward sweep adds them to its head features through
              a zero-initialised 1x1 convolution, so every frame sees past (own memory) and future (backward state).
  cascade     TStab v2 (models/tstab_core.py) on the forward outputs: motion / detail / still stages, real content
              only, scene cuts respected.
Starts exactly as v4 + TStab with better motion: the backward fusion starts at zero, the trunk and TStab load their
trained weights. Frames: YUV, Y 0..1, UV centred."""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldnet_core as core          # noqa: E402
import tstab_core as ts               # noqa: E402
import neuflow_v2 as nf               # noqa: E402

T = core.T
C_MID = 3 * (T // 2)
SWAP_PN = [2, 1, 0, 3, 4]             # time reversed: a p match becomes an n match


class FieldNet5(nn.Module):
    VERSION = 5

    def __init__(self, ch=192):
        super().__init__()
        self.trunk = core.FieldNet3(ch=ch, ndial=7, version=4)
        self.bwd = nn.Conv2d(3 + core.NH, ch, 1)
        nn.init.zeros_(self.bwd.weight); nn.init.zeros_(self.bwd.bias)
        self.ts = ts.TStab()
        self.register_buffer("version", torch.tensor(self.VERSION))
        self.flow_net = None                                  # NeuFlow, frozen, not saved with the checkpoint

    def load_parts(self, v4=None, tstab=None, device="cpu"):
        if v4 is not None:
            sd = torch.load(v4, map_location="cpu", weights_only=True)
            self.trunk.load_state_dict(sd, strict=False)
        if tstab is not None:
            self.ts.load_state_dict(torch.load(tstab, map_location="cpu", weights_only=True))
        return self

    def flow(self, cur, prv, blur=0):
        """prv(x + flow) ~ cur(x), full YUV frames [N,3,H,W]"""
        if self.flow_net is None:
            object.__setattr__(self, "flow_net", nf.load(next(self.parameters()).device))   # not a submodule: frozen, own file
        with torch.autocast(cur.device.type, enabled=False):                  # full precision: sub-pixel accuracy
            return nf.flow_yuv(self.flow_net, cur, prv, blur)

    def state_dict(self, *a, **k):                            # NeuFlow has its own file
        return {kk: v for kk, v in super().state_dict(*a, **k).items() if not kk.startswith("flow_net.")}

    # ---- motion between input frames (centre frame of each window) -------------------------------------------
    @torch.no_grad()
    def motion(self, xw, reverse=False):
        """flow from step t to step t-1 (t+1 when reverse) on the input frames, and an occlusion map -> [B,L,2/1,H,W]"""
        B, L = xw.shape[:2]
        mid = xw[:, :, C_MID:C_MID + 3].float()
        fls = torch.zeros(B, L, 2, *xw.shape[-2:], device=xw.device)
        occ = torch.ones(B, L, 1, *xw.shape[-2:], device=xw.device)
        idx = range(L - 2, -1, -1) if reverse else range(1, L)
        cur = torch.stack([mid[:, t] for t in idx], 1).flatten(0, 1)
        prv = torch.stack([mid[:, t + 1 if reverse else t - 1] for t in idx], 1).flatten(0, 1)
        fl = self.flow(cur, prv, blur=1).view(B, L - 1, 2, *xw.shape[-2:])
        err = (core._blur(cur[:, :1], 2) - core._blur(core.warp(prv[:, :1], fl.flatten(0, 1)), 2)).abs().view(B, L - 1, 1, *xw.shape[-2:])
        for j, t in enumerate(idx):
            fls[:, t], occ[:, t] = fl[:, j], (err[:, j] * 30).clamp(0, 1)
        return fls, occ

    # ---- one sweep of the trunk -------------------------------------------------------------------------------
    def sweep(self, xw, dials, fw, mt, ex, rs, mot, hadd=None, reverse=False, ckpt=False):
        """recurrent run over the steps (forward, or backward in time with time-reversed inputs: window and raw
        frames reversed, p/n matches swapped). Returns outputs and memories in the ORIGINAL frame order."""
        B, L = xw.shape[:2]
        order = range(L - 1, -1, -1) if reverse else range(L)
        outs, hids = [None] * L, [None] * L
        prev = hid = None
        fl, oc = mot
        sw = torch.as_tensor(SWAP_PN, device=xw.device)
        for i, t in enumerate(order):
            x, f_, m_ = xw[:, t], fw[:, t], mt[:, t]
            if reverse:
                x = x.view(B, T, 3, *x.shape[-2:]).flip(1).reshape(x.shape)
                f_ = f_.view(B, 7, 6, *f_.shape[-2:]).flip(1).reshape(f_.shape)
                m_ = m_[:, sw]
            mid = xw[:, t, C_MID:C_MID + 3]
            if prev is None:
                pw, hw, occ = mid, torch.zeros(B, core.NH, *x.shape[-2:], device=x.device), torch.ones(B, 1, *x.shape[-2:], device=x.device)
            else:
                pw, hw, occ = core.warp(prev, fl[:, t]), core.warp(hid, fl[:, t]), oc[:, t]
                cut = rs[:, t + 1] if reverse else rs[:, t]          # a cut between this step and the previous one
                if cut.any():
                    m = cut[:, None, None, None].float()
                    pw, hw, occ = pw * (1 - m) + mid * m, hw * (1 - m), occ * (1 - m) + m
            ha = None if hadd is None else hadd[:, t]
            if ckpt and torch.is_grad_enabled():
                o, h = torch.utils.checkpoint.checkpoint(self.trunk, x, pw, hw, occ, dials, f_, m_, ex[:, t], ha, use_reentrant=False)
            else:
                o, h = self.trunk(x, pw, hw, occ, dials, f_, m_, ex[:, t], ha)
            outs[t], hids[t] = o, h
            prev, hid = o, h
        return torch.stack(outs, 1), torch.stack(hids, 1)

    def run(self, xw, dials, fw, mt, ex, rs, bwd_grad=False, ckpt=True, stab=True):
        """whole v5 on a run [B,L,...] -> (trunk outputs [B,L,3,H,W], stabilized [B,L-2*REACH,3,H,W] or None)"""
        mf, mb = self.motion(xw), self.motion(xw, reverse=True)
        with torch.set_grad_enabled(bwd_grad and torch.is_grad_enabled()):
            ob, hb = self.sweep(xw, dials, fw, mt, ex, rs, mb, reverse=True, ckpt=ckpt)
        B, L = xw.shape[:2]
        hadd = self.bwd(torch.cat([ob, hb], 2).flatten(0, 1).float()).view(B, L, -1, *xw.shape[-2:])
        of, _ = self.sweep(xw, dials, fw, mt, ex, rs, mf, hadd=hadd, ckpt=ckpt)
        if not stab:
            return of, None
        with torch.no_grad():
            fl = ts.run_flows(of.detach().float(), flow_fn=lambda c, d: self.flow(c, d))
        return of, self.ts.forward_run(of.float(), fl)


def build(path=None, device="cpu"):
    net = FieldNet5()
    if path:
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
    return net.to(device)
