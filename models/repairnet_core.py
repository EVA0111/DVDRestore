"""RepairNet: one network for MPEG-2 damage (blocking, mosquito noise, smeared colour), halos / ringing / echoes and
colour bleed, single frame, YUV 4:2:0 in and out. Replaces BlockNet + HaloNet + ChromaNet in one pass.

Layout (sizes for a 720x480 frame):
  input   luma folded 2x2 into 4 channels at half resolution + U + V (already half resolution in 4:2:0)
          + 2 dial planes: halo repair, bleed repair (0 = leave it, 1 = repair; MPEG damage is always repaired)
  U-Net   NAFNet-style blocks (LayerNorm -> 1x1 -> depthwise 3x3 -> SimpleGate -> 1x1, then a gated feed-forward)
          at 1/2 (c), 1/4 (2c), 1/8 (4c) resolution: most work small = fast, ~100 px reach (echoes, colour tails)
  heads   half resolution: 4 luma sub-pixels + U + V corrections;
          full resolution: a small luma refine net for 1-2 px rims and thin ink lines
LayerNorm runs in float32 so fp16 / TensorRT cannot drift. Width and height must be multiples of 16 (the app pads)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LN2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        xf = x.float()
        mu = xf.mean(1, keepdim=True)
        var = (xf - mu).pow(2).mean(1, keepdim=True)
        return ((xf - mu) * torch.rsqrt(var + 1e-6) * self.w + self.b).to(x.dtype)


class NAF(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = LN2d(c), LN2d(c)
        self.c1 = nn.Conv2d(c, 2 * c, 1)
        self.dw = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.c2 = nn.Conv2d(c, c, 1)
        self.c3 = nn.Conv2d(c, 2 * c, 1)
        self.c4 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        a, b = self.dw(self.c1(self.n1(x))).chunk(2, 1)
        x = x + self.c2(a * b) * self.beta
        a, b = self.c3(self.n2(x)).chunk(2, 1)
        return x + self.c4(a * b) * self.gamma


def _stack(c, n):
    return nn.Sequential(*[NAF(c) for _ in range(n)])


class RepairNet(nn.Module):
    def __init__(self, c=64, blocks=(2, 2, 6, 2, 2), dials=2, rc=24):
        super().__init__()
        self.dials = dials
        self.inp = nn.Conv2d(6 + dials, c, 3, padding=1)
        self.e1 = _stack(c, blocks[0]);      self.down1 = nn.Conv2d(c, 2 * c, 2, 2)
        self.e2 = _stack(2 * c, blocks[1]);  self.down2 = nn.Conv2d(2 * c, 4 * c, 2, 2)
        self.mid = _stack(4 * c, blocks[2])
        self.up2 = nn.Sequential(nn.Conv2d(4 * c, 8 * c, 1), nn.PixelShuffle(2))
        self.d2 = _stack(2 * c, blocks[3])
        self.up1 = nn.Sequential(nn.Conv2d(2 * c, 4 * c, 1), nn.PixelShuffle(2))
        self.d1 = _stack(c, blocks[4])
        self.out = nn.Conv2d(c, 6, 3, padding=1)
        self.feat = nn.Sequential(nn.Conv2d(c, 4 * 8, 1), nn.PixelShuffle(2))          # 8 feature planes at full res
        self.refine = nn.Sequential(nn.Conv2d(2 + 8, rc, 3, padding=1), nn.ReLU(inplace=True),
                                    nn.Conv2d(rc, rc, 3, padding=1), nn.ReLU(inplace=True),
                                    nn.Conv2d(rc, 1, 3, padding=1))
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.refine[-1].weight); nn.init.zeros_(self.refine[-1].bias)

    def forward(self, y, uv, dials):
        """y [B,1,H,W], uv [B,2,H/2,W/2] (y = (Y8-16)/219, uv = (C8-128)/224), dials [B,dials] 0..1.
        Returns repaired y, uv in the same scale."""
        B, _, H, W = y.shape
        d = dials[:, :, None, None].to(y.dtype).expand(B, self.dials, H // 2, W // 2)
        x = torch.cat([F.pixel_unshuffle(y, 2), uv, d], 1)
        h = self.inp(x)
        s1 = self.e1(h)
        s2 = self.e2(self.down1(s1))
        m = self.mid(self.down2(s2))
        u = self.d2(self.up2(m) + s2)
        u = self.d1(self.up1(u) + s1)
        r = self.out(u)
        y1 = y + F.pixel_shuffle(r[:, :4], 2)
        uv1 = uv + r[:, 4:]
        y2 = y1 + self.refine(torch.cat([y, y1, self.feat(u)], 1))
        return y2, uv1


if __name__ == "__main__":
    import time
    net = RepairNet().cuda().eval()
    n = sum(p.numel() for p in net.parameters())
    y = torch.rand(1, 1, 480, 720, device="cuda"); uv = torch.rand(1, 2, 240, 360, device="cuda") - 0.5
    dl = torch.ones(1, 2, device="cuda")
    with torch.no_grad():
        for dt in (torch.float32, torch.float16):
            m = net.to(dt)
            for _ in range(5):
                m(y.to(dt), uv.to(dt), dl.to(dt))
            torch.cuda.synchronize(); t = time.time()
            for _ in range(50):
                m(y.to(dt), uv.to(dt), dl.to(dt))
            torch.cuda.synchronize()
            print(f"{n / 1e6:.2f}M params, {dt}: {50 / (time.time() - t):.1f} fps at 720x480 (PyTorch)")
