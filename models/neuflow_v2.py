"""NeuFlow v2 optical flow, inference port for DVD Restore (FieldNet v5 motion estimation).

NeuFlow v2: Zhiyong Zhang, Aniket Gupta, Huaizu Jiang, Hanumant Singh - https://github.com/neufieldrobotics/NeuFlow_v2
Apache License 2.0 (models/neuflow/LICENSE). Weights: models/neuflow/neuflow_mixed.pth (their "mixed" checkpoint).
Changes from the original: one file; no huggingface_hub; the cached grids are built per call for the frame size,
batch and device (no init_bhwd); padding to a multiple of 16; inference returns only the final full-size flow;
input helpers for our YUV frames. Parameter names match the original checkpoint, so it loads unchanged.

Flow convention: flow(img0, img1)[x] = position of x in img1 minus x, i.e. img1(x + flow) ~ img0(x) - the same as
fieldnet_core.est_flow(cur=img0, prv=img1), so it is a drop-in replacement (fieldnet_core.warp(img1, flow) ~ img0)."""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT16, CTX16, ITER16 = 128, 64, 64
FEAT8, CTX8, ITER8 = 128, 64, 64
FEAT1 = 128
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "neuflow", "neuflow_mixed.pth")


def coords_grid(b, h, w, device, dtype):
    ys, xs = torch.meshgrid(torch.arange(h, dtype=dtype, device=device), torch.arange(w, dtype=dtype, device=device), indexing="ij")
    return torch.stack([xs, ys], 0)[None].repeat(b, 1, 1, 1)


class ConvBlock(nn.Module):                                   # backbone_v7.ConvBlock
    def __init__(self, cin, cout, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.1)
        self.norm1 = nn.BatchNorm2d(cout)
        self.norm2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        return self.relu(self.norm2(self.conv2(x)))


class CNNEncoder(nn.Module):                                  # backbone_v7.CNNEncoder
    def __init__(self):
        super().__init__()
        self.block_8_1 = ConvBlock(3, FEAT8 * 2, 8, 4, 2)
        self.block_8_2 = ConvBlock(3, FEAT8, 6, 2, 2)
        self.block_cat_8 = ConvBlock(FEAT8 * 3, FEAT8 + CTX8, 3, 1, 1)
        self.block_16_1 = ConvBlock(3, FEAT16, 6, 2, 2)
        self.block_8_16 = ConvBlock(FEAT8 + CTX8, FEAT16, 6, 2, 2)
        self.block_cat_16 = ConvBlock(FEAT16 * 2, FEAT16 + CTX16 - 2, 3, 1, 1)

    def forward(self, img):
        img = F.avg_pool2d(img, 2, 2)
        x8 = self.block_8_1(img)
        img = F.avg_pool2d(img, 2, 2)
        x8_2 = self.block_8_2(img)
        x8 = self.block_cat_8(torch.cat([x8, x8_2], 1))
        img = F.avg_pool2d(img, 2, 2)
        x16 = self.block_16_1(img)
        x16 = self.block_cat_16(torch.cat([x16, self.block_8_16(x8)], 1))
        B, _, h, w = x16.shape
        ys, xs = torch.meshgrid(torch.arange(h, dtype=x16.dtype, device=x16.device), torch.arange(w, dtype=x16.dtype, device=x16.device), indexing="ij")
        pos = torch.stack([ys - h / 2, xs - w / 2])[None].expand(B, -1, -1, -1)
        return torch.cat([x16, pos], 1), x8


class TransformerLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.merge = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, 2 * dim, bias=False), nn.GELU(), nn.Linear(2 * dim, dim, bias=False))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, source, target):
        m = F.scaled_dot_product_attention(self.q_proj(source), self.k_proj(target), self.v_proj(target))
        m = self.norm1(self.merge(m))
        m = self.norm2(self.mlp(torch.cat([source, m], -1)))
        return source + m


class FeatureAttention(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(dim) for _ in range(num_layers)])
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, f):
        b, c, h, w = f.shape
        f0 = f.flatten(-2).permute(0, 2, 1)
        f1 = torch.cat(f0.chunk(2, 0)[::-1], 0)
        for layer in self.layers:
            f0 = layer(f0, f1)
            f1 = torch.cat(f0.chunk(2, 0)[::-1], 0)
        return self.norm(f0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous())


def global_correlation_softmax(f0, f1):
    b, c, h, w = f0.shape
    grid = coords_grid(b, h, w, f0.device, f0.dtype)
    corr = F.scaled_dot_product_attention(f0.flatten(-2).permute(0, 2, 1), f1.flatten(-2).permute(0, 2, 1),
                                          grid.view(b, 2, -1).permute(0, 2, 1))
    return corr.view(b, h, w, 2).permute(0, 3, 1, 2) - grid


def init_corr(f0, f1):
    b, c, h, w = f0.shape
    corr = torch.matmul(f0.view(b, c, h * w).transpose(1, 2), f1.view(b, c, h * w))
    return corr.view(b * h * w, 1, h, w) / math.sqrt(c)


def lookup(corr, flow, radius=4):
    """local correlation window (2r+1)^2 around x + flow, per pixel"""
    b, _, h, w = flow.shape
    grid = coords_grid(b, h, w, flow.device, flow.dtype)
    coords = (grid + flow).permute(0, 2, 3, 1).reshape(b * h * w, 1, 1, 2)
    r = torch.linspace(-radius, radius, 2 * radius + 1, dtype=flow.dtype, device=flow.device)
    delta = torch.stack(torch.meshgrid(r, r, indexing="ij"), -1).view(1, 2 * radius + 1, 2 * radius + 1, 2)
    c = coords + delta
    H, W = corr.shape[-2:]
    g = torch.cat([2 * c[..., :1] / (W - 1) - 1, 2 * c[..., 1:] / (H - 1) - 1], -1)
    out = F.grid_sample(corr, g.to(corr.dtype), align_corners=True)
    return out.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()


class RConv(nn.Module):                                       # refine.ConvBlock
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(self.conv(x))


class Refine(nn.Module):
    def __init__(self, ctx, it, num_layers, radius, inter):
        super().__init__()
        self.radius = radius
        self.conv1 = RConv((2 * radius + 1) ** 2 + ctx + it + 2 + 1, ctx + it)
        self.conv2 = RConv(ctx + it, inter)
        self.conv_layers = nn.ModuleList([RConv(inter, inter) for _ in range(num_layers)])
        self.conv3 = nn.Conv2d(inter, it + 2, 3, 1, 1, bias=True)

    def forward(self, corrs, context, iter_context, flow):
        b, _, h, w = flow.shape
        rad = torch.full((b, 1, h, w), float(self.radius), dtype=flow.dtype, device=flow.device)
        x = self.conv2(self.conv1(torch.cat([corrs, context, iter_context, flow, rad], 1)))
        for layer in self.conv_layers:
            x = layer(x)
        x = self.conv3(x)
        return torch.clamp(x[:, 2:], -4.0, 4.0), x[:, :2]


class UpSample(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv2d(2 + dim, 256, 3, 1, 1)
        self.conv2 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv3 = nn.Conv2d(512, k * k * 9, 1, 1, 0)

    def forward(self, feature, flow):
        mask = self.conv3(F.relu(self.conv2(F.relu(self.conv1(torch.cat([flow, feature], 1))))))
        b, _, h, w = flow.shape
        k = self.k
        mask = torch.softmax(mask.view(b, 1, 9, k, k, h, w), 2)
        up = F.unfold(flow, [3, 3], padding=1).view(b, 2, 9, 1, 1, h, w)
        up = (mask * up).sum(2).permute(0, 1, 4, 2, 5, 3)
        return up.reshape(b, 2, k * h, k * w)


class NeuFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNNEncoder()
        self.cross_attn_s16 = FeatureAttention(FEAT16 + CTX16, 2)
        self.merge_s8 = nn.Sequential(nn.Conv2d(FEAT16 + FEAT8, FEAT8, 3, 1, 1, bias=False), nn.GELU(),
                                      nn.Conv2d(FEAT8, FEAT8, 3, 1, 1, bias=False), nn.BatchNorm2d(FEAT8))
        self.context_merge_s8 = nn.Sequential(nn.Conv2d(CTX16 + CTX8, CTX8, 3, 1, 1, bias=False), nn.GELU(),
                                              nn.Conv2d(CTX8, CTX8, 3, 1, 1, bias=False), nn.BatchNorm2d(CTX8))
        self.refine_s16 = Refine(CTX16, ITER16, 5, 4, 128)
        self.refine_s8 = Refine(CTX8, ITER8, 5, 4, 96)
        self.conv_s8 = ConvBlock(3, FEAT1, 8, 8, 0)
        self.upsample_s8 = UpSample(FEAT1, 8)

    @staticmethod
    def _split(f, cdim, fdim):
        ctx, feat = torch.split(f, [cdim, fdim], 1)
        return feat, torch.relu(ctx.chunk(2, 0)[0])

    def forward(self, img0, img1, iters_s16=1, iters_s8=8):
        """img0, img1: [B,3,H,W] RGB 0..255, H and W multiples of 16 -> flow [B,2,H,W] (img1(x + flow) ~ img0(x))"""
        img0, img1 = img0 / 255.0, img1 / 255.0
        f16, f8 = self.backbone(torch.cat([img0, img1], 0))
        f16 = self.cross_attn_s16(f16)
        f16, c16 = self._split(f16, CTX16, FEAT16)
        f8, c8 = self._split(f8, CTX8, FEAT8)
        a16, b16 = f16.chunk(2, 0)
        flow = global_correlation_softmax(a16, b16)
        corr = init_corr(a16, b16)
        it = torch.zeros(img0.shape[0], ITER16, *flow.shape[-2:], device=flow.device, dtype=flow.dtype)
        for _ in range(iters_s16):
            it, d = self.refine_s16(lookup(corr, flow), c16, it, flow)
            flow = flow + d
        flow = F.interpolate(flow, scale_factor=2, mode="nearest") * 2
        f16 = F.interpolate(f16, scale_factor=2, mode="nearest")
        f8 = self.merge_s8(torch.cat([f8, f16], 1))
        a8, b8 = f8.chunk(2, 0)
        corr = init_corr(a8, b8)
        c8 = self.context_merge_s8(torch.cat([c8, F.interpolate(c16, scale_factor=2, mode="nearest")], 1))
        it = torch.zeros(img0.shape[0], ITER8, *flow.shape[-2:], device=flow.device, dtype=flow.dtype)
        for _ in range(iters_s8):
            it, d = self.refine_s8(lookup(corr, flow), c8, it, flow)
            flow = flow + d
        return self.upsample_s8(self.conv_s8(img0), flow) * 8


def load(device="cpu", path=WEIGHTS):
    net = NeuFlow()
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    sd = {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"NeuFlow weights do not match: missing {missing[:5]} unexpected {unexpected[:5]}")
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def yuv_to_rgb255(x):
    """our frames [B,3,H,W] (Y 0..1, UV centred) -> RGB 0..255 (BT.601)"""
    y, u, v = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    rgb = torch.cat([y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u], 1)
    return rgb.clamp(0, 1) * 255


@torch.no_grad()
def flow_yuv(net, cur, prv, blur=0):
    """drop-in for fieldnet_core.est_flow(cur, prv) on full YUV frames [B,3,H,W]: prv(x + flow) ~ cur(x).
    blur: box radius applied first (dot crawl must not steer the motion); pads to a multiple of 16 (min 32)."""
    a, b = yuv_to_rgb255(cur.float()), yuv_to_rgb255(prv.float())
    if blur:
        k = 2 * blur + 1
        a = F.avg_pool2d(F.pad(a, (blur,) * 4, mode="replicate"), k, 1)
        b = F.avg_pool2d(F.pad(b, (blur,) * 4, mode="replicate"), k, 1)
    H, W = a.shape[-2:]
    ph, pw = max(32, -(-H // 16) * 16) - H, max(32, -(-W // 16) * 16) - W
    if ph or pw:
        a = F.pad(a, (0, pw, 0, ph), mode="replicate"); b = F.pad(b, (0, pw, 0, ph), mode="replicate")
    fl = net(a, b)
    return fl[..., :H, :W].float()
