"""RelayVSR reference (keyframe Wan + Flash decoder), a straight port of
ComfyUI `comfy/ldm/wan/model_vsr.py` + `comfy_extras/nodes_wan.py`
(RelayVSRConditioning, 1 step euler / simple / cfg 1, VSRDecode) with
comfy-kitchen's eager `window_attention`.

Truth model: the int8 convrot weights DEQUANTIZED (`W = (s * Wq) @ H` per 256
group), bf16 acts for the keyframe Wan, f32 for the small Flash decoder. (ComfyUI
itself also quantizes activations to int8 per row; that is an approximation of
this.)

Usage:
  python -m thinfer_pytorch_ref.wan.gen_relayvsr_ref --video in.mp4 --frames 31 \
      --dit <relayvsr_x2_*.safetensors> --flash <relayvsr_flash_vae_x2_fp32.safetensors> \
      --out relay_ref.safetensors [--seed 42]
"""

import argparse
import json
import math
import subprocess

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

KEYFRAME_INTERVAL = 15
WAN_CONTEXT = 4
WAN_WINDOW = (22, 40)
FLASH_COND_WINDOW = (48, 48)
FLASH_VIDEO_WINDOW = (24, 24)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(8)


def relay_keyframes(frames):
    return list(range(0, frames - 1, KEYFRAME_INTERVAL)) + [frames - 1]


def relay_latent_size(height, width, latent_stride):
    stride = latent_stride * 2
    return math.ceil(height / stride) * 2, math.ceil(width / stride) * 2


# ---------------------------------------------------------------- attention
def _axis_keys(q0, q1, size, window, clamp, device):
    half = window // 2
    low0, low1 = q0 - half, q1 - 1 - half
    low = torch.arange(q0, q1, device=device) - half
    if clamp:
        low0, low1 = min(max(low0, 0), size - window), min(max(low1, 0), size - window)
        low = low.clamp(0, size - window)
    start, stop = max(low0, 0), min(low1 + window, size)
    keys = torch.arange(start, stop, device=device)
    return start, stop, (keys >= low[:, None]) & (keys < low[:, None] + window)


def window_attention(q, k, v, windows, temporal_window=None, clamp=True):
    """comfy-kitchen eager window_attention, one query row per tile."""
    if windows and isinstance(windows[0], int):
        windows = [windows]
    batch, tq, h, w, nh, hd = q.shape
    tk, kvh = k.shape[1], k.shape[4]
    if temporal_window is None:
        temporal_window = tk
    windows = list(windows) if len(windows) > 1 else [windows[0]] * tk
    if clamp:
        windows = [(min(wh, h), min(ww, w)) for wh, ww in windows]
    gqa = {"enable_gqa": True} if nh != kvh else {}
    out = torch.empty_like(q)
    for t in range(tq):
        last = t + tk - tq
        planes = range(max(0, last - temporal_window + 1), last + 1)
        for y in range(h):
            ks, vs, masks = [], [], []
            for p in planes:
                top, bottom, rows = _axis_keys(y, y + 1, h, windows[p][0], clamp, q.device)
                left, right, cols = _axis_keys(0, w, w, windows[p][1], clamp, q.device)
                ks.append(k[:, p, top:bottom, left:right].reshape(batch, -1, kvh, hd))
                vs.append(v[:, p, top:bottom, left:right].reshape(batch, -1, kvh, hd))
                masks.append((rows[:, None, :, None] & cols[None, :, None, :]).flatten(2).flatten(0, 1))
            qt = q[:, t, y].reshape(batch, -1, nh, hd).transpose(1, 2)
            kt = torch.cat(ks, dim=1).transpose(1, 2)
            vt = torch.cat(vs, dim=1).transpose(1, 2)
            o = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=torch.cat(masks, dim=-1), **gqa)
            out[:, t, y] = o.transpose(1, 2).reshape(batch, w, nh, hd)
    return out


# ---------------------------------------------------------------- rope
def rope(pos, dim, theta):
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float64), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def embed_nd(ids, axes_dim, theta=10000.0):
    emb = torch.cat([rope(ids[..., i], axes_dim[i], theta) for i in range(ids.shape[-1])], dim=-3)
    return emb.unsqueeze(1)  # [1, 1, N, d/2, 2, 2]


def apply_rope1(x, freqs):
    # x [B, N, H, D]; freqs [1, N, 1, D/2, 2, 2] (movedim'ed)
    x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    out = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
    return out.reshape(*x.shape).type_as(x)


# ---------------------------------------------------------------- weights
class Weights:
    def __init__(self, path):
        self.f = safe_open(path, "pt", device="cpu")
        self.keys = set(self.f.keys())
        self.meta = self.f.metadata() or {}
        self.h = None

    def hadamard(self, dev):
        if self.h is None:
            h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
            h = h4
            while h.shape[0] < 256:
                h = torch.kron(h, h4)
            self.h = (h / 16.0).to(dev)
        return self.h

    def t(self, k, dtype=torch.bfloat16):
        return self.f.get_tensor(k).to(DEV, dtype)

    def linear_w(self, stem, dtype=torch.bfloat16):
        w = self.f.get_tensor(f"{stem}.weight")
        if w.dtype == torch.int8 and f"{stem}.weight_scale" in self.keys:
            s = self.f.get_tensor(f"{stem}.weight_scale").to(DEV, torch.float32)
            wq = w.to(DEV, torch.float32) * s
            n, kk = wq.shape
            wq = (wq.reshape(n, kk // 256, 256) @ self.hadamard(DEV)).reshape(n, kk)
            return wq.to(dtype)
        return w.to(DEV, dtype)

    def linear(self, x, stem, dtype=torch.bfloat16):
        b = self.t(f"{stem}.bias", dtype) if f"{stem}.bias" in self.keys else None
        return F.linear(x.to(dtype), self.linear_w(stem, dtype), b)


def rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).type_as(x) * w


def layernorm(x, eps=1e-6):
    return F.layer_norm(x.float(), (x.shape[-1],), eps=eps).type_as(x)


# ---------------------------------------------------------------- keyframe Wan
def sinusoidal(dim, pos):
    half = dim // 2
    pos = pos.float()
    s = torch.outer(pos, torch.pow(10000, -torch.arange(half, device=pos.device).float().div(half)))
    return torch.cat([torch.cos(s), torch.sin(s)], dim=1)


def lq_proj(W, video, up):
    # video [1, 3, K, H, W] in [-1, 1]
    b, c, f, h, w = video.shape
    x = video.movedim(2, 1).reshape(b * f, c, h, w).float()
    if up == 4:
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
    x = F.pixel_unshuffle(x, 4 * up).to(torch.bfloat16)
    x = F.conv2d(x, W.t("lq_proj_in.conv1.weight"), W.t("lq_proj_in.conv1.bias"))
    x = F.silu(rms(x.movedim(1, -1), W.t("lq_proj_in.norm1.weight"), 1e-6)).movedim(-1, 1)
    x = F.conv2d(x, W.t("lq_proj_in.conv2.weight"), W.t("lq_proj_in.conv2.bias"), stride=2, padding=1)
    x = F.silu(rms(x.flatten(2).transpose(1, 2), W.t("lq_proj_in.norm2.weight"), 1e-6))
    return W.linear(x, "lq_proj_in.linear").view(b, -1, 3072)


def wan_forward(W, x, lq_pixels, up, taps):
    dim, nh, hd = 3072, 24, 128
    eps = 1e-6
    _, _, kf, hlat, wlat = x.shape
    gh, gw = hlat // 2, wlat // 2
    tokens = gh * gw
    # time embedding at t = 1000
    t = torch.tensor([1000.0], device=DEV)
    e = sinusoidal(256, t).to(torch.bfloat16)
    e = W.linear(F.silu(W.linear(e, "time_embedding.0")), "time_embedding.2")  # [1, dim]
    e0 = W.linear(F.silu(e), "time_projection.1").view(1, 6, dim)
    ctx = W.t("positive_conditioning").unsqueeze(0)
    ctx = W.linear(F.gelu(W.linear(ctx, "text_embedding.0"), approximate="tanh"), "text_embedding.2")
    taps["temb"] = e.float()
    taps["text"] = ctx.float()

    lq_tok = lq_proj(W, lq_pixels, up)
    taps["lq_tokens"] = lq_tok.float()
    pe = F.conv3d(x.float(), W.t("patch_embedding.weight", torch.float32), W.t("patch_embedding.bias", torch.float32), stride=(1, 2, 2))
    hidden = lq_tok + pe.flatten(2).transpose(1, 2).to(torch.bfloat16)
    taps["patch_x"] = hidden.float()

    d = hd
    ids = torch.zeros((kf, gh, gw, 3), device=DEV)
    ids[..., 0] = torch.arange(kf, device=DEV).view(-1, 1, 1)
    ids[..., 1] = torch.arange(gh, device=DEV).view(1, -1, 1)
    ids[..., 2] = torch.arange(gw, device=DEV).view(1, 1, -1)
    freqs = embed_nd(ids.reshape(1, -1, 3), [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]).movedim(1, 2)

    for i in range(30):
        p = f"blocks.{i}"
        mod = W.t(f"{p}.modulation") + e0.unsqueeze(0)  # [1, 1, 6, dim]
        sh, sc, g1, sh2, sc2, g2 = mod.view(1, 6, dim).unbind(1)
        # self-attn (window)
        xin = layernorm(hidden) * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)
        q = rms(W.linear(xin, f"{p}.self_attn.q"), W.t(f"{p}.self_attn.norm_q.weight"), eps)
        k = rms(W.linear(xin, f"{p}.self_attn.k"), W.t(f"{p}.self_attn.norm_k.weight"), eps)
        v = W.linear(xin, f"{p}.self_attn.v")
        q = apply_rope1(q.view(1, -1, nh, hd), freqs)
        k = apply_rope1(k.view(1, -1, nh, hd), freqs)
        grid = (1, kf, gh, gw, nh, hd)
        a = window_attention(q.view(grid), k.view(grid), v.view(grid), WAN_WINDOW, temporal_window=WAN_CONTEXT)
        y = W.linear(a.reshape(1, -1, dim), f"{p}.self_attn.o")
        hidden = hidden + y * g1.unsqueeze(1)
        # cross-attn
        xn = F.layer_norm(hidden.float(), (dim,), W.t(f"{p}.norm3.weight", torch.float32), W.t(f"{p}.norm3.bias", torch.float32), eps).to(torch.bfloat16)
        cq = rms(W.linear(xn, f"{p}.cross_attn.q"), W.t(f"{p}.cross_attn.norm_q.weight"), eps)
        ck = rms(W.linear(ctx, f"{p}.cross_attn.k"), W.t(f"{p}.cross_attn.norm_k.weight"), eps)
        cv = W.linear(ctx, f"{p}.cross_attn.v")
        ca = F.scaled_dot_product_attention(
            cq.view(1, -1, nh, hd).transpose(1, 2), ck.view(1, -1, nh, hd).transpose(1, 2), cv.view(1, -1, nh, hd).transpose(1, 2)
        ).transpose(1, 2).reshape(1, -1, dim)
        hidden = hidden + W.linear(ca, f"{p}.cross_attn.o")
        # ffn
        xin = layernorm(hidden) * (1 + sc2.unsqueeze(1)) + sh2.unsqueeze(1)
        f = W.linear(F.gelu(W.linear(xin, f"{p}.ffn.0"), approximate="tanh"), f"{p}.ffn.2")
        hidden = hidden + f * g2.unsqueeze(1)
        if i == 0:
            taps["block0"] = hidden.float()
    taps["block_last"] = hidden.float()
    hm = W.t("head.modulation") + e.unsqueeze(1)  # [1, 2, dim]
    sh, sc = hm.unbind(1)
    xo = layernorm(hidden) * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)
    out = W.linear(xo, "head.head")  # [1, kf*tokens, 48*4]
    out = out.float().view(1, kf, gh, gw, 1, 2, 2, 48)
    out = torch.einsum("bfhwpqrc->bcfphqwr", out).reshape(1, 48, kf, hlat, wlat)
    return out


# ---------------------------------------------------------------- Flash decoder
class Flash:
    def __init__(self, path):
        self.W = Weights(path)
        cfg = json.loads(self.W.meta["config"])
        self.up = cfg["upscale_factor"]
        self.ls = 16 // self.up

    def t(self, k):
        return self.W.t(k, torch.float32)

    def lin(self, x, stem):
        return self.W.linear(x, stem, torch.float32)

    def grid_rope(self, t, h, w):
        ids = torch.zeros((h, w, 3), device=DEV)
        ids[..., 0] = t
        ids[..., 1] = torch.arange(h, device=DEV).float()[:, None]
        ids[..., 2] = torch.arange(w, device=DEV).float()[None]
        return embed_nd(ids.reshape(1, -1, 3), [16, 24, 24]).movedim(1, 2)

    def project(self, i, x, kstem, vstem, freqs):
        b, s, _ = x.shape
        p = f"backbone.blocks.{i}.attention"
        q = apply_rope1(self.lin(x, f"{p}.query").view(b, s, 8, 64), freqs)
        k = rms(self.lin(x, f"{p}.{kstem}").view(b, s, 2, 64), self.t(f"{p}.key_norm.weight"), 1e-5)
        v = rms(self.lin(x, f"{p}.{vstem}").view(b, s, 2, 64), self.t(f"{p}.value_norm.weight"), 1e-5)
        return q, k, v

    def ffn(self, i, x):
        p = f"backbone.blocks.{i}"
        xn = rms(x, self.t(f"{p}.feed_forward_norm.weight"), 1e-5)
        gu = self.lin(xn, f"{p}.feed_forward.gate_up_proj")
        g, u = gu.chunk(2, dim=-1)
        return x + self.lin(F.silu(g) * u, f"{p}.feed_forward.down")

    def prepare_condition(self, latent, freqs):
        h, w = latent.shape[-2:]
        x = F.conv2d(latent.float(), self.t("condition_encoder.projection.weight"), self.t("condition_encoder.projection.bias"))
        x = rms(x.flatten(2).transpose(1, 2), self.t("condition_encoder.norm.weight"), 1e-5) + self.t("condition_type_embedding")
        out = []
        for i in range(10):
            xn = rms(x, self.t(f"backbone.blocks.{i}.attention_norm.weight"), 1e-5)
            q, k, v = self.project(i, xn, "condition_key", "condition_value", freqs)
            k = apply_rope1(k, freqs).view(1, 1, h, w, 2, 64)
            v = v.view(1, 1, h, w, 2, 64)
            a = window_attention(q.view(1, 1, h, w, 8, 64), k, v, FLASH_COND_WINDOW, clamp=False)
            x = x + self.lin(a.reshape(1, h * w, 512), f"backbone.blocks.{i}.attention.output")
            x = self.ffn(i, x)
            out.append((k, v))
        return out

    def decode_frame(self, lr, freqs, freqs_past, condition, history):
        h, w = lr.shape[-2] // self.ls, lr.shape[-1] // self.ls
        x = lr.float()
        strides = [2, 2, self.ls // 4]
        for li, s in zip([0, 2, 4], strides):
            x = F.silu(F.conv2d(x, self.t(f"lr_encoder.layers.{li}.weight"), self.t(f"lr_encoder.layers.{li}.bias"), stride=s, padding=1))
        x = F.conv2d(x, self.t("lr_encoder.layers.6.weight"), self.t("lr_encoder.layers.6.bias"))
        x = rms(x.flatten(2).transpose(1, 2), self.t("lr_encoder.norm.weight"), 1e-5) + self.t("video_type_embedding")
        plane = (1, 1, h, w, 2, 64)
        nxt = []
        for i in range(10):
            xn = rms(x, self.t(f"backbone.blocks.{i}.attention_norm.weight"), 1e-5)
            q, k, v = self.project(i, xn, "key", "value", freqs)
            (ks, vs), (ke, ve) = condition[i]
            keys, vals = [ks, ke], [vs, ve]
            if history[i] is not None:
                keys.append(apply_rope1(history[i][0], freqs_past).view(plane))
                vals.append(history[i][1].view(plane))
            keys.append(apply_rope1(k, freqs).view(plane))
            vals.append(v.view(plane))
            wins = [FLASH_COND_WINDOW] * 2 + [FLASH_VIDEO_WINDOW] * (len(keys) - 2)
            a = window_attention(q.view(1, 1, h, w, 8, 64), torch.cat(keys, 1), torch.cat(vals, 1), wins, clamp=False)
            x = x + self.lin(a.reshape(1, h * w, 512), f"backbone.blocks.{i}.attention.output")
            x = self.ffn(i, x)
            nxt.append((k, v))
        r = self.lin(F.silu(self.lin(x, "spatial_head.0")), "residual_projection")
        r = F.pixel_shuffle(r.transpose(1, 2).reshape(1, 768, h, w), 16)
        return F.interpolate(lr.float(), scale_factor=self.up, mode="bicubic", align_corners=False) + r, nxt

    def decode(self, lat, lq):
        # lat [1, 48, K, h, w]; lq [F, H, W, 3] in [0, 1]
        frames, lh, lw = lq.shape[:3]
        h, w = lat.shape[-2:]
        kfs = relay_keyframes(frames)
        freqs, freqs_past = self.grid_rope(0, h, w), self.grid_rope(-1, h, w)
        out = torch.empty((frames, 3, lh * self.up, lw * self.up), device=DEV)
        start = self.prepare_condition(lat[:, :, 0], freqs)
        for wi, (first, last) in enumerate(zip(kfs, kfs[1:] or kfs)):
            end = self.prepare_condition(lat[:, :, min(wi + 1, len(kfs) - 1)], freqs)
            cond = list(zip(start, end))
            hist = [None] * 10
            lr_win = lq[first:last + 1].movedim(-1, 1) * 2.0 - 1.0
            lr_win = F.pad(lr_win, (0, w * self.ls - lw, 0, h * self.ls - lh), mode="replicate")
            for i, fr in enumerate(range(first, last + 1)):
                dec, hist = self.decode_frame(lr_win[i:i + 1], freqs, freqs_past, cond, hist)
                if wi == 0 or fr > first:
                    out[fr] = dec[0, :, : out.shape[-2], : out.shape[-1]]
            start = end
        return out  # [-1, 1]


def read_video(path, frames):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    w, h = map(int, probe.stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    vid = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(-1, h, w, 3)
    return vid[:frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=31)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--flash", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    vid_u8 = read_video(a.video, a.frames)
    lq = vid_u8.to(DEV).float() / 255.0  # [F, H, W, 3]
    frames, H, Wd = lq.shape[:3]
    flash = Flash(a.flash)
    up, ls = flash.up, flash.ls
    W = Weights(a.dit)
    kfs = relay_keyframes(frames)
    lh, lw = relay_latent_size(H, Wd, ls)
    g = torch.Generator().manual_seed(a.seed)
    noise = torch.randn((1, 48, len(kfs), lh, lw), generator=g, dtype=torch.float32).to(DEV)
    lqp = lq[kfs].movedim(-1, 0).unsqueeze(0) * 2.0 - 1.0  # [1, 3, K, H, W]
    lqp = F.pad(lqp, (0, lw * ls - Wd, 0, lh * ls - H, 0, 0), mode="replicate")
    taps = {}
    with torch.no_grad():
        v = wan_forward(W, noise, lqp, up, taps)
        latent = noise - v
        out = flash.decode(latent, lq)
    res = {
        "lr_u8": vid_u8.contiguous(),
        "noise": noise.cpu(),
        "velocity": v.cpu(),
        "kf_latent": latent.float().cpu(),
        "flash_out": out.float().cpu(),
    }
    for k, t in taps.items():
        res[f"tap.{k}"] = t.float().cpu().contiguous()
    save_file(res, a.out)
    print("wrote", a.out, {k: tuple(t.shape) for k, t in res.items()})


if __name__ == "__main__":
    main()
