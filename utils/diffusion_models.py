"""Stage 15: lightweight DDPM-style trajectory-window denoiser (PyTorch).

A small temporal 1D-convolutional noise-prediction network over normalized
trajectory windows (B x T x F), trained as a standard DDPM: corrupt a clean
window at a random timestep and predict the injected noise. Evaluation uses
single-step denoising (Mode A): treat an observed window as a noisy sample at
a fixed level and recover x0 in one shot. Optional iterative reverse sampling
(Mode B) is provided but not used by default.

This is stage 15's denoising/gap-filling study -- NOT the full model zoo, and
NOT a new primary false-track classifier.
"""

import math
from typing import Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    raise SystemExit("Stage 15 requires PyTorch. Please install torch before running this stage.")

from utils.sequence_models import resolve_device, make_loader  # noqa: F401  (reuse)

POS_IDX = [0, 1, 2]      # dx, dy, dz
VEL_IDX = [3, 4, 5]      # vx, vy, vz


# =============================================================================
# DDPM noise schedule
# =============================================================================

class NoiseSchedule:
    def __init__(self, num_steps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_steps = int(num_steps)
        betas = torch.linspace(beta_start, beta_end, self.num_steps)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def sqrt_ab(self, t):
        return torch.sqrt(self.alpha_bars[t])

    def sqrt_one_minus_ab(self, t):
        return torch.sqrt(1.0 - self.alpha_bars[t])


def timestep_embedding(t: "torch.Tensor", dim: int) -> "torch.Tensor":
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half, 1))
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros(len(t), 1, device=t.device)], dim=-1)
    return emb


# =============================================================================
# Temporal diffusion denoiser
# =============================================================================

class _ResBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.c1 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.c2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, h):
        r = self.c1(h)
        r = self.act(r)
        r = self.c2(r)
        return self.act(h + r)


class TemporalDiffusionDenoiser(nn.Module):
    def __init__(self, input_dim: int, window_len: int, hidden_dim: int = 128,
                 num_blocks: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.window_len = window_len
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([_ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_proj = nn.Conv1d(hidden_dim, input_dim, 3, padding=1)

    def forward(self, x, t):                      # x: B x T x F, t: B
        h = self.in_proj(x.transpose(1, 2))       # B x hidden x T
        temb = self.time_mlp(timestep_embedding(t, self.hidden_dim))  # B x hidden
        h = h + temb[:, :, None]
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h).transpose(1, 2)   # B x T x F


# =============================================================================
# Corrupt / denoise primitives
# =============================================================================

def forward_corrupt(x0, t_int, sched: NoiseSchedule, eps=None):
    if eps is None:
        eps = torch.randn_like(x0)
    ab = sched.alpha_bars[t_int]
    return math.sqrt(float(ab)) * x0 + math.sqrt(float(1.0 - ab)) * eps, eps


def single_step_denoise(model, x_obs, t_int, sched: NoiseSchedule):
    """Mode A: treat x_obs as a noisy sample at level t_int; recover x0 in one shot."""
    b = x_obs.shape[0]
    t = torch.full((b,), int(t_int), dtype=torch.long, device=x_obs.device)
    eps_pred = model(x_obs, t)
    ab = float(sched.alpha_bars[t_int])
    x0 = (x_obs - math.sqrt(1.0 - ab) * eps_pred) / math.sqrt(ab)
    return x0, eps_pred


def reverse_sample(model, x_start, t_start, sched: NoiseSchedule):
    """Mode B (optional): iterative ancestral sampling from t_start down to 0."""
    x = x_start
    for t_int in range(int(t_start), -1, -1):
        b = x.shape[0]
        t = torch.full((b,), t_int, dtype=torch.long, device=x.device)
        eps = model(x, t)
        ab = float(sched.alpha_bars[t_int])
        beta = float(sched.betas[t_int])
        alpha = float(sched.alphas[t_int])
        x0 = (x - math.sqrt(1.0 - ab) * eps) / math.sqrt(ab)
        if t_int > 0:
            ab_prev = float(sched.alpha_bars[t_int - 1])
            mean = math.sqrt(ab_prev) * x0 + math.sqrt(1.0 - ab_prev) * eps
            x = mean + math.sqrt(beta) * torch.randn_like(x) * 0.0  # deterministic (DDIM-like)
        else:
            x = x0
    return x


# =============================================================================
# Training
# =============================================================================

def train_diffusion(model, train_loader, val_loader, sched: NoiseSchedule, cfg: Dict) -> Dict:
    device = cfg["device"]
    epochs = int(cfg.get("epochs", 20))
    grad_clip = float(cfg.get("grad_clip", 5.0))
    torch.manual_seed(int(cfg.get("seed", 42)))
    model = model.to(device)
    sched.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)),
                           weight_decay=float(cfg.get("weight_decay", 1e-5)))
    loss_fn = nn.MSELoss()

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        tl, n = 0.0, 0
        for (x,) in train_loader:
            x = x.to(device)
            t = torch.randint(0, sched.num_steps, (x.shape[0],), device=device)
            eps = torch.randn_like(x)
            ab = sched.alpha_bars[t][:, None, None]
            x_t = torch.sqrt(ab) * x + torch.sqrt(1.0 - ab) * eps
            loss = loss_fn(model(x_t, t), eps)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tl += loss.item() * len(x)
            n += len(x)
        train_loss = tl / max(n, 1)

        model.eval()
        gen = torch.Generator(device=device).manual_seed(1234 + epoch)
        vl, m = 0.0, 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                t = torch.randint(0, sched.num_steps, (x.shape[0],), device=device, generator=gen)
                eps = torch.randn(x.shape, device=device, generator=gen)
                ab = sched.alpha_bars[t][:, None, None]
                x_t = torch.sqrt(ab) * x + torch.sqrt(1.0 - ab) * eps
                vl += loss_fn(model(x_t, t), eps).item() * len(x)
                m += len(x)
        val_loss = vl / max(m, 1)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"    epoch {epoch + 1:>3}/{epochs}: train {train_loss:.5f}  val {val_loss:.5f}",
              flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    return history


# =============================================================================
# Batched single-step denoise over a window array (numpy in/out)
# =============================================================================

def denoise_windows(model, windows: np.ndarray, denoise_t: int, sched: NoiseSchedule,
                    batch_size: int, device: str) -> np.ndarray:
    """Single-step-denoise a stack of (already-noisy) windows; returns x0_hat."""
    model = model.to(device).eval()
    sched.to(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(
                windows[start:start + batch_size], dtype=np.float32)).to(device)
            x0, _ = single_step_denoise(model, x, denoise_t, sched)
            out.append(x0.cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,) + windows.shape[1:], dtype=np.float32)


def save_diffusion(path: str, model, metadata: Dict) -> None:
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_diffusion(path: str, cfg: Dict, device: str):
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = {**payload.get("metadata", {}), **cfg}
    model = TemporalDiffusionDenoiser(int(meta["input_dim"]), int(meta["window_len"]),
                                      int(meta["hidden_dim"]), int(meta["num_blocks"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
