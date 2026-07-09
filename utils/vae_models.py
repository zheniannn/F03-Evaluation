"""Stage 13: sequence VAE over normalized trajectory windows (PyTorch).

A deliberately simple fully-connected variational autoencoder on flattened
windows (B x T x F -> B x (T*F)). The encoder produces a diagonal-Gaussian
posterior (mu, logvar); the decoder reconstructs the window from a
reparameterized sample. Training minimizes reconstruction MSE plus a
KL-to-standard-normal term with beta weighting and KL annealing.

This is stage 13's probabilistic counterpart to the stage-12 deterministic
autoencoders -- NOT diffusion (stage 14) and NOT the full model zoo. It
reuses the stage-12 window pipeline and device/loader helpers so stage 12
and stage 13 stay directly comparable.
"""

from typing import Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    raise SystemExit("Stage 13 requires PyTorch. Please install torch before running this stage.")

# reuse device / loader helpers so stage 12 and 13 batch identically
from utils.sequence_models import resolve_device, make_loader  # noqa: F401


class SequenceVAE(nn.Module):
    def __init__(self, input_dim: int, window_len: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.window_len = window_len
        self.latent_dim = latent_dim
        flat = input_dim * window_len
        self.enc = nn.Sequential(
            nn.Linear(flat, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, flat),
        )

    def encode(self, x):
        h = self.enc(x.reshape(x.shape[0], -1))
        # clamp logvar so exp(logvar) cannot overflow and destabilize training
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10.0, 10.0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z).reshape(z.shape[0], self.window_len, self.input_dim)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta_eff: float):
    """Total loss, plus the recon and KL parts (all scalar tensors).

    recon_loss = per-sample mean squared error
    kl_loss    = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
    """
    recon_loss = ((recon - x) ** 2).mean()
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta_eff * kl_loss, recon_loss, kl_loss


def train_vae(model: nn.Module, train_loader, val_loader, cfg: Dict) -> Dict:
    """Train the VAE with beta KL annealing; restore best-val-loss weights."""
    device = cfg["device"]
    epochs = int(cfg.get("epochs", 20))
    beta = float(cfg.get("beta", 1e-3))
    anneal = max(int(cfg.get("kl_anneal_epochs", 5)), 1)
    grad_clip = float(cfg.get("grad_clip", 5.0))
    torch.manual_seed(int(cfg.get("seed", 42)))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)),
                           weight_decay=float(cfg.get("weight_decay", 1e-5)))

    history = {k: [] for k in ("train_loss", "val_loss", "train_recon", "val_recon",
                               "train_kl", "val_kl", "beta_eff")}
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        beta_eff = beta * min(1.0, epoch / anneal)
        model.train()
        tl = tr = tk = 0.0
        n = 0
        for (x,) in train_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, rl, kl = vae_loss(recon, x, mu, logvar, beta_eff)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            bs = len(x)
            tl += loss.item() * bs
            tr += rl.item() * bs
            tk += kl.item() * bs
            n += bs
        train_loss, train_recon, train_kl = tl / max(n, 1), tr / max(n, 1), tk / max(n, 1)

        model.eval()
        vl = vr = vk = 0.0
        m = 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss, rl, kl = vae_loss(recon, x, mu, logvar, beta_eff)
                bs = len(x)
                vl += loss.item() * bs
                vr += rl.item() * bs
                vk += kl.item() * bs
                m += bs
        val_loss, val_recon, val_kl = vl / max(m, 1), vr / max(m, 1), vk / max(m, 1)

        for k, v in (("train_loss", train_loss), ("val_loss", val_loss),
                     ("train_recon", train_recon), ("val_recon", val_recon),
                     ("train_kl", train_kl), ("val_kl", val_kl), ("beta_eff", beta_eff)):
            history[k].append(v)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"    epoch {epoch + 1:>3}/{epochs}: loss {train_loss:.5f}/{val_loss:.5f} "
              f"recon {train_recon:.5f}/{val_recon:.5f} kl {train_kl:.5f}/{val_kl:.5f} "
              f"(beta_eff {beta_eff:.5g})", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    return history


def vae_reconstruct(model: nn.Module, x):
    """Deterministic reconstruction from the posterior mean (no sampling)."""
    mu, logvar = model.encode(x)
    return model.decode(mu), mu, logvar


def vae_encode(model: nn.Module, x):
    return model.encode(x)


def vae_window_metrics(model: nn.Module, windows: np.ndarray, batch_size: int,
                       device: str) -> Dict[str, np.ndarray]:
    """Per-window reconstruction error, KL divergence, and latent mean vector.

    recon_error : mean squared error of the mean reconstruction (deterministic)
    kl          : per-sample KL to N(0, I), summed over latent dims (nats, >= 0)
    mu          : posterior mean vector per window (N, latent_dim)
    """
    model = model.to(device).eval()
    recon_errs, kls, mus = [], [], []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(
                windows[start:start + batch_size], dtype=np.float32)).to(device)
            recon, mu, logvar = vae_reconstruct(model, x)
            recon_errs.append(((recon - x) ** 2).mean(dim=(1, 2)).cpu().numpy())
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
            kls.append(kl.cpu().numpy())
            mus.append(mu.cpu().numpy())
    if not recon_errs:
        return {"recon_error": np.array([]), "kl": np.array([]),
                "mu": np.empty((0, model.latent_dim))}
    return {"recon_error": np.concatenate(recon_errs),
            "kl": np.concatenate(kls),
            "mu": np.concatenate(mus)}


def save_vae(path: str, model: nn.Module, metadata: Dict) -> None:
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_vae(path: str, cfg: Dict, device: str) -> nn.Module:
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = {**payload.get("metadata", {}), **cfg}
    model = SequenceVAE(int(meta["input_dim"]), int(meta["window_len"]),
                        int(meta["hidden_dim"]), int(meta["latent_dim"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
