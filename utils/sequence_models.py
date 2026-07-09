"""Stage 12: sequence-prior autoencoder models (PyTorch).

Three deliberately simple model families over normalized trajectory
windows (B x T x F):

  * mlp_dae -- flatten to B x (T*F), 2-layer MLP encoder/decoder;
  * gru_ae  -- GRU encoder (final hidden state -> latent), GRU decoder
    driven by the latent repeated over T;
  * tcn_ae  -- 1D convolutional encoder/decoder over the time axis.

All three are trained as DENOISING autoencoders (Gaussian noise on the
input, MSE against the clean window). This is stage 12's v1 -- not a VAE
(stage 13), not diffusion (stage 14), not the full model zoo.
"""

from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    raise SystemExit("Stage 12 requires PyTorch. Please install torch before running this stage.")

MODEL_NAMES = ["mlp_dae", "gru_ae", "tcn_ae"]


class MlpDenoisingAE(nn.Module):
    def __init__(self, input_dim: int, window_len: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        flat = input_dim * window_len
        self.window_len = window_len
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(flat, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, flat),
        )

    def forward(self, x):                         # x: B x T x F
        b = x.shape[0]
        z = self.encoder(x.reshape(b, -1))
        return self.decoder(z).reshape(b, self.window_len, self.input_dim)


class GruAE(nn.Module):
    def __init__(self, input_dim: int, window_len: int, hidden_dim: int,
                 latent_dim: int, num_layers: int):
        super().__init__()
        self.window_len = window_len
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):                         # x: B x T x F
        _, h = self.encoder(x)
        z = self.to_latent(h[-1])                 # B x latent
        z_seq = z.unsqueeze(1).expand(-1, self.window_len, -1)
        out, _ = self.decoder(z_seq)
        return self.head(out)


class TcnAE(nn.Module):
    def __init__(self, input_dim: int, window_len: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden_dim, latent_dim, 5, padding=2), nn.ReLU(),
            nn.Conv1d(latent_dim, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden_dim, input_dim, 5, padding=2),
        )

    def forward(self, x):                         # x: B x T x F
        return self.net(x.transpose(1, 2)).transpose(1, 2)


def get_model(model_name: str, input_dim: int, window_len: int, cfg: Dict) -> nn.Module:
    hidden = int(cfg.get("hidden_dim", 128))
    latent = int(cfg.get("latent_dim", 32))
    layers = int(cfg.get("num_layers", 2))
    if model_name == "mlp_dae":
        return MlpDenoisingAE(input_dim, window_len, hidden, latent)
    if model_name == "gru_ae":
        return GruAE(input_dim, window_len, hidden, latent, layers)
    if model_name == "tcn_ae":
        return TcnAE(input_dim, window_len, hidden, latent)
    raise ValueError(f"Unknown model '{model_name}' (choices: {MODEL_NAMES})")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def make_loader(windows: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_autoencoder(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                      cfg: Dict) -> Dict:
    """Denoising-MSE training with per-epoch history and best-val restore."""
    device = cfg["device"]
    epochs = int(cfg.get("epochs", 20))
    noise_std = float(cfg.get("noise_std", 0.05))
    torch.manual_seed(int(cfg.get("seed", 42)))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)),
                           weight_decay=float(cfg.get("weight_decay", 1e-5)))
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        for (x,) in train_loader:
            x = x.to(device)
            noisy = x + noise_std * torch.randn_like(x)
            loss = loss_fn(model(noisy), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(x)
            count += len(x)
        train_loss = total / max(count, 1)

        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                total += float(loss_fn(model(x), x)) * len(x)
                count += len(x)
        val_loss = total / max(count, 1)
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


def reconstruction_errors(model: nn.Module, windows: np.ndarray, batch_size: int,
                          device: str) -> np.ndarray:
    """Per-window mean-squared reconstruction error (clean input, no noise)."""
    model = model.to(device).eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(
                windows[start:start + batch_size], dtype=np.float32)).to(device)
            err = ((model(x) - x) ** 2).mean(dim=(1, 2))
            errors.append(err.cpu().numpy())
    return np.concatenate(errors) if errors else np.array([])


def save_model(path: str, model: nn.Module, metadata: Dict) -> None:
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_model(path: str, model_name: str, input_dim: int, window_len: int,
               device: str, cfg: Optional[Dict] = None) -> nn.Module:
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = payload.get("metadata", {})
    model = get_model(model_name, input_dim, window_len, cfg or meta)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
