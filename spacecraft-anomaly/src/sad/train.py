"""Training one LSTM per channel on nominal (train) telemetry, and forecasting the test set."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .data import make_input_windows, make_windows
from .model import LSTMForecaster


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(n_features: int, cfg: Config) -> LSTMForecaster:
    return LSTMForecaster(n_features, cfg.hidden_size, cfg.num_layers, cfg.dropout, cfg.n_predictions)


def train_channel(train: np.ndarray, cfg: Config, device: torch.device | None = None,
                  seed: int = 0, verbose: bool = False) -> tuple[LSTMForecaster, dict]:
    """Fit on train windows; the last `validation_split` of windows (chronological) is used
    for early stopping. Returns the best model (lowest validation MSE) and a history dict."""
    set_seed(seed)
    device = device or default_device()
    X, y = make_windows(train, cfg.l_s, cfg.n_predictions)
    n_val = max(1, int(len(X) * cfg.validation_split))
    Xtr, ytr, Xva, yva = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]
    if len(Xtr) == 0:
        raise ValueError("not enough training windows")

    gen = torch.Generator().manual_seed(seed)
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                           batch_size=cfg.batch_size, shuffle=True, generator=gen)
    va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                           batch_size=512)

    model = build_model(X.shape[2], cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.epochs):
        model.train()
        tot, cnt = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            cnt += len(xb)
        model.eval()
        vtot, vcnt = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                vtot += loss_fn(model(xb), yb).item() * len(xb)
                vcnt += len(xb)
        tr_loss, va_loss = tot / cnt, vtot / vcnt
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        if verbose:
            print(f"  epoch {epoch + 1:3d}  train {tr_loss:.5f}  val {va_loss:.5f}")
        if va_loss < best - cfg.min_delta:
            best, bad = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best
    history["epochs_run"] = len(history["val_loss"])
    return model, history


@torch.no_grad()
def predict_series(model: LSTMForecaster, series: np.ndarray, cfg: Config,
                   device: torch.device | None = None) -> np.ndarray:
    """Forecast every timestep t >= l_s of `series`.

    Window i covers [i, i+l_s) and predicts timesteps i+l_s ... i+l_s+h-1. Each timestep is
    predicted by up to h windows; those predictions are averaged. Returns y_hat of length
    len(series) - l_s, aligned with series[l_s:, 0].
    """
    device = device or next(model.parameters()).device
    model.eval()
    X = make_input_windows(series, cfg.l_s)
    preds = []
    for i in range(0, len(X), 1024):
        preds.append(model(torch.from_numpy(X[i:i + 1024]).to(device)).cpu().numpy())
    P = np.concatenate(preds)                      # (n, h)
    n, h = P.shape
    total = np.zeros(n)
    count = np.zeros(n)
    for k in range(h):
        total[k:] += P[: n - k, k]
        count[k:] += 1
    return (total / count).astype(np.float32)


def save_model(model: LSTMForecaster, path: str | Path, cfg: Config, n_features: int,
               history: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(),
                "n_features": n_features, "history": history}, path)


def load_model(path: str | Path, cfg: Config, device: torch.device | None = None) -> LSTMForecaster:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["n_features"], cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device or default_device())
