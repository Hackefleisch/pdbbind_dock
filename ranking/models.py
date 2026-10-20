"""
ranking.models
--------------
The two pose-ranking models of Luca Friebe's ``rosettaligand-vs-posebuster`` (branch
``test_new_architecture``: ``reweighting/reweight.py`` and ``ranking/ranking.py``), their loss
and the training loop, ported to run on :class:`ranking.data.Grouped` tables.

Models (score per pose, **higher is better**)
    ``Reweight``  a bias-free linear layer over the scaled features: a learned reweighting of
                  the energy terms. With ``init_rosetta`` it starts at the Rosetta weights
                  (mapped into the scaled feature space), i.e. at the interface-delta ranking.
    ``Ranker``    Luca's fully connected net ``n -> 64 -> 32 -> 16 -> 1`` with ReLU.
Loss (``BCEPlusInfoNCE``)
    binary cross-entropy of the score against the hit label (every pose) plus ``beta`` times an
    InfoNCE term per docking run: ``-(logsumexp(s_hits / tau) - logsumexp(s_all / tau))``, i.e.
    a softmax cross-entropy that has to put the probability mass on the hits of the run. Runs
    without a hit contribute to the BCE term only (as in Luca's code).
    Reference: Weng, L. (2021) *Contrastive representation learning*, Lil'Log.
Training (:func:`fit`)
    Adam, one docking run per batch (``batch_runs`` runs per step pool several), a fixed number
    of epochs; after every epoch the validation Success@1 and Success@5 are computed and the
    model of the best epoch (Success@1, ties by Success@5) is kept — Luca's selection rule.
    The untrained model is evaluated as epoch 0 and can be selected (relevant for the Rosetta
    initialisation: training then has to beat the baseline on the validation set).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Grouped, RobustScale, rosetta_weights
from .evaluate import quick_rates


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class Reweight(nn.Module):
    """Bias-free linear reweighting of the (scaled) energy terms."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=False)

    def forward(self, x):
        return self.linear(x)

    def weights(self) -> np.ndarray:
        return self.linear.weight.detach().cpu().numpy().ravel().astype(np.float64)


class Ranker(nn.Module):
    """Luca's fully connected ranking net: n -> 64 -> 32 -> 16 -> 1."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


MODELS = {"reweight": Reweight, "ranker": Ranker}
DEFAULT_LR = {"reweight": 1e-4, "ranker": 1e-5}          # Luca's learning rates


def make_model(kind: str, n_features: int) -> nn.Module:
    return MODELS[kind](n_features)


@torch.no_grad()
def init_rosetta(model: Reweight, feature_cols: list[str], scaler: RobustScale) -> None:
    """Start the reweighting at the Rosetta weights: score = -Σ w_t x_t with x = scale·z + center,
    so the weight on the scaled feature z_t is -w_t·scale_t (the sign flips because the model
    scores higher-is-better). Exact up to the clipping of the features."""
    w = -rosetta_weights(feature_cols) * scaler.scale
    model.linear.weight.copy_(torch.as_tensor(w, dtype=torch.float32).view(1, -1))


def weights_in_raw_units(model: Reweight, scaler: RobustScale, feature_cols: list[str]) -> pd.DataFrame:
    """The learned weights as (a) weights on the scaled features, (b) weights on the raw energy
    terms (score = Σ w_raw·x, higher is better) and (c) the latter flipped to Rosetta's
    lower-is-better sign and normalised to fa_atr = 0.8 for comparison with the defaults."""
    w_scaled = model.weights()
    w_raw = w_scaled / scaler.scale
    rosetta = rosetta_weights(feature_cols)
    w_ros = -w_raw
    atr = [i for i, c in enumerate(feature_cols) if c.endswith("fa_atr")]
    norm = w_ros * (0.8 / w_ros[atr[0]]) if atr and w_ros[atr[0]] != 0 else np.full_like(w_ros, np.nan)
    return pd.DataFrame({"feature": feature_cols, "w_scaled": w_scaled, "w_raw": w_raw,
                         "w_rosetta_sign": w_ros, "w_normalised_fa_atr_0.8": norm, "rosetta_default": rosetta})


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
class BCEPlusInfoNCE(nn.Module):
    """Luca's loss: BCEWithLogits over all poses of the batch + beta · mean InfoNCE over the runs."""

    def __init__(self, beta: float = 1.0, tau: float = 0.5):
        super().__init__()
        self.beta, self.tau = beta, tau

    def forward(self, scores: torch.Tensor, hit: torch.Tensor, seg: np.ndarray) -> tuple[torch.Tensor, float, float]:
        bce = F.binary_cross_entropy_with_logits(scores, hit)
        terms = []
        for a, b in zip(seg[:-1], seg[1:]):
            h = hit[a:b] > 0.5
            if not bool(h.any()):                      # InfoNCE undefined without a positive
                continue
            s = scores[a:b] / self.tau
            terms.append(torch.logsumexp(s, 0) - torch.logsumexp(s[h], 0))
        nce = torch.stack(terms).mean() if terms else scores.new_zeros(())
        return bce + self.beta * nce, float(bce.detach()), float(nce.detach())


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    history: pd.DataFrame          # one row per epoch: loss terms, validation metrics, selected flag
    best_epoch: int
    best_state: dict               # state_dict of the selected epoch (CPU tensors)


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device, batch: int = 1 << 18) -> np.ndarray:
    """Per-pose scores (higher is better) as float64."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(np.ascontiguousarray(X[i:i + batch], dtype=np.float32)).to(device)
        out.append(model(x).squeeze(-1).double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fit(model: nn.Module, train: Grouped, val: Grouped, *, epochs: int = 50, lr: float = 1e-4,
        beta: float = 1.0, tau: float = 0.5, batch_runs: int = 1, patience: int | None = None,
        seed: int = 0, device: torch.device | None = None, log=print) -> FitResult:
    """Train *model* on *train*, select the epoch by validation Success@1 then Success@5."""
    device = device or select_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    loss_fn = BCEPlusInfoNCE(beta, tau)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.from_numpy(train.X).to(device)
    y = torch.from_numpy(train.hit.astype(np.float32)).to(device)
    offsets = train.offsets
    n_groups = train.n_groups

    def validate() -> dict[str, float]:
        return quick_rates(predict(model, val.X, device), val, ks=(1, 5))

    history = []
    best = (-1.0, -1.0)
    best_epoch, best_state = 0, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    m = validate()
    history.append({"epoch": 0, "loss": np.nan, "bce": np.nan, "nce": np.nan, "seconds": 0.0, **{f"val_{k}": v for k, v in m.items()}})
    best = (m["top1"], m["top5"])
    log(f"epoch   0 | untrained | val Success@1 {m['top1']:.2f}  Success@5 {m['top5']:.2f}  AUC {m['auc_pooled']:.4f}")
    since_best = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(n_groups)
        sums = np.zeros(3)
        steps = 0
        for i in range(0, n_groups, batch_runs):
            gs = order[i:i + batch_runs]
            if len(gs) == 1:
                a, b = int(offsets[gs[0]]), int(offsets[gs[0] + 1])
                x, t, seg = X[a:b], y[a:b], np.array([0, b - a])
            else:
                lengths = offsets[gs + 1] - offsets[gs]
                idx = np.concatenate([np.arange(offsets[g], offsets[g + 1]) for g in gs])
                idx_t = torch.from_numpy(idx).to(device)
                x, t, seg = X[idx_t], y[idx_t], np.concatenate([[0], np.cumsum(lengths)])
            opt.zero_grad(set_to_none=True)
            scores = model(x).squeeze(-1)
            loss, bce, nce = loss_fn(scores, t, seg)
            loss.backward()
            opt.step()
            sums += (float(loss), bce, nce)
            steps += 1
        m = validate()
        selected = (m["top1"] > best[0]) or (m["top1"] == best[0] and m["top5"] > best[1])
        if selected:
            best, best_epoch, since_best = (m["top1"], m["top5"]), epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            since_best += 1
        row = {"epoch": epoch, "loss": sums[0] / max(steps, 1), "bce": sums[1] / max(steps, 1), "nce": sums[2] / max(steps, 1),
               "seconds": time.time() - t0, **{f"val_{k}": v for k, v in m.items()}}
        history.append(row)
        log(f"epoch {epoch:3d} | loss {row['loss']:.4f} (bce {row['bce']:.4f} nce {row['nce']:.4f}) | "
            f"val Success@1 {m['top1']:.2f}  Success@5 {m['top5']:.2f}  AUC {m['auc_pooled']:.4f} | "
            f"{row['seconds']:.0f}s{'  *' if selected else ''}")
        if patience is not None and since_best >= patience:
            log(f"no validation improvement for {patience} epochs — stopping")
            break
    hist = pd.DataFrame(history)
    hist["selected"] = hist["epoch"] == best_epoch
    model.load_state_dict(best_state)
    model.to(device)
    return FitResult(hist, best_epoch, best_state)
