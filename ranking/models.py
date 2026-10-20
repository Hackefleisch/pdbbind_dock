"""
ranking.models
--------------
The three ranking approaches compared with the unmodified interface delta, and their training.

``Reweight`` — the physics-side baseline, not ML
    score = −Σ_t w_t x_t over the **raw** interface terms with **w_t ≥ 0** (projected gradient:
    clamp at zero after every step). Two starts, reported as two baselines: ``init="random"``
    (weights drawn uniformly from [0, 1]) and ``init="rosetta"`` (the default weights, so the
    untrained model *is* the interface-delta ranking). All seven weights are free (a ranking is
    invariant to their overall scale, so the learned set is reported normalised to
    ``fa_atr`` = 0.8 as well); ``gauge="fa_atr"`` optionally pins that weight instead. The learned
    weights are Rosetta weights and are written as ``weights.wts``. Trained with the listwise
    InfoNCE term only.
``Ranker`` — Luca Friebe's MLP (``n → 64 → 32 → 16 → 1``, ReLU) on his robust-scaled features,
    trained with his BCE + InfoNCE loss (``rosettaligand-vs-posebuster``, branch
    ``test_new_architecture``).
``TreeRanker`` — LightGBM LambdaRank over the raw interface terms, query groups = docking runs,
    early stopping on the validation NDCG@1 (= Success@1 for binary labels).

Loss (``BCEPlusInfoNCE``)
    ``bce_weight · BCEWithLogits(score, hit)`` over the poses of the batch plus ``beta`` × the mean
    over the runs of ``logsumexp(s/τ) − logsumexp(s_hits/τ)`` — a softmax cross-entropy that
    has to put the probability mass of a run on its hits (Weng 2021, *Contrastive representation
    learning*). Runs without a hit contribute to the BCE term only. The InfoNCE term compares
    poses within their run only.
Training (:func:`fit`)
    Adam, one docking run per optimiser step by default, a fixed number of epochs; after every
    epoch the validation Success@1 / Success@5 are computed and the best epoch is kept
    (Success@1, ties by Success@5; the untrained model is epoch 0 and eligible).
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

from .data import FEATURE_COLS, Grouped, rosetta_weights, term_name
from .evaluate import quick_rates

MODELS = ("reweight", "ranker", "tree")
INITS = ("random", "rosetta")                 # the two reweighting baselines
DEFAULT_LR = {"reweight": 1e-4, "ranker": 1e-5}
DEFAULT_BCE_WEIGHT = {"reweight": 0.0, "ranker": 1.0}
DEFAULT_SCALING = {"reweight": "none", "ranker": "robust", "tree": "none"}


# ---------------------------------------------------------------------------
# torch models
# ---------------------------------------------------------------------------
class Reweight(nn.Module):
    """Non-negative reweighting of the raw interface terms (see module doc)."""

    def __init__(self, feature_cols: list[str] = FEATURE_COLS, init: str = "random", gauge: str | None = None, seed: int = 0):
        super().__init__()
        self.feature_cols = list(feature_cols)
        names = [term_name(c) for c in self.feature_cols]
        self.gauge_idx = names.index(gauge) if gauge not in (None, "none") else None
        defaults = rosetta_weights(self.feature_cols)
        if init == "rosetta":
            w0 = defaults
        elif init == "random":
            w0 = np.random.default_rng(seed).uniform(0.0, 1.0, size=len(self.feature_cols))
            if self.gauge_idx is not None:
                w0[self.gauge_idx] = defaults[self.gauge_idx]
        else:
            raise ValueError(f"unknown init {init!r} (random | rosetta)")
        self.init = init
        self.w = nn.Parameter(torch.as_tensor(w0, dtype=torch.float32))
        self.register_buffer("w_init", torch.as_tensor(w0, dtype=torch.float32))

    def forward(self, x):
        return -(x * self.w).sum(-1, keepdim=True)

    @torch.no_grad()
    def project(self) -> None:
        """Keep the weights non-negative and the gauge weight at its initial value."""
        self.w.clamp_(min=0.0)
        if self.gauge_idx is not None:
            self.w[self.gauge_idx] = self.w_init[self.gauge_idx]

    def weights(self) -> np.ndarray:
        return self.w.detach().cpu().numpy().astype(np.float64)

    def weights_normalised(self) -> np.ndarray:
        """The learned weights scaled so that fa_atr = 0.8 (the ranking is unchanged); as learned if fa_atr is 0."""
        w = self.weights()
        names = [term_name(c) for c in self.feature_cols]
        if "fa_atr" in names and w[names.index("fa_atr")] > 0:
            return w * (0.8 / w[names.index("fa_atr")])
        return w

    def weights_table(self) -> pd.DataFrame:
        return pd.DataFrame({"term": [term_name(c) for c in self.feature_cols], "weight": self.weights(),
                             "weight_fa_atr_0.8": self.weights_normalised(), "rosetta_default": rosetta_weights(self.feature_cols)})

    def rosetta_wts(self) -> str:
        """The learned weights, normalised to fa_atr = 0.8, in the format of a Rosetta ``.wts`` file."""
        return "".join(f"{term_name(c)} {w:.4f}\n" for c, w in zip(self.feature_cols, self.weights_normalised()))


class Ranker(nn.Module):
    """Luca's fully connected ranking net: n -> 64 -> 32 -> 16 -> 1."""

    def __init__(self, n_features: int = len(FEATURE_COLS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
class BCEPlusInfoNCE(nn.Module):
    """``bce_weight · BCEWithLogits`` over the batch + ``beta`` · mean InfoNCE over its runs."""

    def __init__(self, beta: float = 1.0, tau: float = 0.5, bce_weight: float = 1.0):
        super().__init__()
        self.beta, self.tau, self.bce_weight = beta, tau, bce_weight

    def forward(self, scores: torch.Tensor, hit: torch.Tensor, seg: np.ndarray) -> tuple[torch.Tensor, float, float]:
        bce = F.binary_cross_entropy_with_logits(scores, hit) if self.bce_weight else scores.new_zeros(())
        terms = []
        for a, b in zip(seg[:-1], seg[1:]):
            h = hit[a:b] > 0.5
            if not bool(h.any()):                      # InfoNCE undefined without a positive
                continue
            s = scores[a:b] / self.tau
            terms.append(torch.logsumexp(s, 0) - torch.logsumexp(s[h], 0))
        nce = torch.stack(terms).mean() if terms else scores.new_zeros(())
        return self.bce_weight * bce + self.beta * nce, float(bce.detach()), float(nce.detach())


# ---------------------------------------------------------------------------
# training of the torch models
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
        beta: float = 1.0, tau: float = 0.5, bce_weight: float = 1.0, batch_runs: int = 1,
        patience: int | None = None, seed: int = 0, device: torch.device | None = None, log=print) -> FitResult:
    """Train *model* on *train*, select the epoch by validation Success@1 then Success@5."""
    device = device or select_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    loss_fn = BCEPlusInfoNCE(beta, tau, bce_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    project = getattr(model, "project", None)

    X = torch.from_numpy(train.X).to(device)
    y = torch.from_numpy(train.hit.astype(np.float32)).to(device)
    offsets = train.offsets
    n_groups = train.n_groups

    def snapshot() -> dict:
        return copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    def validate() -> dict[str, float]:
        return quick_rates(predict(model, val.X, device), val, ks=(1, 5))

    history = []
    m = validate()
    best, best_epoch, best_state = (m["top1"], m["top5"]), 0, snapshot()
    history.append({"epoch": 0, "loss": np.nan, "bce": np.nan, "nce": np.nan, "seconds": 0.0, **{f"val_{k}": v for k, v in m.items()}})
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
            if not loss.requires_grad:                 # bucket(s) without a hit and no BCE term: nothing to learn from
                continue
            loss.backward()
            opt.step()
            if project is not None:
                project()
            sums += (float(loss), bce, nce)
            steps += 1
        m = validate()
        selected = (m["top1"] > best[0]) or (m["top1"] == best[0] and m["top5"] > best[1])
        if selected:
            best, best_epoch, since_best, best_state = (m["top1"], m["top5"]), epoch, 0, snapshot()
        else:
            since_best += 1
        row = {"epoch": epoch, "loss": sums[0] / max(steps, 1), "bce": sums[1] / max(steps, 1), "nce": sums[2] / max(steps, 1),
               "seconds": time.time() - t0, **{f"val_{k}": v for k, v in m.items()}}
        history.append(row)
        extra = ""
        if isinstance(model, Reweight):
            extra = "  w=" + " ".join(f"{v:.2f}" for v in model.weights_normalised())
        log(f"epoch {epoch:3d} | loss {row['loss']:.4f} (bce {row['bce']:.4f} nce {row['nce']:.4f}) | "
            f"val Success@1 {m['top1']:.2f}  Success@5 {m['top5']:.2f}  AUC {m['auc_pooled']:.4f} | "
            f"{row['seconds']:.0f}s{'  *' if selected else ''}{extra}")
        if patience is not None and since_best >= patience:
            log(f"no validation improvement for {patience} epochs — stopping")
            break
    hist = pd.DataFrame(history)
    hist["selected"] = hist["epoch"] == best_epoch
    model.load_state_dict(best_state)
    model.to(device)
    return FitResult(hist, best_epoch, best_state)


# ---------------------------------------------------------------------------
# tree baseline
# ---------------------------------------------------------------------------
class TreeRanker:
    """LightGBM LambdaRank on the raw interface terms; query groups = docking runs."""

    def __init__(self, n_estimators: int = 2000, learning_rate: float = 0.05, num_leaves: int = 31,
                 min_child_samples: int = 100, early_stopping_rounds: int = 50, seed: int = 0, n_jobs: int = 16):
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                           min_child_samples=min_child_samples, early_stopping_rounds=early_stopping_rounds,
                           seed=seed, n_jobs=n_jobs)
        self.model = None
        self.history: pd.DataFrame | None = None
        self.best_iteration: int | None = None

    def fit(self, train: Grouped, val: Grouped, log=print) -> "TreeRanker":
        import lightgbm as lgb
        p = self.params
        self.model = lgb.LGBMRanker(objective="lambdarank", n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                    num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
                                    random_state=p["seed"], n_jobs=p["n_jobs"], verbose=-1)
        self.model.fit(train.X, train.hit.astype(int), group=train.group_sizes,
                       eval_set=[(val.X, val.hit.astype(int))], eval_group=[val.group_sizes], eval_at=[1, 5, 10],
                       eval_metric="ndcg",
                       callbacks=[lgb.early_stopping(p["early_stopping_rounds"], first_metric_only=True, verbose=False),
                                  lgb.log_evaluation(100)])
        ev = self.model.evals_result_["valid_0"]
        self.history = pd.DataFrame({"iteration": np.arange(1, len(ev["ndcg@1"]) + 1),
                                     **{f"val_ndcg@{k}": ev[f"ndcg@{k}"] for k in (1, 5, 10)}})
        self.best_iteration = int(self.model.best_iteration_ or len(self.history))
        log(f"tree: {self.best_iteration} trees selected (val NDCG@1 {self.history['val_ndcg@1'].iloc[self.best_iteration - 1]:.4f})")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, num_iteration=self.best_iteration).astype(np.float64)

    def importance(self) -> pd.DataFrame:
        return pd.DataFrame({"term": [term_name(c) for c in FEATURE_COLS],
                             "gain": self.model.booster_.feature_importance(importance_type="gain"),
                             "split": self.model.booster_.feature_importance(importance_type="split")})

    def save(self, path) -> None:
        self.model.booster_.save_model(str(path), num_iteration=self.best_iteration)
