"""
affinity.models
---------------
The linear models compared on the per-complex feature vectors (Track 3). Every model maps a
feature vector to a predicted pK and is fitted by least squares on the training complexes:
there is no hyper-parameter and nothing random, the bootstrap over complexes gives the
intervals (``affinity.evaluate``).

``unmodified``       pK ≈ a · idelta_score + b — the interface delta, linearly calibrated. The
                     reference: Pearson r is invariant to a and b, the RMSE needs them.
``size``             pK ≈ a · ligand_n_heavy + b — the ligand-size reference (affinity grows with
                     ligand size in PDBbind, and so does every contact term).
``reweight``         pK ≈ b − Σ_t w_t x_t over the seven raw interface terms, w free
                     (ordinary least squares).
``reweight_nonneg``  the same with w_t ≥ 0 (non-negative least squares on the centred data), so
                     the learned vector is a Rosetta weight set; written normalised to
                     fa_atr = 0.8 as ``weights.wts``.
``reweight_size``    ``reweight`` plus ligand_n_heavy — does the energy add to the size?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from ranking.data import FEATURE_COLS, rosetta_weights, term_name

APPROACHES = ("unmodified", "size", "reweight", "reweight_nonneg", "reweight_size")
REFERENCE = "unmodified"
APPROACH_LABELS = {"unmodified": "unmodified (interface delta, calibrated)",
                   "size": "ligand size (heavy atoms)",
                   "reweight": "reweighted (free sign)",
                   "reweight_nonneg": "reweighted (non-negative)",
                   "reweight_size": "reweighted (free sign) + ligand size"}
APPROACH_FEATURES: dict[str, list[str]] = {"unmodified": ["idelta_score"], "size": ["ligand_n_heavy"],
                                           "reweight": list(FEATURE_COLS), "reweight_nonneg": list(FEATURE_COLS),
                                           "reweight_size": list(FEATURE_COLS) + ["ligand_n_heavy"]}
NONNEG = {"reweight_nonneg"}


@dataclass
class Linear:
    """pK = X @ coef + intercept. For the interface terms the energy weight is −coef
    (pK = b − Σ w x: a positive w means that a more negative term favours binding)."""
    approach: str
    names: list[str]
    coef: np.ndarray
    intercept: float
    n_train: int = 0
    extra: dict = field(default_factory=dict)

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef + self.intercept

    def energy_weights(self) -> np.ndarray:
        """−coef of the interface terms (NaN for other features)."""
        return np.array([-c if n in FEATURE_COLS else np.nan for n, c in zip(self.names, self.coef)])

    def weights_normalised(self) -> np.ndarray:
        """The energy weights scaled so that fa_atr = 0.8 (as learned if fa_atr is 0 or absent)."""
        w = self.energy_weights()
        if "raw_delta_fa_atr" in self.names:
            a = w[self.names.index("raw_delta_fa_atr")]
            if np.isfinite(a) and a != 0:
                return w * (0.8 / a)
        return w

    def weights_table(self) -> pd.DataFrame:
        defaults = dict(zip([term_name(c) for c in FEATURE_COLS], rosetta_weights()))
        return pd.DataFrame({"feature": [term_name(n) if n in FEATURE_COLS else n for n in self.names],
                             "coef_pk_per_unit": self.coef, "energy_weight": self.energy_weights(),
                             "weight_fa_atr_0.8": self.weights_normalised(),
                             "rosetta_default": [defaults.get(term_name(n), np.nan) if n in FEATURE_COLS else np.nan for n in self.names],
                             "intercept": self.intercept})

    def rosetta_wts(self) -> str:
        return "".join(f"{term_name(n)} {w:.4f}\n" for n, w in zip(self.names, self.weights_normalised()) if n in FEATURE_COLS)

    def to_dict(self) -> dict:
        return {"approach": self.approach, "names": self.names, "coef": self.coef.tolist(), "intercept": float(self.intercept),
                "n_train": self.n_train, **self.extra}


def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    A = np.column_stack([X, np.ones(len(X))])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return sol[:-1], float(sol[-1])


def fit_nnls(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """pK = b − Σ w x with w ≥ 0: NNLS on the centred data, the intercept follows from the means."""
    xm, ym = X.mean(axis=0), y.mean()
    w, _ = nnls(-(X - xm), y - ym)
    coef = -w
    return coef, float(ym - xm @ coef)


def fit(approach: str, frame: pd.DataFrame, y) -> Linear:
    """Fit one approach on the training complexes (*frame* holds the feature columns)."""
    names = APPROACH_FEATURES[approach]
    X = frame[names].to_numpy(dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    coef, intercept = (fit_nnls if approach in NONNEG else fit_ols)(X, y)
    return Linear(approach, list(names), coef, intercept, n_train=len(y))
