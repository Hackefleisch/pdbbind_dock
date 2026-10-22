"""
affinity.models
---------------
The linear models compared on the per-complex feature vectors (Track 3). Every model maps a
feature vector to a predicted pK; nothing is tuned and nothing is random, so one fit per cell
and the bootstrap over complexes gives the intervals (``affinity.evaluate``).

Approaches (``APPROACHES``)
    ``constant``         no feature: the training mean (``l2``) or median (``l1``) — the null
                         every other row has to beat;
    ``unmodified``       pK ≈ a · idelta_score + b — the interface delta, linearly calibrated
                         (the reference; Pearson r is invariant to a and b, MAE / RMSE need them);
    ``size``             pK ≈ a · ligand_n_heavy + b — the ligand-size reference;
    ``reweight``         pK ≈ b − Σ_t w_t x_t over the raw interface terms, w free;
    ``reweight_nonneg``  the same with w_t ≥ 0, so the vector is a Rosetta weight set
                         (written normalised to fa_atr = 0.8 as ``weights_nonneg_<split>.wts``);
    ``reweight_size``    ``reweight`` plus ligand_n_heavy.
Term sets (``TERM_SETS``, the three reweightings are fitted once per set, full refit)
    ``all``        the seven interface terms;
    ``no_pair``    without ``fa_pair`` (a knowledge-based potential of mean force);
    ``enthalpic``  without ``fa_pair`` and ``fa_sol`` (the implicit-solvent free energy) —
                   ``fa_atr``, ``fa_rep``, ``fa_elec`` and the two H-bond terms only.
Losses (``LOSSES``)
    ``l2`` least squares (OLS, or NNLS on the centred data for the non-negative model);
    ``l1`` median regression (the loss matching the MAE metric): quantile regression at τ = 0.5,
    solved by IRLS (statsmodels) for the free-sign models and as a linear programme for the
    non-negative one.
Lower bound (``LOWER_TAU``)
    quantile regression at τ = 0.05 gives a linear model of the 5th percentile of pK given the
    features, i.e. a 95 % lower bound; the conformal alternative lives in ``affinity.evaluate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog, nnls

from ranking.data import FEATURE_COLS, rosetta_weights, term_name

TERM_SETS: dict[str, list[str]] = {
    "all": list(FEATURE_COLS),
    "no_pair": [c for c in FEATURE_COLS if term_name(c) != "fa_pair"],
    "enthalpic": [c for c in FEATURE_COLS if term_name(c) not in ("fa_pair", "fa_sol")],
}
APPROACHES = ("constant", "unmodified", "size", "reweight", "reweight_nonneg", "reweight_size")
TERM_APPROACHES = ("reweight", "reweight_nonneg", "reweight_size")
NONNEG = {"reweight_nonneg"}
REFERENCE, NULL = "unmodified", "constant"
LOSSES = ("l1", "l2")
LOWER_TAU = 0.05
APPROACH_LABELS = {"constant": "constant (training median / mean)",
                   "unmodified": "unmodified (interface delta, calibrated)",
                   "size": "ligand size (heavy atoms)",
                   "reweight": "reweighted (free sign)",
                   "reweight_nonneg": "reweighted (non-negative)",
                   "reweight_size": "reweighted (free sign) + ligand size"}
TERM_LABELS = {"all": "all seven terms", "no_pair": "without fa_pair", "enthalpic": "enthalpic (no fa_pair, no fa_sol)"}


def approach_features(approach: str, terms: str = "all") -> list[str]:
    if approach == "constant":
        return []
    if approach == "unmodified":
        return ["idelta_score"]
    if approach == "size":
        return ["ligand_n_heavy"]
    cols = list(TERM_SETS[terms])
    if approach == "reweight_size":
        cols.append("ligand_n_heavy")
    if approach in TERM_APPROACHES:
        return cols
    raise ValueError(f"unknown approach {approach!r}")


def predictors(approaches=APPROACHES, term_sets: tuple[str, ...] = tuple(TERM_SETS)) -> list[tuple[str, str]]:
    """(approach, term set) pairs; the term set only varies for the reweightings."""
    out = []
    for a in approaches:
        out += [(a, t) for t in term_sets] if a in TERM_APPROACHES else [(a, "all")]
    return out


def predictor_key(approach: str, terms: str = "all") -> str:
    return f"{approach}:{terms}" if approach in TERM_APPROACHES and terms != "all" else approach


def split_key(key: str) -> tuple[str, str]:
    a, _, t = key.partition(":")
    return a, (t or "all")


@dataclass
class Linear:
    """pK = X @ coef + intercept. For the interface terms the energy weight is −coef
    (pK = b − Σ w x: a positive w means that a more negative term favours binding)."""
    approach: str
    names: list[str]
    coef: np.ndarray
    intercept: float
    n_train: int = 0
    loss: str = "l2"
    terms: str = "all"
    tau: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return predictor_key(self.approach, self.terms)

    def predict(self, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
        X = frame[self.names].to_numpy(dtype=np.float64) if isinstance(frame, pd.DataFrame) else np.asarray(frame, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(len(X), -1)
        return X @ self.coef + self.intercept

    def energy_weights(self) -> np.ndarray:
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
        return {"approach": self.approach, "terms": self.terms, "loss": self.loss, "tau": self.tau, "names": self.names,
                "coef": self.coef.tolist(), "intercept": float(self.intercept), "n_train": self.n_train, **self.extra}


# ---------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------
def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    if X.shape[1] == 0:
        return np.zeros(0), float(y.mean())
    A = np.column_stack([X, np.ones(len(X))])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return sol[:-1], float(sol[-1])


def fit_nnls(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """pK = b − Σ w x with w ≥ 0: NNLS on the centred data, the intercept follows from the means."""
    if X.shape[1] == 0:
        return np.zeros(0), float(y.mean())
    xm, ym = X.mean(axis=0), y.mean()
    w, _ = nnls(-(X - xm), y - ym)
    coef = -w
    return coef, float(ym - xm @ coef)


def fit_quantile(X: np.ndarray, y: np.ndarray, tau: float, nonneg: bool = False) -> tuple[np.ndarray, float]:
    """Linear quantile regression (τ = 0.5: median regression, the L1 fit). Free sign: IRLS
    (statsmodels); ``nonneg`` (coef ≤ 0, i.e. energy weights ≥ 0): the equivalent linear programme."""
    n, p = X.shape
    if p == 0:
        return np.zeros(0), float(np.quantile(y, tau))
    if not nonneg:
        import statsmodels.api as sm
        res = sm.QuantReg(y, sm.add_constant(X, has_constant="add")).fit(q=tau, max_iter=5000)
        params = np.asarray(res.params, dtype=np.float64)
        return params[1:], float(params[0])
    A = sp.hstack([sp.csr_matrix(X), sp.csr_matrix(np.ones((n, 1))), sp.identity(n, format="csr"), -sp.identity(n, format="csr")], format="csr")
    c = np.concatenate([np.zeros(p + 1), np.full(n, tau), np.full(n, 1 - tau)])
    bounds = [(None, 0.0)] * p + [(None, None)] + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A, b_eq=y, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"quantile LP failed: {res.message}")
    return res.x[:p], float(res.x[p])


def fit(approach: str, frame: pd.DataFrame, y, loss: str = "l2", terms: str = "all", tau: float | None = None) -> Linear:
    """Fit one approach on the training complexes: ``loss`` ``l2`` / ``l1`` for the point
    predictor, or the ``tau``-quantile model when ``tau`` is given (the loss is then ignored)."""
    names = approach_features(approach, terms)
    X = frame[names].to_numpy(dtype=np.float64) if names else np.zeros((len(frame), 0))
    y = np.asarray(y, dtype=np.float64)
    nonneg = approach in NONNEG
    if tau is not None:
        coef, b = fit_quantile(X, y, tau, nonneg)
    elif loss == "l1":
        coef, b = fit_quantile(X, y, 0.5, nonneg)
    elif loss == "l2":
        coef, b = (fit_nnls if nonneg else fit_ols)(X, y)
    else:
        raise ValueError(f"unknown loss {loss!r}")
    return Linear(approach, list(names), coef, b, n_train=len(y), loss=loss, terms=terms, tau=tau)
