# Track 3 — score vs. affinity: the interface delta and its reweighting against pK (`affinity/`)

> Methods-level documentation of the `affinity` package: the question, the fixed setup, the
> approaches, the evaluation and the results. Source files:
> [`affinity/data.py`](../affinity/data.py), [`affinity/models.py`](../affinity/models.py),
> [`affinity/evaluate.py`](../affinity/evaluate.py), [`affinity/train.py`](../affinity/train.py),
> [`affinity/report.py`](../affinity/report.py). Paper context:
> [PAPER1_STORY_INVENTORY.md](PAPER1_STORY_INVENTORY.md), story S7 and Appendix A.3 (step 1).
> Written 2026-10-21, when the legacy grid of the zip era was replaced (git history before that
> date); revised 2026-10-22 from correlation to absolute prediction (MAE against the constant
> predictor, 95 % lower bounds, median-regression fits, term ablation). Companion of
> [ranking.md](ranking.md) (S5) and [degeneracy.md](degeneracy.md) (S6).

## 0. Headline

**Binding affinity cannot be calculated from the Rosetta interface delta of a docked pose. On
the leakage-controlled LP-PDBBind split the calibrated interface delta of the best `apo_relax`
· `perturb` pose predicts pK with a mean absolute error of 1.49 [1.44, 1.54], the constant
predictor (the training median) with 1.56 [1.51, 1.61], and the ligand's heavy-atom count with
1.46.** A median-regression reweighting of the seven terms reaches 1.45–1.46 (ΔMAE −0.03 to
−0.04 to the calibrated score, −0.10 to the constant), the same as the size alone; adding the
terms to the size gains 0.01 pK. A 95 % lower bound per complex has to sit 2.4 pK below the
prediction (2.0 for the size), so a 10 µM binder (pK ≥ 5) can be certified for 9–14 % of the
test complexes and a 1 µM binder for 2–3 %; the constant predictor certifies pK ≥ 3.7 for every
complex and nothing above it. On the CASF-2016 core set (HAC test, n = 271) the numbers are
1.74 / 1.50 / 1.50 / 1.39 (constant / unmodified / size / reweighted): there the energy beats the
size, and the reweighting gains 0.11 pK. The pose is not the limit: the best *hit* (oracle)
predicts worse than the best-scoring pose (1.54 vs 1.49), the relaxed holo complex with the
default weights is worse than the constant (1.60), and the reweighting repairs both (1.47). The
term ablation costs nothing for `fa_pair` and at most 0.01 pK for the enthalpic set, but the
non-negative enthalpic fit degenerates to `fa_atr` alone — a contact count. The learned weights
are the same on every split: `fa_rep` and `fa_pair` off, `fa_sol` and `fa_elec` at a third to a
half of their default, the hydrogen-bond terms at a quarter (§6.8).

![affinity headline figure](../data/figures/affinity/headline_scatter.png)

![affinity lower-bound figure](../data/figures/affinity/lower_bound.png)

`data/figures/affinity/headline_scatter.png` — predicted vs experimental pK on the headline cell
for the calibrated interface delta, the reweighted terms and the heavy-atom count;
`lower_bound.png` — the share of test complexes whose 95 % conformal lower bound lies above a
pK threshold, against the share whose experimental pK does; `pools.png` the MAE per pool ×
aggregation; `weights.png` the learned non-negative weights per split against the defaults.
Regenerate with `uv run python -m affinity.report`.

## 1. Question and setup

**Can the binding affinity be predicted from the Rosetta interface delta of a docked pose, and
how much does a linear reweighting of its interface terms add?** This is step 1 of Appendix A.3:
the single-pose baseline that step 2 (the sampled-density free energy, not implemented, open
Paper 1 / Paper 2 decision) would have to beat. The question is absolute prediction, not
ranking or enrichment (decided 2026-10-22): the headline metric is the **MAE in pK** against
the **constant predictor** (the null: the training median), every approach also gets a **95 %
lower bound per complex**, and the paired ΔMAE to the calibrated interface delta on the same
complexes says whether reweighting helps. Pearson r is kept as a secondary column because the
CASF scoring power is reported that way.

| Element | Decision (2026-10-21) |
|---|---|
| Labels | pK = −log10(value / M) of the **exact** (`=`) binding data of `INDEX_general_PL.2020R1.lst`; inequalities (`<`, `<=`, `>`, `~`) are dropped. **Headline `kdki`: Kd and Ki pooled** (6,708 of the 11,444 complete complexes; Kd 3,579, Ki 3,129); **IC50 separately** (4,579; trained and tested on IC50 complexes only, it is assay-dependent); the Kd and the Ki half of the headline test set are also reported |
| Features | the **seven raw interface-delta terms** of the aggregated pose (`raw_delta_{fa_atr, fa_rep, fa_sol, fa_elec, fa_pair, hbond_bb_sc, hbond_sc}`; `idelta_score` is their weighted sum with the default weights, verified to 1e-4 on the extraction), plus the ligand heavy-atom count `ligand_n_heavy` as the size reference |
| Pools | which poses of a complex are aggregated: **one docking run** (complex × protocol, 300 poses; headline `apo_relax` · `perturb` as in S5, `relax` · `perturb` as the realistic holo setting), the **perturb + std runs of one receptor state** (600 poses — the "with Standard poses" ablation), the **relaxed holo complex** the docking started from (`relax_ref`, the native pose after FastRelax) and the **crystal complex** as prepared (`crystal`, unrelaxed) |
| Aggregations | `best` — the pose with the lowest interface delta, the pose a user would take (**headline**); `mean10` — mean of the terms over the 10 lowest-interface-delta poses; `mean` — mean over the poses with a non-positive interface delta (the legacy grid's filtered mean; the plain mean over all 300 poses is dominated by clashing poses and correlates ≈ 0 with pK); `best_hit` — oracle, the lowest-interface-delta pose among the hits (`rmsd_to_crystal` ≤ 2 Å), i.e. the score when the pose is right |
| Splits | **the same three non-random PDBbind splits as S5** (`docking_analysis/splits.csv`): LP-PDBBind (headline; built for leakage-controlled affinity prediction, largest test set), HAC (test = the CASF-2016 core set, the standard scoring-power benchmark), CleanSplit. Point models are fitted on the **training** complexes, the conformal slack is calibrated on the **validation** complexes (the models have no hyper-parameters, so val is free for calibration), evaluation on the test complexes. The CASF-2016 core complexes outside a split's training and validation sets are its external test (for HAC that is its test set). The Astex overlap exclusion of S5 is irrelevant here (no external set is evaluated) |
| Predictors | `constant` (the null) · `unmodified` · `size` · `reweighted` (free sign) · `non-negative` · `reweighted + size`; the three reweightings are fitted once per **term set** (all seven terms / without `fa_pair` / enthalpic only, i.e. without `fa_pair` and `fa_sol`), always a full refit; every point predictor is fitted with both **losses** (`l1` median regression = the loss matching the MAE, the headline; `l2` least squares for the r tables) — §3 |
| Metrics | **MAE in pK** on the test complexes (headline), RMSE, Pearson r (the CASF "scoring power", secondary), Spearman ρ; percentile-bootstrap 95 % intervals over the evaluation complexes (2,000 resamples) and the **paired ΔMAE to the calibrated interface delta and to the constant predictor** on the same resamples. **95 % lower bounds** per complex: the one-sided split-conformal bound (point prediction minus a slack calibrated on the validation complexes) and the τ = 0.05 quantile-regression model, each judged by its coverage on the test complexes, its mean slack and the share of complexes it certifies above pK 5 / 6 — §4 |
| Headline cell | Kd + Ki · `apo_relax_docking_perturb` · best pose · LP-PDBBind test complexes (n = 1,874), `l1` fits with all seven terms |

What the rewrite lifted from the legacy grid (inventory S7): exact Kd only → Kd + Ki with IC50
separate; random 80/20 split → the three leakage-aware splits plus the CASF core; no intervals →
bootstrap intervals and paired differences; non-negative weights with an MAE loss trained by
gradient descent → exact median regression and least squares, free-sign and non-negative;
relax + apo_relax perturb poses only → six pools including the std runs and the native
structures; uncorrected RMSDs and totals → the corrected extraction; correlation only → MAE
against the constant predictor with calibrated lower bounds. Not carried over: the RMSD
clustering (plain RMSD, cluster-representative aggregation) — it belongs to the ensemble
question of A.3 step 2.

## 2. Data

- **Labels** (`load_labels`): every line of the PDBbind index is parsed into measure, relation,
  value and pK; 19,037 entries, 18,767 exact. Among the complete docked complexes: Kd 3,579 ·
  Ki 3,129 · IC50 4,579 exact (`labels.csv`). pK distributions: Kd mean 5.9 (sd 1.8), Ki 6.8
  (2.1), IC50 6.6 (1.6).
- **Feature table** (`build_features` → `data/affinity/features.parquet`, ~40 s): one row per
  complete complex × pool × aggregation with the aggregated interface delta, RMSD, the seven
  terms, `n_poses`, `n_agg` (poses the aggregation used) and `n_hits`. The seed pools recompute
  the interface delta from the terms with the default weights: the stored `idelta_score` of five
  relax reference seeds (2f5t, 3cr5, 3gk4, 5er4, 3cr4) and one crystal row is corrupt (values down
  to −4.6e8 with sane terms), which made the Pearson r of the native pose meaningless until fixed.
- **Split counts** (Kd + Ki; train / validation = calibration / test / CASF core outside train +
  val): LP-PDBBind 3,672 / 811 / 1,874 / 109; HAC 6,331 / 103 / 271 / 271 (= core); CleanSplit
  4,658 / 513 / 151 / 267. IC50: LP-PDBBind 2,116 / 727 / 1,669; the HAC and CleanSplit test sets
  contain no IC50 complex, so IC50 is evaluated on LP-PDBBind only.
- The unmodified score is also correlated with pK on **all** labelled complete complexes per
  pool × aggregation × label set, with no split and no fit (`baseline_all.csv`, table 8 of the
  report) — the numbers comparable with the legacy grid's r ≈ 0.39 / 0.38 / 0.42.

## 3. Predictors (`affinity.models`)

Every predictor maps a per-complex feature vector to a predicted pK; nothing is tuned and
nothing is random, so one fit per cell and the bootstrap over complexes gives the intervals.

- **`constant`** — no feature: the training median (`l1`) or mean (`l2`). The null. With a
  label spread of ≈ 1.9 pK it predicts every complex to within 1.5 pK on average, and every other
  row is judged by its paired ΔMAE to this one.
- **`unmodified`** — pK ≈ a · `idelta_score` + b, the interface delta calibrated to pK. The
  reference. Pearson r is invariant to a and b; MAE and RMSE need them, and without them the
  interface delta would predict pK ≈ 11 for a typical −15 REU (the fitted slope is ≈ 0.2 pK per
  REU, i.e. one REU of interface delta is worth ≈ 0.28 kcal/mol of binding free energy).
- **`size`** — pK ≈ a · `ligand_n_heavy` + b. Affinity grows with ligand size in PDBbind, and so
  does every contact term, so this is the reference any energy-based score has to beat (see the
  caveat in §7: this holds for the co-crystallised binders of PDBbind, not for diverse ligand
  sets).
- **`reweighted`** (free sign) — pK ≈ b − Σ_t w_t x_t over the raw terms of the term set. The
  energy weights w = −coef; a positive w means a more negative term favours binding. Reported
  normalised to fa_atr = 0.8 like the ranking weights of S5.
- **`non-negative`** — the same with w_t ≥ 0, so the vector is a Rosetta weight set
  (`weights_nonneg_<split>.wts`, the `l1` fit with all terms).
- **`reweighted + size`** — the free-sign fit plus `ligand_n_heavy`: does the energy add to the
  size?

**Term sets** (the three reweightings, full refit per set): `all` — the seven interface terms;
`no_pair` — without `fa_pair`, a knowledge-based potential of mean force over everything,
configurations included; `enthalpic` — without `fa_pair` and `fa_sol`, the implicit-solvent
free energy, leaving `fa_atr`, `fa_rep`, `fa_elec` and the two H-bond terms. The ablation is
the single-pose answer to the question whether an ensemble estimator (A.3 step 2) should
start from an enthalpy-only energy: `fa_sol` is a potential of mean force over water the
sampler never touches (the MM/GBSA precedent keeps it), `fa_pair` is the term that also
averages over configurations.

**Losses**: `l2` — ordinary least squares, NNLS on the centred data for the non-negative model;
`l1` — median regression (quantile regression at τ = 0.5, the loss that matches the MAE), by
IRLS for the free-sign models and as a linear programme for the non-negative one. The
**lower-bound model** is the same linear form fitted at τ = 0.05: a linear predictor of the
5th percentile of pK given the features.

## 4. Evaluation (`affinity.evaluate`)

Per cell (label set × split × pool × aggregation) every predictor is fitted on the training
complexes, the conformal slack is calibrated on the validation complexes, and the point
predictions and the bounds are evaluated on the test complexes of the split, on the Kd and the
Ki half of the headline test set, and on the CASF-2016 core complexes outside the training and
validation sets. Evaluation or calibration sets with fewer than 20 complexes are skipped.

- **Point metrics:** MAE (headline), RMSE, Pearson r with percentile-bootstrap 95 % intervals
  over complexes (2,000 resamples, seed 0), Spearman ρ as a point estimate; the paired ΔMAE and
  Δr to the calibrated interface delta and the paired ΔMAE to the constant predictor on the
  same resamples.
- **95 % lower bounds per complex:** `conformal` — L = prediction − s, with s the
  ⌈(m + 1)(1 − α)⌉-th smallest signed residual (prediction − pK) on the m validation complexes
  (one-sided split conformal: P(pK ≥ L) ≥ 95 % for exchangeable complexes, no model
  assumption); `QR` — the τ = 0.05 quantile-regression model. Each bound is judged by its
  **coverage** on the test complexes (share with pK ≥ L, nominal 95 %, with its bootstrap
  interval), its mean **slack** (prediction − L: how much has to be given away to be 95 % safe)
  and the share of complexes it **certifies** above pK 5 and pK 6 (a 10 µM / 1 µM binder
  certified at 95 %) — the use case of a lower bound.

## 5. How to run

```bash
uv run python -m affinity.train                 # feature cache + the default grid (6 pools, both losses, three term sets), ~15 min
uv run python -m affinity.train --pool all      # all ten pools (adds the crystal-receptor and the std runs)
uv run python -m affinity.train --loss l1 --terms all   # a smaller grid
uv run python -m affinity.report                # tables + figures -> data/figures/affinity/
```

`affinity.train` writes `data/affinity/{features.parquet, labels.csv, baseline_all.csv,
results.csv, weights.csv, weights_nonneg_<split>.wts, predictions.parquet, results.json,
results.md, train.log}`; `affinity.report` writes `data/figures/affinity/affinity_tables.md`
(ten tables), `affinity_results.csv`, `headline_scatter.png`, `lower_bound.png`, `pools.png`,
`weights.png`.

## 6. Results (run of 2026-10-22)

The default grid: 2 label sets × 3 splits × (4 run pools × 4 aggregations + 2 reference
structures) = 108 cells × 2 losses × 12 predictors, 4,680 result rows (`data/affinity/results.csv`,
collected with the per-complex predictions, bounds and fitted weights in
`data/figures/affinity/affinity_tables.md`, ten tables; 8.5 min on one core). Cells below: MAE
in pK [95 % bootstrap interval] on the evaluation complexes of the `l1` (median-regression) fits;
`ΔMAE` the paired difference to the calibrated interface delta or to the constant predictor on
the same complexes, negative = better.

### 6.1 Headline cell — Kd + Ki, `apo_relax` · `perturb`, best pose, LP-PDBBind test complexes

| predictor | n | MAE [95 % CI] | ΔMAE vs unmodified [95 % CI] | ΔMAE vs constant [95 % CI] | RMSE | r | conformal LB: slack | coverage % | ≥ pK 5 % | ≥ pK 6 % | QR LB: slack | coverage % | ≥ pK 5 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| constant (training median / mean) | 1874 | 1.56 [1.51, 1.61] | +0.06 [+0.03, +0.10] | — | 1.90 | nan | 2.40 | 91.5 | 0.0 | 0.0 | 3.10 | 95.9 | 0.0 |
| unmodified (interface delta, calibrated) | 1874 | 1.49 [1.44, 1.54] | — | -0.06 [-0.10, -0.03] | 1.84 | 0.308 | 2.38 | 93.7 | 9.3 | 2.2 | 2.75 | 96.5 | 4.5 |
| ligand size (heavy atoms) | 1874 | 1.46 [1.41, 1.51] | -0.03 [-0.07, +0.00] | -0.10 [-0.13, -0.06] | 1.79 | 0.418 | 2.00 | 93.8 | 8.3 | 1.5 | 2.72 | 97.8 | 0.5 |
| reweighted (free sign) | 1874 | 1.46 [1.41, 1.51] | -0.03 [-0.05, -0.01] | -0.10 [-0.13, -0.06] | 1.81 | 0.372 | 2.35 | 95.1 | 10.0 | 2.4 | 2.58 | 96.9 | 6.0 |
| reweighted (non-negative) | 1874 | 1.45 [1.41, 1.50] | -0.04 [-0.06, -0.02] | -0.10 [-0.14, -0.06] | 1.80 | 0.373 | 2.35 | 95.5 | 10.5 | 2.4 | 2.60 | 97.0 | 5.8 |
| reweighted (free sign) + ligand size | 1874 | 1.45 [1.40, 1.50] | -0.04 [-0.07, -0.02] | -0.11 [-0.15, -0.06] | 1.80 | 0.429 | 2.07 | 95.0 | 13.6 | 2.7 | 2.36 | 96.9 | 9.0 |
| reweighted (free sign), without fa_pair | 1874 | 1.46 [1.41, 1.51] | -0.03 [-0.05, -0.01] | -0.10 [-0.13, -0.06] | 1.81 | 0.372 | 2.35 | 95.1 | 10.0 | 2.4 | 2.58 | 97.0 | 6.2 |
| reweighted (free sign), enthalpic (no fa_pair, no fa_sol) | 1874 | 1.47 [1.42, 1.52] | -0.02 [-0.05, +0.00] | -0.08 [-0.13, -0.05] | 1.82 | 0.371 | 2.36 | 95.7 | 8.5 | 2.4 | 2.59 | 97.2 | 4.7 |
| reweighted (non-negative), without fa_pair | 1874 | 1.45 [1.41, 1.50] | -0.04 [-0.06, -0.02] | -0.10 [-0.14, -0.06] | 1.80 | 0.373 | 2.35 | 95.5 | 10.5 | 2.4 | 2.60 | 97.0 | 5.8 |
| reweighted (non-negative), enthalpic (no fa_pair, no fa_sol) | 1874 | 1.47 [1.42, 1.52] | -0.02 [-0.05, -0.00] | -0.09 [-0.13, -0.05] | 1.82 | 0.370 | 2.42 | 95.9 | 7.8 | 2.5 | 2.63 | 97.1 | 4.1 |
| reweighted (free sign) + ligand size, without fa_pair | 1874 | 1.45 [1.40, 1.50] | -0.04 [-0.07, -0.02] | -0.11 [-0.15, -0.07] | 1.80 | 0.430 | 2.08 | 95.0 | 13.7 | 2.7 | 2.36 | 97.0 | 9.1 |
| reweighted (free sign) + ligand size, enthalpic (no fa_pair, no fa_sol) | 1874 | 1.46 [1.41, 1.51] | -0.04 [-0.07, -0.00] | -0.10 [-0.14, -0.06] | 1.80 | 0.424 | 2.12 | 94.9 | 11.6 | 3.0 | 2.44 | 97.1 | 5.1 |

### 6.2 Headline cell across the splits, and the lower bounds

| predictor | LP-PDBBind (n = 1,874) MAE [95 % CI] | LP-PDBBind ΔMAE vs unmodified [95 % CI] | HAC (n = 271) MAE [95 % CI] | HAC ΔMAE vs unmodified [95 % CI] | CleanSplit (n = 151) MAE [95 % CI] | CleanSplit ΔMAE vs unmodified [95 % CI] |
|---|---|---|---|---|---|---|
| constant (training median / mean) | 1.56 [1.51, 1.61] | +0.06 [+0.03, +0.10] | 1.74 [1.60, 1.89] | +0.25 [+0.16, +0.33] | 1.85 [1.65, 2.07] | +0.23 [+0.12, +0.35] |
| unmodified (interface delta, calibrated) | 1.49 [1.44, 1.54] | — | 1.50 [1.37, 1.63] | — | 1.62 [1.44, 1.80] | — |
| ligand size (heavy atoms) | 1.46 [1.41, 1.51] | -0.03 [-0.07, +0.00] | 1.50 [1.37, 1.63] | +0.00 [-0.10, +0.10] | 1.51 [1.35, 1.68] | -0.11 [-0.23, +0.01] |
| reweighted (free sign) | 1.46 [1.41, 1.51] | -0.03 [-0.05, -0.01] | 1.39 [1.27, 1.51] | -0.11 [-0.17, -0.05] | 1.50 [1.33, 1.67] | -0.12 [-0.19, -0.05] |
| reweighted (non-negative) | 1.45 [1.41, 1.50] | -0.04 [-0.06, -0.02] | 1.38 [1.26, 1.51] | -0.11 [-0.17, -0.05] | 1.50 [1.33, 1.67] | -0.12 [-0.19, -0.05] |
| reweighted (free sign) + ligand size | 1.45 [1.40, 1.50] | -0.04 [-0.07, -0.02] | 1.37 [1.25, 1.49] | -0.13 [-0.20, -0.06] | 1.45 [1.29, 1.62] | -0.16 [-0.25, -0.08] |

Conformal 95 % lower bound (slack calibrated on the validation complexes: 811 / 103 / 513):

| predictor | LP-PDBBind slack (pK) | LP-PDBBind coverage % [95 % CI] | LP-PDBBind ≥ pK 5 % | HAC slack (pK) | HAC coverage % [95 % CI] | HAC ≥ pK 5 % | CleanSplit slack (pK) | CleanSplit coverage % [95 % CI] | CleanSplit ≥ pK 5 % |
|---|---|---|---|---|---|---|---|---|---|
| constant (training median / mean) | 2.40 | 91.5 [90.3, 92.7] | 0.0 | 3.34 | 93.7 [90.8, 96.7] | 0.0 | 3.26 | 91.4 [86.8, 95.4] | 0.0 |
| unmodified (interface delta, calibrated) | 2.38 | 93.7 [92.6, 94.8] | 9.3 | 2.91 | 94.1 [91.1, 96.7] | 5.5 | 2.68 | 92.7 [88.7, 96.7] | 5.3 |
| ligand size (heavy atoms) | 2.00 | 93.8 [92.7, 94.9] | 8.3 | 3.53 | 99.3 [98.2, 100.0] | 0.7 | 2.74 | 96.7 [94.0, 99.3] | 4.0 |
| reweighted (free sign) | 2.35 | 95.1 [94.1, 96.1] | 10.0 | 3.05 | 98.5 [97.0, 99.6] | 4.8 | 2.60 | 96.0 [92.7, 98.7] | 7.9 |
| reweighted (non-negative) | 2.35 | 95.5 [94.5, 96.4] | 10.5 | 3.05 | 98.5 [97.0, 99.6] | 4.8 | 2.59 | 96.0 [92.7, 98.7] | 7.9 |
| reweighted (free sign) + ligand size | 2.07 | 95.0 [94.0, 95.9] | 13.6 | 2.98 | 98.2 [96.3, 99.6] | 6.3 | 2.50 | 96.7 [93.4, 99.3] | 8.6 |

Quantile-regression (τ = 0.05) lower bound:

| predictor | LP-PDBBind slack (pK) | LP-PDBBind coverage % [95 % CI] | LP-PDBBind ≥ pK 5 % | HAC slack (pK) | HAC coverage % [95 % CI] | HAC ≥ pK 5 % | CleanSplit slack (pK) | CleanSplit coverage % [95 % CI] | CleanSplit ≥ pK 5 % |
|---|---|---|---|---|---|---|---|---|---|
| constant (training median / mean) | 3.10 | 95.9 [94.9, 96.7] | 0.0 | 3.29 | 93.7 [90.8, 96.7] | 0.0 | 3.22 | 90.1 [85.4, 94.7] | 0.0 |
| unmodified (interface delta, calibrated) | 2.75 | 96.5 [95.6, 97.3] | 4.5 | 2.80 | 93.4 [90.4, 96.3] | 8.1 | 2.75 | 93.4 [89.4, 97.4] | 5.3 |
| ligand size (heavy atoms) | 2.72 | 97.8 [97.1, 98.5] | 0.5 | 2.81 | 95.9 [93.4, 98.2] | 1.8 | 2.82 | 94.7 [91.4, 98.0] | 2.0 |
| reweighted (free sign) | 2.58 | 96.9 [96.1, 97.7] | 6.0 | 2.62 | 96.7 [94.5, 98.5] | 11.1 | 2.60 | 98.0 [95.4, 100.0] | 8.6 |
| reweighted (non-negative) | 2.60 | 97.0 [96.1, 97.7] | 5.8 | 2.60 | 96.7 [94.5, 98.5] | 10.3 | 2.61 | 97.4 [94.7, 100.0] | 7.9 |
| reweighted (free sign) + ligand size | 2.36 | 96.9 [96.1, 97.5] | 9.0 | 2.49 | 95.9 [93.4, 98.2] | 12.9 | 2.47 | 96.7 [93.4, 99.3] | 11.3 |

### 6.3 Term ablation — all seven terms / without fa_pair / enthalpic only, full refit

| predictor | LP-PDBBind (n = 1,874) MAE [95 % CI] | LP-PDBBind ΔMAE vs unmodified [95 % CI] | HAC (n = 271) MAE [95 % CI] | HAC ΔMAE vs unmodified [95 % CI] | CleanSplit (n = 151) MAE [95 % CI] | CleanSplit ΔMAE vs unmodified [95 % CI] |
|---|---|---|---|---|---|---|
| constant (training median / mean) | 1.56 [1.51, 1.61] | +0.06 [+0.03, +0.10] | 1.74 [1.60, 1.89] | +0.25 [+0.16, +0.33] | 1.85 [1.65, 2.07] | +0.23 [+0.12, +0.35] |
| unmodified (interface delta, calibrated) | 1.49 [1.44, 1.54] | — | 1.50 [1.37, 1.63] | — | 1.62 [1.44, 1.80] | — |
| reweighted (free sign) | 1.46 [1.41, 1.51] | -0.03 [-0.05, -0.01] | 1.39 [1.27, 1.51] | -0.11 [-0.17, -0.05] | 1.50 [1.33, 1.67] | -0.12 [-0.19, -0.05] |
| reweighted (free sign), without fa_pair | 1.46 [1.41, 1.51] | -0.03 [-0.05, -0.01] | 1.38 [1.26, 1.51] | -0.11 [-0.17, -0.05] | 1.50 [1.33, 1.67] | -0.12 [-0.19, -0.05] |
| reweighted (free sign), enthalpic (no fa_pair, no fa_sol) | 1.47 [1.42, 1.52] | -0.02 [-0.05, +0.00] | 1.38 [1.26, 1.51] | -0.11 [-0.19, -0.04] | 1.48 [1.31, 1.65] | -0.14 [-0.23, -0.04] |
| reweighted (non-negative) | 1.45 [1.41, 1.50] | -0.04 [-0.06, -0.02] | 1.38 [1.26, 1.51] | -0.11 [-0.17, -0.05] | 1.50 [1.33, 1.67] | -0.12 [-0.19, -0.05] |
| reweighted (non-negative), without fa_pair | 1.45 [1.41, 1.50] | -0.04 [-0.06, -0.02] | 1.38 [1.26, 1.51] | -0.11 [-0.17, -0.05] | 1.50 [1.33, 1.67] | -0.12 [-0.19, -0.05] |
| reweighted (non-negative), enthalpic (no fa_pair, no fa_sol) | 1.47 [1.42, 1.52] | -0.02 [-0.05, -0.00] | 1.38 [1.26, 1.51] | -0.11 [-0.19, -0.04] | 1.49 [1.32, 1.67] | -0.13 [-0.22, -0.03] |
| reweighted (free sign) + ligand size | 1.45 [1.40, 1.50] | -0.04 [-0.07, -0.02] | 1.37 [1.25, 1.49] | -0.13 [-0.20, -0.06] | 1.45 [1.29, 1.62] | -0.16 [-0.25, -0.08] |
| reweighted (free sign) + ligand size, without fa_pair | 1.45 [1.40, 1.50] | -0.04 [-0.07, -0.02] | 1.37 [1.25, 1.49] | -0.13 [-0.20, -0.06] | 1.46 [1.29, 1.62] | -0.16 [-0.25, -0.08] |
| reweighted (free sign) + ligand size, enthalpic (no fa_pair, no fa_sol) | 1.46 [1.41, 1.51] | -0.04 [-0.07, -0.00] | 1.38 [1.25, 1.50] | -0.12 [-0.21, -0.03] | 1.45 [1.28, 1.61] | -0.17 [-0.28, -0.07] |

The same ablation for `relax` · `perturb` and the relaxed holo complex is in table 3 of
`affinity_tables.md`; on the relaxed holo complex the enthalpic free-sign fit is the best model
(1.44 on LP-PDBBind, against 1.47 with all terms).

### 6.4 Pools × aggregations — Kd + Ki, LP-PDBBind test complexes

`pose RMSD` / `pose ≤ 2 Å`: the symmetry-corrected RMSD to the crystal ligand of the aggregated
pose on the test complexes (for the means: the mean RMSD of the averaged poses); `slack` and
`≥ pK 5` of the conformal bound for the unmodified / the reweighted model.

| pool | aggregation | n | pose RMSD median (Å) | pose ≤ 2 Å (%) | constant MAE | unmodified MAE [95 % CI] | ligand size MAE [95 % CI] | reweighted ΔMAE [95 % CI] | non-negative ΔMAE [95 % CI] | reweighted + size ΔMAE [95 % CI] | conformal slack unmodified / reweighted | ≥ pK 5 % unmodified / reweighted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apo_relax · perturb | best pose | 1,874 | 1.78 | 53 | 1.56 | 1.49 [1.44, 1.54] | 1.46 [1.41, 1.51] | -0.03 [-0.05, -0.01] | -0.04 [-0.06, -0.02] | -0.04 [-0.07, -0.02] | 2.38 / 2.35 | 9.3 / 10.0 |
| apo_relax · perturb | mean of 10 best | 1,874 | 3.42 | 23 | 1.56 | 1.51 [1.46, 1.56] | 1.46 [1.41, 1.51] | -0.03 [-0.06, -0.01] | -0.05 [-0.07, -0.03] | -0.05 [-0.08, -0.02] | 2.36 / 2.44 | 8.8 / 7.3 |
| apo_relax · perturb | mean (interface delta ≤ 0) | 1,874 | 6.03 | 0 | 1.56 | 1.52 [1.47, 1.57] | 1.46 [1.41, 1.51] | -0.01 [-0.04, +0.02] | +0.00 [-0.02, +0.03] | -0.04 [-0.08, -0.00] | 2.38 / 2.47 | 2.3 / 5.9 |
| apo_relax · perturb | best hit (oracle) | 1,718 | 0.89 | 100 | 1.57 | 1.54 [1.49, 1.59] | 1.44 [1.39, 1.48] | -0.03 [-0.06, -0.00] | -0.03 [-0.06, -0.01] | -0.07 [-0.10, -0.04] | 2.38 / 2.31 | 9.1 / 12.1 |
| relax · perturb | best pose | 1,874 | 0.74 | 69 | 1.56 | 1.50 [1.45, 1.56] | 1.46 [1.41, 1.51] | -0.05 [-0.07, -0.03] | -0.04 [-0.06, -0.03] | -0.05 [-0.08, -0.03] | 2.41 / 2.38 | 9.0 / 9.7 |
| relax · perturb | mean of 10 best | 1,874 | 2.62 | 36 | 1.56 | 1.51 [1.46, 1.56] | 1.46 [1.41, 1.51] | -0.05 [-0.07, -0.02] | -0.06 [-0.08, -0.04] | -0.05 [-0.08, -0.02] | 2.43 / 2.39 | 8.5 / 8.5 |
| relax · perturb | mean (interface delta ≤ 0) | 1,874 | 5.88 | 0 | 1.56 | 1.53 [1.48, 1.57] | 1.46 [1.41, 1.51] | -0.01 [-0.04, +0.01] | -0.01 [-0.03, +0.02] | -0.04 [-0.07, -0.00] | 2.38 / 2.42 | 2.1 / 5.5 |
| relax · perturb | best hit (oracle) | 1,803 | 0.57 | 100 | 1.56 | 1.53 [1.48, 1.58] | 1.44 [1.40, 1.49] | -0.05 [-0.07, -0.03] | -0.05 [-0.07, -0.03] | -0.08 [-0.10, -0.05] | 2.40 / 2.35 | 9.2 / 11.0 |
| apo_relax · perturb + std | best pose | 1,874 | 0.81 | 67 | 1.56 | 1.49 [1.44, 1.54] | 1.46 [1.41, 1.51] | -0.03 [-0.05, -0.02] | -0.03 [-0.05, -0.02] | -0.05 [-0.07, -0.03] | 2.38 / 2.22 | 9.4 / 12.3 |
| apo_relax · perturb + std | mean of 10 best | 1,874 | 1.44 | 60 | 1.56 | 1.49 [1.44, 1.54] | 1.46 [1.41, 1.51] | -0.03 [-0.05, -0.01] | -0.03 [-0.06, -0.01] | -0.04 [-0.07, -0.01] | 2.44 / 2.31 | 9.0 / 11.2 |
| apo_relax · perturb + std | mean (interface delta ≤ 0) | 1,874 | 4.32 | 0 | 1.56 | 1.52 [1.47, 1.57] | 1.46 [1.41, 1.51] | -0.02 [-0.05, +0.00] | -0.02 [-0.04, +0.01] | -0.04 [-0.07, -0.01] | 2.34 / 2.38 | 6.6 / 7.6 |
| apo_relax · perturb + std | best hit (oracle) | 1,830 | 0.60 | 100 | 1.56 | 1.52 [1.47, 1.57] | 1.45 [1.40, 1.50] | -0.04 [-0.07, -0.02] | -0.05 [-0.07, -0.03] | -0.07 [-0.09, -0.04] | 2.35 / 2.23 | 9.2 / 12.8 |
| relax · perturb + std | best pose | 1,874 | 0.51 | 80 | 1.56 | 1.55 [1.50, 1.60] | 1.46 [1.41, 1.51] | -0.06 [-0.08, -0.04] | -0.06 [-0.08, -0.05] | -0.08 [-0.10, -0.06] | 2.63 / 2.34 | 5.8 / 10.1 |
| relax · perturb + std | mean of 10 best | 1,874 | 0.54 | 82 | 1.56 | 1.55 [1.50, 1.60] | 1.46 [1.41, 1.51] | -0.07 [-0.09, -0.05] | -0.07 [-0.09, -0.05] | -0.08 [-0.10, -0.06] | 2.63 / 2.32 | 6.1 / 10.9 |
| relax · perturb + std | mean (interface delta ≤ 0) | 1,874 | 3.70 | 0 | 1.56 | 1.54 [1.49, 1.59] | 1.46 [1.41, 1.51] | -0.05 [-0.07, -0.03] | -0.06 [-0.07, -0.04] | -0.06 [-0.08, -0.03] | 2.40 / 2.40 | 7.7 / 7.8 |
| relax · perturb + std | best hit (oracle) | 1,863 | 0.48 | 100 | 1.56 | 1.56 [1.50, 1.61] | 1.46 [1.41, 1.51] | -0.08 [-0.10, -0.07] | -0.08 [-0.09, -0.06] | -0.09 [-0.11, -0.07] | 2.58 / 2.34 | 6.1 / 10.5 |
| relaxed holo complex (docking reference) | best pose | 1,874 | 0.29 | 96 | 1.56 | 1.60 [1.55, 1.65] | 1.46 [1.41, 1.51] | -0.13 [-0.15, -0.10] | -0.13 [-0.15, -0.11] | -0.13 [-0.15, -0.10] | 2.48 / 2.34 | 6.9 / 10.3 |
| crystal complex (input, unrelaxed) | best pose | 1,874 | 0.00 | 100 | 1.56 | 1.57 [1.52, 1.62] | 1.46 [1.41, 1.51] | -0.11 [-0.15, -0.06] | -0.10 [-0.14, -0.06] | -0.11 [-0.15, -0.06] | 2.45 / 2.32 | 0.0 / 10.6 |

### 6.5 The Kd and the Ki half of the headline test set, and IC50

LP-PDBBind, `apo_relax` · `perturb`, best pose; the Kd + Ki models evaluated on each half:

| predictor | Kd + Ki (n = 1,874) MAE [95 % CI] | Kd + Ki ΔMAE vs unmodified [95 % CI] | Kd only (n = 921) MAE [95 % CI] | Kd only ΔMAE vs unmodified [95 % CI] | Ki only (n = 953) MAE [95 % CI] | Ki only ΔMAE vs unmodified [95 % CI] |
|---|---|---|---|---|---|---|
| constant (training median / mean) | 1.56 [1.51, 1.61] | +0.06 [+0.03, +0.10] | 1.47 [1.40, 1.54] | +0.04 [-0.01, +0.09] | 1.64 [1.57, 1.71] | +0.09 [+0.03, +0.14] |
| unmodified (interface delta, calibrated) | 1.49 [1.44, 1.54] | — | 1.43 [1.36, 1.50] | — | 1.55 [1.48, 1.62] | — |
| ligand size (heavy atoms) | 1.46 [1.41, 1.51] | -0.03 [-0.07, +0.00] | 1.40 [1.33, 1.47] | -0.03 [-0.08, +0.02] | 1.51 [1.45, 1.58] | -0.03 [-0.08, +0.01] |
| reweighted (free sign) | 1.46 [1.41, 1.51] | -0.03 [-0.05, -0.01] | 1.40 [1.33, 1.47] | -0.03 [-0.06, +0.00] | 1.51 [1.45, 1.58] | -0.03 [-0.06, -0.00] |
| reweighted (non-negative) | 1.45 [1.41, 1.50] | -0.04 [-0.06, -0.02] | 1.40 [1.33, 1.47] | -0.03 [-0.07, -0.00] | 1.51 [1.44, 1.58] | -0.04 [-0.07, -0.01] |
| reweighted (free sign) + ligand size | 1.45 [1.40, 1.50] | -0.04 [-0.07, -0.02] | 1.38 [1.31, 1.45] | -0.06 [-0.09, -0.02] | 1.52 [1.45, 1.59] | -0.03 [-0.07, +0.00] |

IC50, trained and tested on the IC50 complexes (LP-PDBBind; the HAC and CleanSplit test sets
contain no IC50 complex):

| predictor | LP-PDBBind (n = 1,669) MAE [95 % CI] | LP-PDBBind ΔMAE vs unmodified [95 % CI] |
|---|---|---|
| constant (training median / mean) | 1.24 [1.20, 1.29] | +0.13 [+0.10, +0.16] |
| unmodified (interface delta, calibrated) | 1.11 [1.07, 1.15] | — |
| ligand size (heavy atoms) | 1.09 [1.05, 1.13] | -0.02 [-0.06, +0.01] |
| reweighted (free sign) | 1.10 [1.06, 1.14] | -0.02 [-0.04, +0.00] |
| reweighted (non-negative) | 1.09 [1.05, 1.13] | -0.02 [-0.04, -0.00] |
| reweighted (free sign) + ligand size | 1.06 [1.02, 1.10] | -0.05 [-0.08, -0.03] |

### 6.6 CASF-2016 core complexes outside the training and calibration sets — Kd + Ki, `apo_relax` · `perturb`, best pose

For HAC the core set *is* the test set (§6.2); for the other splits the core complexes that fell
into their training or validation set are removed (109 of 271 remain outside LP-PDBBind's, 267
outside CleanSplit's).

| predictor | trained on LP-PDBBind (n = 109) MAE [95 % CI] | LP-PDBBind ΔMAE vs unmodified [95 % CI] | trained on HAC (n = 271) MAE [95 % CI] | HAC ΔMAE vs unmodified [95 % CI] | trained on CleanSplit (n = 267) MAE [95 % CI] | CleanSplit ΔMAE vs unmodified [95 % CI] |
|---|---|---|---|---|---|---|
| constant (training median / mean) | 1.76 [1.52, 2.01] | +0.31 [+0.14, +0.48] | 1.74 [1.60, 1.89] | +0.25 [+0.16, +0.33] | 1.75 [1.60, 1.90] | +0.25 [+0.16, +0.34] |
| unmodified (interface delta, calibrated) | 1.46 [1.24, 1.66] | — | 1.50 [1.37, 1.63] | — | 1.49 [1.36, 1.63] | — |
| ligand size (heavy atoms) | 1.55 [1.35, 1.74] | +0.10 [-0.05, +0.26] | 1.50 [1.37, 1.63] | +0.00 [-0.10, +0.10] | 1.51 [1.38, 1.64] | +0.02 [-0.08, +0.12] |
| reweighted (free sign) | 1.44 [1.23, 1.64] | -0.01 [-0.11, +0.08] | 1.39 [1.27, 1.51] | -0.11 [-0.17, -0.05] | 1.38 [1.26, 1.51] | -0.11 [-0.17, -0.05] |
| reweighted (non-negative) | 1.42 [1.22, 1.62] | -0.03 [-0.12, +0.05] | 1.38 [1.26, 1.51] | -0.11 [-0.17, -0.05] | 1.38 [1.26, 1.51] | -0.11 [-0.17, -0.05] |
| reweighted (free sign) + ligand size | 1.43 [1.24, 1.62] | -0.03 [-0.14, +0.09] | 1.37 [1.25, 1.49] | -0.13 [-0.20, -0.06] | 1.38 [1.26, 1.50] | -0.11 [-0.19, -0.04] |

### 6.7 The unmodified interface delta on all labelled complete complexes (no split, no fit)

Pearson r of −`idelta_score` with pK over every labelled complete complex (the best-pose rows of
all ten pools plus the filtered mean of the std runs; the full table is table 8 of
`affinity_tables.md`) — the numbers comparable with the legacy grid's r ≈ 0.39 / 0.38 / 0.42:

| pool | aggregation | pose ≤ 2 Å (%) | Kd + Ki r [95 % CI] (n) | Kd r [95 % CI] (n) | Ki r [95 % CI] (n) | IC50 r [95 % CI] (n) |
|---|---|---|---|---|---|---|
| apo_relax · perturb | best pose | 50 | 0.428 [0.409, 0.447] (6,708) | 0.364 [0.337, 0.391] (3,579) | 0.455 [0.430, 0.480] (3,129) | 0.408 [0.380, 0.433] (4,579) |
| relax · perturb | best pose | 69 | 0.453 [0.434, 0.472] (6,708) | 0.389 [0.360, 0.416] (3,579) | 0.485 [0.461, 0.509] (3,129) | 0.441 [0.416, 0.466] (4,579) |
| apo_relax · perturb + std | best pose | 67 | 0.448 [0.428, 0.467] (6,708) | 0.389 [0.360, 0.416] (3,579) | 0.472 [0.447, 0.496] (3,129) | 0.435 [0.409, 0.460] (4,579) |
| apo_relax · perturb + std | mean (interface delta ≤ 0) | 1 | 0.330 [0.309, 0.351] (6,708) | 0.287 [0.258, 0.316] (3,579) | 0.351 [0.323, 0.379] (3,129) | 0.315 [0.289, 0.340] (4,577) |
| relax · perturb + std | best pose | 83 | 0.450 [0.429, 0.470] (6,708) | 0.389 [0.360, 0.418] (3,579) | 0.484 [0.459, 0.508] (3,129) | 0.443 [0.419, 0.469] (4,579) |
| relax · perturb + std | mean (interface delta ≤ 0) | 2 | 0.460 [0.441, 0.479] (6,708) | 0.374 [0.345, 0.401] (3,579) | 0.517 [0.493, 0.540] (3,129) | 0.430 [0.405, 0.454] (4,579) |
| relaxed holo complex (docking reference) | best pose | 97 | 0.401 [0.374, 0.427] (6,708) | 0.343 [0.307, 0.381] (3,579) | 0.441 [0.405, 0.476] (3,129) | 0.405 [0.367, 0.446] (4,579) |
| crystal complex (input, unrelaxed) | best pose | 100 | 0.011 [-0.013, 0.040] (6,708) | -0.035 [-0.065, 0.000] (3,579) | 0.047 [0.008, 0.101] (3,129) | 0.087 [0.067, 0.112] (4,579) |
| crystal · std | best pose | 84 | 0.454 [0.434, 0.473] (6,708) | 0.395 [0.367, 0.423] (3,579) | 0.483 [0.459, 0.507] (3,129) | 0.446 [0.422, 0.471] (4,579) |
| crystal · std | mean (interface delta ≤ 0) | 63 | 0.470 [0.451, 0.488] (6,706) | 0.380 [0.351, 0.409] (3,577) | 0.525 [0.501, 0.548] (3,129) | 0.441 [0.418, 0.464] (4,578) |
| crystal · perturb | best pose | 63 | 0.454 [0.435, 0.473] (6,708) | 0.386 [0.359, 0.414] (3,579) | 0.488 [0.464, 0.511] (3,129) | 0.439 [0.414, 0.464] (4,579) |
| relax · std | best pose | 87 | 0.452 [0.431, 0.471] (6,708) | 0.391 [0.362, 0.419] (3,579) | 0.485 [0.461, 0.509] (3,129) | 0.447 [0.424, 0.471] (4,579) |
| relax · std | mean (interface delta ≤ 0) | 63 | 0.470 [0.452, 0.488] (6,707) | 0.383 [0.354, 0.411] (3,578) | 0.522 [0.498, 0.545] (3,129) | 0.435 [0.412, 0.458] (4,579) |
| apo_relax · std | best pose | 71 | 0.445 [0.426, 0.465] (6,708) | 0.387 [0.360, 0.415] (3,579) | 0.469 [0.444, 0.493] (3,129) | 0.434 [0.408, 0.459] (4,579) |
| apo_relax · std | mean (interface delta ≤ 0) | 27 | 0.322 [0.301, 0.343] (6,707) | 0.282 [0.253, 0.312] (3,578) | 0.338 [0.310, 0.366] (3,129) | 0.307 [0.283, 0.331] (4,577) |

### 6.8 Learned energy weights — Kd + Ki, `apo_relax` · `perturb`, best pose, l1 fits

Energy weights w (pK = b − Σ w · term) normalised to fa_atr = 0.8 like the ranking weights of
S5; the free-sign fits can give negative weights; `—` = term not in the set. For comparison, the
S5 *ranking* weights (LP-PDBBind, Rosetta start) are fa_rep 0.50, fa_sol 0.33, fa_elec 0.33,
fa_pair 0.14, hbond_bb_sc 1.29, hbond_sc 1.08.

| model · term set · split | fa_atr | fa_rep | fa_sol | fa_elec | fa_pair | hbond_bb_sc | hbond_sc | intercept (pK) | n train |
|---|---|---|---|---|---|---|---|---|---|
| Rosetta default | 0.80 | 0.40 | 0.60 | 0.42 | 0.80 | 1.30 | 1.30 | — | — |
| reweighted (non-negative), all seven terms · LP-PDBBind | 0.80 | -0.00 | 0.24 | 0.22 | -0.00 | 0.39 | 0.30 | 2.91 | 3,672 |
| reweighted (non-negative), all seven terms · HAC | 0.80 | 0.02 | 0.33 | 0.19 | -0.00 | 0.36 | 0.38 | 3.45 | 6,331 |
| reweighted (non-negative), all seven terms · CleanSplit | 0.80 | 0.11 | 0.30 | 0.22 | -0.00 | 0.07 | 0.26 | 3.33 | 4,658 |
| reweighted (non-negative), without fa_pair · LP-PDBBind | 0.80 | -0.00 | 0.24 | 0.22 | — | 0.39 | 0.30 | 2.91 | 3,672 |
| reweighted (non-negative), without fa_pair · HAC | 0.80 | 0.02 | 0.33 | 0.19 | — | 0.36 | 0.38 | 3.45 | 6,331 |
| reweighted (non-negative), without fa_pair · CleanSplit | 0.80 | 0.11 | 0.30 | 0.22 | — | 0.07 | 0.26 | 3.33 | 4,658 |
| reweighted (non-negative), enthalpic (no fa_pair, no fa_sol) · LP-PDBBind | 0.80 | -0.00 | — | 0.07 | — | -0.00 | -0.00 | 2.93 | 3,672 |
| reweighted (non-negative), enthalpic (no fa_pair, no fa_sol) · HAC | 0.80 | -0.00 | — | -0.00 | — | -0.00 | -0.00 | 3.52 | 6,331 |
| reweighted (non-negative), enthalpic (no fa_pair, no fa_sol) · CleanSplit | 0.80 | 0.02 | — | -0.00 | — | -0.00 | -0.00 | 3.38 | 4,658 |
| reweighted (free sign), all seven terms · LP-PDBBind | 0.80 | -0.16 | 0.24 | 0.21 | 0.01 | 0.41 | 0.30 | 2.99 | 3,672 |
| reweighted (free sign), all seven terms · HAC | 0.80 | 0.03 | 0.33 | 0.19 | -0.17 | 0.35 | 0.38 | 3.47 | 6,331 |
| reweighted (free sign), all seven terms · CleanSplit | 0.80 | 0.11 | 0.30 | 0.23 | -0.09 | 0.07 | 0.26 | 3.32 | 4,658 |
| reweighted (free sign), without fa_pair · LP-PDBBind | 0.80 | -0.16 | 0.24 | 0.21 | — | 0.41 | 0.30 | 2.99 | 3,672 |
| reweighted (free sign), without fa_pair · HAC | 0.80 | 0.02 | 0.33 | 0.19 | — | 0.36 | 0.38 | 3.45 | 6,331 |
| reweighted (free sign), without fa_pair · CleanSplit | 0.80 | 0.11 | 0.30 | 0.22 | — | 0.07 | 0.26 | 3.32 | 4,658 |
| reweighted (free sign), enthalpic (no fa_pair, no fa_sol) · LP-PDBBind | 0.80 | -0.22 | — | 0.07 | — | -0.08 | -0.24 | 3.08 | 3,672 |
| reweighted (free sign), enthalpic (no fa_pair, no fa_sol) · HAC | 0.80 | -0.07 | — | -0.00 | — | -0.32 | -0.40 | 3.67 | 6,331 |
| reweighted (free sign), enthalpic (no fa_pair, no fa_sol) · CleanSplit | 0.80 | 0.05 | — | 0.09 | — | -0.68 | -0.54 | 3.48 | 4,658 |
| reweighted (free sign) + ligand size, all seven terms · LP-PDBBind | 0.80 | -0.19 | 0.55 | 0.38 | 0.43 | 1.18 | 1.42 | 2.25 | 3,672 |
| reweighted (free sign) + ligand size, all seven terms · HAC | 0.80 | -0.06 | 0.63 | 0.31 | -0.26 | 1.30 | 1.37 | 2.86 | 6,331 |
| reweighted (free sign) + ligand size, all seven terms · CleanSplit | 0.80 | 0.04 | 0.59 | 0.34 | -0.09 | 0.75 | 1.19 | 2.77 | 4,658 |
| reweighted (free sign) + ligand size, without fa_pair · LP-PDBBind | 0.80 | -0.18 | 0.54 | 0.39 | — | 1.10 | 1.40 | 2.24 | 3,672 |
| reweighted (free sign) + ligand size, without fa_pair · HAC | 0.80 | -0.06 | 0.63 | 0.31 | — | 1.32 | 1.38 | 2.86 | 6,331 |
| reweighted (free sign) + ligand size, without fa_pair · CleanSplit | 0.80 | 0.04 | 0.59 | 0.34 | — | 0.75 | 1.20 | 2.76 | 4,658 |
| reweighted (free sign) + ligand size, enthalpic (no fa_pair, no fa_sol) · LP-PDBBind | 0.80 | -0.39 | — | 0.15 | — | 0.04 | 0.10 | 2.56 | 3,672 |
| reweighted (free sign) + ligand size, enthalpic (no fa_pair, no fa_sol) · HAC | 0.80 | -0.26 | — | 0.00 | — | 0.16 | -0.06 | 3.21 | 6,331 |
| reweighted (free sign) + ligand size, enthalpic (no fa_pair, no fa_sol) · CleanSplit | 0.80 | -0.04 | — | 0.10 | — | -0.56 | -0.21 | 3.07 | 4,658 |
| lower-bound model (τ = 0.05), non-negative, all terms · LP-PDBBind | 0.80 | 0.07 | 0.21 | 0.14 | -0.00 | -0.00 | 0.30 | 0.52 | 3,672 |
| lower-bound model (τ = 0.05), non-negative, all terms · HAC | 0.80 | 0.12 | 0.34 | 0.18 | -0.00 | -0.00 | 0.52 | 0.57 | 6,331 |
| lower-bound model (τ = 0.05), non-negative, all terms · CleanSplit | 0.80 | 0.17 | 0.35 | 0.21 | -0.00 | -0.00 | 0.40 | 0.45 | 4,658 |

Raw coefficients (pK per raw unit) of the LP-PDBBind l1 models of the headline cell — the
calibration slope of the interface delta, −0.208 pK per REU, says one REU of interface delta is
worth 0.28 kcal/mol of binding free energy on average:

| predictor | idelta_score | ligand_n_heavy | fa_atr | fa_rep | fa_sol | fa_elec | fa_pair | hbond_bb_sc | hbond_sc | intercept |
|---|---|---|---|---|---|---|---|---|---|---|
| unmodified (interface delta, calibrated) | -0.2083 | — | — | — | — | — | — | — | — | 3.19 |
| ligand size (heavy atoms) | — | 0.0989 | — | — | — | — | — | — | — | 3.54 |
| reweighted (free sign) | — | — | -0.1244 | 0.0256 | -0.0367 | -0.0334 | -0.0017 | -0.0645 | -0.0460 | 2.99 |
| reweighted (non-negative) | — | — | -0.1354 | 0.0000 | -0.0413 | -0.0379 | 0.0000 | -0.0659 | -0.0515 | 2.91 |
| reweighted (free sign) + ligand size | — | 0.0622 | -0.0942 | 0.0222 | -0.0644 | -0.0450 | -0.0508 | -0.1389 | -0.1675 | 2.25 |

### 6.9 Pearson r of the least-squares (l2) fits — the CASF-comparable numbers

| predictor | LP-PDBBind (n = 1,874) r [95 % CI] | LP-PDBBind Δr vs unmodified [95 % CI] | HAC (n = 271) r [95 % CI] | HAC Δr vs unmodified [95 % CI] | CleanSplit (n = 151) r [95 % CI] | CleanSplit Δr vs unmodified [95 % CI] |
|---|---|---|---|---|---|---|
| constant (training median / mean) | — | — | — | — | — | — |
| unmodified (interface delta, calibrated) | 0.308 [0.270, 0.346] | — | 0.520 [0.429, 0.597] | — | 0.493 [0.381, 0.598] | — |
| ligand size (heavy atoms) | 0.418 [0.380, 0.457] | +0.110 [+0.070, +0.150] | 0.535 [0.444, 0.616] | +0.014 [-0.097, +0.127] | 0.639 [0.557, 0.717] | +0.146 [+0.029, +0.265] |
| reweighted (free sign) | 0.374 [0.334, 0.412] | +0.066 [+0.047, +0.087] | 0.625 [0.557, 0.685] | +0.104 [+0.048, +0.171] | 0.617 [0.530, 0.697] | +0.124 [+0.061, +0.198] |
| reweighted (non-negative) | 0.373 [0.333, 0.411] | +0.065 [+0.046, +0.086] | 0.626 [0.558, 0.685] | +0.106 [+0.050, +0.171] | 0.613 [0.526, 0.695] | +0.120 [+0.059, +0.192] |
| reweighted (free sign) + ligand size | 0.426 [0.388, 0.461] | +0.118 [+0.096, +0.140] | 0.629 [0.563, 0.689] | +0.109 [+0.045, +0.182] | 0.653 [0.575, 0.723] | +0.160 [+0.090, +0.241] |

## 7. Reading

- **In absolute terms nothing predicts.** On LP-PDBBind every predictor lies within 0.11 pK of
  the constant: the calibrated interface delta gains 0.06 pK over knowing nothing, the best
  reweighting 0.10, the heavy-atom count 0.10. A residual of 1.5 pK is a factor of 30 in Kd. The
  gap to the constant is larger on HAC and CleanSplit (0.25–0.35 pK) because their test
  distributions differ from the training median (the CASF core is enriched in strong binders),
  not because the models are better there.
- **A usable lower bound does not exist at this accuracy.** The conformal slack is 2.4 pK for the
  energy models and 2.0 for the size (the 95th percentile of the signed residual), so a complex
  predicted at pK 6.5 is certified above pK 4. On LP-PDBBind the bounds certify a 10 µM binder
  for 9–14 % of the complexes and a 1 µM binder for 2–3 %, against 80 % and 66 % of complexes that
  *are* such binders (`lower_bound.png`); below pK 3.7 the constant predictor's single bound
  certifies more complexes than any model, because the models spread their bounds around it.
  Coverage is 94–96 % on LP-PDBBind for the models (the constant 91.5 %: the validation and test
  distributions of that split differ) and conservative on HAC and CleanSplit (96–99 %, small
  calibration sets of 103 and 513 complexes); the quantile-regression bound is slightly more
  conservative (96–98 %) and certifies fewer complexes.
- **Reweighting helps a little and consistently.** ΔMAE −0.03 to −0.04 pK on LP-PDBBind, −0.11
  to −0.12 on HAC and CleanSplit, −0.13 on the relaxed holo complex, intervals excluding zero in
  every case but one. Free-sign and non-negative fits coincide (the free fit only turns `fa_pair`
  negative on two splits) and give the same weight set everywhere: `fa_rep` and `fa_pair` off,
  `fa_sol` 0.24–0.33, `fa_elec` ≈ 0.2, the H-bond terms 0.3–0.4 (CleanSplit drops
  `hbond_bb_sc`). The non-negative vector of the `l1` fit is written as
  `data/affinity/weights_nonneg_<split>.wts`.
- **Ligand size is as good as the energy on LP-PDBBind and worse on the CASF core.** Size
  1.46 vs reweighted 1.45–1.46 on LP-PDBBind; on HAC size 1.50 = unmodified 1.50 vs reweighted
  1.39, and on the core complexes outside the LP-PDBBind training set size is 0.10 pK worse than
  the unmodified score. The core set is built from clusters of ligands per target spanning an
  affinity range, i.e. a miniature of the diverse-ligand regime in which the size relation of
  PDBbind (one fitting binder per pocket) is expected to collapse — the first in-data hint for
  that hypothesis. The real test needs ligand series with new docking (inventory S7 to-do,
  Appendix B run E).
- **The pose is not the limiting factor; the default weights are.** The best hit (oracle, a
  correct pose by construction) predicts worse than the best-scoring pose (1.54 vs 1.49), the
  relaxed holo complex with the default weights is worse than the constant (1.60 vs 1.56) and so
  is the unrelaxed crystal complex (1.57); the reweighting takes all three to 1.46–1.47, the
  same as the docked pose. The information the interface delta carries about affinity does not
  depend on the pose being right, and with the default weights the native pose carries less of
  it than a docked one.
- **Term ablation.** Dropping `fa_pair` changes nothing (its weight is zero in every fit).
  Dropping `fa_sol` as well costs ≤ 0.01 pK on LP-PDBBind and nothing on HAC or CleanSplit, but
  the non-negative enthalpic fit puts every weight except `fa_atr` at zero on all three splits,
  i.e. it becomes a contact count, and the free-sign enthalpic fit gives the H-bond terms
  negative weights. For A.3 step 2 this means: an enthalpy-only energy loses nothing measurable at
  the single-pose level, but what survives of it is `fa_atr`; `fa_sol` and the H-bond terms carry
  the 0.02 pK that distinguishes the energy from a contact count. The weights were refitted per
  set, as decided; the default weights on the reduced sets were not evaluated.
- **Aggregations and pools.** The mean of the 10 best and the filtered mean are no better than
  the best pose in MAE (1.51–1.52 vs 1.49); the std runs change nothing. On all complexes without a
  split the filtered mean over the near-native ensemble of the std runs has the highest Pearson r
  (0.47 vs 0.45 for the best pose) — the only hint that an ensemble average carries information
  the single best pose does not, the question of A.3 step 2.
- **Label sets.** Ki labels are harder than Kd (constant 1.64 vs 1.47 pK) and the models gain
  the same 0.03–0.04 on both halves; IC50 has the tightest distribution (constant 1.24) and the
  largest gain of the calibrated score over the constant (−0.13), with size again equal to the
  energy.
- **Caveats.** PDBbind's label noise (different assays and laboratories) bounds the attainable
  MAE well above zero, but not at 1.5 pK; one set of exact labels, inequalities dropped; the HAC
  and CleanSplit test sets are small (271 / 151, intervals of ±0.13 to ±0.2 pK) and their
  calibration sets smaller (103 / 513); the split decides the number, so cross-paper comparisons
  need the same split; the size baseline is a property of PDBbind (S7 to-do); no clustering, no
  ensemble estimator (A.3 step 2 open).
