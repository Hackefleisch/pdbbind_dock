# Track 3 — score vs. affinity: the interface delta and its reweighting against pK (`affinity/`)

> Methods-level documentation of the `affinity` package: the question, the fixed setup, the
> approaches, the evaluation and the results. Source files:
> [`affinity/data.py`](../affinity/data.py), [`affinity/models.py`](../affinity/models.py),
> [`affinity/evaluate.py`](../affinity/evaluate.py), [`affinity/train.py`](../affinity/train.py),
> [`affinity/report.py`](../affinity/report.py). Paper context:
> [PAPER1_STORY_INVENTORY.md](PAPER1_STORY_INVENTORY.md), story S7 and Appendix A.3 (step 1).
> Written 2026-10-21, when the legacy grid of the zip era was replaced (git history before that
> date). Companion of [ranking.md](ranking.md) (S5) and [degeneracy.md](degeneracy.md) (S6).

## 0. Headline

**The interface delta of the best-scoring pose is a weak predictor of binding affinity, and on
a leakage-controlled split it is a worse one than the ligand's heavy-atom count.** On the
LP-PDBBind test complexes (Kd + Ki, n = 1,874) the calibrated interface delta of the best
`apo_relax` · `perturb` pose reaches Pearson r = 0.31 [0.27, 0.35]; the heavy-atom count alone
reaches 0.42 [0.38, 0.46]. A least-squares reweighting of the seven terms adds +0.065
[+0.046, +0.085] (r = 0.37) and still trails the size, and the terms added to the size gain
+0.007 over the size alone (0.425 vs 0.418): what the interface delta knows about affinity is
mostly how big the ligand is. The number is split-dependent — on the CASF-2016 core set (HAC test,
n = 271) the same four quantities are 0.52 / 0.54 / 0.62 / 0.63, and on all 6,708 labelled
complexes without a split the unmodified score reaches 0.43–0.45 (the legacy r ≈ 0.39 on exact
Kd, recomputed: 0.36–0.39) — but the ordering is not. The pose is not the limit: the best *hit*
(oracle) correlates no better than the best-scoring pose (0.28 vs 0.31), the relaxed holo
complex itself gives 0.28, and the std runs give the same as the perturb runs. The unrelaxed
crystal complex is useless without minimisation (r = −0.11, clashes; reweighted 0.41). The
learned weight set is the same on every split: fa_rep and fa_pair switched off, fa_sol and
fa_elec halved, the hydrogen-bond terms down to a third of their default (§6.6).

![affinity headline figure](../data/figures/affinity/headline_scatter.png)

`data/figures/affinity/headline_scatter.png` — predicted vs experimental pK on the headline cell
for the calibrated interface delta, the reweighted terms and the heavy-atom count; `pools.png`
the Pearson r per pool × aggregation for the three; `weights.png` the learned non-negative
weights per split against the defaults. Regenerate with `uv run python -m affinity.report`.


## 1. Question and setup

**How well does the Rosetta interface delta of a docked pose track the experimental binding
affinity, and how much does a linear reweighting of its seven interface terms add?** This is
step 1 of Appendix A.3: the single-pose baseline that step 2 (the sampled-density free energy,
not implemented, open Paper 1 / Paper 2 decision) would have to beat. The reference is the
*unmodified* interface delta, linearly calibrated to pK; everything else is reported as the
paired difference in Pearson r to it on the same complexes.

| Element | Decision (2026-10-21) |
|---|---|
| Labels | pK = −log10(value / M) of the **exact** (`=`) binding data of `INDEX_general_PL.2020R1.lst`; inequalities (`<`, `<=`, `>`, `~`) are dropped. **Headline `kdki`: Kd and Ki pooled** (6,708 of the 11,444 complete complexes; Kd 3,579, Ki 3,129); **IC50 separately** (4,579; trained and tested on IC50 complexes only, it is assay-dependent); the Kd and the Ki half of the headline test set are also reported |
| Features | the **seven raw interface-delta terms** of the aggregated pose (`raw_delta_{fa_atr, fa_rep, fa_sol, fa_elec, fa_pair, hbond_bb_sc, hbond_sc}`; `idelta_score` is their weighted sum with the default weights, verified to 1e-4 on the extraction), plus the ligand heavy-atom count `ligand_n_heavy` as the size reference |
| Pools | which poses of a complex are aggregated: **one docking run** (complex × protocol, 300 poses; headline `apo_relax` · `perturb` as in S5, `relax` · `perturb` as the realistic holo setting), the **perturb + std runs of one receptor state** (600 poses — the "with Standard poses" ablation), the **relaxed holo complex** the docking started from (`relax_ref`, the native pose after FastRelax) and the **crystal complex** as prepared (`crystal`, unrelaxed) |
| Aggregations | `best` — the pose with the lowest interface delta, the pose a user would take (**headline**); `mean10` — mean of the terms over the 10 lowest-interface-delta poses; `mean` — mean over the poses with a non-positive interface delta (the legacy grid's filtered mean; the plain mean over all 300 poses is dominated by clashing poses and correlates ≈ 0 with pK); `best_hit` — oracle, the lowest-interface-delta pose among the hits (`rmsd_to_crystal` ≤ 2 Å), i.e. the score when the pose is right |
| Splits | **the same three non-random PDBbind splits as S5** (`docking_analysis/splits.csv`): LP-PDBBind (headline; built for leakage-controlled affinity prediction, largest test set), HAC (test = the CASF-2016 core set, the standard scoring-power benchmark), CleanSplit. Training complexes = train ∪ val of the split: the models are least squares with nothing to select. The CASF-2016 core complexes outside a split's training set are its external test (for HAC that is its test set). The Astex overlap exclusion of S5 is irrelevant here (no external set is evaluated) |
| Approaches | `unmodified` · `size` · `reweighted` (free sign) · `non-negative` · `reweighted + size` — §3 |
| Metrics | Pearson r between the predicted and the experimental pK (the CASF "scoring power"), Spearman ρ, RMSE and MAE in pK after the linear fit; percentile-bootstrap 95 % intervals over the evaluation complexes (2,000 resamples) and the **paired Δr to the calibrated interface delta** on the same resamples |
| Headline cell | Kd + Ki · `apo_relax_docking_perturb` · best pose · LP-PDBBind test complexes (n = 1,874) |

What the rewrite lifted from the legacy grid (inventory S7): exact Kd only → Kd + Ki with IC50
separate; random 80/20 split → the three leakage-aware splits plus the CASF core; no intervals →
bootstrap intervals and paired differences; non-negative weights with an MAE loss trained by
gradient descent → closed-form least squares, free-sign and non-negative; relax + apo_relax
perturb poses only → six pools including the std runs and the native structures; uncorrected
RMSDs and totals → the corrected extraction. Not carried over: the RMSD clustering (plain RMSD,
cluster-representative aggregation) — it belongs to the ensemble question of A.3 step 2.

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
- **Split counts** (Kd + Ki): LP-PDBBind train 4,483 (train + val) / test 1,874 / CASF core
  outside the training set 109; HAC train 6,434 / test 271 (= core); CleanSplit train 5,171 /
  test 151 / core outside training 267. IC50: LP-PDBBind 2,843 / 1,669; HAC 4,579 / 0 and
  CleanSplit 4,048 / 0 (neither test set contains an IC50 complex, so IC50 is LP-PDBBind only).
- The unmodified score is also correlated with pK on **all** labelled complete complexes per
  pool × aggregation × label set, with no split and no fit (`baseline_all.csv`, table 7 of the
  report) — the numbers comparable with the legacy grid's r ≈ 0.39 / 0.38 / 0.42.

## 3. Approaches (`affinity.models`)

All five map a per-complex feature vector to a predicted pK and are fitted by least squares on
the training complexes; nothing is tuned and nothing is random, so one fit per cell and the
bootstrap over complexes gives the intervals.

- **`unmodified`** — pK ≈ a · `idelta_score` + b. The reference. Pearson r is invariant to a and
  b (it is the correlation of −interface delta with pK); the calibration only makes the RMSE
  meaningful.
- **`size`** — pK ≈ a · `ligand_n_heavy` + b. Affinity grows with ligand size in PDBbind, and so
  does every contact term, so this is the reference any energy-based score has to beat.
- **`reweighted`** (free sign) — pK ≈ b − Σ_t w_t x_t over the seven raw terms, ordinary least
  squares. The energy weights w = −coef; a positive w means a more negative term favours
  binding. Reported normalised to fa_atr = 0.8 like the ranking weights of S5.
- **`non-negative`** — the same with w_t ≥ 0 (non-negative least squares on the centred data,
  the intercept follows from the means), so the vector is a Rosetta weight set
  (`weights_nonneg_<split>.wts`).
- **`reweighted + size`** — the free-sign fit plus `ligand_n_heavy`: does the energy add to the
  size?

## 4. Evaluation (`affinity.evaluate`)

Per cell (label set × split × pool × aggregation) the five models are evaluated on the test
complexes of the split, on the Kd and the Ki half of the headline test set, and on the CASF-2016
core complexes outside the training set. Pearson r, Spearman ρ, RMSE and MAE with their
percentile-bootstrap 95 % intervals over complexes (2,000 resamples, seed 0); the paired Δr and
ΔRMSE to the calibrated interface delta are computed on the same resamples. Evaluation sets with
fewer than 20 complexes are skipped.

## 5. How to run

```bash
uv run python -m affinity.train                 # feature cache + the default grid (6 pools), ~10 min on 1 core
uv run python -m affinity.train --pool all      # all ten pools (adds the crystal-receptor and the std runs)
uv run python -m affinity.report                # tables + figures -> data/figures/affinity/
```

`affinity.train` writes `data/affinity/{features.parquet, labels.csv, baseline_all.csv,
results.csv, weights.csv, weights_nonneg_<split>.wts, predictions.parquet, results.json,
results.md, train.log}`; `affinity.report` writes `data/figures/affinity/affinity_tables.md`
(eight tables), `affinity_results.csv`, `headline_scatter.png`, `pools.png`, `weights.png`.

## 6. Results (run of 2026-10-21)

The default grid: 2 label sets × 3 splits × (4 run pools × 4 aggregations + 2 reference
structures) = 108 cells, 1,170 result rows (`data/affinity/results.csv`, collected with the
per-complex predictions and the fitted weights in `data/figures/affinity/affinity_tables.md`,
eight tables). Cells below: Pearson r [95 % bootstrap interval] on the evaluation complexes;
`Δr` the paired difference to the calibrated interface delta on the same complexes.

### 6.1 Headline — Kd + Ki, `apo_relax` · `perturb`, best pose, test complexes of each split

| approach | LP-PDBBind (n = 1,874) r [95 % CI] | LP-PDBBind Δr [95 % CI] | HAC (n = 271) r [95 % CI] | HAC Δr [95 % CI] | CleanSplit (n = 151) r [95 % CI] | CleanSplit Δr [95 % CI] |
|---|---|---|---|---|---|---|
| unmodified (interface delta, calibrated) | 0.308 [0.270, 0.346] | — | 0.520 [0.429, 0.597] | — | 0.493 [0.381, 0.598] | — |
| ligand size (heavy atoms) | 0.418 [0.380, 0.457] | +0.110 [+0.070, +0.150] | 0.535 [0.444, 0.616] | +0.014 [-0.097, +0.127] | 0.639 [0.557, 0.717] | +0.146 [+0.029, +0.265] |
| reweighted (free sign) | 0.373 [0.332, 0.411] | +0.065 [+0.046, +0.085] | 0.624 [0.556, 0.684] | +0.104 [+0.048, +0.170] | 0.616 [0.528, 0.696] | +0.123 [+0.060, +0.199] |
| reweighted (non-negative) | 0.372 [0.332, 0.410] | +0.064 [+0.045, +0.084] | 0.626 [0.558, 0.685] | +0.105 [+0.050, +0.170] | 0.613 [0.526, 0.694] | +0.120 [+0.058, +0.194] |
| reweighted (free sign) + ligand size | 0.425 [0.388, 0.460] | +0.117 [+0.096, +0.138] | 0.629 [0.562, 0.689] | +0.109 [+0.045, +0.181] | 0.653 [0.576, 0.723] | +0.160 [+0.089, +0.241] |

The other best-pose pools (`relax` · `perturb`, perturb + std of either state, relaxed holo
complex, crystal complex) are in table 2 of `affinity_tables.md`; on LP-PDBBind their unmodified
r is 0.331 / 0.331 / 0.319 / 0.277 / −0.107 and their reweighted r 0.387 / 0.386 / 0.378 /
0.396 / 0.408.

### 6.2 Pools × aggregations — Kd + Ki, LP-PDBBind test complexes

`pose RMSD` / `pose ≤ 2 Å`: the symmetry-corrected RMSD to the crystal ligand of the aggregated
pose on the test complexes (for the means: the mean RMSD of the averaged poses).

| pool | aggregation | n | pose RMSD median (Å) | pose ≤ 2 Å (%) | unmodified r [95 % CI] | ligand size r | reweighted Δr [95 % CI] | non-negative Δr [95 % CI] | reweighted + size Δr [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|
| apo_relax · perturb | best pose | 1,874 | 1.78 | 53 | 0.308 [0.270, 0.346] | 0.418 | +0.065 [+0.046, +0.085] | +0.064 [+0.045, +0.084] | +0.117 [+0.096, +0.138] |
| apo_relax · perturb | mean of 10 best | 1,874 | 3.42 | 23 | 0.271 [0.231, 0.310] | 0.418 | +0.091 [+0.066, +0.116] | +0.093 [+0.071, +0.115] | +0.146 [+0.119, +0.172] |
| apo_relax · perturb | mean (interface delta ≤ 0) | 1,874 | 6.03 | 0 | 0.195 [0.153, 0.234] | 0.418 | +0.138 [+0.103, +0.172] | +0.126 [+0.095, +0.156] | +0.217 [+0.183, +0.253] |
| apo_relax · perturb | best hit (oracle) | 1,718 | 0.89 | 100 | 0.280 [0.238, 0.321] | 0.447 | +0.091 [+0.066, +0.117] | +0.091 [+0.068, +0.115] | +0.145 [+0.118, +0.172] |
| relax · perturb | best pose | 1,874 | 0.74 | 69 | 0.331 [0.289, 0.371] | 0.418 | +0.056 [+0.040, +0.073] | +0.052 [+0.037, +0.067] | +0.097 [+0.079, +0.115] |
| relax · perturb | mean of 10 best | 1,874 | 2.62 | 36 | 0.308 [0.267, 0.345] | 0.418 | +0.094 [+0.072, +0.115] | +0.086 [+0.068, +0.104] | +0.128 [+0.105, +0.151] |
| relax · perturb | mean (interface delta ≤ 0) | 1,874 | 5.88 | 0 | 0.206 [0.165, 0.245] | 0.418 | +0.145 [+0.112, +0.176] | +0.125 [+0.097, +0.152] | +0.213 [+0.178, +0.247] |
| relax · perturb | best hit (oracle) | 1,803 | 0.57 | 100 | 0.316 [0.276, 0.356] | 0.439 | +0.071 [+0.055, +0.088] | +0.065 [+0.050, +0.081] | +0.111 [+0.092, +0.130] |
| apo_relax · perturb + std | best pose | 1,874 | 0.81 | 67 | 0.331 [0.292, 0.371] | 0.418 | +0.055 [+0.037, +0.072] | +0.054 [+0.037, +0.071] | +0.098 [+0.079, +0.117] |
| apo_relax · perturb + std | mean of 10 best | 1,874 | 1.44 | 60 | 0.314 [0.274, 0.352] | 0.418 | +0.070 [+0.051, +0.090] | +0.068 [+0.051, +0.087] | +0.114 [+0.092, +0.135] |
| apo_relax · perturb + std | mean (interface delta ≤ 0) | 1,874 | 4.32 | 0 | 0.237 [0.197, 0.276] | 0.418 | +0.112 [+0.082, +0.142] | +0.103 [+0.077, +0.130] | +0.183 [+0.151, +0.216] |
| apo_relax · perturb + std | best hit (oracle) | 1,830 | 0.60 | 100 | 0.306 [0.267, 0.346] | 0.425 | +0.073 [+0.053, +0.092] | +0.072 [+0.054, +0.091] | +0.118 [+0.096, +0.140] |
| relax · perturb + std | best pose | 1,874 | 0.51 | 80 | 0.319 [0.277, 0.359] | 0.418 | +0.059 [+0.044, +0.074] | +0.058 [+0.043, +0.073] | +0.095 [+0.079, +0.112] |
| relax · perturb + std | mean of 10 best | 1,874 | 0.54 | 82 | 0.315 [0.272, 0.354] | 0.418 | +0.070 [+0.055, +0.084] | +0.066 [+0.052, +0.080] | +0.101 [+0.085, +0.118] |
| relax · perturb + std | mean (interface delta ≤ 0) | 1,874 | 3.70 | 0 | 0.316 [0.276, 0.356] | 0.418 | +0.067 [+0.050, +0.084] | +0.059 [+0.046, +0.072] | +0.115 [+0.095, +0.137] |
| relax · perturb + std | best hit (oracle) | 1,863 | 0.48 | 100 | 0.310 [0.268, 0.350] | 0.418 | +0.077 [+0.061, +0.091] | +0.073 [+0.058, +0.087] | +0.109 [+0.091, +0.126] |
| relaxed holo complex (docking reference) | best pose | 1,874 | 0.29 | 96 | 0.277 [0.237, 0.317] | 0.418 | +0.118 [+0.098, +0.138] | +0.117 [+0.099, +0.135] | +0.139 [+0.119, +0.159] |
| crystal complex (input, unrelaxed) | best pose | 1,874 | 0.00 | 100 | -0.107 [-0.162, -0.062] | 0.418 | +0.515 [+0.468, +0.574] | +0.508 [+0.460, +0.567] | +0.538 [+0.491, +0.596] |

### 6.3 The Kd and the Ki half of the headline test set, and IC50

LP-PDBBind, `apo_relax` · `perturb`, best pose; the Kd + Ki model evaluated on each half:

| approach | Kd + Ki (n = 1,874) r [95 % CI] | Kd + Ki Δr [95 % CI] | Kd only (n = 921) r [95 % CI] | Kd only Δr [95 % CI] | Ki only (n = 953) r [95 % CI] | Ki only Δr [95 % CI] |
|---|---|---|---|---|---|---|
| unmodified (interface delta, calibrated) | 0.308 [0.270, 0.346] | — | 0.302 [0.245, 0.360] | — | 0.288 [0.233, 0.341] | — |
| ligand size (heavy atoms) | 0.418 [0.380, 0.457] | +0.110 [+0.070, +0.150] | 0.359 [0.303, 0.412] | +0.057 [-0.003, +0.119] | 0.481 [0.424, 0.532] | +0.192 [+0.141, +0.244] |
| reweighted (free sign) | 0.373 [0.332, 0.411] | +0.065 [+0.046, +0.085] | 0.354 [0.295, 0.413] | +0.053 [+0.023, +0.084] | 0.373 [0.317, 0.424] | +0.084 [+0.057, +0.110] |
| reweighted (non-negative) | 0.372 [0.332, 0.410] | +0.064 [+0.045, +0.084] | 0.353 [0.294, 0.412] | +0.051 [+0.022, +0.081] | 0.373 [0.319, 0.424] | +0.085 [+0.058, +0.110] |
| reweighted (free sign) + ligand size | 0.425 [0.388, 0.460] | +0.117 [+0.096, +0.138] | 0.405 [0.349, 0.458] | +0.103 [+0.072, +0.135] | 0.432 [0.377, 0.481] | +0.144 [+0.115, +0.173] |

IC50, trained and tested on the IC50 complexes (LP-PDBBind; the HAC and CleanSplit test sets
contain no IC50 complex):

| approach | LP-PDBBind (n = 1,669) r [95 % CI] | LP-PDBBind Δr [95 % CI] |
|---|---|---|
| unmodified (interface delta, calibrated) | 0.419 [0.379, 0.454] | — |
| ligand size (heavy atoms) | 0.514 [0.477, 0.550] | +0.095 [+0.055, +0.138] |
| reweighted (free sign) | 0.471 [0.435, 0.506] | +0.052 [+0.030, +0.075] |
| reweighted (non-negative) | 0.475 [0.439, 0.510] | +0.056 [+0.037, +0.077] |
| reweighted (free sign) + ligand size | 0.544 [0.511, 0.578] | +0.126 [+0.098, +0.153] |

### 6.4 CASF-2016 core complexes outside the training set — Kd + Ki, `apo_relax` · `perturb`, best pose

For HAC the core set *is* the test set (§6.1); for the other splits the core complexes that
fell into their training set are removed (109 of 271 remain outside LP-PDBBind's training set,
267 outside CleanSplit's).

| approach | trained on LP-PDBBind (n = 109) r [95 % CI] | LP-PDBBind Δr [95 % CI] | trained on HAC (n = 271) r [95 % CI] | HAC Δr [95 % CI] | trained on CleanSplit (n = 267) r [95 % CI] | CleanSplit Δr [95 % CI] |
|---|---|---|---|---|---|---|
| unmodified (interface delta, calibrated) | 0.483 [0.334, 0.609] | — | 0.520 [0.429, 0.597] | — | 0.514 [0.427, 0.596] | — |
| ligand size (heavy atoms) | 0.632 [0.502, 0.738] | +0.149 [+0.008, +0.286] | 0.535 [0.444, 0.616] | +0.014 [-0.097, +0.127] | 0.525 [0.433, 0.609] | +0.011 [-0.105, +0.120] |
| reweighted (free sign) | 0.553 [0.422, 0.661] | +0.070 [-0.001, +0.151] | 0.624 [0.556, 0.684] | +0.104 [+0.048, +0.170] | 0.623 [0.555, 0.685] | +0.108 [+0.048, +0.178] |
| reweighted (non-negative) | 0.551 [0.420, 0.660] | +0.068 [-0.000, +0.148] | 0.626 [0.558, 0.685] | +0.105 [+0.050, +0.170] | 0.622 [0.554, 0.684] | +0.108 [+0.049, +0.177] |
| reweighted (free sign) + ligand size | 0.617 [0.492, 0.720] | +0.134 [+0.053, +0.227] | 0.629 [0.562, 0.689] | +0.109 [+0.045, +0.181] | 0.624 [0.554, 0.686] | +0.110 [+0.044, +0.183] |

### 6.5 The unmodified interface delta on all labelled complete complexes (no split, no fit)

Pearson r of −`idelta_score` with pK over every labelled complete complex (the best-pose rows of
all ten pools plus the filtered mean of the std runs; the full table is table 7 of
`affinity_tables.md`):

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

### 6.6 Learned energy weights — Kd + Ki, `apo_relax` · `perturb`, best pose

Energy weights w (pK = b − Σ w · term) normalised to fa_atr = 0.8 like the ranking weights of
S5; the free-sign fit can give negative weights. For comparison, the S5 *ranking* weights
(LP-PDBBind, Rosetta start) are fa_rep 0.50, fa_sol 0.33, fa_elec 0.33, fa_pair 0.14,
hbond_bb_sc 1.29, hbond_sc 1.08.

| approach · split | fa_atr | fa_rep | fa_sol | fa_elec | fa_pair | hbond_bb_sc | hbond_sc | intercept (pK) | n train |
|---|---|---|---|---|---|---|---|---|---|
| Rosetta default | 0.80 | 0.40 | 0.60 | 0.42 | 0.80 | 1.30 | 1.30 | — | — |
| reweighted (non-negative) · LP-PDBBind | 0.80 | 0.00 | 0.28 | 0.18 | 0.00 | 0.41 | 0.35 | 3.15 | 4,483 |
| reweighted (non-negative) · HAC | 0.80 | 0.00 | 0.31 | 0.20 | 0.00 | 0.31 | 0.29 | 3.56 | 6,434 |
| reweighted (non-negative) · CleanSplit | 0.80 | 0.07 | 0.28 | 0.18 | 0.00 | 0.00 | 0.35 | 3.47 | 5,171 |
| reweighted (free sign) · LP-PDBBind | 0.80 | -0.02 | 0.28 | 0.18 | -0.45 | 0.40 | 0.35 | 3.17 | 4,483 |
| reweighted (free sign) · HAC | 0.80 | -0.03 | 0.31 | 0.20 | -0.71 | 0.29 | 0.28 | 3.59 | 6,434 |
| reweighted (free sign) · CleanSplit | 0.80 | 0.07 | 0.28 | 0.18 | -0.44 | -0.05 | 0.33 | 3.48 | 5,171 |
| reweighted (free sign) + ligand size · LP-PDBBind | 0.80 | -0.03 | 0.53 | 0.28 | -0.30 | 1.20 | 1.33 | 2.54 | 4,483 |
| reweighted (free sign) + ligand size · HAC | 0.80 | -0.07 | 0.58 | 0.30 | -0.61 | 1.20 | 1.21 | 3.04 | 6,434 |
| reweighted (free sign) + ligand size · CleanSplit | 0.80 | 0.03 | 0.54 | 0.29 | -0.48 | 0.75 | 1.22 | 2.95 | 5,171 |

## 7. Reading

- **Ligand size beats the energy on the leakage-controlled split.** On LP-PDBBind the heavy-atom
  count (r = 0.418) is a better affinity predictor than the interface delta (0.308) and than its
  best reweighting (0.373); adding the seven terms to the size gains +0.007. On the CASF-2016 core
  (HAC test) the terms do carry something beyond size (reweighted 0.624 vs size 0.535), but that
  test set shares protein families with the training set and has 271 complexes. The LP-PDBBind
  numbers are the ones to quote; the CASF number (unmodified 0.52) is the one comparable with the
  literature `[CITE]`.
- **The pose is not the limiting factor.** The best hit (oracle, always a correct pose) correlates
  no better than the best-scoring pose (0.280 vs 0.308 on apo_relax · perturb, 0.316 vs 0.331 on
  relax · perturb), the relaxed holo complex gives 0.277, and pooling the std run (a near-native
  pose in 67–80 % of the complexes) changes nothing. The 0.3–0.45 is what the Rosetta interface
  delta knows about affinity when the pose is right — the T2 result from the other side: the
  energy function, not the sampler, is the bottleneck.
- **Reweighting is a consistent but small gain.** +0.05 to +0.12 in r depending on the split and
  pool, intervals excluding zero everywhere except relax · perturb on HAC. The non-negative and
  the free-sign fit coincide (the free fit only turns fa_pair negative) and give the same weight set
  on the three splits: fa_rep off (repulsion is a pose-quality term, not an affinity term), fa_pair
  off (as in the S5 ranking weights), fa_sol and fa_elec halved, hbond terms at a quarter to a
  third of their default. With the size in the model the H-bond weights return to their defaults
  and fa_sol to 0.55: the energy terms then describe the *quality* of the contacts and the size
  their *number*.
- **Aggregations.** For the perturb runs the mean of the 10 best and the filtered mean are worse
  than the best pose (they average in wrong poses); for the std runs, where most poses are
  near-native, the filtered mean is as good as or slightly better than the best pose (all
  complexes, no split: 0.470 vs 0.452 for relax · std, 0.460 vs 0.450 for relax · perturb + std).
  That is the only hint in the data that an ensemble average carries information the single best
  pose does not — the question of A.3 step 2, not evidence for it.
- **Label sets.** Ki correlates with size much more than Kd does (0.481 vs 0.359 on the LP-PDBBind
  halves) while the interface delta is the same on both (0.29 / 0.30); IC50 behaves like Kd + Ki
  with higher numbers (unmodified 0.42, size 0.51, reweighted + size 0.54).
- **Relaxation is required for the native pose.** The unrelaxed crystal complex scores −0.11
  because fa_rep dominates (clashes after protonation); the reweighting recovers 0.41 by switching
  fa_rep off. The relaxed holo complex (0.28) is no better than a docked pose — the docked pose
  has been minimised against the same score function and is at least as well packed.
- **Caveats.** Pearson r on pK with PDBbind's label noise (different assays and laboratories; the
  attainable r is well below 1); one set of exact labels, inequalities dropped; the HAC and
  CleanSplit test sets are small (271 / 151 complexes, intervals of ±0.08 to ±0.11); the split
  decides the number, so cross-paper comparisons need the same split; no clustering, no ensemble
  estimator (A.3 step 2 open).

