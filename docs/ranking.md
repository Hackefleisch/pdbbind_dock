# Track 2 — pose-ranking models (`ranking/`)

> Methods-level documentation of the `ranking` package: the pose-ranking models of Luca
> Friebe's thesis repository ported to the full extraction, what the port changes, how runs
> are evaluated, and the results. Source files: [`ranking/data.py`](../ranking/data.py),
> [`ranking/models.py`](../ranking/models.py), [`ranking/evaluate.py`](../ranking/evaluate.py),
> [`ranking/train.py`](../ranking/train.py), [`ranking/report.py`](../ranking/report.py).
> Paper context: [PAPER1_STORY_INVENTORY.md](PAPER1_STORY_INVENTORY.md), story S5 and §6.1.
> Written 2026-10-19.

## 1. Origin and credit

The models come from Luca Friebe's repository
[`lucafrb/rosettaligand-vs-posebuster`](https://github.com/lucafrb/rosettaligand-vs-posebuster).
Its `main` branch is the thesis-era code on the old docking results (150 poses per protocol);
the branch **`test_new_architecture`** (last commit `ecce837`, "reweighter and added prints")
applies the pipelines to the current 300-pose docking: the numbers in its
`ML_pipeline_description.md` are on 388 PoseBusters and 84 Astex complexes, and its Rosetta
baseline is reproduced exactly by this port with the stored labels (§7.1). The branch holds
four self-contained pipelines:

| Pipeline | Content | Ported |
|---|---|---|
| `classification/` | tinygrad MLP, good/bad pose (BCE) | no — judged "inappropriate" in his description |
| `regression/` | tinygrad MLP, predicts the RMSD (MSE/Huber) | no — same |
| `ranking/` | `data_loader.py` + `ranking.py`: fully connected net `n → 64 → 32 → 16 → 1`, InfoNCE + BCE loss, one complex per batch, best epoch by Success@1/5; `test_ranking.py` | **yes** (`Ranker`) |
| `reweighting/` | `reweight.py`: bias-free `nn.Linear(n, 1)` on the same features and loss; `test_reweighting.py` | **yes** (`Reweight`) |

His reported results (`ML_pipeline_description.md`, Success@1 / Success@5 in %, relax · perturb
poses, labels = stored `rmsd_to_input` ≤ 2 Å, random split by complex, PDBbind = validation fold):

| | PoseBusters | Astex | PDBbind |
|---|---|---|---|
| Rosetta score | 70.88 / 83.25 | 84.52 / 94.05 | 71.08 / 84.41 |
| Reweighter | 74.48 / 86.60 | 84.52 / 94.05 | 73.29 / 84.96 |
| Ranker | 74.74 / 85.31 | 86.90 / 94.05 | 73.39 / 85.34 |

The port (≈ 1,000 lines in `ranking/`, half of it docstrings and CLI) covers the two models, the loss, the data preprocessing,
the model selection and the Success@k evaluation, rewritten against `docked.parquet`. Luca is
credited in the package and module docstrings; his repository remains the reference for the
thesis-era numbers (inventory §3).

## 2. What the port keeps and what it changes

| Aspect | Luca's branch `test_new_architecture` | Port (`ranking/`) |
|---|---|---|
| Data | `datasets/*.pkl`: pandas frames built from the H5 files, one protocol (`docking_perturb_pose_relax` = `relax` · `perturb`), columns `pdb_id, protocol, run_id, rmsd_to_input` + 25 energy terms | `data/docking_stats/<set>_full/docked.parquet`, any protocol(s) (`--protocol`), complete complexes only |
| Features | 25 requested terms; those with zero standard deviation dropped → in practice the **seven non-zero interface-delta terms** (`fa_atr, fa_rep, fa_sol, fa_elec, fa_pair, hbond_bb_sc, hbond_sc`; his saved models have 7 inputs); `fa_rep` clipped at 50; `RobustScaler` fitted on all data; scaled values clipped to ±10 | the same seven terms as `raw_delta_*` (`--features delta`, default); optionally the 20 constraint-free `raw_*` total terms (`total`) or both; the same clips; the scaler is fitted on the **training** poses only and stored as JSON |
| Label | stored `rmsd_to_input` ≤ 2 Å: no symmetry correction, and for `apo_relax` the input ligand is the placed-back one, not the crystal pose | symmetry-corrected `rmsd_to_crystal` ≤ 2 Å (`--rmsd-col`, `--cutoff`; `rmsd_to_input_legacy` reproduces his labels) |
| Ranking unit | complex (`pdb_id`), 300 poses | docking run = complex × protocol, 300 poses (identical for one protocol) |
| PDBbind split | `StratifiedGroupKFold(5, shuffle, seed 0)` by complex on his RMSD relevance bins; fold 0 = validation; the "PDBbind" numbers are on that fold | **LP-PDBBind** train / val / test (default, from `docking_analysis/splits.csv`); `hac` (CASF-2016 core as test, CASF validation set as val); `cleansplit` (10 % of train held out as val); `random` = his split |
| External sets | PoseBusters and Astex evaluated with a model trained on all of PDBbind; **43 of the 84 Astex entries are PDBbind complexes** (same PDB id), no PoseBusters entry is | the overlapping entries are removed from train and val (`--no-exclude-overlap` keeps them) |
| Models | `Reweight`: `nn.Linear(n, 1, bias=False)`; `Ranker`: `n → 64 → 32 → 16 → 1`, ReLU | identical; `--init rosetta` starts the reweighting at the Rosetta weights (§4) |
| Loss | `BCEWithLogitsLoss` over the batch + β · InfoNCE per complex, τ = 0.5, β = 1; one complex per batch; Adam, lr 1e-4 (reweight) / 1e-5 (ranker); 50 epochs | identical defaults (`--beta --tau --lr --epochs --batch-runs`); optional `--patience` |
| Model selection | best validation Success@1, ties broken by Success@5; `torch.compile` | identical rule; the untrained model is epoch 0 and eligible; no compile |
| Evaluation | Success@1 / @5 (`test_*.py`: sort by score, hit if RMSD ≤ 2 Å), pooled ROC AUC on validation | Success@1 / 5 / 10 per run with the benchmark definition, percentile-bootstrap 95 % CIs, paired difference to the interface delta on the same runs, pooled and per-run AUC; the Rosetta baselines (`idelta_score`, corrected `total_score`) evaluated on the same runs |
| Environment | TensorFlow / Keras / tinygrad pinned but unused (stray `keras` import in `reweight.py`), MPS device | the project venv (torch is already a dependency), CUDA or CPU |

## 3. Data

- **Poses.** `load_poses(set, protocols, feature_cols)` reads the meta columns (`pdb, protocol,
  seq_id, total_score, idelta_score, rmsd_to_crystal, rmsd_to_input` and their `*_legacy`
  values) plus the feature columns of the requested protocol(s), keeps the complete complexes
  (`complexes.complete`) and sorts by (pdb, protocol, seq_id), so every docking run is a
  contiguous block of 300 rows (`Grouped.offsets`). One protocol of PDBbind is 3.43 M rows
  (11,444 complexes) and loads in under a second.
- **Features.** `delta`: `raw_delta_{fa_atr, fa_rep, fa_sol, fa_elec, fa_pair, hbond_bb_sc,
  hbond_sc}`, the only interface-delta terms that are not identically zero; the interface
  delta is their weighted sum with the default weights (`scfx_weights.json`; verified to
  1e-4). `total`: the 20 `raw_*` terms without the four constraint terms (restraint energies)
  and `chainbreak` (always zero); the corrected `total_score` is their weighted sum. The delta
  `fa_rep` is capped at 50 before scaling (99 % of the poses are below 100, the maximum is
  34,000); `RobustScale` = (x − median) / IQR of the training poses, clipped to ±10.
- **Labels.** Hit ⇔ `rmsd_to_crystal` ≤ 2 Å. relax · perturb: 5.7 % of the poses are hits,
  521 of the 11,444 runs have none (the InfoNCE term skips them, the BCE term uses them);
  apo_relax · perturb: 4.3 % and 1,317 runs without a hit.
- **PDBbind split** (complete complexes): LP-PDBBind train 5,882 / val 1,566 / test 3,572,
  424 unassigned (dropped); HAC test 271 (CASF-2016 core) / val 178; CleanSplit test 151,
  1,949 unassigned. The 43 Astex overlaps fall into LP-PDBBind train/val for 32 of them; those
  are dropped from training (`results.json: overlap_excluded`).

## 4. Models and training

- `Reweight` scores a pose as **w · z** (z = scaled features, no bias). A learned reweighting
  of the energy terms; `weights.csv` reports the weights on the scaled features, on the raw
  terms, and flipped to Rosetta's lower-is-better sign and normalised to `fa_atr = 0.8` for
  comparison with the defaults.
- `Ranker` is Luca's MLP `n → 64 → 32 → 16 → 1` with ReLU, no dropout, no weight decay (he
  found neither necessary).
- **Loss** (`BCEPlusInfoNCE`): `BCEWithLogits(score, hit)` over all poses of the batch plus
  β × mean over the runs of `logsumexp(s/τ) − logsumexp(s_hits/τ)`, the softmax cross-entropy
  that puts the probability mass of a run on its hits (after Weng 2021, *Contrastive
  representation learning*). Runs without a hit contribute to the BCE term only.
- **Training** (`fit`): Adam; one docking run per optimiser step by default (`--batch-runs`
  pools several); fixed number of epochs (50). After every epoch the validation Success@1 and
  Success@5 are computed and the best epoch is kept (Success@1, ties by Success@5). The
  untrained model is evaluated as epoch 0 and can be selected.
- **Rosetta initialisation** (`--init rosetta`, reweight only): with z = (x − median)/IQR the
  interface delta Σ w_t x_t equals Σ (w_t IQR_t) z_t + const, so the weights −w_t·IQR_t
  reproduce the interface-delta ranking exactly (up to the clipped poses). Epoch 0 of such a
  run therefore shows the baseline on the validation set, and training has to beat it there
  to be selected — the cleanest form of the "can weights fix it" test.

## 5. Evaluation

- Success@k per docking run as in `docking_analysis.benchmark`: poses sorted by the score
  (stable sort, lower is better; model scores are negated), success when the rank of the
  first hit is < k. `train.py` prints a **sanity line** with the full-set baseline Success@k
  of the training protocol, which must equal the benchmark tables (relax · perturb:
  71.98 / 84.55 / 88.38 for the interface delta).
- The rates are percentages of runs; `[95 % CI]` is the percentile bootstrap over runs
  (2,000 resamples); `Δ` is the paired difference to the interface delta on the same runs with
  its bootstrap interval — the number that decides whether a model helps.
- AUC: pooled over all poses of a set (Luca's `roc_auc`) and the mean of the within-run AUCs.
  The pooled AUC of `total_score` is ≈ 0.5 by construction (totals are not comparable across
  complexes); the per-run AUC is the meaningful one for it.
- Test sets: the PDBbind test complexes of the split, PoseBusters (388 complete) and Astex (84
  complete), each for the training protocol(s) and for any `--eval-protocols`, which shows
  whether a model trained on relax poses transfers to the apo_relax setting and vice versa.

## 6. How to run

```bash
uv run python -m ranking.train --model reweight --init rosetta                     # relax·perturb, delta, LP-PDBBind
uv run python -m ranking.train --model ranker --eval-protocols relax_docking_perturb apo_relax_docking_perturb
uv run python -m ranking.train --model reweight --init rosetta --protocol apo_relax_docking_perturb
uv run python -m ranking.train --model reweight --split random --rmsd-col rmsd_to_input_legacy \
    --no-exclude-overlap --run-name luca_setup_reweight                            # Luca's setup
uv run python -m ranking.report                                                    # all runs -> data/figures/ranking/
```

A run takes 15–30 min on one GPU (the per-run Python loop dominates, not the model). Outputs
under `data/ranking/<run>/`: `results.md` / `results.json` (rates, intervals, AUCs, split
counts, excluded entries, the full epoch history), `per_run.parquet` (per run and scorer:
`n_hit, first_hit_rank, top1, top5, top10, auc`), `model.pt`, `scaler.json`, `history.csv`,
`weights.csv` (reweight), `train.log`, `config.json`. `ranking.report` writes
`data/figures/ranking/ranking_tables.md` and `ranking_results.csv`.

## 7. Results (2026-10-19)

Ten runs (`data/ranking/<run>/`, collected in `data/figures/ranking/ranking_tables.md` and
`ranking_results.csv`), one seed each, 50 epochs, defaults of §4 unless stated. Rates are
Success@k in percent of docking runs; `Δ [lo, hi]` is the paired difference to the interface
delta on the same runs with its bootstrap 95 % interval.

### 7.1 Validation of the port: Luca's setup

`luca_setup_reweight` / `luca_setup_ranker`: relax · perturb, labels = stored
`rmsd_to_input_legacy` ≤ 2 Å, random split by complex (validation fold = 2,289 complexes,
reported as "PDBbind"), no Astex exclusion. Success@1 / Success@5:

| Scorer | PoseBusters (388) | Astex (84) | PDBbind fold |
|---|---|---|---|
| Rosetta interface delta — his description | 70.88 / 83.25 | 84.52 / 94.05 | 71.08 / 84.41 |
| Rosetta interface delta — port | 70.9 / 83.2 | 84.5 / 94.0 | 71.5 / 83.7 |
| Reweighter — his description | 74.48 / 86.60 | 84.52 / 94.05 | 73.29 / 84.96 |
| Reweighter — port | 75.0 / 86.1 | 85.7 / 92.9 | 74.3 / 84.9 |
| Ranker — his description | 74.74 / 85.31 | 86.90 / 94.05 | 73.39 / 85.34 |
| Ranker — port | 74.7 / 85.3 | 85.7 / 94.0 | 74.0 / 85.6 |

The baselines on the external sets are identical and the models agree within one complex
(0.3 points on PoseBusters, 1.2 on Astex); the PDBbind fold is a different random draw over
a slightly different complex set, hence ±1 point. The port reproduces his pipeline. In this
setup the gains are +2.8 / +2.4 points Success@1 on the PDBbind fold (reweighter / ranker,
intervals [+1.6, +3.9] and [+1.4, +3.5]) and +4.1 / +3.9 on PoseBusters.

**Label effect isolated** (`luca_split_corrected_labels_*`: the same random split and no Astex
exclusion, but `rmsd_to_crystal` labels): baseline 72.2 / 84.8 on the fold, reweighter 74.5
(+2.4 [+1.2, +3.5]), ranker 74.5 (+2.4 [+1.4, +3.3]); PoseBusters +2.6 [+0.3, +4.9] for both,
Astex −1.2 for both. The corrected labels change nothing about the gain on a random split; the
difference to §7.2 comes from the split.

### 7.2 Corrected labels, LP-PDBBind split, Astex excluded from training

Success@1 with Δ to the interface delta; PDBbind = the 3,572 LP-PDBBind test complexes.

**Trained and evaluated on `relax` · `perturb`**

| Scorer | PDBbind test | PoseBusters | Astex |
|---|---|---|---|
| interface delta (baseline) | 75.3 | 72.9 | 85.7 |
| corrected total score | 74.2 (−1.0 [−2.2, +0.1]) | 72.4 (−0.5 [−3.6, +2.6]) | 83.3 (−2.4 [−8.3, +3.6]) |
| reweight, Rosetta init | 75.6 (+0.4 [−0.5, +1.3]) | 75.3 (+2.3 [+0.3, +4.4]) | 84.5 (−1.2 [−3.6, +0.0]) |
| reweight, random init | 75.7 (+0.4 [−0.4, +1.3]) | 75.5 (+2.6 [+0.3, +4.9]) | 84.5 (−1.2 [−4.8, +2.4]) |
| ranker | 75.8 (+0.6 [−0.3, +1.5]) | 75.8 (+2.8 [+0.8, +4.9]) | 83.3 (−2.4 [−6.0, +0.0]) |
| reweight on the 20 total terms, Rosetta init | 74.2 (−1.0 [−2.2, +0.1]) | 72.2 (−0.8 [−3.9, +2.3]) | 83.3 (−2.4 [−8.3, +3.6]) |

Success@5: PDBbind 88.1 → 88.2–88.4 (Δ ≤ +0.3, intervals include 0); PoseBusters 83.5 →
85.8–86.3 (+2.3 to +2.8, intervals exclude 0); Astex 91.7 → 90.5–92.9. Per-run AUC on the
PDBbind test complexes: interface delta 0.909, ranker 0.916, reweighters 0.896–0.905.

**Trained and evaluated on `apo_relax` · `perturb`** (the paper's headline setting)

| Scorer | PDBbind test | PoseBusters | Astex |
|---|---|---|---|
| interface delta (baseline) | 59.5 | 57.5 | 64.3 |
| corrected total score | 55.8 (−3.7 [−4.9, −2.4]) | 55.7 (−1.8 [−5.4, +1.8]) | 65.5 (+1.2 [−7.1, +9.5]) |
| reweight, Rosetta init | 59.6 (+0.1 [−0.9, +1.2]) | 58.0 (+0.5 [−2.6, +3.6]) | 67.9 (+3.6 [−3.6, +10.7]) |
| ranker | 60.5 (+1.1 [+0.1, +2.0]) | 57.5 (+0.0 [−2.6, +2.6]) | 66.7 (+2.4 [−3.6, +9.5]) |

Success@5: PDBbind 76.2 → 76.4 (reweight) / 77.3 (ranker, +1.1 [+0.3, +1.9]); PoseBusters
71.9 → 72.9 / 73.7. Per-run AUC: interface delta 0.887, ranker 0.905, reweight 0.881.

**Transfer between receptor states** (`--eval-protocols`): the relax-trained ranker on
apo_relax poses gains +1.3 [+0.3, +2.3] on PDBbind, +1.0 on PoseBusters and +6.0
[−1.2, +13.1] on Astex, i.e. as much as the model trained on apo_relax poses itself; the
relax-trained reweighters give +0.7 [−0.3, +1.7] (random init) and −1.5 [−2.5, −0.5]
(Rosetta init). The apo_relax-trained reweighter on relax poses gains +1.0 [+0.1, +1.9] on
PDBbind and +2.6 [+0.5, +4.9] on PoseBusters, the apo_relax-trained ranker +0.3 and +1.8.

**Learned weights** (reweighters on the seven interface terms, flipped to Rosetta's sign and
normalised to `fa_atr` = 0.8; Rosetta default in the last column):

| Term | relax, Rosetta init | relax, random init | apo_relax, Rosetta init | Rosetta |
|---|---|---|---|---|
| fa_atr | 0.80 | 0.80 | 0.80 | 0.80 |
| fa_rep | 0.64 | 0.69 | 0.59 | 0.40 |
| fa_sol | 0.38 | 0.25 | 0.36 | 0.60 |
| fa_elec | 0.27 | 0.33 | 0.31 | 0.42 |
| fa_pair | −0.13 | −0.07 | −0.14 | 0.80 |
| hbond_bb_sc | 0.53 | 0.98 | 0.76 | 1.30 |
| hbond_sc | 0.57 | 0.83 | 0.48 | 1.30 |

### 7.3 Reading

- **On the PDBbind test complexes the learned heads do not move Success@1 beyond ±1 point.**
  All relax · perturb intervals include zero; on apo_relax · perturb only the ranker's
  +1.1 [+0.1, +2.0] excludes it. This is the S5 result on corrected data.
- **The gain depends on the split, not on the labels.** On a random split by complex the same
  heads gain +2.4 points on the held-out fold, with corrected labels as well as with Luca's
  (§7.1); on the LP-PDBBind test complexes, whose proteins are unseen in training, the gain is
  +0.4 to +0.6. What the heads learn beyond the Rosetta weights is largely protein-specific
  and does not transfer to new protein families — the leakage-aware split is what turns Luca's
  "+2–4 points" into "no measurable improvement".
- **PoseBusters is the only set with a consistent gain**, +2.3 to +2.8 points Success@1 and
  Success@5 for all three relax heads (intervals exclude zero) — about ten of 388 complexes.
  Astex (84 complexes) cannot resolve differences of a few points; its intervals span ±5–10.
- **Total-energy features are worse than the interface terms** (−1.0 on PDBbind), so the
  receptor-internal terms add noise, not ranking signal; this supports the choice of the seven
  interface terms in Appendix A.1.
- **The reweighted term vector is nearly the same across runs and initialisations** (more
  weight on repulsion, less on solvation, electrostatics and hydrogen bonds, `fa_pair` dropped
  to ≈ 0) and ranks like the default weights: the linear optimum is flat. This is the
  reweighting-side view of the energy degeneracy that A.1 / A.2 will quantify.
- **Caveats.** One seed per configuration (the two reweight initialisations differ by ≈ 0.1
  point on PDBbind, a rough noise floor); the LP-PDBBind test complexes have a higher baseline
  Success@1 than the full set (75.3 vs. 72.0 on relax · perturb, 59.5 vs. 53.1 on
  apo_relax · perturb), so the split is cleaner, not harder, for the ranking question; the
  final split decision (inventory §4, item 3) is still open — `--split hac` / `cleansplit` give
  the alternatives; a tree reference head (S5 b) is not implemented.
