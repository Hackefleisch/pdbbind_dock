# Track 2 — pose ranking: approaches vs. the unmodified interface delta (`ranking/`)

> Methods-level documentation of the `ranking` package: the question, the fixed setup, the
> four approaches, the evaluation and the results. Source files:
> [`ranking/data.py`](../ranking/data.py), [`ranking/models.py`](../ranking/models.py),
> [`ranking/evaluate.py`](../ranking/evaluate.py), [`ranking/train.py`](../ranking/train.py),
> [`ranking/report.py`](../ranking/report.py). Paper context:
> [PAPER1_STORY_INVENTORY.md](PAPER1_STORY_INVENTORY.md), story S5 and §6.1.
> Written 2026-10-19 (port of Luca Friebe's models), revised 2026-10-20 to the decided setup.

## 1. Question and setup

**Can RosettaLigand's pose ranking be repaired, and by how much?** The reference is the
*unmodified* Rosetta interface delta (`idelta_score`, the weighted sum of the interface energy
terms with the default weights). Everything else is reported as the paired difference to it.
The setup was fixed on 2026-10-20:

| Element | Decision |
|---|---|
| Features | the **seven non-zero raw interface-delta terms** `raw_delta_{fa_atr, fa_rep, fa_sol, fa_elec, fa_pair, hbond_bb_sc, hbond_sc}` and nothing else; `idelta_score` is their weighted sum (verified to 1e-4 on the extraction) |
| Label | hit ⇔ symmetry-corrected `rmsd_to_crystal` ≤ 2 Å |
| Training pool | the `relax` · `perturb` and `apo_relax` · `perturb` runs of the PDBbind training complexes. Not the crystal state (its structures are not relaxed; they may carry signals that are not about ranking) and not the std runs (they only add near-identical positives) |
| Training bucket | one bucket per **complex × receptor state**: all poses docked from the same starting structure. With the default pool that is one perturb run (300 poses); if the std protocol is pooled in (`--protocol … relax_docking_std`) it joins the perturb run of the same state (600 poses). relax and apo_relax buckets of a complex stay separate. Poses are never compared across complexes |
| Headline | the Δ Success@k to the unmodified ranking on **`apo_relax` · `perturb`**; `relax` · `perturb` is reported as well |
| Splits | all three non-random PDBbind splits of `docking_analysis/splits.csv`: **LP-PDBBind** (train / val / test), **HAC** (test = CASF-2016 core, val = CASF validation set), **CleanSplit** (10 % of train held out as val). The 43 Astex entries that are PDBbind complexes are removed from train and val in every split |
| Approaches | `unmodified` · `tree` · `reweighted` (random start) · `reweighted` (Rosetta start; all weights free, non-negative) · `ranker` — §4 |
| Tables | one row per approach, one column per k ∈ {1, 5, 10}, cells = Δ Success@k [95 % CI]; one table per split for the PDBbind test complexes, one per external set (PoseBusters, Astex) |

Thesis-era numbers and the random split by complex are no longer used anywhere.

## 2. Origin and credit

The MLP ranker, its loss and its feature preprocessing come from Luca Friebe's repository
[`lucafrb/rosettaligand-vs-posebuster`](https://github.com/lucafrb/rosettaligand-vs-posebuster),
branch `test_new_architecture` (`ranking/data_loader.py`, `ranking/ranking.py`,
`reweighting/reweight.py`, the latter a bias-free `nn.Linear` on the same scaled features
trained with the same loss). His classification and regression pipelines were not ported (he
judged them inappropriate for the task). The port was validated on 2026-10-19 before the setup
was changed: with his labels, split and training set it reproduced the numbers of his
`ML_pipeline_description.md` within one complex on PoseBusters and Astex (git history of this
page). What is still his in the current code: the `Ranker` architecture and the
`BCEPlusInfoNCE` loss (both unchanged), the robust scaling with the `fa_rep` cap and the ±10
clip (now fitted on the training poses only), the model-selection rule (best validation
Success@1, ties by Success@5).

Facts about his setup that matter for reading his repository: his 25 requested features are the
interface-delta terms and reduce to the seven non-zero ones (his saved models have 7 inputs);
his labels were the stored, uncorrected `rmsd_to_input`; his split was a random
`StratifiedGroupKFold` by complex, and his "PDBbind" numbers are its validation fold; all of
PDBbind was in his training set although 43 of the 84 Astex entries are PDBbind complexes; the
checked-in `datasets/astex_test_set.pkl` is stale (72 complexes × 150 poses).

## 3. Data

- **Poses.** `load_poses(set, protocols)` reads the meta columns and the seven features of the
  requested protocol(s) from `docked.parquet`, keeps the complete complexes and sorts by
  (pdb, protocol, seq_id); `Grouped.build(…, group_by="state" | "protocol")` turns a frame into
  contiguous buckets. The pooled PDBbind training frame has 6.9 M rows and loads in seconds.
- **Hits.** relax · perturb: 5.7 % of the poses, 521 of 11,444 runs without a hit;
  apo_relax · perturb: 4.3 %, 1,317 runs without a hit. Buckets without a hit contribute to
  the BCE term of the ranker only; the InfoNCE term and the LambdaRank objective skip them.
- **Splits** (complete PDBbind complexes): LP-PDBBind train 5,882 / val 1,566 / test 3,572 (424
  unassigned dropped); HAC test 271 / val 178 / train 10,992; CleanSplit test 151, 1,949
  unassigned, val = 10 % of train. Minus the Astex overlaps in train / val.
- **Preprocessing** (`Preprocess`): `none` — raw energies — for the reweighters and the tree;
  `robust` for the ranker (delta `fa_rep` capped at 50, median / IQR per term fitted on the
  training poses, scaled values clipped to ±10; Luca's choices, fitted on train only). Stored
  as `preprocess.json` of the run.

## 4. Approaches

- **`reweighted`** — the physics-side baseline, not ML. Score = Σ_t w_t x_t over the seven raw
  interface terms with **w_t ≥ 0** (projected gradient: clamped at zero after every step), no
  scaling, no clipping, no bias, so the learned vector *is* a Rosetta weight set. All seven
  weights are free. A ranking is invariant to their overall scale, so the vector is reported
  normalised to `fa_atr` = 0.8 (`weights.wts`, next to the raw learned values in
  `weights.csv`); the scale itself is not meaningless during training, because scaling the
  weights by c equals dividing the InfoNCE temperature by c, so the optimiser also uses it to
  sharpen the softmax (`--gauge fa_atr` pins the scale instead; not used). Two starts, reported
  as two baselines: **random** (all weights drawn from U(0, 1)) and **Rosetta** (the default
  weights, so epoch 0 *is* the unmodified ranking and training has to beat it on the validation
  complexes to be selected). Trained with the listwise InfoNCE term
  only (τ = 0.5): the BCE term of Luca's loss acts on absolute score levels, which raw energies
  cannot calibrate without a bias. Adam, lr 1e-4, one bucket per step, 50 epochs, best epoch by
  validation Success@1 then Success@5.
- **`ranker`** — Luca's MLP `7 → 64 → 32 → 16 → 1` (ReLU) on the robust-scaled features, loss
  `BCEWithLogits(score, hit)` over the bucket + InfoNCE over the bucket (τ = 0.5, β = 1), Adam
  lr 1e-5, one bucket per step, 50 epochs, same selection rule. The InfoNCE term compares poses
  within their bucket only; the BCE term is pointwise (no comparison between systems).
- **`tree`** — LightGBM `LGBMRanker`, objective `lambdarank`, query groups = the training
  buckets, labels = hit, raw interface terms, 31 leaves, learning rate 0.05, min 100 poses per
  leaf, up to 2,000 trees with early stopping (50 rounds) on the validation NDCG@1 — which for
  binary labels equals Success@1, i.e. the same selection criterion as the torch models.
  Feature importance (gain) is written as `importance.csv`.

## 5. Evaluation

- Success@k per docking run with the definition of `docking_analysis.benchmark` (stable sort by
  the score, success when the first hit has rank < k); `train.py` prints the unmodified
  Success@k on all complete PDBbind runs of each training protocol as a sanity line that must
  equal the benchmark tables (relax · perturb 71.98 / 84.55 / 88.38, apo_relax · perturb
  53.07 / 69.68 / 75.60).
- Rates in percent of runs; `[95 % CI]` percentile bootstrap over runs (2,000 resamples); `Δ`
  the paired difference to the unmodified ranking on the same runs with its bootstrap interval —
  the number that decides whether an approach helps.
- Whole first-hit-rank distribution (`rank_shift`): the mean reciprocal rank of the first hit
  (MRR; runs without a hit count 0), the percentage of runs with a hit under both scorers
  whose first hit moves to a better / the same / a worse rank, and the mean shift of
  log2(rank + 1). These tell whether an approach that does not move Success@1 still pushes
  hits up the list.
- Evaluation groups are docking runs (complex × protocol), also when the training buckets
  pooled several protocols. Evaluated: the PDBbind test complexes of the split, PoseBusters
  (388 complete), Astex (84 complete), for both training protocols.

## 6. How to run

```bash
uv run python -m ranking.train --model reweight --init random     # all three splits, ~50 min on one GPU
uv run python -m ranking.train --model reweight --init rosetta
uv run python -m ranking.train --model ranker
uv run python -m ranking.train --model tree                       # ~10 min, CPU
uv run python -m ranking.report                                   # tables of all runs
```

One run = one approach × split under `data/ranking/<approach>_<split>/` (`results.md` /
`results.json`, `per_run.parquet`, `model.pt` or `model.txt`, `preprocess.json`, `history.csv`,
`train.log`, `config.json`, plus `weights.csv` + `weights.wts` for the reweighters and
`importance.csv` for the tree). `ranking.report` writes `data/figures/ranking/ranking_tables.md`
and `ranking_results.csv` and only reads runs of the current schema (`"schema": 2` in
`results.json`); the run directories of the first pass of 2026-10-19 (Luca's original setup,
scaled signed reweighting, single-protocol training) are ignored by it and can be deleted.

## 7. Results (runs of 2026-10-20)

Twelve runs (four approaches × three splits; `data/ranking/<approach>_<split>/`, collected in
`data/figures/ranking/ranking_tables.md` and `ranking_results.csv`), one seed each. Cells are the
paired Δ Success@k to the unmodified interface delta on the same docking runs, in points, with
the percentile-bootstrap 95 % interval; the unmodified row gives the absolute Success@k.

### 7.1 Headline — `apo_relax` · `perturb`, PDBbind test complexes of each split

| approach | LP-PDBBind (n = 3,572) Δ@1 | Δ@5 | Δ@10 | HAC / CASF core (n = 271) Δ@1 | Δ@5 | Δ@10 | CleanSplit (n = 151) Δ@1 | Δ@5 | Δ@10 |
|---|---|---|---|---|---|---|---|---|---|
| unmodified (interface delta) | 59.5 | 76.2 | 81.9 | 59.0 | 74.5 | 80.1 | 49.7 | 68.9 | 74.2 |
| tree (LambdaRank) | +0.4 [−0.6, +1.5] | −0.4 [−1.3, +0.5] | +0.1 [−0.6, +0.9] | −0.4 [−5.2, +4.1] | −0.7 [−3.7, +2.2] | −0.4 [−3.3, +2.6] | +0.7 [−4.6, +6.0] | +0.7 [−4.6, +6.0] | +4.0 [+0.7, +7.9] |
| reweighted, random start | +1.5 [+0.6, +2.4] | +1.3 [+0.5, +2.0] | +0.8 [+0.2, +1.5] | −1.8 [−4.4, +0.4] | +1.5 [−1.1, +4.1] | +2.6 [+0.4, +4.8] | −0.7 [−4.6, +3.3] | +2.0 [−2.6, +6.0] | +3.3 [+0.0, +6.6] |
| reweighted, Rosetta start | +1.5 [+0.6, +2.3] | +1.1 [+0.4, +1.9] | +0.9 [+0.3, +1.5] | −0.7 [−3.3, +1.8] | +2.2 [−0.7, +5.2] | +2.2 [+0.0, +4.4] | −0.7 [−4.6, +3.3] | +2.0 [−2.6, +6.0] | +3.3 [+0.0, +6.6] |
| ranker (MLP) | +0.8 [−0.2, +1.6] | +1.2 [+0.4, +1.9] | +0.9 [+0.3, +1.5] | −0.7 [−4.1, +2.6] | +3.7 [+1.1, +6.3] | +3.0 [+0.0, +5.9] | +2.6 [−2.0, +7.3] | +3.3 [−0.7, +7.9] | +4.6 [+2.0, +7.9] |

### 7.2 `relax` · `perturb`, PDBbind test complexes of each split

| approach | LP-PDBBind Δ@1 | Δ@5 | Δ@10 | HAC Δ@1 | Δ@5 | Δ@10 | CleanSplit Δ@1 | Δ@5 | Δ@10 |
|---|---|---|---|---|---|---|---|---|---|
| unmodified (interface delta) | 75.3 | 88.1 | 91.6 | 78.2 | 87.5 | 91.5 | 69.5 | 84.1 | 89.4 |
| tree (LambdaRank) | −1.3 [−2.3, −0.4] | −1.8 [−2.6, −1.0] | −0.8 [−1.4, −0.2] | −3.0 [−6.6, +0.4] | +0.7 [−2.2, +3.7] | +0.4 [−1.5, +2.2] | −0.7 [−5.3, +4.0] | +2.0 [−1.3, +6.0] | +2.6 [+0.7, +5.3] |
| reweighted, random start | +0.2 [−0.5, +1.0] | +0.4 [−0.2, +1.0] | +0.1 [−0.3, +0.5] | −0.4 [−2.6, +1.8] | +0.4 [−0.7, +1.5] | +0.4 [−1.5, +2.2] | +0.7 [−2.6, +4.0] | +2.0 [−0.7, +5.3] | +1.3 [+0.0, +3.3] |
| reweighted, Rosetta start | +0.3 [−0.5, +1.0] | +0.4 [−0.1, +1.0] | +0.1 [−0.3, +0.5] | +0.4 [−1.5, +2.2] | +1.1 [−0.7, +3.0] | +0.0 [−1.5, +1.5] | +0.7 [−2.6, +4.0] | +2.0 [−0.7, +5.3] | +1.3 [+0.0, +3.3] |
| ranker (MLP) | −0.1 [−0.9, +0.8] | −0.2 [−0.8, +0.4] | +0.1 [−0.4, +0.6] | +2.2 [−0.4, +4.8] | +1.5 [+0.4, +3.0] | +0.0 [−1.1, +1.1] | +3.3 [−0.7, +7.3] | +1.3 [−1.3, +4.0] | +1.3 [+0.0, +3.3] |

### 7.3 External sets, `apo_relax` · `perturb` (training set = the split's training complexes)

PoseBusters (388 complexes; unmodified 57.5 / 71.9 / 77.3): no approach moves Success@1
(−0.5 to +1.5, all intervals include zero); Success@10 gains +2.6 to +3.4 for the reweighters
and the ranker (intervals excluding zero for every split), Success@5 +1.3 to +3.6. Astex (84
complexes; unmodified 64.3 / 79.8 / 88.1): every Δ@1 is positive (+0.0 to +3.6) but every
interval spans ±5 to ±8 points. The full tables, including the first-hit-rank statistics, are
in `ranking_tables.md`.

### 7.4 Learned Rosetta weights (`weights.wts`, normalised to `fa_atr` = 0.8)

| split · start | fa_rep | fa_sol | fa_elec | fa_pair | hbond_bb_sc | hbond_sc | selected epoch | learned fa_atr (raw scale) |
|---|---|---|---|---|---|---|---|---|
| Rosetta default | 0.40 | 0.60 | 0.42 | 0.80 | 1.30 | 1.30 | — | 0.80 |
| LP-PDBBind · random | 0.51 | 0.33 | 0.34 | 0.04 | 1.28 | 1.07 | 14 | 0.35 |
| LP-PDBBind · Rosetta | 0.50 | 0.33 | 0.33 | 0.14 | 1.29 | 1.08 | 10 | 0.36 |
| HAC · random | 0.49 | 0.37 | 0.32 | 1.06 | 1.42 | 1.02 | 1 | 0.42 |
| HAC · Rosetta | 0.49 | 0.35 | 0.36 | 0.02 | 1.29 | 0.93 | 18 | 0.38 |
| CleanSplit · random | 0.51 | 0.36 | 0.37 | 0.04 | 1.34 | 0.97 | 35 | 0.37 |
| CleanSplit · Rosetta | 0.51 | 0.36 | 0.37 | 0.04 | 1.34 | 0.97 | 35 | 0.37 |

Five of the six runs converge to the same vector, from both starts and on every split:
relative to Rosetta's weights, `fa_rep` up (0.5 vs 0.4), `fa_sol` and `fa_elec` down (0.35 vs
0.6 and 0.42), `fa_pair` switched off (0.8 → 0), `hbond_bb_sc` unchanged, `hbond_sc` slightly
down (1.0 vs 1.3). The exception, HAC · random, was selected at epoch 1 (its validation set has
176 complexes) and still carries its random `fa_pair`. The overall scale shrank in every run
(learned `fa_atr` 0.35–0.42 instead of 0.80): with the InfoNCE temperature fixed at 0.5 the
optimiser softened the softmax by about a factor of two, which is why the run with `fa_atr`
pinned at 0.8 (git history of this page) ended up ±0.6 on LP-PDBBind where the free run gains
+1.5 — the free scale is a better-conditioned optimisation, not a different model. Tree
importance (gain) is dominated by fa_atr, fa_elec and fa_rep; fa_pair contributes nothing.

### 7.5 Reading

- **The ranking is not repaired by any of the three approaches; the best of them gains about
  1.5 points.** On the headline setting the Δ Success@1 on the PDBbind test complexes lies
  between −1.8 and +2.6 points over all approaches and splits. The largest split (LP-PDBBind,
  3,572 complexes) gives tree +0.4 [−0.6, +1.5], reweighted +1.5 [+0.6, +2.4] from either start,
  ranker +0.8 [−0.2, +1.6]; the reweighting is the only approach whose interval excludes zero
  there, and it does not reproduce on HAC (−1.8 / −0.7) or CleanSplit (−0.7), whose test sets
  are too small to resolve ±2 points. On `relax` · `perturb` nothing moves (−3.0 to +3.3,
  intervals including zero except the tree's small losses on LP-PDBBind).
- **Success@5 / @10 move by a point or two at most**, and only for the ranker consistently
  (LP-PDBBind +1.2 / +0.9, HAC +3.7 / +3.0, CleanSplit +3.3 / +4.6 at k = 5 / 10; PoseBusters
  +2 to +3.6). The MLP pushes some hits a few ranks up without putting them first.
- **A non-negative reweighting of the seven terms is worth about one to one and a half points
  on apo_relax poses and nothing on relax poses.** Both starts find the same weight set
  (§7.4: more repulsion, less solvation and electrostatics, `fa_pair` off), so the two starts
  are one result and the vector is a genuine, if marginal, alternative Rosetta weight set for
  ranking. That the optimum is reached from a random start in a few epochs, and that it buys so
  little, says the linear information in the seven terms is nearly exhausted by the defaults.
- **The tree is no better than the linear reweighting** on the headline setting and slightly
  worse on `relax` · `perturb` (−1.3 on LP-PDBBind): non-linear interactions of the seven terms
  do not carry the missing information either.
- **Caveats.** One seed per run; HAC and CleanSplit test sets are small (271 and 151
  complexes, intervals of ±4 to ±6 points) — the LP-PDBBind column is the one to quote; the
  HAC · random reweighting was selected at epoch 1 on a 176-complex validation set and is the
  one run whose weights are not converged.
