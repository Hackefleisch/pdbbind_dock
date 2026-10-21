# Track 2, S6 — why the ranking resists repair: energy degeneracy (A.1) and the per-complex oracle (A.2)

> Methods-level documentation of `ranking/degeneracy.py` and `ranking/degeneracy_report.py`:
> the question, the definitions, the outputs and the results. Paper context:
> [PAPER1_STORY_INVENTORY.md](PAPER1_STORY_INVENTORY.md), story S6 and Appendix A.1 / A.2.
> Written 2026-10-21. Companion of [ranking.md](ranking.md) (S5), whose result it explains.

## 0. Headline

**RosettaLigand's pose ranking cannot be repaired by reweighting the seven interface terms,
because the information is not in them.** A linear classifier whose weights are chosen *per
complex* and fitted on the very poses it is evaluated on ranks a correct pose first in 67.9 %
of the apo_relax · perturb test runs, against 66.3 % for the unmodified interface delta, and
the S5 models (66.5–67.7 %) already sit at that ceiling (figure, panel a). The decoys that win
are not energetically degenerate copies of correct poses: only 11 % of the Top-1 failures have
a best-scoring pose that is nearer to a correct pose than the correct poses are to each other;
the typical winner lies 2.6 thresholds away in energy space and scores 2.8 REU *better* than
its nearest correct pose (panel b); energy-degenerate decoys are 5 % of all incorrect poses,
1.3 % when they must also score at least as well as a correct pose (panel c). The energy
function has genuine minima at wrong placements, and telling them from the correct basin needs
information the seven terms do not carry.

![S6 headline figure](../data/figures/degeneracy/headline.png)

`data/figures/degeneracy/headline.png` — regenerate with
`uv run python -m ranking.degeneracy_report --headline-only` (a: Success@1 with 95 % bootstrap
intervals on the 3,048 LP-PDBBind test runs with ≥ 2 hits; b: hexbin of the best-scoring pose
of the 3,431 Top-1 failures, distance to the nearest correct pose over the run threshold
against the interface-delta difference to that pose; c: pose-level shares per receptor state).

## 1. Question and setup

S5 found that a non-negative reweighting of the seven interface terms, a LambdaRank tree and
an MLP ranker gain at most about 1.5 points of Success@1 over the unmodified interface delta.
S6 asks **why**: are the incorrect poses energetically indistinguishable from the correct ones
(A.1), and how well could *any* linear weighting of the seven terms do if it were allowed to
be chosen per complex (A.2)?

| Element | Decision (2026-10-21) |
|---|---|
| Unit of analysis | one **docking run** = one complex × one protocol (300 poses). The Perturb run of each receptor state (`crystal`, `relax`, `apo_relax` · `perturb`); the std runs are near-native by construction and are not used. All three sets (PDBbind, PoseBusters, Astex); the headline is `apo_relax` · `perturb` on PDBbind |
| Energy vector | the same seven raw interface-delta terms as the rest of the track, **z-scored over the run's own poses** (a term without variance in the run drops out). Robustness check `--terms total`: the whole-complex `raw_*` energies of the same seven terms |
| Correct / incorrect / far | symmetry-corrected `rmsd_to_crystal` ≤ 2 Å / > 2 Å / > 5 Å |
| Distances | Euclidean on the z-scores (main text) and Mahalanobis (Euclidean after whitening with the run's own covariance; eigen-directions below 1e-6 of the largest dropped) — both reported |
| Skipped by decision | the cross-complex permutation control of A.1 |
| Added by decision | the per-run summary of A.1 (fraction of degenerate incorrect poses, histogram, table by protocol); three example complexes (hard / median / easy) for A.2; UMAP / t-SNE as illustration only |

## 2. Definitions

**A.1 — nearest neighbours in energy space.** For every pose of a run, the distance to the
nearest *other* correct pose (`d_correct`) and to the nearest other incorrect pose
(`d_incorrect`). The three populations of the spec are

- incorrect → nearest correct: `d_correct` of the incorrect poses — the quantity *d_i*;
- correct → nearest correct: `d_correct` of the correct poses — the within-basin spread;
- incorrect → nearest incorrect: the density of the decoy cloud.

The run's **degeneracy threshold** is its median correct–correct distance. An incorrect pose
is **degenerate** when its *d_i* lies below the threshold (it is nearer to a correct pose than
the correct poses are to each other); an **energy-degenerate decoy** is degenerate *and* > 5 Å
(not a near-miss). The score differences Δ interface delta and Δ total score to the nearest
correct pose (Euclidean) are kept per pose, negative = the decoy scores better, so the claim
"similar energy vector *and* similar-or-better score *and* large RMSD" can be checked
explicitly. Runs without a correct pose are excluded (counted); runs with exactly one correct
pose have no correct–correct reference and are excluded from the percentages.

**A.2 — per-run oracle.** A logistic regression (L2, C = 1, balanced classes) on the z-scores
of that run alone, i.e. the best linear separation of correct from incorrect poses *for this
complex* — strictly more freedom than any global reweighting, Rosetta's own weights or the S5
models. Reported in-sample (the optimistic bound: 7 weights fitted on the 300 poses they are
evaluated on) and cross-validated (stratified k-fold, k = min(5, n_hit), pooled out-of-fold
scores; the honest estimate). Besides the AUC, the rank of the first hit under every score
gives Success@k with the benchmark's definition. The unmodified interface delta and total
score are evaluated on the same poses as the reference. Runs with fewer than two hits have no
oracle.

## 3. Data and counts

`ranking.degeneracy` reads `docked.parquet` of each set through `ranking.data.load_poses`
(complete complexes only). Counts of the Perturb runs:

| set | complexes | runs | runs without a hit (excluded) | runs with one hit (no reference) | runs analysed (≥ 2 hits) |
|---|---|---|---|---|---|
| PDBbind | 11,444 | 34,332 | 2,372 | 1,962 | 29,998 (crystal 10,394 / relax 10,430 / apo_relax 9,174) |
| PoseBusters | 388 | 1,164 | 79 | 69 | 1,016 |
| Astex | 84 | 252 | 8 | 3 | 241 |

Hits per `apo_relax` · `perturb` PDBbind run: median 7, quartiles 2 and 19; 5,015 runs have
≥ 10 hits. The pose-level percentages below are pooled over the incorrect poses of the
analysed runs (2.6–2.9 M per PDBbind protocol); the run-level percentages are over the
analysed runs, so the Top-1 rates here are those of the runs with ≥ 2 hits and are higher than
the benchmark's rates over all runs (apo_relax · perturb: 62.6 vs 53.1 %).

## 4. How to run, outputs

```bash
uv run python -m ranking.degeneracy                 # three perturb protocols, three sets; ~1.5 min on 20 cores
uv run python -m ranking.degeneracy --terms total   # robustness check on the raw_* totals
uv run python -m ranking.degeneracy_report          # figures, tables, examples, UMAP / t-SNE; ~2 min
```

`data/ranking/degeneracy/` (`_total/` for the second term set): `runs.parquet` (one row per
run with ≥ 1 hit: counts, the medians of the three populations, the degenerate fractions, the
best-scoring pose's RMSD / distance / score difference, the AUCs and first-hit ranks of every
score, the in-sample oracle coefficients), `poses.parquet` (one row per pose of those runs:
distances, degenerate flags, score differences, oracle scores), `summary.csv` / `summary.md`
(the headline tables per set × protocol for both metrics), `config.json`.

`data/figures/degeneracy/`: `degeneracy_tables.md` (every table, including the oracle vs the
S5 approaches and the examples), `nn_distance_density_<set>[_ma].png`,
`hexbin_<set>_<protocol>.png`, `degenerate_fraction_<set>.png`, `oracle_auc_<set>.png`,
`oracle_vs_degeneracy_<set>.png`, `oracle_gap.csv`, `examples.png` / `examples.csv`,
`embedding_pooled.png`. The report joins the schema-2 runs of `ranking.train`
(`per_run.parquet`) for the gap table; `--gap-split` picks the split (default LP-PDBBind),
`--example-quantiles` the hard / median / easy runs (default: lowest, middle, highest
cross-validated oracle AUC among the runs with `--example-min-hits` 10 hits), `--no-embedding`
skips UMAP / t-SNE.

## 5. Results (PDBbind unless stated; Euclidean distance, interface-delta terms)

### 5.1 A.1 — strict energy degeneracy is a minority phenomenon

Incorrect poses of the analysed runs, pooled:

| protocol | % degenerate (d_i < threshold) | % degenerate **and > 5 Å** (energy-degenerate decoys) | % degenerate, > 5 Å and scoring ≤ its nearest correct pose | % of the > 5 Å poses that are degenerate | per-run fraction degenerate, median [IQR] |
|---|---|---|---|---|---|
| crystal · perturb | 8.4 | 3.7 | 0.9 | 5.7 | 0.03 [0.00, 0.10] |
| relax · perturb | 7.4 | 3.2 | 0.8 | 4.9 | 0.02 [0.00, 0.08] |
| **apo_relax · perturb** | **10.2** | **5.0** | **1.3** | 7.6 | 0.03 [0.00, 0.12] |

PoseBusters and Astex give the same picture (energy-degenerate decoys 2.7–4.3 % of the
incorrect poses; 0.5–1.0 % also scoring at least as well as their nearest correct pose). About
a third (31–33 %) of the degenerate poses score as well as or better than their nearest correct
pose; the other two thirds are near in energy *profile* but worse in score.

The expected "dense band at low *d_i* spanning the whole RMSD range" is **not** in the data.
The density plot (`nn_distance_density_pdbbind.png`) shows the correct–correct and the
incorrect–incorrect nearest-neighbour distances peaking at 0.3–0.4 (z-units) while the
incorrect → correct distances peak at 1.5–2 and have a long tail: the correct poses form a
tight cluster that the incorrect poses mostly do not enter. In the hexbin
(`hexbin_pdbbind_apo_relax_docking_perturb.png`) the incorrect poses sit at one to four
thresholds from the nearest correct pose at every RMSD; the region left of the threshold line
is sparsely populated and, in the score-difference panel, is the only region where decoys score
*better* than their nearest correct pose (red), i.e. degeneracy and score inversion coincide
exactly where the spec's pitfall says they must be stated together. Per run
(`degenerate_fraction_pdbbind.png`), the degenerate fraction is below 0.1 for three quarters of
the runs, with a thin tail of runs where almost all incorrect poses are degenerate (these are
the runs with the lowest oracle AUC, §5.3).

### 5.2 A.1 — the Top-1 failures are *not* degenerate copies of the correct poses

The best-scoring pose by interface delta, per analysed run:

| protocol | % Top-1 failures (best pose not a hit) | % runs whose best pose is a degenerate decoy | … degenerate and > 5 Å | of the Top-1 failures: % degenerate | % > 5 Å | % 2–5 Å (near-miss) |
|---|---|---|---|---|---|---|
| crystal · perturb | 29.3 | 3.0 | 0.8 | 10.3 | 36.9 | 63.1 |
| relax · perturb | 22.8 | 2.4 | 0.4 | 10.7 | 32.5 | 67.5 |
| **apo_relax · perturb** | **37.4** | **4.3** | **1.3** | **11.4** | 40.0 | 60.0 |

Only about one Top-1 failure in ten is an energy-degenerate decoy in the nearest-neighbour
sense. The typical winning decoy lies **2.6–2.7 thresholds** away from the nearest correct
pose in energy space and scores a median **2.6–2.8 REU better** than that pose; 60–68 % of
them are 2–5 Å near-misses. So the ranking fails on poses that the seven terms *genuinely rate
better* with a *distinct* energy profile, not on poses they cannot tell apart.

### 5.3 A.2 — the per-complex oracle barely moves Success@1

Per-run AUC (median [IQR] over the analysed runs) and Success@1 / @5 of the same runs:

| protocol | AUC interface delta | AUC total score | AUC oracle, cross-validated | AUC oracle, in-sample | % runs oracle CV AUC < 0.9 / < 0.8 | Success@1 interface delta / oracle CV / oracle in-sample | Success@5 interface delta / oracle CV |
|---|---|---|---|---|---|---|---|
| crystal · perturb | 0.927 [0.849, 0.978] | 0.810 | 0.947 [0.894, 0.981] | 0.976 [0.944, 0.994] | 26.9 / 7.3 | 70.7 / 67.4 / 75.1 | 86.9 / 90.6 |
| relax · perturb | 0.935 [0.863, 0.982] | 0.888 | 0.953 [0.903, 0.983] | 0.979 [0.948, 0.995] | 23.8 / 5.8 | 77.2 / 71.5 / 78.6 | 90.3 / 92.5 |
| **apo_relax · perturb** | 0.909 [0.824, 0.968] | 0.851 | 0.938 [0.877, 0.975] | 0.973 [0.940, 0.992] | 32.6 / 10.6 | **62.6 / 58.2 / 66.4** | 81.8 / 85.8 |

- **AUC is high everywhere** (median 0.91–0.94 for the unmodified interface delta): the bulk
  of the decoy cloud is easy to separate from the hits. The per-run oracle raises the median
  AUC to 0.94–0.95 cross-validated and 0.97–0.98 in-sample.
- **Success@1 is where the oracle fails to deliver.** Weights chosen freely *per complex* and
  fitted *on the evaluated poses* raise Success@1 by only **+3.8 / +1.4 / +4.4 points**
  (apo_relax / relax / crystal); on apo_relax they fix the Top-1 of 13.4 % of the runs and break
  it for 9.5 %. The cross-validated oracle is *below* the unmodified score (−4.4 / −5.7 / −3.3
  points): with a median of 7 hits per run the folds hold one to two positives each, and the
  fitted direction does not generalise within the run. The gap narrows with more hits (runs
  with ≥ 20 hits, apo_relax: 72.3 / 73.7 / 79.8 %), but the in-sample ceiling stays within
  8 points of the unmodified ranking.
- **One run in three has a cross-validated oracle AUC below 0.9, one in ten below 0.8** —
  complexes where no linear use of the seven terms separates correct from incorrect poses.
- **Oracle AUC and the A.1 degenerate fraction agree on which complexes are hopeless:**
  Spearman ρ = −0.67 to −0.69 on every protocol (`oracle_vs_degeneracy_pdbbind.png`); the
  runs with a high degenerate fraction are the runs with the lowest oracle AUC.
- **The per-complex optimal weights agree in sign only for three terms.** In-sample oracle
  coefficients on the z-scores (apo_relax; positive = a larger value favours hits): `fa_atr`
  median −1.65 (positive in 12 % of the runs), `fa_rep` −1.89 (3 %), `fa_elec` −0.89 (21 %),
  but `fa_sol` −0.13 (46 % positive), `fa_pair` 0.00 (47 %), `hbond_bb_sc` −0.17 (43 %),
  `hbond_sc` +0.01 (51 %): for solvation, the pair term and both hydrogen-bond terms the
  optimal direction is a coin flip across complexes. This is consistent with the S5 reweighting
  lowering `fa_sol` and `fa_elec` and switching `fa_pair` off, and it is the structural reason
  why no single weight vector does much better than the defaults.

### 5.4 The gap between the oracle and the S5 approaches on the same runs

Mean per-run AUC / Success@1 (%) on the runs with ≥ 2 hits among the LP-PDBBind test complexes
(the approaches were trained on that split's training complexes) and on the external sets
(`oracle_gap.csv`; all splits in the CSV):

| set · protocol | runs | unmodified | tree | reweighted (random) | reweighted (Rosetta) | ranker | oracle CV | oracle in-sample |
|---|---|---|---|---|---|---|---|---|
| PDBbind · apo_relax | 3,048 | 0.886 / 66.3 | 0.908 / 66.5 | 0.896 / 67.7 | 0.896 / 67.6 | 0.901 / 66.9 | 0.914 / 60.2 | 0.962 / 67.9 |
| PDBbind · relax | 3,359 | 0.910 / 78.6 | 0.926 / 77.3 | 0.918 / 78.9 | 0.918 / 78.9 | 0.922 / 78.6 | 0.934 / 71.6 | 0.968 / 78.2 |
| PoseBusters · apo_relax | 312 | 0.876 / 67.6 | 0.899 / 66.7 | 0.890 / 68.3 | 0.891 / 68.6 | 0.892 / 67.0 | 0.918 / 62.2 | 0.956 / 71.2 |
| Astex · apo_relax | 77 | 0.881 / 70.1 | 0.894 / 70.1 | 0.886 / 71.4 | 0.886 / 71.4 | 0.890 / 71.4 | 0.912 / 66.2 | 0.955 / 71.4 |

The trained approaches close part of the AUC gap (0.886 → 0.896–0.908 against the
cross-validated oracle's 0.914) but their Success@1 (66.5–67.7 %) already sits at the
**in-sample** oracle's 67.9 %: on the headline setting the global models have reached the
ceiling of per-complex linear weights for the Top-1 criterion.

### 5.5 Example complexes (`examples.png`, `examples.csv`)

Chosen on `apo_relax` · `perturb` among the 5,015 PDBbind runs with ≥ 10 hits by the
cross-validated oracle AUC (hard = lowest, easy = highest, median = the middle of the sorted
list; `--example-quantiles` changes the rule). Ranks are 0-based.

| example | pdb | ligand heavy atoms | hits | AUC interface delta | AUC oracle CV / in-sample | first-hit rank: interface delta / oracle CV / in-sample | best-scoring pose | AUC of the S5 models (LP-PDBBind test) |
|---|---|---|---|---|---|---|---|---|
| hard | `4nhx` | 10 | 13 | 0.388 | 0.366 / 0.629 | 23 / 22 / 16 | 4.9 Å, 3.8 REU better than the best hit, 2.4 thresholds from the nearest correct pose; 23 decoys (median 4.1 Å) outscore every hit; 15 % of the incorrect poses degenerate | tree 0.52, reweighted 0.38, ranker 0.45 |
| median | `4tjw` | 22 | 13 | 0.991 | 0.936 / 0.993 | 0 / 0 / 0 | a 1.4 Å hit; no degenerate incorrect pose | 0.99 for all |
| easy | `6udi` | 49 | 12 | 1.000 | 1.000 / 1.000 | 0 / 0 / 0 | a 0.1 Å hit | not a test complex of the split |

What they mean in practice: for `4nhx` (a ten-heavy-atom fragment) the seven terms rank the
correct poses *below* the decoys (AUC 0.39), and the best linear weights for this very complex
still leave the first hit at rank 16 — no reweighting, learned or hand-tuned, can rescue it,
and the S5 models indeed score 0.38–0.52. The median complex is a success: the unmodified
interface delta puts a hit first with a near-perfect AUC (0.99); its cross-validated oracle AUC
of 0.94 only shows the noise of fitting on 13 positives. The easy complex is a textbook funnel
(every score perfect). Note that "hard by AUC" and "Top-1 failure" are different things: in
the lowest decile of the cross-validated oracle AUC the unmodified score fails Top-1 in 69 %
of the runs, the in-sample oracle in 64 %.

### 5.6 UMAP / t-SNE (illustration only, `embedding_pooled.png`)

30,000 poses of 100 random `apo_relax` · `perturb` PDBbind runs (z-scored per run), embedded
together. The 1,587 correct poses are not spread uniformly — they concentrate in one region of
either map, the low-energy end — but that region is shared with thousands of incorrect poses,
and the RMSD colouring shows no boundary. The per-run UMAP panels of `examples.png` show the
same at the complex level: in the hard case the hits sit inside the decoy cloud, in the easy
case they form a separate corner.

### 5.7 Robustness

| variant (apo_relax · perturb, PDBbind) | % incorrect poses degenerate | % energy-degenerate decoys | % runs: best pose a degenerate decoy | oracle AUC CV / in-sample (median) | Success@1 interface delta / oracle CV / in-sample | ρ (oracle CV, fraction degenerate) |
|---|---|---|---|---|---|---|
| interface terms, Euclidean (main text) | 10.2 | 5.0 | 4.3 | 0.938 / 0.973 | 62.6 / 58.2 / 66.4 | −0.67 |
| interface terms, Mahalanobis | 9.9 | 4.8 | 3.7 | (same oracle) | | −0.66 |
| `raw_*` totals of the seven terms, Euclidean | 11.5 | 5.5 | 6.0 | 0.924 / 0.969 | 62.6 / 56.1 / 64.6 | −0.68 |

The Mahalanobis distance changes nothing (the terms are correlated, but the correct–correct
threshold scales with them). The whole-complex totals are slightly *more* degenerate and give
a *weaker* oracle (in-sample Success@1 64.6 vs 66.4), most clearly on the unrelaxed crystal
receptor (oracle in-sample Success@1 58.9 vs 75.1 with the interface terms), the A.1/A.2
counterpart of the benchmark's finding that the total score collapses on the crystal receptor.

### 5.8 Reading

- **The information is not in the seven terms — for the top of the list.** Linear weights
  chosen per complex and fitted on the evaluated poses rank a correct pose first in 66 % of
  the apo_relax runs with ≥ 2 hits (unmodified: 63 %); honestly cross-validated they do worse
  than the defaults. That is the upper bound of any reweighting of these terms, and the S5
  approaches sit at it (§5.4). The S5 result is therefore a statement about the term set, not
  about the models.
- **The failure mode is not nearest-neighbour degeneracy.** Energy-degenerate decoys exist
  (5 % of the incorrect poses on apo_relax · perturb, 1.3 % also scoring at least as well as a
  correct pose) and they mark the hopeless complexes (ρ ≈ −0.7 with the oracle AUC), but they
  are responsible for about one Top-1 failure in ten. The typical Top-1 failure is a 2–5 Å
  near-miss, or a far decoy, with an energy profile of its own that the seven terms rate
  2.6–2.8 REU *better* than the nearest correct pose.
- **AUC and Success@1 disagree because the decoy cloud is easy and the top is hard.** Per-run
  AUCs of 0.9+ mean the hits beat 90 % of the decoys; the ranking is decided by the remaining
  few, and those are separable from the hits by no linear combination of the seven terms in a
  third of the runs (oracle CV AUC < 0.9), nor by the tree (non-linear) in S5.
- **The per-complex optimal weights are inconsistent across complexes** for solvation, the
  pair term and both hydrogen-bond terms (sign flips in ~half of the complexes): a global
  weight set can at best average over them, which is what S5's learned vector does.
- Sampling-side reading (S1/S2): on apo_relax · perturb the sampler produces more distinct,
  well-scoring states; A.1/A.2 show these states are *distinct* in energy space and scored
  better, i.e. the energy function has genuine minima there, not an inability to tell them
  apart.

### 5.9 Caveats

- One docking run is the unit; the threshold is the run's median correct–correct distance and
  depends on how tightly the hits cluster. Runs with one hit (953 on apo_relax · perturb) are
  excluded from every percentage; the run-level Top-1 rates are over runs with ≥ 2 hits.
- The cross-validated oracle is biased downward for runs with few hits (one to two positives
  per fold) and is not a fair estimate of a per-complex scorer's generalisation there; the
  in-sample oracle is the quantity to quote as the ceiling, with the note that 7 weights on
  300 poses can overfit slightly.
- The logistic regression maximises the likelihood, not Success@1. The exact Top-1 ceiling of a
  linear scorer per run (does *any* weight vector put a hit first?) is a linear-programming
  feasibility question that was not computed; it would be at least as high as the in-sample
  oracle's 66 %.
- Score differences to the nearest correct pose refer to the Euclidean nearest neighbour also
  in the Mahalanobis rows.
- UMAP / t-SNE are illustrations of z-scored seven-dimensional vectors pooled over runs; the
  pooled map has no quantitative meaning beyond the overlap it shows.
