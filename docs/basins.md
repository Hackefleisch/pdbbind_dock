# Track 3, A.3 step 2a — the basins of a docking run (`affinity/basins.py`)

> Methods-level documentation of `affinity/basins.py`: the question, the definitions, the outputs
> and the descriptive results. Source files: [`affinity/basins.py`](../affinity/basins.py),
> [`docking/ligand_rmsd.py`](../docking/ligand_rmsd.py) (`LigandSymmetry.pairwise_rmsd`),
> [`docking/result.py`](../docking/result.py) (`Result.docked_ligand_xyz`). Paper context:
> [PAPER1_STORY_INVENTORY.md](PAPER1_STORY_INVENTORY.md), story S7 and Appendix A.3 step 2a
> (item 1 of step 2, decided 2026-10-22; all of A.3 is Paper 1 material). Written 2026-10-22.
> Companion of [affinity.md](affinity.md) (S7, step 1), whose §8 motivates it, and of
> [degeneracy.md](degeneracy.md) (S6), which shares the unit of analysis.

## 0. Headline

**At 1 Å the perturb runs are almost completely fragmented: a typical PDBbind run of 300 poses
falls into 267–272 basins, 91–92 % of them single poses, and the basin of the best-scoring pose
holds a median of 2–4 poses (1 % of the run). At 2 Å a run has 187–193 basins (76 % singletons)
and the best basin 6–10 poses (2–3 %).** The 5 Å-perturbed starts rarely end in the same minimum:
the minimised landscape is rugged at the 1 Å scale, and 300 trajectories do not populate 1 Å
basins — the population of a 1 Å basin is almost always 0 or 1, so the 1 Å cutoff of the decision
sits at the resolution limit of the sampling, and 2 Å is the cutoff at which populations (~10)
and widths exist for most runs. The native basin exists in 86–95 % of the PDBbind runs (99.9 % of
the runs with a hit at 1 Å); when it is not the best basin (24–40 % of the runs that have one) it
sits at median index 3–4 and 1.5–1.9 REU above the minimum — the correct minimum is usually a
few REU away from the wrong one, as S6 found in energy space. The largest basin is the best basin
in only 29–54 % of the runs and native in 42–75 %. Population is a *pose-correctness* signal: on
PDBbind `apo_relax` · `perturb` at 1 Å the best basin has a median population of 4 when the best
pose is a hit and 1 when it is not (Spearman ρ 0.55 with the hit flag), and runs with many hits
have far fewer basins (ρ −0.77 between the number of basins and the number of hits); the number
of basins also grows with the ligand size (ρ 0.45). The share of runs whose best basin is native
reproduces the benchmark's Success@1 exactly (53.1 / 65.9 / 72.0 % for `apo_relax` / `crystal` /
`relax` · `perturb` on PDBbind). Whether any of this carries affinity information beyond the
minimum and the ligand size is the next step (the test of S7's to-do, affinity.md §8.2 protocol);
these tables and the stored RMSD matrices are its input.

## 1. Question and setup

A.3 step 2 asks whether the **distribution** of the sampled poses carries information about the
affinity that the single best pose does not. [affinity.md](affinity.md) §8 showed that the
distribution of the pose *energies* does not (every spread statistic is a scale effect of the
minimum and the ligand size) and named what a second attempt has to change: measure basin width
geometrically, treat the populations as catchments of the perturbed starts, and report the gain
beyond the minimum and the size. Item 1 of step 2 — the pose representation — offered "a reduced
coordinate from the clustering"; step 2a is that choice made discrete and taken to its descriptive
end. Every docking run is reduced to its **basins**: groups of poses within a cutoff RMSD of a
founder, the founder being the lowest-scoring pose of the group. The basins carry the three
quantities the ensemble question needs — population (the catchment of the start distribution),
depth (the founder's score) and width (the RMSD spread of the members) — and the native flag.

| Element | Decision (2026-10-22) |
|---|---|
| Unit of analysis | one **docking run** = one complex × one protocol, 300 poses. The perturb run of each receptor state (`crystal`, `relax`, `apo_relax` · `perturb`); all three sets; complete complexes only (11,444 / 388 / 84). The std runs start at the crystal placement and are not clustered (`--protocols` can add them) |
| Coordinates | the heavy-atom coordinates of the ligand of every pose, reconstructed from the line diffs of the HDF5 file against the docking reference seed (`Result.docked_ligand_xyz`) — the same coordinates the symmetry-corrected `rmsd_to_crystal` of the extraction is computed from |
| Distance | the symmetry-corrected, **in-place** heavy-atom RMSD between every pair of poses of the run (`LigandSymmetry.pairwise_rmsd`; 44,850 pairs per run), the same definition as `rmsd_to_crystal`. Stored as float16 (`rmsd/<set>.h5`) so that other cutoffs or clusterings need no recomputation |
| Order | the interface delta (`idelta_score`, lower is better): the ranking score of the track and the score behind the affinity headline's "best pose". `--score total` orders by the constraint-free total instead |
| Clustering | greedy **leader** clustering on the ordered poses: the best pose founds basin 0; every following pose joins the first founder (in score order) within the cutoff, or founds the next basin. The founder is the basin's minimum, so "the minimum defines the basin" |
| Cutoffs | **1 Å** (the first test of the decision) and **2 Å** (the hit cutoff and the legacy grid's value, the reference), both from the same matrix |
| Hit / native | a pose is a hit when its symmetry-corrected `rmsd_to_crystal` ≤ 2 Å; a basin is **native** when its founder is a hit; the **native basin** of a run is the lowest native basin |
| Not done here | the test of the basin statistics against pK (S7's to-do: population / width of the best and the native basin, basins within a few REU of the minimum, basin-weighted log-sum-exp estimators = Paper 2's S2 / S3, through the partial-correlation protocol of affinity.md §8.2 and as features of the affinity grid) — the next step and the go / no-go for step 2b |

## 2. Definitions

### 2.1 Pairwise symmetry-corrected RMSD

For a run with poses $x_1 \dots x_N$ (each $n$ heavy atoms × 3) and the automorphism group
$\{P_m\}$ of the ligand's heavy-atom graph (element labels only, as in `docking.ligand_rmsd`),

$$d_{ij} = \min_m \sqrt{\tfrac{1}{n}\,\lVert P_m x_i - x_j \rVert^2}.$$

`LigandSymmetry.pairwise_rmsd` computes all $N^2$ distances per automorphism from one Gram matrix,
$\lVert P_m x_i - x_j\rVert^2 = \lVert x_i\rVert^2 + \lVert x_j\rVert^2 - 2\langle P_m x_i, x_j\rangle$
(a permutation of the atoms leaves the norm unchanged), and takes the minimum over $m$. The
matrix is symmetric because the automorphisms form a group. It agrees with the per-pair `rmsd`
to 1e-12 Å and costs 0.01 s per run for a typical ligand (median 4 automorphisms) and 6.5 s for
a ligand at the cap of 10,000 automorphisms (21 PDBbind ligands). The stored float16 copy is
exact to 0.001 Å below 2 Å and to 0.008 Å at 16 Å.

### 2.2 Leader clustering

Sort the poses by the score (stable; poses without coordinates are left out and get basin −1).
The first pose founds basin 0. Every following pose is compared with the founders in founder
order and joins the **first** founder whose distance is ≤ the cutoff; if there is none, it founds
the next basin. Consequences:

- the founder of a basin is its lowest-scoring pose, and basins are numbered by their founder's
  score: basin 0 is the basin of the best pose of the run;
- every member lies within the cutoff of its founder (`d_founder` ≤ cutoff), but two members
  can be up to twice the cutoff apart; a pose within the cutoff of several founders joins the
  lowest-scoring one, not the nearest one;
- the clustering is a function of the order and the matrix only — no seed, no iteration;
- a run's best basin is native if and only if its best pose is a hit, so the share of runs
  with a native best basin equals the benchmark's Success@1 of the score (§6 checks this).

### 2.3 Basin statistics (`basins.parquet`, one row per run × cutoff × basin)

| Column | Definition |
|---|---|
| `basin` | basin index (0 = the basin of the best pose) |
| `founder_seq_id`, `founder_rank` | the founder pose and its score rank in the run (0 = best) |
| `population`, `population_frac` | members including the founder, and their share of the clustered poses |
| `founder_score`, `founder_total`, `gap` | the founder's score (interface delta) and constraint-free total; `gap` = founder score − run minimum (0 for basin 0) |
| `founder_rmsd_to_crystal`, `native` | the founder's RMSD to the crystal ligand; `native` = it is ≤ 2 Å |
| `n_hits`, `min_rmsd_to_crystal` | hits among the members; the members' lowest RMSD to the crystal ligand |
| `width_mean`, `width_max` | mean / maximum RMSD of the members to the founder (the founder excluded; NaN for a singleton) |
| `width_pair_mean` | mean pairwise RMSD among the members (NaN for a singleton) |
| `mean_score` | mean score of the members |

### 2.4 Run statistics (`runs.parquet`, one row per run × cutoff)

`n_basins`, `n_singletons`, `n_basins_within2` / `within5` (basins whose founder lies within 2 / 5
REU of the run minimum), `n_native_basins`, `n_basins_with_hit`, `n_hits`, `e_min`; the best basin
(`best_rmsd_to_crystal`, `best_hit`, `best_population[_frac]`, `best_width_mean`, `best_width_max`,
`best_n_hits`); the native basin (`native_basin` = its index or −1, `native_gap`,
`native_population[_frac]`, `native_width_mean`, `native_founder_rank`, `native_hits_frac` = the
share of the run's hits that lie in it); the population distribution (`pop_entropy` =
−Σ p ln p over the basins, `n_eff` = exp of it); the largest basin (`largest_basin`,
`largest_population`, `largest_is_best`, `largest_native`); the ligand (`n_automorphisms`,
`automorphisms_capped`, `n_heavy`).

### 2.5 Pose columns (`poses.parquet`, one row per pose)

`score`, `total`, `rmsd_to_crystal`, `rank` (score rank in the run), `hit`, `valid` (coordinates
available), `d_best` (RMSD to the best pose of the run), `d_nearest` (RMSD to the nearest other
pose), and per cutoff `c` the columns `basin_<c>`, `founder_<c>`, `d_founder_<c>`.

## 3. Data and counts

`affinity.basins` reads the HDF5 files of the complete complexes of every set directly (the
coordinates are not in the extraction tables); the scores and the symmetry-corrected
`rmsd_to_crystal` come from the same auto-corrected reader as the extraction, and match
`docked.parquet` exactly (§8).

| Set | Complete complexes | Runs (3 perturb protocols) | Poses | Basin rows (1 Å + 2 Å) |
|---|---|---|---|---|
| PDBbind | 11,444 | 34,332 | 10,299,600 | 15.3 M |
| PoseBusters | 388 | 1,164 | 349,200 | 0.5 M |
| Astex Diverse | 84 | 252 | 75,600 | 0.1 M |
| **Total** | 11,916 | **35,748** | **10,724,400** | **15,964,106** |

Every complete complex of every set was processed (0 errors, no pose without coordinates). On
PDBbind `apo_relax` · `perturb` alone there are 3,005,733 basins at 1 Å (91.5 % singletons;
population quantiles: median 1, 90 % ≤ 1, 99 % ≤ 4, max 71) and 2,181,222 at 2 Å (76.5 %
singletons; median 1, 90 % ≤ 3, 99 % ≤ 9, max 148). Native basins are 2.0 % of the 1 Å basins
and 0.65 % of the 2 Å basins; at 2 Å a native basin has a median population of 6 against 1 for a
non-native one and is slightly narrower (mean RMSD to the founder 1.38 vs 1.58 Å).

## 4. Outputs

All under `data/affinity/basins/` (`--score total` → `basins_total/`):

| File | Content |
|---|---|
| `poses.parquet` | one row per pose of every clustered run (§2.5) |
| `basins.parquet` | one row per basin per cutoff (§2.3) |
| `runs.parquet` | one row per run per cutoff (§2.4) |
| `rmsd/<set>.h5` | dataset `<pdb>/<protocol>`: the condensed upper triangle (scipy `squareform` order) of the run's pairwise RMSD matrix, float16, pose order = `seq_id`; `affinity.basins.load_rmsd_matrix(set, pdb, protocol)` returns the square float32 matrix |
| `summary.csv` / `summary.md` | the medians [IQR] and percentages per set × protocol × cutoff (§6) |
| `config.json` | the call, counts, errors, git hash, elapsed time |
| `basins.log` | the console output of the run |

## 5. How to run

```bash
uv run python -m affinity.basins                                   # three perturb protocols, three sets, 1 and 2 Å
uv run python -m affinity.basins --limit 20 --out-dir /tmp/basins  # smoke test
uv run python -m affinity.basins --score total                     # founders by the constraint-free total
uv run python -m affinity.basins --cutoffs 1 1.5 2 --no-matrices   # other cutoffs, without the matrix store
```

The full run (three sets, three perturb protocols, two cutoffs, matrices stored) took 29.7 min
with 22 workers on a 12-core / 24-thread machine (6.8 files per second; run of 2026-10-23, git
`8ccb9ef`). The module pins every worker to one BLAS thread (`OMP_NUM_THREADS` etc. and
`threadpoolctl`), without which the Gram products of the large automorphism groups spawn a thread
pool per worker and the machine is oversubscribed (2.5 files per second). Loading and
decompressing the structures dominates; the pairwise RMSD costs 0.01–0.1 s per run, 6.5 s for a
ligand at the automorphism cap. The output is 4.0 G: the matrices 3.2 G (`rmsd/pdbbind.h5` 3.10 G,
`posebuster.h5` 0.11 G, `astex.h5` 0.02 G), `basins.parquet` 0.55 G, `poses.parquet` 0.42 G,
`runs.parquet` 4 M. Start a full run detached from the terminal session (`setsid nohup … &`);
a run bound to an interactive session dies with it.

## 6. Results

### 6.1 Cross-check with the benchmark

The share of runs whose best basin is native must equal the benchmark's Success@1 of the
interface delta (`data/figures/benchmark/success_<state>_docking_perturb_…_idelta_score_2A.csv`),
and the share of runs with at least one hit its `any_hit`. Both agree to the second decimal on
every set × receptor state (PDBbind: 53.07 / 65.89 / 71.98 % and 88.49 / 95.33 / 95.45 % for
`apo_relax` / `crystal` / `relax`; PoseBusters 57.47 / 67.27 / 72.94; Astex 64.29 / 80.95 / 85.71).

### 6.2 Summary tables (`summary.md`)

Number of basins per docking run:

| set | protocol | cutoff (Å) | runs | basins per run, median [IQR] | singletons per run, median | % of basins singletons | basins within 2 REU of the minimum, median (q75) | within 5 REU | effective number of basins, median [IQR] |
|---|---|---|---|---|---|---|---|---|---|
| pdbbind | `crystal_docking_perturb` | 1 | 11,444 | 267 [243, 283] | 244 | 91 | 2 (6) | 16 (46) | 250.2 [213.0, 274.8] |
| pdbbind | `crystal_docking_perturb` | 2 | 11,444 | 187 [145, 228] | 137 | 76 | 2 (4) | 9 (29) | 142.7 [93.6, 197.1] |
| pdbbind | `relax_docking_perturb` | 1 | 11,444 | 267 [243, 283] | 244 | 91 | 2 (5) | 11 (42) | 250.5 [212.2, 275.6] |
| pdbbind | `relax_docking_perturb` | 2 | 11,444 | 187 [145, 228] | 136 | 76 | 1 (4) | 7 (25) | 143.0 [93.1, 196.9] |
| pdbbind | `apo_relax_docking_perturb` | 1 | 11,444 | 272 [249, 286] | 251 | 92 | 3 (7) | 25 (60) | 258.9 [224.8, 280.7] |
| pdbbind | `apo_relax_docking_perturb` | 2 | 11,444 | 193 [150, 233] | 141 | 76 | 2 (6) | 16 (39) | 153.1 [101.0, 206.4] |
| posebuster | `crystal_docking_perturb` | 1 | 388 | 268 [240, 286] | 245 | 91 | 2 (5) | 12 (39) | 251.0 [208.8, 280.2] |
| posebuster | `crystal_docking_perturb` | 2 | 388 | 181 [132, 227] | 131 | 76 | 2 (4) | 6 (24) | 135.5 [77.4, 193.5] |
| posebuster | `relax_docking_perturb` | 1 | 388 | 268 [239, 285] | 246 | 91 | 2 (5) | 8 (36) | 252.8 [208.1, 277.8] |
| posebuster | `relax_docking_perturb` | 2 | 388 | 180 [134, 228] | 128 | 76 | 1 (3) | 4 (23) | 136.8 [84.1, 196.5] |
| posebuster | `apo_relax_docking_perturb` | 1 | 388 | 272 [245, 288] | 251 | 92 | 3 (6) | 20 (52) | 259.4 [220.1, 283.3] |
| posebuster | `apo_relax_docking_perturb` | 2 | 388 | 186 [142, 233] | 134 | 77 | 2 (5) | 13 (33) | 143.2 [93.8, 205.5] |
| astex | `crystal_docking_perturb` | 1 | 84 | 252 [227, 274] | 220 | 88 | 1 (4) | 8 (28) | 226.8 [188.3, 261.9] |
| astex | `crystal_docking_perturb` | 2 | 84 | 162 [121, 199] | 114 | 71 | 1 (2) | 4 (13) | 106.9 [69.8, 160.2] |
| astex | `relax_docking_perturb` | 1 | 84 | 248 [224, 271] | 218 | 88 | 1 (2) | 6 (20) | 223.0 [182.5, 260.4] |
| astex | `relax_docking_perturb` | 2 | 84 | 158 [123, 200] | 108 | 71 | 1 (2) | 4 (10) | 101.1 [69.4, 157.4] |
| astex | `apo_relax_docking_perturb` | 1 | 84 | 256 [235, 275] | 231 | 89 | 2 (5) | 18 (39) | 233.2 [199.9, 264.8] |
| astex | `apo_relax_docking_perturb` | 2 | 84 | 167 [124, 204] | 112 | 71 | 1 (3) | 11 (25) | 118.5 [75.8, 166.8] |

The basin of the best-scoring pose (`native` = its founder is within 2 Å of the crystal ligand; the share of runs whose best basin is native equals the benchmark's Success@1 of the score):

| set | protocol | cutoff (Å) | best basin: population, median [IQR] | share of the run, median | width (mean RMSD to the founder), median [IQR] | width (max), median | % runs best basin native | % runs largest basin = best | % runs largest basin native | largest basin population, median (q75) |
|---|---|---|---|---|---|---|---|---|---|---|
| pdbbind | `crystal_docking_perturb` | 1 | 3 [1, 8] | 1 % | 0.49 [0.30, 0.67] | 0.79 | 65.9 | 43.2 | 59.6 | 6 (10) |
| pdbbind | `crystal_docking_perturb` | 2 | 10 [4, 22] | 3 % | 1.24 [0.88, 1.50] | 1.92 | 65.9 | 50.8 | 57.6 | 14 (26) |
| pdbbind | `relax_docking_perturb` | 1 | 4 [2, 8] | 1 % | 0.48 [0.31, 0.66] | 0.79 | 72.0 | 47.6 | 61.8 | 6 (10) |
| pdbbind | `relax_docking_perturb` | 2 | 10 [4, 22] | 3 % | 1.22 [0.87, 1.48] | 1.92 | 72.0 | 53.9 | 58.8 | 14 (26) |
| pdbbind | `apo_relax_docking_perturb` | 1 | 2 [1, 5] | 1 % | 0.54 [0.34, 0.72] | 0.81 | 53.1 | 29.4 | 41.9 | 4 (7) |
| pdbbind | `apo_relax_docking_perturb` | 2 | 6 [2, 16] | 2 % | 1.35 [1.04, 1.58] | 1.92 | 53.1 | 37.2 | 42.4 | 12 (21) |
| posebuster | `crystal_docking_perturb` | 1 | 3 [1, 9] | 1 % | 0.44 [0.27, 0.66] | 0.77 | 67.3 | 46.4 | 64.4 | 6 (10) |
| posebuster | `crystal_docking_perturb` | 2 | 12 [4, 27] | 4 % | 1.32 [0.92, 1.53] | 1.94 | 67.3 | 53.4 | 62.9 | 16 (32) |
| posebuster | `relax_docking_perturb` | 1 | 4 [2, 9] | 1 % | 0.47 [0.28, 0.66] | 0.84 | 72.9 | 51.3 | 68.0 | 6 (11) |
| posebuster | `relax_docking_perturb` | 2 | 12 [5, 30] | 4 % | 1.25 [0.91, 1.47] | 1.94 | 72.9 | 59.5 | 65.2 | 16 (31) |
| posebuster | `apo_relax_docking_perturb` | 1 | 2 [1, 6] | 1 % | 0.54 [0.34, 0.74] | 0.81 | 57.5 | 34.3 | 46.6 | 4 (8) |
| posebuster | `apo_relax_docking_perturb` | 2 | 8 [3, 20] | 3 % | 1.38 [1.08, 1.57] | 1.94 | 57.5 | 44.8 | 50.5 | 13 (23) |
| astex | `crystal_docking_perturb` | 1 | 6 [2, 12] | 2 % | 0.48 [0.29, 0.62] | 0.77 | 81.0 | 54.8 | 75.0 | 7 (13) |
| astex | `crystal_docking_perturb` | 2 | 18 [7, 34] | 6 % | 1.21 [0.85, 1.45] | 1.94 | 81.0 | 65.5 | 70.2 | 20 (36) |
| astex | `relax_docking_perturb` | 1 | 5 [3, 13] | 2 % | 0.48 [0.32, 0.66] | 0.86 | 85.7 | 50.0 | 67.9 | 7 (16) |
| astex | `relax_docking_perturb` | 2 | 16 [9, 34] | 5 % | 1.22 [0.93, 1.43] | 1.96 | 85.7 | 70.2 | 69.0 | 20 (38) |
| astex | `apo_relax_docking_perturb` | 1 | 3 [1, 7] | 1 % | 0.56 [0.41, 0.78] | 0.90 | 64.3 | 38.1 | 54.8 | 6 (11) |
| astex | `apo_relax_docking_perturb` | 2 | 11 [5, 24] | 4 % | 1.31 [1.07, 1.49] | 1.94 | 64.3 | 46.4 | 51.2 | 18 (31) |

The native basin (the lowest basin whose founder is a hit):

| set | protocol | cutoff (Å) | % runs with a native basin | % of the runs with a hit | native basins per run, median | % native basin is not the best (of runs with one) | its index, median (q75) | its gap to the minimum (REU), median [IQR] | native basin population, median (q75) | its width, median | share of the run's hits in it, median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pdbbind | `crystal_docking_perturb` | 1 | 95.3 | 99.9 | 5 | 30.8 | 3 (11) | 1.57 [0.71, 2.93] | 4 (9) | 0.49 | 0.42 |
| pdbbind | `crystal_docking_perturb` | 2 | 93.9 | 98.5 | 1 | 29.8 | 3 (9) | 1.64 [0.74, 3.06] | 12 (25) | 1.22 | 1.00 |
| pdbbind | `relax_docking_perturb` | 1 | 95.4 | 99.9 | 5 | 24.5 | 3 (10) | 1.47 [0.67, 2.73] | 4 (9) | 0.49 | 0.43 |
| pdbbind | `relax_docking_perturb` | 2 | 94.3 | 98.8 | 1 | 23.7 | 3 (8) | 1.52 [0.69, 2.80] | 13 (25) | 1.20 | 1.00 |
| pdbbind | `apo_relax_docking_perturb` | 1 | 88.4 | 99.9 | 4 | 40.0 | 4 (14) | 1.81 [0.80, 3.25] | 3 (6) | 0.55 | 0.40 |
| pdbbind | `apo_relax_docking_perturb` | 2 | 86.3 | 97.5 | 1 | 38.5 | 4 (12) | 1.86 [0.82, 3.32] | 9 (20) | 1.32 | 1.00 |
| posebuster | `crystal_docking_perturb` | 1 | 96.1 | 100.0 | 6 | 30.0 | 3 (11) | 1.57 [0.85, 2.58] | 5 (10) | 0.46 | 0.35 |
| posebuster | `crystal_docking_perturb` | 2 | 94.8 | 98.7 | 1 | 29.1 | 2 (7) | 1.69 [0.91, 2.57] | 16 (33) | 1.30 | 0.99 |
| posebuster | `relax_docking_perturb` | 1 | 95.1 | 100.0 | 6 | 23.3 | 4 (9) | 1.72 [1.10, 2.50] | 5 (11) | 0.48 | 0.42 |
| posebuster | `relax_docking_perturb` | 2 | 94.3 | 99.2 | 1 | 22.7 | 3 (7) | 1.73 [1.11, 2.61] | 15 (33) | 1.22 | 1.00 |
| posebuster | `apo_relax_docking_perturb` | 1 | 87.9 | 99.4 | 5 | 34.6 | 4 (15) | 1.80 [0.94, 3.55] | 3 (7) | 0.55 | 0.33 |
| posebuster | `apo_relax_docking_perturb` | 2 | 86.1 | 97.4 | 1 | 33.2 | 3 (12) | 2.20 [1.06, 3.51] | 12 (25) | 1.35 | 0.97 |
| astex | `crystal_docking_perturb` | 1 | 97.6 | 100.0 | 6 | 17.1 | 3 (5) | 1.34 [0.74, 1.73] | 6 (13) | 0.55 | 0.37 |
| astex | `crystal_docking_perturb` | 2 | 97.6 | 100.0 | 1 | 17.1 | 2 (5) | 1.48 [0.76, 1.73] | 19 (37) | 1.24 | 1.00 |
| astex | `relax_docking_perturb` | 1 | 98.8 | 100.0 | 7 | 13.3 | 4 (10) | 1.95 [1.38, 2.38] | 5 (15) | 0.48 | 0.36 |
| astex | `relax_docking_perturb` | 2 | 97.6 | 98.8 | 1 | 12.2 | 2 (8) | 2.00 [1.30, 2.52] | 17 (39) | 1.22 | 1.00 |
| astex | `apo_relax_docking_perturb` | 1 | 94.0 | 100.0 | 5 | 31.6 | 3 (9) | 1.93 [0.55, 3.74] | 3 (10) | 0.55 | 0.39 |
| astex | `apo_relax_docking_perturb` | 2 | 92.9 | 98.7 | 1 | 30.8 | 2 (6) | 1.77 [0.53, 3.63] | 13 (31) | 1.30 | 0.97 |

### 6.3 Dependence on size and on sampling success (PDBbind, `apo_relax` · `perturb`, 1 Å)

| Quantity | Value |
|---|---|
| Spearman ρ (number of basins, heavy atoms) | 0.45 |
| median number of basins for ≤ 15 / 15–20 / 20–25 / 25–30 / 30–40 / > 40 heavy atoms | 242 / 258 / 267 / 274 / 281 / 287 |
| Spearman ρ (number of basins, number of hits) | −0.77 |
| median number of basins when the best pose is a hit / is not | 264 / 280 |
| median population of the best basin when the best pose is a hit / is not | 4 / 1 |
| Spearman ρ (population of the best basin, best pose is a hit) | 0.55 |
| Spearman ρ (population of the best basin, heavy atoms) | −0.13 |
| runs whose best basin is a singleton | 41.3 % |
| runs where the native basin is not the best and lies ≤ 2 / ≤ 5 REU above the minimum | 19.2 % / 31.8 % |
| runs consisting only of singletons | 0.6 % |

## 7. Reading

**The sampled landscape is rugged at the 1 Å scale.** A perturb trajectory starts up to 5 Å
from the crystal placement with a random orientation and a resampled conformer, walks 500
Monte-Carlo steps at T = 5 on the softened grid score, takes the best pose it visited through one
high-resolution cycle and minimises. The end points of 300 such trajectories almost never coincide
within 1 Å: 92 % of the 1 Å basins are single poses, and even the best-scoring pose has company
in its basin in only 59 % of the `apo_relax` runs. The energy landscape the sampler reports is a
field of hundreds of shallow local minima, not a handful of funnels. This has three consequences
for the ensemble question:

1. **Resolution.** With 300 trajectories a 1 Å basin cannot be *populated*; the population is a
   Bernoulli outcome, not a count. At 2 Å the best and the native basin hold ~10 poses, which is
   a count with ~30 % Poisson error. The test of the basin statistics (next step) should run on
   both cutoffs, with 2 Å expected to be the usable one; the stored matrices allow 1.5 Å or a
   nearest-founder assignment without recomputation.
2. **Width is often undefined.** The width of a singleton is NaN: at 1 Å the best basin has a
   width in 59–75 % of the runs, at 2 Å in 83–92 %. A width-weighted estimator (Paper 2's S3)
   needs a fallback for singletons or the 2 Å cutoff.
3. **Everything scales with size and with sampling success.** The number of basins grows with
   the heavy-atom count (ρ 0.45; 242 for the smallest ligands, 287 for the largest) and shrinks
   with the number of hits (ρ −0.77), because hits cluster and misses do not. As in
   affinity.md §8.3, any basin statistic has to be tested for its gain *beyond* the minimum and
   the size, and its correlation with the hit count means it will also track the ranking quality
   of the run.

**Population is a pose-correctness signal, as expected from consensus docking.** The best basin
is four times larger when the best pose is correct (median 4 vs 1 at 1 Å), and runs that re-dock
well have fewer, larger basins. Picking the largest basin instead of the best score would be
worse (largest basin native in 42 % of the `apo_relax` runs against Success@1 of 53 %), but the
population is information the interface delta does not contain and could complement it in the
ranking track; this is noted for S6 / T2 and not pursued here. It is also the kind of signal that
could *confound* an affinity test: a per-complex population statistic will partly measure how
well the complex docks, which S7's diagnostics showed to be unrelated to pK (hit fraction, partial
ρ 0.03).

**The native minimum is near the wrong one.** When the native basin is not the best (24–40 % of
the runs with one), its founder is the 3rd–4th basin and 1.5–1.9 REU above the minimum (IQR
0.7–3.3); a fifth to a third of all runs have the native basin within 2 / 5 REU of a wrong
minimum. This is the geometric counterpart of S6's finding that the winning decoys score 2.8 REU
better than their nearest correct pose: the correct basin is there, close in energy, and the score
does not single it out.

**What the populations measure.** The starts of the perturb protocol lie within 5 Å of the
crystal placement, so a basin's population is the catchment of that neighbourhood, biased towards
the native region for weak and strong binders alike; the unbiased catchments need Appendix B
run A with recorded starts. On the existing data the populations are what they are: a descriptive
map of where 300 perturbed starts end, with the native basin holding 3–13 of them.

**Next step (S7's to-do).** From `runs.parquet` and `basins.parquet`: population and width of
the best and the native basin, the number of basins within 2 / 5 REU of the minimum, the
population entropy, and the basin-weighted log-sum-exp estimators (weight = population, Paper 2's
S2; weight = width, S3) per labelled complex, through the partial-correlation protocol of
affinity.md §8.2 (ρ with pK, with the heavy-atom count, partial ρ given size and minimum, the
matched weak / strong comparison) and as additional features in the affinity grid (MAE against the
calibrated interface delta, the lower bounds). A positive result is the go for step 2b; a null
result closes the ensemble question on the existing samples and hands it to run A.

## 8. Validation

Checks performed on 2026-10-22 (smoke test on four complexes per set) and 2026-10-23 (the full run):

- `LigandSymmetry.pairwise_rmsd` against the per-pair `LigandSymmetry.rmsd` on every pair of a
  run: identical to 3e-15 Å; symmetric to 3e-15 Å; zero diagonal. The stored float16 matrix
  differs from the float64 one by at most 0.0074 Å (relative 5e-4).
- `leader_clusters` against a plain double loop over the ordered poses and the founders: identical
  labels at both cutoffs (13gs · `apo_relax` · `perturb`: 286 basins at 1 Å, 239 at 2 Å).
- `score`, `total` and `rmsd_to_crystal` of `poses.parquet` against `docked.parquet` of the
  extraction, matched on (pdb, protocol, seq_id): identical.
- Invariants over all runs of the smoke test: populations sum to 300; founder scores increase
  with the basin index; every founder has `d_founder` = 0 and every member `d_founder` ≤ cutoff;
  `width_mean` ≤ cutoff; `best_hit` ⟺ `best_rmsd_to_crystal` ≤ 2 Å ⟺ `native_basin` = 0.
- The refactoring of `docking/result.py` (the coordinate accessors) leaves the extraction tables
  of five files from the three sets byte-identical (`extract_one` before / after).
- Full run (2026-10-23): the share of runs with a native best basin and the share of runs with a
  hit equal the benchmark's Success@1 and `any_hit` on every set × receptor state (§6.1); all
  35,748 runs clustered, no pose without coordinates, 0 errors.
