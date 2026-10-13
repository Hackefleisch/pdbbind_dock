# RosettaLigand: Preliminary Paper Outline (Paper 1 — characterisation)

> **Two-paper split (decided 2026-09-22).** This is **Paper 1**: the characterisation paper
> (target: JCIM / JCTC). Everything about ensemble thermodynamics, entropy estimation, learned
> Boltzmann-consistent reweighting and the ML-facing dataset release has moved to
> **Paper 2** — see [PAPER2_CONCEPT.md](PAPER2_CONCEPT.md). Paper 1 keeps only a baseline
> affinity correlation and points to Paper 2 in the outlook.
>
> **Working document — follows "Paper Structure Idea 3.0"** from the Confluence page
> *RosettaLigand vs Posebuster*
> (https://meilerlab.atlassian.net/wiki/spaces/~898177309/pages/3534225411).
>
> Built on two source documents in the project:
> - **Luca Friebe, B.Sc. thesis** — *Assessment and improvement of RosettaLigand's
>   docking performance on state-of-the-art deep learning benchmarks* (Leipzig, May 2025).
> - **Josephine Eboh, presentation** — *The impact of the initial ligand position on
>   docking outcomes using RosettaLigand.*
>
> Plus the **Felipe** affinity/ensemble notes and the cluster-scoring discussion (Confluence + prior analysis).

---

## How to use this file with Claude Code

This is a **scaffold**, not a finished draft. Numbers carried over from the thesis and the
presentation reflect the *older* runs; the current automated pipeline re-docked everything with
stricter filters and more samples, so **most quantitative claims must be refreshed from the repo
before they go into the paper.** Markers used throughout:

- `[FROM REPO]` — pull the current value/figure from code or results; the placeholder is the
  last known value and may be stale.
- `[VERIFY]` — a claim that was true in a source document but should be re-confirmed against the
  latest data.
- `[TODO]` — content that does not exist yet and needs to be written/run.
- `[CITE]` — citation needed; where it says "cite own work", prefer Meiler-lab / Eisenhuth
  publications per the idea-3 brief.
- `[COMPUTE]` — needs **new docking/relax runs**, not just re-analysis. All such items are
  collected in **Appendix C**; Paper 2's compute list is in
  [PAPER2_CONCEPT.md](PAPER2_CONCEPT.md) §4, and the two overlap (C3/C5 there = A/B here).

Repos to check for the latest code and results:
- Analysis / ML: `github.com/lucafrb/rosettaligand-vs-posebuster`
- Docking pipeline: `github.com/Hackefleisch/pdbbind_dock`

Suggested first pass for Claude Code: (1) read the results directory and regenerate the headline
tables in §4; (2) replace every `[FROM REPO]` placeholder; (3) flag any claim where the new
pipeline contradicts the thesis-era number.

---

## Core thesis of the paper (idea 3.0, expanded)

RosettaLigand is **widely used and trusted** — it powers REvoLd, posted the best scoring results
at CACHE Challenge #2, and Rosetta remains state-of-the-art in adjacent areas like protein design
— **yet its standalone docking-and-scoring behaviour has never been characterised at scale on
modern, leakage-controlled benchmarks.** People reach for it without a clear picture of *when and
why* it works.

This paper closes that gap. We dock a large, carefully filtered benchmark, quantify performance,
and then turn to the central diagnostic finding: **RosettaLigand samples near-native poses well
but ranks them poorly** — it routinely *overrates incorrect conformations*. We show that this
ranking failure is hard to fix by reweighting or by learning a new scoring head (gains are real
but small), and we argue from the energy landscape (dimensionality reduction) that the cause is a
**dense field of near-degenerate local minima** that the current score cannot separate. Finally we
report the baseline affinity (Kd) correlation of the default score and point to ensemble
thermodynamics (Paper 2) as the route forward. The raw docking output (Rosetta energy scores plus
energetically minimised complexes spanning the full RMSD range) is deposited with this paper; the
curated ML benchmark built on it is Paper 2's contribution.

The arc, in one line: *characterise → expose the sampling/scoring gap → try to fix scoring and
explain why it resists → baseline affinity correlation → outlook (Paper 2).*

### Story update 2026-10-09 — three tracks

The paper is told as **three tracks**, each with a lead contributor. The sections below have not
yet been re-ordered to match; the detailed spec of the new analyses is in **Addendum D**.

| Track | Question | Lead | Status |
|---|---|---|---|
| **T1 — Benchmark** | How well does RosettaLigand work on PDBbind, PoseBusters and Astex? *Side aim:* **what is the mode by which RosettaLigand samples near-crystal structures** — when does it find the crystal basin at all? | Paul / Luca; side aim **Josephine** | data exists; §4.1–§4.4 |
| **T2 — Fix the ranking** | Can the failure mode (picking the right pose) be repaired? Reweighting + ML heads → **mild improvements only**. Then: *why?* UMAP/t-SNE **plus the new energy-degeneracy analysis** (Addendum D.1) and the **per-complex oracle probe** (Addendum D.2) | **Luca**; analysis Paul | partly done; new analyses `[TODO]` |
| **T3 — Affinity** | Can score ↔ binding free energy be improved? Simple reweighting of the best pose first (**prelim data: small gain**), then exploit T1/T2: RosettaLigand is a sampler of physically plausible poses biased toward low energy, so fit a **density over the sampled poses, separate basins from flat regions, and map back to an equilibrium free energy with Bayesian uncertainty** (Addendum D.3) | Paul / Felipe | prelim; `[TODO]` |

The logic that links them: T1 establishes that the sampler works, T2 establishes that the energies
cannot tell right from wrong, and T3 turns both facts around — if the samples are physically
plausible but the energies are not discriminative, use the **distribution of samples** rather than
the energy of the single best one.

> **Framing note for the writing pass:** the reweighting "failure" is a *result*, not a
> disappointment. It is evidence about the shape of the energy landscape. Write it as such.

---

## Working title (pick / refine)

- *"How good is RosettaLigand, really? A large-scale characterisation of physics-based
  protein–ligand docking and its scoring limits"*
- *"RosettaLigand at scale: strong sampling, weak ranking, and what that tells us about the
  scoring function"*
- `[TODO]` decide title with co-authors.

**Author line / contributions** `[TODO]` — note Luca (thesis: benchmark + ML scoring),
Josephine (starting-position analysis), Felipe (affinity/ensemble analysis), Paul
(pipeline, ensemble scoring, integration).

---

## Abstract `[TODO]`

One paragraph, written last. Should hit: (1) RosettaLigand's prominence vs. the lack of a
systematic performance characterisation; (2) the large-scale benchmark across PDBBind, Astex
Diverse, and PoseBusters; (3) the headline finding that sampling succeeds where scoring/ranking
fails; (4) the reweighting/ML result and the landscape explanation; (5) the affinity correlation
and ensemble outlook; (6) the released dataset.

---

## 1. Introduction

- **1.1 Why protein–ligand docking matters.** HTS vs. virtual HTS, structure-based drug design,
  cost/time reduction. *(Source: Luca §2.1; classical examples — HIV protease/Viracept, β2-AR,
  ACE/Captopril.)* `[CITE]`
- **1.2 The deep-learning wave and its cracks.** DiffDock, EquiBind, TankBind predict poses in
  seconds but PoseBusters showed AI methods often produce physically invalid poses and
  generalise poorly to novel sequences. *(Source: Luca §1, §2.4; Buttenschoen et al. 2024.)*
- **1.3 The case for RosettaLigand — and the open question.** Physics-based, induced-fit, full
  flexibility; embedded in widely used pipelines (REvoLd) and competitive in community challenges
  (CACHE #2 best scoring); Rosetta still central to protein design. **But nobody has pinned down
  its real docking/scoring performance at scale.** This is the gap. *(idea 3.0; cite own/lab work
  heavily here)* `[CITE]`
- **1.4 Contributions.** Enumerate:
  1. Largest controlled RosettaLigand benchmark to date across three datasets and six protocols.
  2. Systematic separation of **sampling** ability from **scoring/ranking** ability (Top-1/5/150).
  3. Starting-position sensitivity analysis (Josephine): how initial RMSD, perturbation, and the
     high-resolution step shape what gets found.
  4. An honest attempt to repair scoring (reweighting + three ML heads) and a landscape-level
     explanation of why it resists.
  5. Baseline affinity (Kd) correlation of the default score (min vs. mean-of-N vs. simple
     reweighting); the ensemble / entropy–enthalpy treatment is deferred to Paper 2.
  6. Deposited raw docking output: scores + minimised complexes across all RMSD ranges.

---

## 2. Background

- **2.1 The docking problem.** Lock-and-key → induced-fit; conformation + orientation + affinity;
  minimise binding energy. *(Luca §2.2.)*
- **2.2 Traditional docking.** Sampling + scoring; rigid vs. flexible. *(Luca §2.3.)*
- **2.3 Scoring-function taxonomy.** Force-field, empirical, knowledge-based, consensus, ML-based.
  *(Luca §2.3.2.)*
- **2.4 RosettaLigand in detail.** Two-stage protocol — low-resolution grid sampling (TRANSFORM,
  replacing TransRot; DeLuca et al. 2015) and high-resolution full-atom refinement (Monte-Carlo
  Metropolis / gradient minimisation). Energy = weighted sum of van der Waals, implicit solvation,
  explicit H-bonding, electrostatics. **Note the pre-Talaris2013 score function** used by
  RosettaLigand (relevant later: `fa_pair` carries residue–residue electrostatics, `fa_elec` is
  the ligand-specific Coulombic term). *(Luca §2.3.3; Meiler & Baker 2006.)* `[VERIFY]` exact
  score function / flags against the pipeline config.
- **2.5 Deep-learning docking methods.** EquiBind (E(3)-equivariant GNN, blind, rigid receptor),
  TankBind (trigonometry-aware, pocket + pose), DiffDock / DiffDock-L (diffusion + confidence
  model). *(Luca §2.4.)*
- **2.6 Benchmarks and datasets.** PDBBind (scale), Astex Diverse (standard set, used by
  PoseBusters), PoseBusters (explicitly disjoint from PDBBind; physical/chemical validity suite —
  stereochemistry, bond lengths/angles, aromatic planarity, clashes), and why PLINDER is too large
  for a physics-based method at this scale. *(Luca §2.5.)*

---

## 3. Methods

> Most of this section now reflects the **new automated pipeline** (Confluence: PreProcessing,
> Filters, Docking Fails), which supersedes the thesis's earlier preparation. Cross-check every
> step against `pdbbind_dock`. `[VERIFY]`

- **3.1 Dataset construction and filtering.**
  - PDBBind filter cascade (Confluence "Filters"): start 19,535 → remove peptides, covalent,
    incomplete, redundant, resolution > 2.5 Å, NMR, ligand valence errors, QED < 0.1, heavy-atom
    bounds (<8 or >500), rare elements, protein > 1000 AA, co-ligands, mutated residues →
    **11,518 before docking**. `[FROM REPO]` confirm final counts.
  - Note on resolution semantics (2.5 Å ≠ atoms can be 2.5 Å off). *(Confluence.)*
- **3.2 Structure preprocessing.**
  - **Receptors:** keep TER/ATOM/HETATM only; drop chains with no atom within 7 Å of any ligand
    atom; strip hydrogens, waters, and co-factors outside 7 Å — rely on Rosetta's implicit water
    and automatic hydrogen placement.
  - **Ligands:** from PDBBind 2020+ SDF (kekulised, 3D). RDKit load → sanitise → remove H →
    re-protonate (Rocco's functions) → define aromatic rings → embed to heavy-atom coords →
    ETKDGv3 conformers gated on rotatable bonds (<8 → 50, <12 → 200, >12 → 300), keeping and
    rigidly aligning to the input conformation → add to Rosetta as residue.
  - Acknowledge the known error sources: protonation and kekulé-structure ambiguities.
    *(Confluence "PreProcessing"; Rocco's reprotonation.)* `[VERIFY]`
- **3.3 Docking protocols.** Six protocols = {Standard, Perturb} × {PDB-Structure, Relax,
  Relax-Ligaway}. Perturb displaces the ligand by up to 5 Å for a harder, more realistic test;
  Relax pre-optimises protein+ligand (mean prep ≈ 4.7 h `[VERIFY]`). *(Luca §4.1; Fig. "docking
  protocols".)* Describe the receptor preparation exactly as `docking/runner.py` does it:
  - **Relax:** 10 FastRelax seeds of the complex with the ligand in place; backbone coordinate
    constraints kept on throughout (`AtomCoordinateCstMover`, no ramp-down).
  - **Relax-Ligaway:** ligand translated 500 Å along the ligand jump, then 10 FastRelax seeds of the
    whole system (same constraints), then the ligand is translated back into the pocket.
  - Docking for each receptor state starts from the **lowest-total-score** seed of that state.
  - **Wording:** Relax-Ligaway is a *holo-derived receptor relaxed without the ligand*, **not an apo
    structure**. The pocket keeps its holo shape (backbone constrained). Do not call it "apo" in
    the text.
  - **Wording:** in all three receptor states the Standard protocol starts from the crystal ligand
    position, and Perturb is still pocket-centred. Neither is blind docking; say so explicitly
    when comparing against DL methods.
- **3.4 Sample counts.** `[FROM REPO]` — thesis used ~150 poses/PDB; the new pipeline increased
  samples to ~200–300 (Confluence ToDo "Increase number of samples to 200-300"). State the final
  number actually used.
- **3.5 Metrics.**
  - RMSD to crystal **and** to input (Rosetta-specific). 2 Å success cutoff. *(Luca §3.1.1.)*
  - **Top-1 / Top-5 / Top-150** by Interface Delta Score — operationalises "can the score *pick*
    the good pose it found?" *(Luca §4.1.1.)*
  - PoseBusters validity suite (PB-valid). *(Luca §3.1.2.)*
  - Total Score vs. Interface Delta Score, and why the latter better reflects pose quality.
    *(Luca §3.1.3–3.1.4.)*
- **3.6 Starting-position experiment (Josephine).** From a reference complex (case study 4j1p,
  then five more complexes): generate 1000 starting points with the RosettaLigand protocol,
  select several with varying starting RMSD, generate 150 conformations from each. Variants:
  with/without **initial perturbation**; starting points generated with/without the
  **high-resolution docking step**. *(Eboh slides 9, 12, 16, 22.)*
- **3.7 Scoring-improvement methods.**
  - Interface-delta feature set: the 7 non-zero terms — `fa_atr`, `fa_elec`, `fa_pair`, `fa_rep`,
    `fa_sol`, `hbond_bb_sc`, `hbond_sc`. Engineered features: `atr_rep_ratio`, `atr_rep_sum`,
    `hbond_total` (then drop the two raw hbond terms). RobustScaler. *(Luca §4.5.4.)*
  - **Baselines:** Random Forest Regressor, Gradient Boosting Regressor.
  - **MLP heads (three losses):** regression (MSE, predict RMSD), classification (BCE, good/bad at
    2 Å → confidence used for ranking), learning-to-rank (LambdaRank-style pairwise; also
    AllRank LambdaRank and NeuralNDCG). *(Luca §4.5.)*
  - **Reweighting:** bias-free linear layer = one learned weight per term. *(Luca §4.5.8.)*
  - **Hyperparameter / ablation plan** `[TODO]` (Confluence ToDos): Bayesian optimisation; layer
    count and width tuned jointly; categorical layer sizes [8,16,32,64,128,256] rather than
    ranges; ~30 epochs sufficient (early stopping ~15). State final protocol. `[FROM REPO]`
  - **Train/test split.** DiffDock split for comparability; thesis: 347 test PDBs (363 − 16
    failed), ~2.57 M docked complexes for training. `[FROM REPO]` update counts; the random-split
    caveat (Confluence ToDo "current training uses random splits — do something cleverer") should
    be addressed or at least flagged. Also "filter out positive scores" before training
    (Confluence ToDo). `[VERIFY]`
- **3.8 Energy-landscape analysis.** tSNE / UMAP (and possibly t-NEB) on the per-pose energy-term
  vectors to visualise why scoring struggles. Per-PDB embedding (energies are not comparable
  across PDBs) **or** normalise each pose by the sum of its terms to get relative contributions.
  *(Confluence Notes/ToDos.)* `[TODO]` finalise the approach actually run.
  - **The embedding is the illustration; the statistic is the evidence.** Reviewers discount UMAP
    pictures. The quantitative claims come from the **energy-degeneracy analysis** and the
    **per-complex oracle probe** — see **Addendum D.1 / D.2**.
- **3.9 Baseline affinity correlation.**
  - Correlation of score vs. experimental Kd/affinity: minimum Interface Delta vs. mean of N
    lowest vs. reweighted score. *(Felipe: r ≈ 0.392 min, 0.376 mean-of-10, 0.418 reweight —
    `[FROM REPO]` refresh.)*
  - Restrict the headline correlation to Kd/Ki; report IC50 separately (not a thermodynamic
    quantity).
  - **Moved to Paper 2:** ensemble / cluster scoring, population and width weights, entropy
    estimation, the bound-vs-separated thermodynamic cycle. Keep only one sentence here pointing
    there, so Paper 1 does not pre-empt Paper 2's results.

---

## 4. Results

> Headline structure mirrors the idea-3 arc. Regenerate **all** tables/figures from the repo;
> the values below are last-known and largely thesis-era unless marked otherwise. `[FROM REPO]`

- **4.1 RosettaLigand samples near-native poses well (all-pose view).**
  - PDBBind % ≤ 2 Å by protocol *(Luca §4.1; `[FROM REPO]`)*:
    | Protocol | Standard | Perturb |
    |---|---|---|
    | PDB-Structure | 93.2% | 77.3% |
    | Relax | 93.6% | 78.2% |
    | Relax-Ligaway | 90.2% | 71.1% |
  - Astex: all-150-pose **82.4% ≤ 2 Å** (Perturb-Relax). PoseBusters: **80.7% ≤ 2 Å**.
    *(Luca §4.2.1, §4.3.)* `[VERIFY]` against new dataset (Astex final 84, PoseBuster final 388).
  - Takeaway: as a **sampler**, RosettaLigand is strong and beats reported DL methods on raw RMSD.
  - **Caveat that must be stated, and ideally answered with new runs.** All six protocols are
    seeded at (Standard) or within 5 Å of (Perturb) the crystal ligand position, i.e. this is
    re-docking with a known pocket, whereas DiffDock/EquiBind numbers are usually blind. Either
    restrict the comparison to pocket-conditioned DL baselines, or run **Appendix C / A
    (pocket-box docking from random starts)** `[COMPUTE]` and report that as the comparable
    number. Reviewers will ask.
- **4.2 But scoring/ranking lags — the central gap.**
  - Top-1 vs Top-5 vs all-150: Top-1 is substantially worse, exposing the sampling–scoring
    discrepancy. *(Luca §4.1.1, §4.2.1, Fig. 4.5; §4.3.)* `[FROM REPO]` current Top-k table.
  - Frame explicitly: *the correct pose is in the bag; the score won't reliably pull it out.*
- **4.3 Starting position governs what is found (Josephine) — T1 side aim: the mode by which
  RosettaLigand reaches the crystal basin.**
  - Frame the section as *when and how does the sampler find near-crystal structures at all?*,
    not as a separate study. The answer feeds T2 (what the score then does with those poses) and
    T3 (how trustworthy the sampled distribution is).
  - Success rate collapses as starting RMSD grows (4j1p: 1.92 Å → 7.33%; 3.14 Å → 4.67%;
    ≥ 5.7 Å → ~0%). *(Eboh slide 10.)*
  - **Initial perturbation** redistributes rather than rescues success — helps a little at
    mid-range starts, hurts at near-native starts. *(Eboh slides 13–14.)*
  - **Skipping the high-resolution step when generating starting points** markedly raises success
    (1.38 Å → 19.33%, 1.75 Å → 22.67% without perturbation). *(Eboh slide 18.)* `[VERIFY]`
    interpretation.
  - Generalises across six complexes. **Key qualitative result:** RosettaLigand *can* sample the
    correct conformation from diverse starts, **but often overrates incorrect ones** — same gap as
    §4.2, seen from the sampling side. *(Eboh slides 22, 24.)*
- **4.4 Physical/chemical validity (PoseBusters suite).**
  - On RMSD-only, RosettaLigand tops all PoseBusters-reported methods; on RMSD ≤ 2 Å **and**
    PB-valid it ranks **3rd, behind Vina and Gold**. *(Luca §4.2.1, §4.3.)* `[VERIFY]` — the new
    pipeline fixed RDKit/kekulé issues (Confluence: "RDKit Probleme fixen" ✔), so PB-validity
    likely improved. **Re-run and report the corrected numbers** — this is one of the biggest
    expected changes vs. the thesis.
  - Failure-mode breakdown: previously kekulisation (RDKit) and `minimum_distance_to_protein`
    dominated. *(Luca §4.2/§4.3 figures.)* `[FROM REPO]` updated failure histogram.
- **4.5 Can we fix the scoring? (reweight + ML).**
  - Last-known PDBBind Top-1 / Top-5 *(Luca §4.5; `[FROM REPO]`)*:
    | Method | Top-1 | Top-5 |
    |---|---|---|
    | RosettaLigand score fn (baseline) | best | — |
    | Random Forest Reg. | 51.52% | 63.1% |
    | Gradient Boosting Reg. | 50.41% | 63.9% |
    | MLP Regression | 49.58% | 62.0% |
    | MLP Classification | 49.86% | 62.53% |
    | **LambdaRank-style pairwise** | **52.3%** | **66.1%** |
    | LambdaRank (AllRank) | 36.64% | 49.0% |
    | NeuralNDCG (AllRank) | 38.3% | 51.8% |
    | Reweight (linear) | 51.24% | 65.84% |
  - **Story:** regression/classification are dead ends (can't learn RMSD from the terms); reweight
    gets *close* but doesn't beat the native score; the **LambdaRank-style ranker** is the only
    approach that edges ahead, and only by ~1–2% on Top-5. Gains are real but small. *(Luca §4.5,
    §5.2.)* `[FROM REPO]` add Astex/PoseBuster columns and the latest tuned numbers.
- **4.6 Why scoring resists improvement: the energy landscape.**
  - **Main evidence (new, Addendum D):** the energy-degeneracy density plot and scatter (D.1) —
    many poses that are arbitrarily wrong carry an energy profile indistinguishable from a correct
    one — plus the per-complex oracle probe (D.2), which bounds what *any* reweighting could
    achieve. Together these turn §4.5's small gains from "our models were not good enough" into
    "the term set does not contain the information".
  - tSNE/UMAP of energy-term vectors → `[TODO]` describe what the embedding shows; illustration
    for D.1, not evidence on its own. Working
    hypothesis (idea 3.0): **too many near-degenerate local minima** with near-identical energy
    signatures, so no reweighting of these terms can cleanly separate native from decoy.
  - Optional heat-map deliverable (Confluence ToDo): aggregation (y) × term-selection (x), doubled
    on x for trained vs. standard weights; two maps (MAE and Pearson r) — to test whether more
    aggregation helps as more entropic terms are dropped. `[TODO]`
- **4.7 Affinity (T3).**
  - Score vs. Kd: min vs. mean-of-N vs. reweighted (Felipe's r-values above). `[FROM REPO]`
  - Reweighting fitted on the best pose against affinity: small gain, prelim data `[FROM REPO]`.
  - Then the sampled-density free energy with Bayesian error bars (**Addendum D.3**), if the
    Paper 1 / Paper 2 boundary is settled that way.
  - Otherwise short and descriptive. Felipe's testable predictions (ensemble >
    minimum; open vs. deep pockets; ΔH / TΔS decomposition) and the negative result on raw
    score-distribution free energies all move to **Paper 2**, where they are the motivation for
    estimating the sampler's density.

---

## 5. Discussion

- **5.1 Sampling vs. scoring, unified.** §4.2 (ranking) and §4.3 (overrating incorrect poses) are
  the same phenomenon. RosettaLigand's search is good; its energy ordering is the bottleneck.
- **5.2 The bottleneck is structural, not a tuning problem.** Modest ML/reweight gains + the
  landscape embedding (§4.6) argue the limitation is intrinsic to the term set: near-degenerate
  minima, and entropy/enthalpy entangled within single terms (`fa_sol`). You cannot reweight your
  way to a clean ΔH/TΔS split when the terms were never built to represent them independently
  *(Felipe; Jens)*.
- **5.3 Implications for practice.** Concrete guidance tied to the docking questions: dock into
  relaxed vs. crystal; Total Score vs. Interface Delta; how many samples to reach a sub-2 Å pose;
  how tightly RL stays near a near-native input. *(Confluence "Docking Questions" 1–6.)* `[TODO]`
  answer each from the data.
- **5.4 Limitations.** RDKit/kekulé and protonation ambiguities; random-split training; PB-valid
  third-place; Astex/PoseBuster sizes after filtering; single-conformation entropy not yet modelled.

---

## 6. Conclusion and Outlook

- **6.1 Conclusion.** RosettaLigand is a strong, physically grounded **sampler** whose **ranking**
  is its weak point; we quantified this across three benchmarks, showed it is hard to repair by
  reweighting/ML, and explained it via the energy landscape. It remains a sound default for
  protein–ligand docking when sampling breadth matters.
- **6.2 Outlook.** One paragraph pointing to Paper 2: treat the docking engine as a
  non-equilibrium sampler, estimate its sampling density, and combine enthalpy from the classical
  score with entropy from the ensemble. *(Felipe; Paper 2.)*
- **6.3 Deposited data.** Rosetta energy scores + energetically minimised complexes across all
  RMSD ranges (raw HDF5 output). The curated ML benchmark (splits, tasks, loaders) is Paper 2.

---

## 7. Data and code availability `[TODO]`

- ML/analysis repo: `github.com/lucafrb/rosettaligand-vs-posebuster`
- Docking pipeline: `github.com/Hackefleisch/pdbbind_dock`
- Failed-ID lists are tracked in-repo for re-checking/re-running.
- Overleaf draft: `overleaf.com/project/68c2ac9790bd64a27261ba21`

---

## Appendix A — Dataset accounting `[FROM REPO]`

Final docked-set sizes after the failure cascade (Confluence "Docking Fails"):

| Dataset | After filters / start | Final docked |
|---|---|---|
| PDBBind | 11,518 | **11,444** |
| Astex | 85 | **84** |
| PoseBuster | 428 | **388** |

PDBBind docking-fail reasons: valence error after reprotonation (21), Rosetta internal (26), atom
naming (11), PDB-header-line differences (15), corrupted h5 (1). PoseBuster: Rosetta internal (24),
header differences (3), unreadable SDF (2), unusual energy term (5), unfinished (6). `[VERIFY]`

---

## Appendix B — Open research questions to answer from the data (Confluence)

1. Dock into crystal or relax? Total Score or Interface Delta? `[TODO]`
2. How good is RL at recovering a pocket from an **unbound** complex? `[TODO]` — **the current
   data cannot answer this.** Relax-Ligaway is a holo receptor relaxed without the ligand, not an
   unbound structure. Answering it needs **Appendix C / B (true apo receptors)** `[COMPUTE]`;
   otherwise reformulate the question as "recovering a pocket that was relaxed without its
   ligand" and say so plainly.
3. How closely does RL stick to a near-native input (does it stay near crystal)? — informs whether
   to optimise **sampling** or **scoring**. `[TODO]`
4. How often does RL find a ≤ 2 Å pose, and how many samples does it need? `[TODO]` — answerable
   from existing data by sub-sampling the 300 poses per protocol (success-vs-N curve); no new
   compute needed.
5. How well does the score-fn **rank** complexes? `[TODO → §4.2/4.5]`
6. How well does the score-fn **correlate with affinity**? `[TODO → §4.7]`

---

## Addendum D — New analyses from the 2026-10-09 story update

Written as a spec to be folded into §3/§4 later. Nothing here has been implemented or tested.
All three run on **existing data**; none needs new docking.

### D.1 Energy-degeneracy analysis — "wrong poses with right energies" (T2)

**Claim to establish:** a large fraction of clearly incorrect poses have energy profiles nearly
identical to a correct one, so *no* reweighting of these terms can separate them. This is the
quantitative version of the UMAP picture.

**Procedure, per complex:**

1. Split the poses into **correct** (RMSD ≤ 2 Å) and **incorrect** (> 2 Å). Complexes with no
   correct pose are excluded from this analysis (report how many).
2. Take the per-pose **energy vector**. Default: the `raw_delta_*` interface terms restricted to
   the non-zero set (`fa_atr`, `fa_elec`, `fa_pair`, `fa_rep`, `fa_sol`, `hbond_bb_sc`,
   `hbond_sc`). Report the `raw_*` total-energy version as a robustness check.
3. **Standardise within the complex** (z-score each term over that complex's poses). Energies are
   not comparable across complexes, and the terms differ in scale by orders of magnitude; without
   this, the distance is dominated by `fa_atr`.
4. For every incorrect pose *i*, compute the distance to its **nearest correct pose**:
   $d_i = \min_{j \in \text{correct}} \lVert \mathbf{e}_i - \mathbf{e}_j \rVert$.
   Euclidean on standardised terms as the default; **Mahalanobis** using the within-complex
   covariance as the principled variant (the terms are strongly correlated) `[TODO]` decide which
   goes in the main text.
5. Keep per incorrect pose: $d_i$, its RMSD, its Δ(total score) and Δ(interface delta) to that
   nearest correct pose, and the complex id.

**Reference scales that must be plotted alongside, or the number means nothing:**

- **correct–correct** nearest-neighbour distances (the within-basin spread). If incorrect poses sit
  at the same distance from a correct pose as correct poses do from each other, the energies are
  genuinely non-discriminative. *This comparison is the result.*
- **incorrect–incorrect** nearest-neighbour distances (the decoy cloud's own density).
- a **cross-complex permutation** control: distances to correct poses of a *different* complex,
  which sets the "unrelated energy profile" scale.

**Figures.**

- **Density plot** of $d_i$ for the three populations above, pooled over complexes (the plot the
  story asks for).
- **Scatter:** $d_i$ (x) vs. RMSD (y), coloured by Δ(interface delta), hexbin because of the point
  count. The expected signal is a dense band at low $d_i$ spanning the **whole** RMSD range — the
  "arbitrarily wrong but energetically identical" population.
- Optional per-complex summary: fraction of incorrect poses whose $d_i$ falls below the median
  correct–correct distance. One number per complex, histogram over complexes, and a table by
  protocol.

**Headline statistics to report:**

- % of incorrect poses that are **energy-degenerate** with a correct one, defined as
  $d_i$ below the complex's median correct–correct distance **and** RMSD > 5 Å (not a near-miss);
- % of complexes where the **best-scoring** pose is such a degenerate decoy — this connects the
  analysis directly to the Top-1 failures in §4.2;
- the same split by protocol and by receptor state.

**Pitfalls to avoid:**

- The distance and the score difference are correlated by construction. State the claim as
  *similar energy vector* **and** *similar or better total score* **and** *large RMSD*; the
  Δ-score column makes that explicit.
- Standardisation must use that complex's own poses only; any global scaling reintroduces
  cross-complex incomparability.
- Report whether the degenerate decoys concentrate in particular protocols (they may be an
  artefact of perturbed starts) or in particular target classes.

### D.2 Per-complex oracle probe — an upper bound on what reweighting can achieve (T2)

A short, strong companion to D.1. For each complex **separately**, fit a linear classifier on the
energy terms to separate correct from incorrect poses, with cross-validation **within** that
complex, and report AUC.

**Why it matters:** this classifier is allowed to choose the best possible weights *for that one
complex* — strictly more freedom than any global reweighting, including Rosetta's own weights and
Luca's learned ones. Its AUC is therefore an **upper bound** on the per-complex performance any
reweighting scheme could reach. If the median oracle AUC is low, the §4.5 result ("reweighting
only helps a little") stops being a statement about our models and becomes a statement about the
term set. That is the sentence the discussion needs.

Report: distribution of oracle AUC over complexes; its correlation with the D.1 degeneracy
fraction; and the gap between oracle AUC and the AUC actually achieved by the global reweighting
and the LambdaRank head.

### D.3 Affinity track — from best-pose reweighting to a sampled-density free energy (T3)

**Step 1 (have prelim data).** Fit a reweighting of the terms on the lowest-energy pose per
complex against the reported affinity. Expect a modest gain over the default score
`[FROM REPO]` refresh the number. Report it as the baseline, with the leakage-aware split.

**Step 2 (the idea).** T1 and T2 together say: the sampler produces physically plausible poses and
concentrates them where its own energy is low, but the energies of individual poses do not
identify the right one. So use the **distribution** of samples rather than the energy of the best
one:

1. Represent each pose in a low-dimensional pose space (ligand rigid-body coordinates in the
   receptor frame + torsions, or a reduced coordinate from the clustering) `[TODO]` choose.
2. **Fit a smooth density / energy surface** to the sampled poses (kernel density, a mixture
   model, or a GP on the energies). The point of the fit is to distinguish **basins** (narrow,
   deep, heavily sampled) from **flat regions** (broad, shallow, sparsely sampled but entropically
   large) — the information a single minimum throws away.
3. Integrate $e^{-\beta E}$ over the fitted surface to get an effective $Z$, and map back to a
   free energy. Correct for the sampler's own bias toward low energy, otherwise the enthalpy is
   counted twice (this is the failure mode of the earlier naive attempt; see
   [PAPER2_CONCEPT.md](PAPER2_CONCEPT.md) §2.2).
4. **Bayesian uncertainty:** place a posterior on the fitted surface (GP posterior, or a Bayesian
   bootstrap over poses) and propagate it to a posterior over ΔG per complex. Report predicted
   affinity *with error bars*, and check that the error bars are calibrated — complexes with few
   distinct basins should come out visibly less certain. This is a genuine selling point: almost
   no docking score reports a calibrated uncertainty.

**Paper 1 / Paper 2 boundary — open decision.** This step 2 is the descriptive, non-learned
sibling of Paper 2's estimator ladder (S3–S5). Suggested line, to be agreed with co-authors:

- **Paper 1** keeps step 1, plus *one* fitted-density estimator with Bayesian error bars,
  evaluated descriptively on the existing data. It ends on "the ensemble carries information the
  single best pose does not".
- **Paper 2** keeps the trained version: learned proposal density, the bound/separated
  thermodynamic cycle, the Boltzmann-consistent objective, the estimator comparison and the
  calibration tests.
- If step 2 turns out to work very well, revisit the split — it may belong entirely to Paper 2.

---

## Appendix C — Additional compute this paper needs `[COMPUTE]`

Everything else in this outline runs on the **existing** 11,444 × 1,800 pose dataset. These are
the only items that require new cluster runs. A/B are shared with Paper 2 (= C3/C5 in
[PAPER2_CONCEPT.md](PAPER2_CONCEPT.md) §4), so run them once and use them in both.

| ID | Run | Needed for | Priority | Notes |
|---|---|---|---|---|
| **A** | **Pocket-box docking from random starts** — random ligand position/orientation inside a pocket box, random conformer, no crystal-pose seeding; ~300 poses per complex | §4.1/§4.2 fair comparison against blind DL methods; the only version of "docking power" that is not re-docking | **High** | Start on Astex + PoseBusters (472 complexes) and a PDBbind subset; extend if it works. Also = Paper 2 / C3 |
| **B** | **True apo receptors** for targets with an apo PDB entry | Appendix B question 2 (pocket recovery from an unbound structure) | Medium | Requires an apo↔holo matching step `[TODO]`; without it, that question must be reworded |
| **C** | **Re-run the PoseBusters validity suite** on the current poses | §4.4 corrected PB-valid ranking after the RDKit/kekulé fix | **High** | Analysis-level compute only, no docking |
| **D** | Optional: **more samples on a subset** (e.g. 1,000 poses for a few hundred complexes) | §4.1 saturation check — does the success-vs-N curve flatten before 300? | Low | The 300-pose sub-sampling curve may already answer it |

Not needed for Paper 1: unconstrained relaxation and thermal basin sampling (Paper 2 / C1, C2).

---

## Open decisions before drafting prose

- [ ] Confirm final dataset sizes and sample counts from `pdbbind_dock`.
- [ ] Re-run PoseBusters validity post RDKit/kekulé fix and report corrected ranking.
- [ ] Lock the final ML/reweight numbers (tuned) across all three datasets.
- [ ] Produce the tSNE/UMAP landscape figure and write its interpretation (§4.6).
- [ ] Decide and run the final ensemble/affinity estimator; refresh Felipe's r-values (§4.7).
- [ ] Replace random splits with a leakage-aware split, or justify keeping them.
- [ ] Settle title, author order, and contribution statement.
- [ ] Agree the Paper 1 / Paper 2 boundary with co-authors: Josephine's starting-position results
      are presented in full here; Paper 2 only cites them (and uses the dataset-wide Standard vs.
      Perturb comparison) as evidence that the sampler is non-ergodic.
- [ ] Decide where the raw data is deposited (e.g. Zenodo) so Paper 2 can build on the same DOI.
- [ ] Decide on Appendix C runs A and B (shared with Paper 2) — budget, subset, and who launches
      them. Run A is the one that most changes how §4.1/§4.2 can be phrased.
- [ ] Re-order §3–§5 into the three tracks of the 2026-10-09 story update (T1 benchmark +
      sampling mode, T2 ranking + why it resists, T3 affinity).
- [ ] Settle whether Addendum D.3 step 2 (sampled-density free energy) is Paper 1 or Paper 2.
- [ ] Decide how much of the lab's own RosettaLigand work to cite up front (idea-3 brief: a lot).
