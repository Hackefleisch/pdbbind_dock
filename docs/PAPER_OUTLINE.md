# RosettaLigand: Preliminary Paper Outline

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
look at affinity (Kd) correlation and lay out an ensemble / entropy–enthalpy framework (Felipe's
analysis) as the route forward. A reusable by-product is **a dataset of Rosetta energy scores plus
energetically minimised complexes spanning the full RMSD range.**

The arc, in one line: *characterise → expose the sampling/scoring gap → try to fix scoring and
explain why it resists → connect to affinity → outlook on ensemble thermodynamics.*

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
  5. Affinity (Kd) correlation and an ensemble / entropy–enthalpy outlook (Felipe).
  6. A released dataset: scores + minimised complexes across all RMSD ranges.

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
  Relax pre-optimises protein+ligand (mean prep ≈ 4.7 h `[VERIFY]`); Ligaway starts the ligand
  away from the pocket then re-seats it. *(Luca §4.1; Fig. "docking protocols".)*
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
- **3.9 Affinity and ensemble scoring.**
  - Correlation of score vs. experimental Kd/affinity: minimum Interface Delta vs. mean of N
    lowest vs. reweighted score. *(Felipe: r ≈ 0.392 min, 0.376 mean-of-10, 0.418 reweight —
    `[FROM REPO]` refresh.)*
  - **Ensemble / cluster scoring** as a canonical-ensemble approximation: cluster poses;
    population-based weights (compare `n`, `ln(1+n)`, and a spread-penalised variant); per-cluster
    enthalpic centre (mean/best score) and entropic width (intra-cluster RMSD spread). Note that
    several Rosetta terms carry **local** entropic signal (`fa_sol` = desolvation enthalpy **and**
    entropic water release), so naive averaging double-handles entropy — motivates the cluster
    treatment. *(Felipe notes; cluster-weighting analysis; aggregation × term-selection design.)*
    `[TODO]` settle the final ensemble estimator.

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
- **4.2 But scoring/ranking lags — the central gap.**
  - Top-1 vs Top-5 vs all-150: Top-1 is substantially worse, exposing the sampling–scoring
    discrepancy. *(Luca §4.1.1, §4.2.1, Fig. 4.5; §4.3.)* `[FROM REPO]` current Top-k table.
  - Frame explicitly: *the correct pose is in the bag; the score won't reliably pull it out.*
- **4.3 Starting position governs what is found (Josephine).**
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
  - tSNE/UMAP of energy-term vectors → `[TODO]` describe what the embedding shows. Working
    hypothesis (idea 3.0): **too many near-degenerate local minima** with near-identical energy
    signatures, so no reweighting of these terms can cleanly separate native from decoy.
  - Optional heat-map deliverable (Confluence ToDo): aggregation (y) × term-selection (x), doubled
    on x for trained vs. standard weights; two maps (MAE and Pearson r) — to test whether more
    aggregation helps as more entropic terms are dropped. `[TODO]`
- **4.7 Affinity correlation and ensemble scoring.**
  - Score vs. Kd: min vs. mean-of-N vs. reweighted (Felipe's r-values above). `[FROM REPO]`
  - Ensemble/cluster estimator results `[TODO]`. Felipe's testable predictions to address:
    (a) ML-QM-rescored *ensemble* > ML-QM-rescored *minimum* ⇒ ensembles carry entropic info;
    (b) bigger gains for **open active sites** than deep pockets ⇒ conformational-entropy
    hypothesis; (c) decompose into ΔH (better QM) and TΔS (ensemble stats), fit against ITC.
  - Honest negative result to include: using the raw score distribution to estimate binding free
    energy **worsens** affinity prediction — docking samples are biased toward the score's own
    energy landscape. Frame as motivation for the geometry-based ensemble weighting, which adds
    information the score cannot see. *(Prior cluster-scoring analysis.)*

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
- **6.2 Outlook.** Ensemble thermodynamics: treat clusters as macrostates, approximate Boltzmann
  weights from populations, read local entropy from intra-cluster spread; sample
  entropy–enthalpy explicitly and validate ΔH / TΔS against ITC. *(Felipe.)*
- **6.3 Released dataset.** Rosetta energy scores + energetically minimised complexes across all
  RMSD ranges — reusable for training interaction kernels and diffusion models (likely binned by
  RMSD). *(Confluence "Follow up Projects".)*

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
2. How good is RL at recovering a pocket from an **unbound** complex? `[TODO]`
3. How closely does RL stick to a near-native input (does it stay near crystal)? — informs whether
   to optimise **sampling** or **scoring**. `[TODO]`
4. How often does RL find a ≤ 2 Å pose, and how many samples does it need? `[TODO]`
5. How well does the score-fn **rank** complexes? `[TODO → §4.2/4.5]`
6. How well does the score-fn **correlate with affinity**? `[TODO → §4.7]`

---

## Open decisions before drafting prose

- [ ] Confirm final dataset sizes and sample counts from `pdbbind_dock`.
- [ ] Re-run PoseBusters validity post RDKit/kekulé fix and report corrected ranking.
- [ ] Lock the final ML/reweight numbers (tuned) across all three datasets.
- [ ] Produce the tSNE/UMAP landscape figure and write its interpretation (§4.6).
- [ ] Decide and run the final ensemble/affinity estimator; refresh Felipe's r-values (§4.7).
- [ ] Replace random splits with a leakage-aware split, or justify keeping them.
- [ ] Settle title, author order, and contribution statement.
- [ ] Decide how much of the lab's own RosettaLigand work to cite up front (idea-3 brief: a lot).
