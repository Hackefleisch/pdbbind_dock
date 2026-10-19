# Paper 2 — Concept: Enthalpy from the Scorer, Entropy from the Sampler

> **Status:** concept, not yet an outline. Written 2026-09-22 after deciding the two-paper split.
> Paper 1 (characterisation of RosettaLigand, JCIM/JCTC) is planned in
> [PAPER1_STORY_INVENTORY.md](PAPER1_STORY_INVENTORY.md). This paper targets an ML venue.
>
> **Headline is decided by results, not now** (see §9). If the method works convincingly, the
> method is the headline. If it does not, the dataset/benchmark is the headline and the method
> becomes its flagship baseline plus an honest analysis.
>
> Markers: `[VERIFY]` claim to confirm against data or literature; `[TODO]` not yet done;
> `[COMPUTE]` needs new docking/relax runs.

---

## 1. One-paragraph pitch

Physics-based docking engines and deep generative docking models both produce **ensembles** of
poses, but they are scored one pose at a time. Scoring functions capture enthalpy reasonably well
and entropy poorly. We argue that the missing entropy is already in the ensemble: it is in how
the sampler distributes its samples and how wide each basin is. We propose a framework in which a
**classical scoring function provides the enthalpy** and the **sampler's (estimated) density
provides the entropy**, combined through an importance-weighted partition function. The
framework is trained with a Boltzmann-consistent objective that learns pose discrimination and
binding free energy together. We test it on a new dataset of about 11,400 PDBbind complexes × 1,800
RosettaLigand poses with decomposed energy terms, three receptor states and paired bound/separated
relaxations. The dataset is noisy and biased (it comes from docking), but **consistently** so,
which makes it a controlled testbed for ensemble scoring.

---

## 2. The central idea, stated carefully

### 2.1 Softmax over poses = Boltzmann distribution

A per-pose energy $E_\theta(x)$, softmaxed over all poses of a complex and trained with
cross-entropy against the native pose, is maximum-likelihood fitting of a Boltzmann distribution
$p(x) \propto e^{-\beta E_\theta(x)}$. Log-sum-exp pooling over the ensemble is then a free energy
$F = -\beta^{-1}\ln Z$.

### 2.2 Why raw pose counts are neither pure bias nor pure entropy

The pose set is drawn from the sampler's proposal density $q(x)$, not from the Boltzmann
distribution. The exact identity is

$$Z = \int e^{-\beta E(x)}\,dx = \mathbb{E}_{x\sim q}\!\left[\frac{e^{-\beta E(x)}}{q(x)}\right].$$

Consequences, which drive the whole design:

- If $q$ is known, **raw counts cancel**. Each basin contributes roughly
  (basin volume) × $e^{-\beta E}$. The entropy is in **basin width/volume**, not in population.
- A naive LSE over the raw samples implicitly assumes $q$ is uniform. That is why our earlier
  attempt to get free energies from the raw score distribution **worsened** affinity prediction
  (Paper 1 era, cluster-scoring analysis): the samples are concentrated where Rosetta's own
  energy is low, so enthalpy is counted twice.
- RosettaLigand does minimise-from-random-start (plus MC), so $q$ is roughly a basin's
  **catchment volume** under the starting distribution. That is related to entropy but is not a
  Boltzmann population, and it depends on the start (Josephine's result, §6.3).
- A **learned** generator trained on crystal structures (DiffDock-type, flow-matching) has
  $\log q(x) \approx -\beta F_{\text{learned}}(x) + \text{const}$. For such a generator the density
  *is* a free-energy model and can be combined with physics as a product of experts.

**Resulting claim to test:** *estimate the sampler's density $q$ (or basin widths) with a learned
density model, and use it to turn a biased docking ensemble into a free-energy estimate.
Enthalpy comes from the classical score; entropy comes from the ensemble geometry.*

### 2.3 The additive-constant problem and the thermodynamic cycle

A softmax is invariant to a constant shift per complex, so pose supervision alone never fixes an
absolute free energy and can never yield a $K_d$. Two anchors break the degeneracy:

1. experimental affinity (supervision);
2. the **separated-state relaxation** (ligand 500 Å away), computed with the same atoms, terms and
   protocol. This closes the cycle:

$$\hat{\Delta G} = F_{\text{bound}} - F_{\text{separated}} + c(\text{ligand}).$$

Use **means, not sums**, inside $Z$ for both states. Otherwise the sample-count ratio (1800 vs. 10)
adds about $k_BT\ln 180 \approx 3$ kcal/mol of pure artefact. $c(\cdot)$ absorbs the standard
state, solution-phase ligand conformational entropy and solvation errors; it depends only on
ligand descriptors (rotatable bonds, heavy atoms).

### 2.4 Temperature: a gauge, and a test

Rosetta energy units are not kcal/mol. With a learned linear head only $\beta\mathbf{w}$ is
identifiable, so fixing $\beta = 1/0.593$ just sets the units of $\mathbf{w}$. It is **not** a
physical constraint in itself.

What *is* physical: the **same** $\mathbf{w}$ must explain pose discrimination (a relative,
within-complex quantity) and affinity (an absolute, across-complex quantity). Fit an effective
temperature from each separately:

- $\beta_{\text{pose}}$: the scale at which the pose softmax is best calibrated;
- $\beta_{\text{aff}}$: the scale at which $\hat{\Delta G}$ matches experiment.

**Temperature-consistency test.** If $\beta_{\text{pose}} \approx \beta_{\text{aff}}$, the score is
behaving like a free energy. The gap between them measures how far the score is from one. Either
outcome is a result.

---

## 3. What we have (verified in `docking/runner.py` and the H5 files)

| Item | Per complex | Notes |
|---|---|---|
| Docked poses | 1,800 = 3 receptor states × 2 protocols × 300 | `results` matrix, 57 columns |
| Receptor states | crystal / relax / apo_relax | one representative each (lowest total score) is docked into |
| Bound relax | 10 FastRelax seeds | backbone coordinate constraints kept on (no ramp-down) |
| Separated relax (`apo_relax`) | 10 FastRelax seeds, ligand 500 Å away | same constraints; ligand moved back in afterwards |
| Energy terms | `raw_<term>` (whole system) and `raw_delta_<term>` (interface) | total − interface = intramolecular part → strain |
| Structures | full PDB for relax poses; per-pose diffs for docked poses | ligand + sidechain microstates recoverable |
| Labels | RMSD to crystal / to input; PDBbind Kd/Ki/IC50 | |

**Facts that constrain the design:**

- `apo_relax` is a **holo-derived receptor relaxed without the ligand**, not an apo structure.
  Never call it apo in the paper.
- **All three** receptor states are docked with the Standard protocol starting at the crystal
  ligand position, so all 900 Standard poses are near-native by construction. Perturb is still
  pocket-centred. Nothing here is blind docking. The Standard poses must be down-weighted or
  excluded from $Z$ (see §5).
- Backbone constraints during relaxation mean the 10 seeds are probably tightly clustered, and the
  separated state cannot reorganise much. So unbound entropy and reorganisation cost are
  underestimated. This motivates the extra compute in §4.
- The `raw_coordinate_constraint` column must be excluded from any energy used in $Z$ or the
  cycle `[VERIFY]` whether it is non-zero in the stored relax scores.

---

## 4. Additional compute (what "more compute" buys, in priority order)

Each item states what it is for. None of them is needed to start (§6.1 runs on existing data).
**C3 and C5 are shared with Paper 1** (= runs A and B in
[PAPER1_STORY_INVENTORY.md](PAPER1_STORY_INVENTORY.md) Appendix B), so they only need to be
launched once and serve both papers. C1, C2 and C4 are
Paper 2 only.

**C1. Unconstrained relaxations, more seeds. `[COMPUTE]` — high priority, cheap.**
Re-run bound and separated relaxations **without** backbone coordinate constraints (or with them
ramped down), about 50 seeds per state instead of 10. *Why:* the separated state enters every
affinity prediction and is the least-sampled part of the cycle. With constraints on, it cannot
express reorganisation, so $F_{\text{separated}}$ is closer to "min of 10 near-identical
structures" than to an ensemble average. Keep the constrained runs as an ablation.

**C2. Local thermal sampling around docked minima. `[COMPUTE]` — high priority for the entropy claim.**
For each complex, pick representatives of the top-K pose clusters (e.g. K ≈ 20) and run short
Metropolis MC at $k_BT$ in the Rosetta score, with ligand rigid-body + torsion moves and interface
sidechain moves but **no final minimisation**. *Why:* docking outputs are quenched minima, which
tell us where basins are but not how wide they are. Thermal sampling inside each basin gives its
width directly (fluctuation or quasi-harmonic entropy). This is the cleanest way to measure
"entropy from the sampler" without guessing $q$.

**C3. Start-controlled docking for estimating $q$. `[COMPUTE]` — medium priority.**
A docking run whose starting distribution is known and **not** anchored to the crystal pose:
random ligand placement and orientation in a defined pocket box, random conformer. *Why:* with a
known start distribution, the output density is a well-defined function of the sampler, and it can
be learned and importance-corrected. It is also the realistic deployment setting and gives a clean
headline subset with no near-native leak. Several hundred poses per complex, possibly on a subset
first.

**C4. Learned-generator ensembles. `[COMPUTE]` — optional, needed for the product-of-experts claim.**
Sample poses with a DL generator that has a tractable or estimable likelihood (flow-matching or
diffusion model with ODE log-likelihood; DiffDock-L as the widely known reference), then minimise
and score them with Rosetta. *Why:* this tests the "generator density is a learned free energy"
half of the framework and makes the paper relevant beyond Rosetta. Watch for data leakage: most DL
generators were trained on PDBbind, so evaluate on PoseBusters and time-split complexes only.

**C5. True apo receptors. `[COMPUTE]` — optional.**
For the subset of targets with an apo crystal structure in the PDB, dock into it. *Why:* makes the
realistic-setting claim honest. Not required for the core method.

---

## 5. Method

### 5.1 Energy head (enthalpy)

$$E_\theta(x) = \mathbf{w}_{\text{int}}^\top \mathbf{t}^{\text{int}}(x) + \mathbf{w}_{\text{strain}}^\top \big(\mathbf{t}^{\text{tot}}(x) - \mathbf{t}^{\text{int}}(x)\big)$$

- **Linear in raw terms.** Energy is extensive; a nonlinear head breaks the bound − separated
  difference silently (loss still falls, free energies become meaningless).
- **Per-pose only.** No set-derived features (rank, z-score, energy minus set minimum) anywhere
  near $Z$.
- Sign constraints on physically repulsive terms. Regularise toward the default Rosetta weights
  ([docking_analysis/scfx_weights.json](../docking_analysis/scfx_weights.json)) so the learned
  deviations stay interpretable.
- Use total energy (not interface only) inside $Z$, so receptor-state strain is weighted by the
  softmax itself rather than by hand-set priors over crystal/relax/apo_relax. Restrict internal
  terms to pocket-proximal residues if they can be recomputed `[TODO]`, since far-field terms are
  pure noise.

### 5.2 Entropy estimators (the object of study)

All are plugged into the same importance-weighted $Z$. Ranked from simplest to the proposed one:

| ID | Estimator | Entropy source | Compute |
|---|---|---|---|
| S0 | min energy | none (current practice) | existing data |
| S1 | LSE over raw poses | none; implicitly assumes uniform $q$ (known to hurt) | existing data |
| S2 | LSE over clusters, cluster weight = population | catchment volume (Felipe's `n`, `ln(1+n)`) | existing data |
| S3 | LSE over clusters, weight = basin width from intra-cluster spread | geometric width of the docked cluster | existing data |
| S4 | LSE over clusters, width from thermal sampling | quasi-harmonic / fluctuation entropy | **needs C2** |
| S5 | importance weights $1/\hat q(x)$ from a learned density model | the sampler's estimated density | Perturb runs now; **C3** for the clean version |
| S6 | S5 + generator log-likelihood as a second expert | learned free energy of a DL generator | **needs C4** |

S0–S3 are reachable on the existing data; S4–S6 are what the extra compute in §4 buys.

For S5: a density model over the pose representation (ligand rigid body in the receptor frame
+ ligand torsions + interface χ angles), either a normalising flow or a kNN
(Kozachenko–Leonenko) estimator on symmetry-corrected RMSD as the simple baseline. Fit it on
samples from a **known start distribution** (Perturb runs now, C3 later).

Clustering is joint on ligand RMSD (~1 Å) **and** interface χ, so distinct sidechain states under
one ligand placement count as separate microstates (the legacy clustering in
`docking_analysis/clustering.py` clusters on plain ligand RMSD only).

### 5.3 Losses

- **Pose loss.** KL between a soft target $q^*_i \propto \exp(-\text{RMSD}_i^2 / 2\sigma^2)$,
  $\sigma \approx 1.5$ Å, and the model's weighted Boltzmann distribution over poses. Soft target
  rather than a hard 2 Å cutoff; the marginal over the native basin rather than
  per-positive terms, so redundant near-native poses do not compete with each other.
- **Null class.** A learned background energy $E_\varnothing(\text{ligand descriptors})$ appended
  to every set; complexes with no pose under 2 Å get target $p_\varnothing = 1$. At inference,
  $p_\varnothing$ is a "should I trust this docking run" score.
- **Affinity loss.** Huber($\hat{\Delta G}$, $RT\ln K_d$) with the cycle of §2.3, weighted per
  complex by the bootstrap uncertainty of $F_{\text{separated}}$.
- **Schedule.** Pose loss alone first; anneal the affinity loss in once the native basin is
  found. Otherwise the affinity term pins a decoy's energy to the experimental ΔG early on.
- **Standard-protocol poses** (near-native by construction): kept in the pose loss, excluded from
  or down-weighted in the free-energy $Z$. Density/weights estimated from Perturb (or C3) runs only.

---

## 6. Experiments

### 6.1 Pre-flight diagnostics (existing data, no training)

1. **Zero-cost cycle baseline:** default weights,
   LSE(10 bound relax) − LSE(10 separated relax) vs. log Kd. This is the number everything must
   beat. If r ≈ 0, stop and debug.
2. **Separation cleanliness:** interface energy in separated states exactly 0; identical atom
   composition between bound and separated.
3. **Seed diversity and bootstrap:** energy and RMSD spread across the 10 seeds; bootstrap
   error on $F_{\text{separated}}$. This sets the accuracy floor and decides how urgent C1 is.
4. **Redundant receptor states:** backbone RMSD relax vs. apo_relax representative; flag targets
   where the two collapse to one state.
5. **RMSD conventions:** symmetry-corrected ligand RMSD, superposed on the receptor frame of that
   protocol `[VERIFY]` against what `rmsd_to_crystal` currently does.
6. **Strain vs. RMSD:** near-native poses should show lower strain. If not, the receptor
   preparation is the problem, not the scoring.

### 6.2 Main experiments

- **E1 — Docking power** of the learned linear head (S0 pooling) vs. default Rosetta and Paper 1's
  ML heads (RF, GBR, MLP, LambdaRank). Establishes that the Boltzmann objective is at least as
  good as the best reranker while staying linear and interpretable.
- **E2 — Entropy estimators S0–S6 on affinity** (scoring and ranking power). **Core result:**
  does entropy estimated from the ensemble add signal beyond the enthalpy? Report partial
  correlation of the entropic term with log Kd given the enthalpic term.
  *Compute:* S0–S3 on existing data; S4 needs **C2**, the clean S5 needs **C3**, S6 needs **C4**.
- **E3 — Temperature consistency** ($\beta_{\text{pose}}$ vs. $\beta_{\text{aff}}$), §2.4.
- **E4 — Calibration:** reliability diagrams for $\max_i p_i$ vs. top-1 correctness and
  $p_\varnothing$ vs. docking failure, **without** post-hoc recalibration.
- **E5 — Felipe's predictions:** (a) ensemble > minimum; (b) larger ensemble gain for open or
  shallow pockets than for deep ones (needs a pocket-openness descriptor `[TODO]`);
  (c) where ITC data exist, the ΔH / TΔS split against experiment `[VERIFY]` coverage.
- **E6 — Ablations:** interface-only vs. total energy; with/without separated reference; raw
  counts vs. estimated $q$; constrained vs. unconstrained relax (**needs C1**); Standard poses
  in/out of $Z$; free vs. fixed β (diagnostic only).
- **E7 — Learned weights:** compare $\mathbf{w}$ with the default weights. Large shifts on one term
  usually mean that term is standing in for missing physics (desolvation, entropy). Link to
  Paper 1's landscape analysis.

### 6.3 Where Josephine's work enters

Her starting-position results (success collapses with starting RMSD; perturbation redistributes
rather than rescues success) are direct evidence that **the sampler is non-ergodic and $q$ depends
on initialisation**. That is exactly why raw counts are not Boltzmann populations and why $q$ must
be estimated. In Paper 2 this is a short motivating section: *"Is the docking engine an
equilibrium sampler? No."* It cites Paper 1 for her detailed experiments and adds the dataset-wide
version (Standard vs. Perturb populations across all complexes). She is a co-author.

---

## 7. Evaluation protocol

- **Splits:** protein-family clustering at 30–50 % sequence identity (or an established
  leak-proof PDBbind split such as LP-PDBBind `[VERIFY]`). A random split is reported only as a
  secondary number.
- **External tests:** CASF-2016 core set (`hacsplit == "test"` in `docking_analysis/splits.csv`,
  with its leakage caveat);
  PoseBusters (disjoint from PDBbind by construction); Astex.
- **Labels:** headline on Kd/Ki only; IC50 separately or with assay type fed into $c(\cdot)$.
- **Metrics (CASF decomposition):** docking power (top-1/top-5 at 2 Å), scoring power (Pearson r),
  ranking power (Spearman within target), plus calibration (E4).
- **Headline subset:** Perturb poses on relaxed receptors now; the C3 start-controlled runs once
  they exist (**needs C3** for the fully leak-free version). Crystal + Standard is a positive
  control only.

---

## 8. The dataset contribution (always part of the paper)

Independent of which way the headline goes, the paper releases:

- ~11,400 PDBbind complexes + Astex + PoseBusters, 1,800 poses each (~20 M poses), full-RMSD-range
  coverage including decoys, with per-term raw and interface energies and recoverable structures;
- paired bound / separated relaxation ensembles (and C1–C3 where run);
- fixed splits, task definitions (docking power, affinity, ensemble free energy, calibration),
  loaders, and baselines (default Rosetta, Paper 1 heads, S0–S6).

Uses beyond this paper worth naming: training interaction kernels or energy-based rescorers on
physically minimised decoys; RMSD-binned data for diffusion/flow models; studying sampler bias.

---

## 9. Headline decision rule (fix before looking at test results)

**Method-headline** if, on the leak-proof split and the external tests:

1. the best entropy estimator (S3–S6) beats S0 **and** the learned-weights minimum in scoring
   power, by more than the bootstrap confidence interval;
2. the entropic term carries significant partial correlation with log Kd given enthalpy;
3. calibration (E4) is near-diagonal without recalibration, or temperature consistency (E3)
   holds within error.

Otherwise it is a **dataset-headline** paper: benchmark + tasks + the Boltzmann framework as the
flagship baseline, with the negative/partial results written up as findings about sampler bias.

**Venue:** method-headline → ICML (deadline ~late January `[VERIFY]`); dataset-headline →
NeurIPS Datasets & Benchmarks track (deadline ~May `[VERIFY]`). ICLR 2027 is too close to be
realistic.

---

## 10. Order of work

1. Pre-flight diagnostics §6.1 on existing data. Decide go/no-go and how urgent C1 is.
2. Launch C1 (unconstrained relax, more seeds) on the cluster; cheap and needed either way.
3. Joint ligand-RMSD + χ clustering; estimators S0–S3; learned linear head with pose loss (E1).
4. Add null class, then anneal in the affinity loss (E2, E3, E4).
5. C2 thermal sampling on a subset → S4. C3 start-controlled docking → S5.
6. Apply the headline decision rule (§9). Only then decide on C4/C5.
7. Package the dataset (splits, loaders, task definitions) in parallel with 3–5.

---

## 11. Risks and honest limitations

| Risk | Consequence | Mitigation |
|---|---|---|
| Rosetta's score is mostly enthalpic with crude desolvation | $E_\theta$ is an effective PMF, not a first-principles energy | Say so; the framework is about the ensemble, not the force field |
| PDBbind affinity noise caps scoring power (~0.6 r for strong methods `[VERIFY]`) | Small absolute gains | Report CIs; emphasise calibration and consistency tests |
| Unbound state under-sampled | Noisy $F_{\text{separated}}$ dominates error | C1, bootstrap weighting |
| Near-native leak via Standard protocol | Inflated docking power and $Z_{\text{bound}}$ | Exclude from $Z$; headline on Perturb / C3 |
| Density estimation in pose space is hard (high-dim, symmetric) | S5 noisy | S3/S4 as fallbacks; kNN on RMSD as baseline |
| ML reviewers see "a linear model on 20 features" | Weak main-track reception | Lead with the sampler-density idea and the dataset; the linear head is a deliberate constraint |
| DL generators trained on PDBbind | Leakage in C4 | Evaluate C4 on PoseBusters / time split only |

---

## 12. Open decisions

- [ ] Paper 1 / Paper 2 boundary agreed with co-authors (Josephine's section, Felipe's ensemble
      material moves here).
- [ ] **Boundary moved on 2026-10-09:** Paper 1's story update gives it a third track (affinity),
      ending in a fitted-density free energy with Bayesian error bars — see
      [PAPER1_STORY_INVENTORY.md](PAPER1_STORY_INVENTORY.md) Appendix A.3. That is the
      descriptive sibling of S3–S5 here. Decide which paper owns it before either is drafted; if it works well, it probably
      belongs here and Paper 1 stops after the best-pose reweighting.
- [ ] Paper 1's energy-degeneracy analysis (inventory Appendix A.1) and oracle probe (A.2) are the cleanest
      motivation for this paper's premise ("the energies cannot rank; use the ensemble"). Cite
      them rather than repeating them.
- [ ] Which leak-proof split to adopt.
- [ ] Scope and budget for C1–C3 (subset first, or full PDBbind?).
- [ ] Whether to include a DL generator (C4) and which one.
- [ ] Author list and contributions (Paul, Felipe, Josephine, Luca, Rocco, Chris, Jens, …).
