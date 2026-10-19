# docs

Methods-level documentation and the planning documents of the paper. Keep the
methods pages in sync with the code; each page states the authoritative source
file(s) it documents.

## Methods

- [dataset_preparation.md](dataset_preparation.md) — how `pdbbind_cleaned` is
  filtered and built from the raw PDBbind v2020 download
  (source: [`scripts/filter_pdbbind.py`](../scripts/filter_pdbbind.py)).

## Paper 1 (characterisation of RosettaLigand)

- [abstract.tex](abstract.tex) — the current abstract (work in progress; `abstract.pdf`
  is its rendered copy).
- [PAPER_OUTLINE.md](PAPER_OUTLINE.md) — section scaffold, the three-track story, the
  specification of the new analyses (Addendum D) and the extra compute (Appendix C).
- [PAPER1_STORY_INVENTORY.md](PAPER1_STORY_INVENTORY.md) — per story: what the data
  supports today (with the full-set success tables), the caveats, and what is still
  missing. The most recent of the planning documents; start here.

## Paper 2 (deferred)

- [PAPER2_CONCEPT.md](PAPER2_CONCEPT.md) — concept of the follow-up paper on ensemble
  thermodynamics. Nothing in the code implements it yet.

## Pages still to write

- `docking_protocol.md` — the relax + dock procedure, the RosettaScripts protocols,
  score function, and per-complex sampling settings
  (source: `docking/runner.py`, `docking/xml_protocols/`; the verified facts are
  collected in PAPER1_STORY_INVENTORY.md §0.2).
- `benchmarks.md` — the PoseBusters / Astex Diverse evaluation sets and how they
  differ from the PDBbind set; the success-rate metrics of `docking_analysis/benchmark.py`.
- `reweighting.md` — the energy-term reweighting model, aggregation strategies,
  clustering, and evaluation (source: `docking_analysis/`), once the affinity grid is
  ported to the full extraction.

See [../CLAUDE.md](../CLAUDE.md) for the code-level overview and
[../README.md](../README.md) for the project summary.
