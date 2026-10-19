# docs

Methods-level documentation for the project — the material that will feed into
the eventual research paper. Keep these in sync with the code; each page should
state the authoritative source file(s) it documents.

## Contents

- [dataset_preparation.md](dataset_preparation.md) — how `pdbbind_cleaned` is
  filtered and built from the raw PDBbind v2020 download
  (source: [`scripts/filter_pdbbind.py`](../scripts/filter_pdbbind.py)).
- [PAPER_OUTLINE.md](PAPER_OUTLINE.md) / [PAPER2_CONCEPT.md](PAPER2_CONCEPT.md) — the
  paper plans; [PAPER1_STORY_INVENTORY.md](PAPER1_STORY_INVENTORY.md) — what the data
  supports today and what is still missing, per story.

## Suggested future pages

- `docking_protocol.md` — the relax + dock procedure, the RosettaScripts
  protocols, score function, and per-complex sampling settings
  (source: `docking/runner.py`, `docking/xml_protocols/`).
- `reweighting.md` — the energy-term reweighting model, aggregation strategies,
  clustering, and evaluation (source: `docking_analysis/`).
- `benchmarks.md` — the PoseBusters / Astex Diverse evaluation sets and how they
  differ from the PDBbind training subset.

See [../CLAUDE.md](../CLAUDE.md) for the code-level overview and
[../README.md](../README.md) for the project summary.
