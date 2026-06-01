# Changelog

All notable changes to BEEP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — 0.13.0.dev

### Added

- **`nm_sampling` workflow** — per-functional force-RMSE benchmark on
  normal-mode-displaced geometries. For each binding site, BEEP
  computes a cheap Hessian (default `hf/def2-svp`, configurable),
  diagonalises it via Psi4/qcelemental, classifies every vibrational
  mode as **intermolecular / bending / stretching** by
  fragment-centre-of-mass projection, picks the lowest-frequency modes
  in each band up to per-band caps (defaults `3 / 2 / 1`), and
  generates ± displaced geometries at per-band RMS Cartesian
  amplitudes (defaults `0.08 / 0.05 / 0.03 Å`). At every displacement
  it submits a CCSD(T)/aug-cc-pvtz reference gradient plus a DFT
  gradient per functional in the geom_benchmark pool, then reports
  **per-Cartesian-component force RMSE** vs CCSD(T) grouped by
  functional category. Same per-group log layout as the
  `geom_benchmark` trajectory output. Invoke via `beep --config
  nm_sampling.json`; `beep --schema nm_sampling` for the full config.

  *Why force RMSE on displacements and not just along the trajectory.*
  Trajectory sampling probes a one-dimensional path; normal-mode
  displacements span the soft, chemically-relevant degrees of freedom
  off equilibrium. R2SCAN-3c wins on the H2/W1 sanity benchmark at
  18.5 meV/Å; HF-3c last at 805 meV/Å — both consistent with
  expectation. Single-metric ranking (no z-score machinery) — the
  per-group summary picks winners by raw RMSE.

- **Trajectory analysis in `geom_benchmark`** (default-on). For each
  DFT functional, BEEP now submits SP + gradient calculations at every
  geometry along the reference optimization trajectory and reports the
  per-Cartesian-component **RMSD of the force** (meV/Å) vs the
  reference. Combined with the existing equilibrium-geometry RMSD via
  a z-score-weighted ranking (weights configurable via
  `score_weights`). Same per-group table layout as the existing
  `BENCHMARK RESULTS` section; appears immediately below it in the
  workflow log. Inspired by the MLP test-set validation in Bovolenta
  et al. 2025 (A&A, in press; arXiv 2508.14219), Appendix C.2 /
  Fig C.1 / Table C.1. Set `trajectory_analysis: false` in the
  workflow config to keep the legacy eq-geometry-only behaviour.

  *Why force RMSD and not energy MAE/RMSE.* Absolute total energies
  carry method-specific offsets (correlation, basis, BSSE) that
  dominate any cross-method comparison and aren't relevant to
  geometry quality. Gradient deviations reflect PES shape, which is
  what geometry optimization actually cares about. For relative-energy
  comparison use the `energy_benchmark` workflow. RMSD (rather than
  MAE) is used to penalise occasional large failures — a single bad
  gradient step is exactly the failure mode that derails a real
  geometry optimization.

- **`combined_zscore_ranking` is weight-driven.** The metrics combined
  are taken from `weights.keys()`, so the same function serves the
  2-metric geom-benchmark case (`rmsd_eq` + `rmsd_force`) and any
  future N-metric variant without modification.

### Changed

- `geom_benchmark` now performs additional SP + gradient submissions
  per functional × reference-trajectory step when
  `trajectory_analysis` is enabled (the new default). Existing 0.12.x
  configs that don't set the field will pick up this behaviour
  automatically; opt out with `trajectory_analysis: false`.

- `GeomBenchmarkConfig.score_weights` now defaults to
  `{"rmsd_eq": 1.0, "rmsd_force": 1.0}` (equal weighting of the two
  metrics that survive the geom-benchmark physical filter).

## [0.12.0] — 2026-05-30

The first BEEP release in the qcportal 0.63+ era. The QCFractal stack
underwent a major rewrite between 0.15 and 0.50+, and this release
catches BEEP up — workflows, dataset I/O, spec representation, and CLI
all have new shapes. **This is a breaking release**: existing 0.6.x
configs and the legacy `launch_*.py` scripts no longer work.

### Breaking changes

- **Requires qcportal ≥ 0.63 and a QCFractal server ≥ 0.63.** Old
  qcportal-0.15 installations are no longer supported.
- **Single CLI entry point.** The standalone `launch_sampling.py`,
  `launch_energy.py`, `launch_be_hess.py`, and `launch_pre_exp.py`
  scripts have been replaced by a single command:
  ```
  beep --config <input.json>
  ```
  Workflows are selected by the `"workflow"` field in the JSON
  configuration. See `beep --workflows` to list available workflows and
  `beep --schema <workflow>` to dump a workflow's full field schema.
- **JSON-driven configuration with Pydantic validation.** Each workflow
  now has a Pydantic `*Config` model in `beep/models/` that validates
  the input JSON before execution. Type errors surface at parse time
  rather than mid-run.
- **`ReactionDataset` split by stoichiometry.** qcportal 0.63's
  `ReactionDataset` can only hold one reaction definition, so BEEP
  creates **four separate datasets per cluster**:
  `be_<MOLECULE>_<CLUSTER>_<METHOD>_<BASIS>_<stoich>` where `<stoich>`
  is one of `bsse`, `be_nocp`, `ie`, or `de`. Old 0.15-style datasets
  with multiple stoichs in one collection are no longer compatible.
- **Separated-pair dispersion specs.** Dispersion-corrected DFT
  (e.g. `b3lyp-d3bj/def2-tzvpd`) is now stored as a **separated pair**
  — a bare DFT spec on `psi4` plus a bare dispersion spec on `dftd3`
  or `dftd4` — matching the qcportal 0.63 migration layout. The
  recombined composite column is computed at fetch time. Integrated
  specs (method-with-dispersion-suffix carrying a basis) are skipped
  with a warning.
- **Lowercase spec convention.** All QCSpec names (`method`, `basis`,
  level-of-theory strings) are now normalized to lowercase at the
  model layer. Dataset names remain uppercase (BEEP convention).
- **Python ≥ 3.10 required.** Previous releases supported 3.8.
- `stoich="default"` / `stoich="int"` (qcportal 0.15 aliases) are
  gone — use `"bsse"` and `"ie"` respectively.

### Added

- **Conservative auto-recovery in `check_jobs_status`.** Parent
  `ReactionRecord` services stuck at ERROR are auto-reset *only when
  `record.children_errors` is empty*, restoring the qcportal-0.15 UX
  where resetting an errored child externally was enough to make the
  workflow continue without a be_hess restart. Real chemistry-error
  services are left alone for human inspection. Opt out via
  `auto_recover_services=False`.
- **Refinement-completion monitor in `sampling` workflow.** After the
  per-cluster sampling loop, the workflow polls all refinement-LOT
  optimizations and exits only when they reach terminal state. Max
  wait one week.
- **`fetch_reaction_values` helper** in
  `beep/adapters/qcfractal_adapter.py`. Returns a DataFrame indexed by
  entry name with composite columns built from separated-pair specs.
- **New tutorial notebook**:
  `tutorial/data_generation/05_restart_errored_reactions.ipynb` —
  walks through identifying and recovering from infrastructure vs
  chemistry errors across all four stoichiometry datasets, with regex
  classification.
- **Regression test suite** (`tests/`) — 255 tests covering the
  qcportal-0.63 adapter, workflows, model validation, and the new
  recovery / restart logic.
- `examples/` directory with template JSON configs for each workflow.
- `beep --workflows` lists available workflow names.
- `beep --schema <workflow>` dumps the full Pydantic schema (every
  field, type, default, description).
- `networkx` added as a direct runtime dependency (qcelemental's
  `align(atoms_map=True)` path, used by BEEP's symmetric RMSD filter,
  requires it but doesn't declare it itself).
- `pyproject.toml` packaging; `setup.py`/`setup.cfg` deprecated.

### Changed

- **All workflows refactored** to read a single JSON config:
  `sampling`, `be_hess`, `extract`, `pre_exp`, `geom_benchmark`,
  `energy_benchmark`. QCFractal I/O isolated into
  `beep/adapters/qcfractal_adapter.py`; pure computation logic moved
  to `beep/core/`.
- `fetch_reaction_values` column labels are lowercase (matching the
  lowercase spec convention).
- Tutorial notebooks 01-04 updated to the new qcportal 0.63 API and
  the `beep --config` CLI. Markdown text rewritten throughout (typos
  fixed; legacy CLI flags removed; per-stoich `ReactionDataset` split
  documented).
- Docs (`docs/index.rst`, `docs/getting_started.rst`) updated for
  qcportal 0.63+ and Python 3.10+. `readthedocs.yml` migrated to RTD
  v2 schema (ubuntu-24.04, python 3.10, pip-based install).

### Fixed

- **Sampling re-run at a new LOT.** Pre-existing entries are now
  submitted at the new sampling LOT (server-side dedupe handles the
  same-LOT case as a no-op). Previously the workflow short-circuited
  on "all entries already exist" and submitted zero new opts.
- **Atom-fragment ZPVE extraction.** Single-atom fragments
  (e.g. `C`, `N`, `O` on water clusters) no longer crash
  `zpve_correction` with `RuntimeError: Missing hessian`. The fix
  distinguishes `None` (genuinely missing) from `0.0` (valid atomic
  ZPVE).
- **`check_jobs_status` infinite-loop.** A transient HTTP 500 on
  `reset_records` no longer wedges the polling loop. Reset is
  best-effort: each service ID is reset at most once per call
  regardless of success/failure.
- **`get_zpve_mol` skips incomplete hessian records.** A "complete"-
  tagged hessian record whose `properties` is `None` (server
  inconsistency, mid-write state) no longer crashes downstream with
  `AttributeError: 'NoneType' object has no attribute 'get'`. The
  function now queries with `status=RecordStatusEnum.complete` and
  defensively filters records with null properties.
- **`get_job_ids` handles missing entries.** The `sampling` workflow
  no longer crashes with `PortalRequestError: Missing N entries` when
  pose generation returns fewer structures than requested.
- **be_hess parent-service ERROR recovery.** Workflow no longer logs
  "Some jobs have ERROR" indefinitely after the polling loop exits;
  conservative auto-recovery transitions stale-state parents back to
  COMPLETE when their children are clean.
- **`fetch_reaction_values` integrated-spec handling.** Integrated
  dispersion specs (e.g. `pbe-d3bj_def2-tzvp` as a single spec) are
  skipped with a warning rather than producing duplicate columns.
  BEEP no longer supports the integrated layout — only the
  separated-pair form matching the qcportal 0.63 migration.
- Stoichiometry-type lookup is consistent across `fetch_reaction_values`
  consumers; `extract` workflow's bare-DFT column drop heuristic uses
  lowercase suffixes.
- Pre-exp / extract / be_hess level-of-theory case mismatch (validators
  upper-cased what the server stored lowercase) — all flipped to
  lowercase.

### Removed

- `launch_sampling.py`, `launch_energy.py`, `launch_be_hess.py`,
  `launch_pre_exp.py` standalone CLI scripts — replaced by the unified
  `beep --config <input.json>` CLI.
- `setup.py` (replaced by `pyproject.toml`).
- Legacy U-prefix unrestricted-spec handling in `geom_benchmark` —
  unrestricted treatment now comes via keywords or multiplicity.
- Dead `query_results` adapter helper.

### Notes for users upgrading from 0.6.x

1. **Server side**: ensure your QCFractal server is on the 0.63+ stack.
   Data from older qcportal-0.15 servers needs to be migrated (a
   migration path exists for ReactionDatasets — see the `migration/`
   directory if present).
2. **Client side**: install the `beep-0.12` conda environment from the
   `qcfractal-0.64` clone (see README) and `pip install .` on top.
3. **Config files**: convert your old argparse-based command lines to
   the new JSON format. `beep --schema <workflow>` is the easiest way
   to generate a starting template; the `examples/` directory has one
   per workflow.
4. **Existing datasets**: ReactionDatasets created by BEEP 0.6.x cannot
   be read directly. The data on the v0.63 server should already be in
   the four-stoich layout (one dataset per stoichiometry type) if the
   migration script was used.

[0.12.0]: https://github.com/QCMM/beep/releases/tag/v0.12.0
