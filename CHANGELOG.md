# Changelog

All notable changes to BEEP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — 0.15.0.dev

### Added

- **Many-Body Expansion binding-energy workflows (`mbe` / `mbe_extract`),
  ported from the standalone `beep-mbe` package (v0.1.0 @ `44a90e6`).** These
  provide an alternative route to binding energies on the *same* binding sites
  produced by `sampling` / `be_hess`, re-evaluated at a (typically higher)
  level of theory via n-body fragmentation on a qcmanybody `ManybodyDataset`
  plus a monomer `SinglepointDataset`.
  - `mbe` submits and (optionally) monitors the many-body computations.
  - `mbe_extract` assembles per-site binding energies (`be_data/total_be.csv`)
    plus n-body decomposition tables (`decomp__<spec>.csv`,
    `contrib__<spec>.csv`) and a text report.
  - `mbe_extract` can optionally apply a **read-only** ZPVE correction borrowed
    from a prior `be_hess` run on the same sites (`total_be_zpve.csv`); it never
    submits or mutates the `be_hess` datasets, and sites without a usable
    Hessian are reported as `NaN`.
  - New adapter helpers (`get_or_create_manybody_dataset`,
    `mbe_levels_to_qc_specifications`, `build_manybody_specification`,
    `wait_for_manybody_completion`, `wait_for_dataset_records`) are strictly
    additive; the existing `wait_for_completion` used by `sampling` /
    `geom_benchmark` is unchanged.
  - Config uses BEEP's Pydantic-v2 schema (nested `server.*`, object-style
    `levels`); the standalone package's flat JSON format is not supported.
    See `examples/mbe.json` and `examples/mbe_extract.json`.
  - `pandas` is now an explicit runtime dependency.

- **MBE truncation-error estimate in `mbe_extract`.** Each site/spec now gets a
  symmetric error bar on the binding energy from a geometric extrapolation of
  the uncomputed n-body tail (`bar = |Δn|·r/(1−r)`, `r = |Δn/Δn−1|`), written to
  `be_data/convergence__<spec>.csv` and rendered as `BE ± bar (converged?)` in
  the report. The bar is a magnitude only — no signed `BE_∞` — because the sign
  of the next term is not predictable. 2-body-only runs report `n/a` (no
  convergence information); a non-shrinking series is flagged not converged. The
  `converged` flag uses a configurable `convergence_tol` (default 0.05). The
  existing `total_be` / `decomp` / `contrib` CSVs are unchanged.
- **ZPVE correction is now a friction-free toggle.** `mbe_extract`'s
  `zpve.hessian_clusters` is optional; when omitted, the `be_<MOL>_*` datasets
  are auto-discovered from the server (still strictly read-only). Turning ZPVE
  on is just `"zpve": {"enabled": true}`.
- **MBE level validation.** `mbe` now requires contiguous body-order indices
  (`1..N`, no gaps), matching qcmanybody's expectation. Per-tier levels of
  theory (a distinct method/basis per body order) and per-site selection
  (`entries`, omit for all sites of the cluster) are documented in the examples.

## [0.14.0] — 2026-07-07

### Added

- **Raw force-delta arrays persisted to disk** for both `nm_sampling` and
  `geom_benchmark` trajectory analysis. The per-functional flat arrays of
  per-Cartesian-component force deltas (and per-atom energy deltas) are
  now saved alongside the metrics JSON as a single NumPy `.npz` file,
  loadable with `np.load(path)` → `{functional_name → array}`. Lets
  users plot their own per-functional histograms of `(F_DFT - F_ref)`
  without re-running the workflow.

- **Atom sampling and atom be_hess no longer crash on missing entries.**
  `qcportal>=0.63` raises `PortalRequestError` (HTTP 400 "Missing N
  entries: …") — not `KeyError` — when an entry name isn't in the
  dataset. Both `sampling.py` and `be_hess.py` caught only `KeyError`
  on the OptimizationDataset lookup before falling through to the
  atoms_collection, so single-atom adsorbates (e.g. C, N, O) crashed
  the workflow instead of being routed to the SinglepointDataset
  fallback. Fixed at the adapter layer: `fetch_opt_record` and
  `fetch_atom_molecule` now translate the specific "Missing entries"
  PortalRequestError into `KeyError` while letting genuine
  server/network/auth failures propagate, so the workflow's
  `except KeyError` does the right thing and unrelated transient
  failures aren't silently misrouted as "atom not found". Five
  regression tests added.

- **Clearer per-LOT submission log in `be_hess` / `energy_benchmark`.**
  `compute_be_dft_energies` used to log `Existing N  Submitted M` per
  LOT, where both `N` and `M` come from `ds.submit()`'s
  `InsertCountsMetadata` — *newly* linked vs *newly* submitted in this
  call. On re-runs where every reaction was already linked to the
  dataset from a prior run, both would correctly be 0, but
  "Existing 0  Submitted 0" reads as "nothing on the server" rather
  than "nothing changed." The line now distinguishes the two cases:
  if both metrics are 0, it logs "all reactions already linked to the
  dataset (no new submissions)"; otherwise it logs newly-submitted +
  newly-linked counts explicitly.

- **Refinement summary table in `sampling` workflow.** After refinement
  polling finishes, the workflow now logs a per-cluster breakdown of
  COMPLETE / ERROR / TOTAL refinement opts in the same layout as the
  sampling summary.

- **Functional averaging and over/under-bind tag in `energy_benchmark`.**
  Two related additions to the BE per-group MAE table:

  - New `functional_averages: List[List[str]]` config field. Each group
    is a list of `method_basis` LOT strings. For every group, the
    workflow computes the mean BE per binding site across the listed
    functionals and reports it as one additional row labelled
    `DFT_Average_<N>` under a dedicated "Averages" group in the per-group
    output. Members not present in the benchmark set are dropped with a
    warning. Empty default; configs without the field work unchanged.

  - The per-group MAE table now carries a `Bias` column and a
    human-readable tag indicating whether each functional (or DFT
    average) over- or under-binds the reference. Sign convention:
    negative bias → overbinds, positive bias → underbinds. Tag
    thresholds (kcal/mol): `|bias| < 0.05` → balanced; `< 0.5` → mild
    over/underbind; else → strong over/underbind. Surfaces in the
    standard BE table and the gCP-corrected BE table when
    `gcp_correction=true`.

- **Errored-record reporting at the end of each `be_hess` section.**
  After each completion check (BE pass 1, BE pass 2, Hessian),
  `be_hess` now logs the record IDs that ended in ERROR as a single
  space-separated line, making them easy to copy-paste into a reset
  command. No log noise when there are no errors. Driven by a new
  `report_errored_records(client, ids, logger, section_name)` helper
  in the adapter, reusable by any workflow that wants the same
  end-of-section error summary.

### Fixed

- **Trajectory + SP gradient lookup falls back to `return_result`.**
  `fetch_sp_energy_gradient` and `get_optimization_trajectory` in the
  adapter previously read only `properties.return_gradient`. That field
  is optional per the qcelemental spec: psi4 populates it, but molpro
  (used for the F12 reference geometries in `geom_benchmark` /
  `nm_sampling`) leaves it unset and instead writes the gradient to
  `return_result` on gradient-driver records. The reader was silently
  returning `None` for the gradient on those records, so the
  trajectory force-RMSE / nm_sampling force-RMSE analyses on F12-stack
  data were collapsing to no-data and producing silent zeros. Both
  helpers now try `properties.return_gradient` first and fall back to
  `return_result` when the spec driver is gradient.

- **`pre_exp` moments of inertia now COM-shifted.**
  `get_moments_of_inertia` in `core/pre_exponential.py` evaluated the
  inertia tensor in the input frame without first translating to the
  centre of mass, so any COM offset in the geometry inflated all three
  eigenvalues via the parallel-axis theorem. Compounding this,
  `workflows/pre_exp.py` was passing the raw, un-aligned coordinates
  to `get_moments_of_inertia` while computing (and then discarding) the
  output of `align_to_z_axis`. Drift observed: ~18% on H2O,
  <1% on NH3/CH3OH/CH3CN where the optimiser happened to land near
  origin, ~0 on linear molecules where `Ia=0` is preserved. Two
  coordinated fixes:
  1. `get_moments_of_inertia` now COM-shifts unconditionally before
     building the tensor, so the function is correct regardless of
     what the caller passes.
  2. The `pre_exp` workflow now passes the aligned coordinates (the
     output of `align_to_z_axis`) into `get_moments_of_inertia`, so
     the principal-axis frame is used downstream rather than
     computed-and-discarded.

- **`compute_hessian` reuses existing Hessian records regardless of
  spec keywords.** The DB now holds a mix of migrated Hessians (stored
  with empty `keywords={}`) and newly-submitted ones (stored with
  `keywords={"function_kwargs": {"dertype": 1}}`), and
  `add_singlepoints`' server-side `find_existing` matches strictly on
  the keyword dict. Migrated records were therefore missed and
  re-submitted. `compute_hessian` now does a keyword-agnostic
  pre-query on `(molecule_id, method, basis, driver=hessian, status=complete)`
  before calling `add_singlepoints`, treats every matching record as
  reusable, and submits new Hessians only for the molecules without a
  hit. Eliminates redundant — and individually expensive — Hessian
  re-computes on migrated species. The 2022-era keyword on new
  submissions is unchanged, so future-pure-v0.14 datasets keep the
  full spec.

- **`be_hess` and `energy_benchmark` no longer wipe existing reaction
  datasets on every run.** `create_or_load_reaction_dataset` and
  `create_reaction_dataset` were unconditionally deleting and
  recreating each per-stoichiometry ReactionDataset (`_bsse`,
  `_be_nocp`, `_ie`, `_de`) on every workflow launch, which wiped
  every spec/entry registration the dataset had accumulated from
  previous runs at other LOTs. Underlying singlepoint records
  survived (`delete_records=False`) but were left orphaned. Both
  helpers are now true get-or-create: the dataset is returned
  unchanged if it exists, only created when missing. qcportal 0.63+
  `add_specification` and `add_entry` are already idempotent on the
  dataset, so a second `be_hess`/`energy_benchmark` run at a new LOT
  now layers its specs on top of the existing dataset instead of
  destroying prior LOT data.

### Changed

- **Reverted unconditional SCF-accelerator override in `be_hess`.**
  0.13.0 unconditionally injected `scf_initial_accelerator: NONE` into
  the BE single-point keywords to bypass Psi4's default ADIIS warmup.
  That changed the QCSpecification hash and broke `find_existing` cache
  hits against records on the server that were submitted without the
  keyword. 0.14.0 reverts to the pre-fix behaviour: no
  scf_initial_accelerator override is sent, Psi4's default SCF
  acceleration is used, and existing records continue to match. Users
  who hit ADIIS-induced SCF convergence failures should handle them
  per-record (e.g. reset the failed records) rather than via a
  workflow-wide keyword.

- **`sampling` workflow log order.** The `SAMPLING SUMMARY` table (binding
  sites per cluster) now appears immediately after the sampling loop
  finishes and *before* the refinement-polling block, instead of after.
  Makes the sampling vs refinement stages distinguishable in the log;
  pairs with the new refinement summary table at the very end.

- **Plot generation now opt-in across all workflows.** `geom_benchmark`
  and `energy_benchmark` (alongside `extract`, which was already opt-in)
  now expose a `generate_plots: bool` config field defaulting to
  `False`. JSON results are always written; SVG plots only when the
  flag is set. Set `"generate_plots": true` in the workflow JSON to
  restore the previous always-plot behavior.

- **Uniform output directory layout across every workflow.** All
  workflows now write to `<cwd>/<molecule>/` with a single shape:
  ```
  <molecule>/
  ├── <workflow>_<molecule>.log         (workflow log)
  ├── <workflow>_<molecule>.json        (copy of input config)
  └── data/                             (only if the workflow produces outputs)
      ├── <misc files>                  (CSV / NPZ / xyz / molden / dat)
      ├── json/                         (all JSON outputs)
      └── plots/                        (all SVG / PNG plots)
  ```
  Replaces the per-workflow `<molecule>/<workflow>/` nesting and the
  ad-hoc layouts that previously varied between workflows. Downstream
  consumers that hardcode the old paths (e.g. notebooks reading
  `<molecule>/<workflow>/json_data/...`) need updating to the new
  `<molecule>/data/json/...` form.

## [0.13.0] — 2026-06-08

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

### Fixed

- `be_hess`: bypass Psi4's default ADIIS warmup by setting
  `scf_initial_accelerator: NONE` in the SCF keywords. On Psi4 1.10
  the ADIIS path triggered intermittent SCF convergence failures on a
  fraction of BE records. Verified on H2O / B3LYP / def2-SVP that pure
  DIIS reaches the same converged energy (Δ < 1e-13 Ha) without the
  ADIIS markers in the Psi4 output. No effect on already-submitted
  records — only new submissions pick up the override.

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
