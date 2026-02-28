![logo_BEEP](https://user-images.githubusercontent.com/7481702/178007641-1f4260b3-dd34-4e39-9a51-286b076c5ea8.png)

# BEEP — Binding Energy Evaluation Platform

BEEP is a binding energy evaluation platform and database for molecules on interstellar ice-grain mantles.
It automates the full computational pipeline — from structure sampling through geometry optimization, binding energy computation, and data extraction — using a [QCFractal](https://github.com/MolSSI/QCFractal) (v0.15) server as the computation backend.

## Installation

BEEP requires Python 3.9+ and installs via pip:

```bash
pip install .
```

This pulls in the core dependencies: `qcportal`, `qcelemental`, `pydantic`, `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, and `tqdm`.

**Note:** To run computations you also need a running QCFractal server (v0.15) with compute workers. See the [QCFractal documentation](https://docs.qcarchive.molssi.org) for server setup.

## Usage

All workflows are driven through a single CLI entry point with a JSON configuration file:

```bash
beep --config input.json
```

The JSON file must contain a `"workflow"` key that selects which workflow to run.

## Workflows

| Workflow | Description |
|----------|-------------|
| `sampling` | Generate binding site candidates via random molecular placement, RMSD filtering, and geometry optimization |
| `be_hess` | Submit binding energy and Hessian computations for optimized structures |
| `extract` | Extract binding energies, apply ZPVE corrections, and generate summary tables and plots |
| `pre_exp` | Compute pre-exponential factors from vibrational analysis over a temperature range |
| `geom_benchmark` | Benchmark DFT geometry optimizations against high-level reference geometries |
| `energy_benchmark` | Benchmark DFT binding energies against high-level reference values (CBS extrapolation) |

Each workflow has a corresponding example configuration in the [`examples/`](examples/) directory:

```
examples/
  sampling.json
  be_hess.json
  extract.json
  pre_exp.json
  geom_benchmark.json
  energy_benchmark.json
```

## Output

Each workflow creates an organized output directory under the molecule name:

```
N2/
  sampling/       # binding site structures, site_finder debug data, log
  be_hess/        # log, config copy
  extract/        # CSV tables, SVG plots, log
```

Output folders contain a log file and a copy of the input configuration for reproducibility.

## Tutorials

The [`tutorial/`](tutorial/) directory contains Jupyter notebooks for:

- **Data query** — how to query the BEEP database using QCPortal
- **Data generation** — step-by-step walkthrough of the full BEEP protocol

## Running tests

```bash
pip install .[test]
pytest
```

### Copyright

Copyright (c) 2022, Stefan Vogt-Geisse, Giulia M. Bovolenta

#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
