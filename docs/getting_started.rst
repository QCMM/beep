Getting Started
===============

Prerequisites
-------------

- Python 3.9 or later
- A running `QCFractal <https://docs.qcarchive.molssi.org>`_ (v0.15) server with compute workers (required for running computations; not needed for data extraction or analysis)

Installation
------------

Clone the repository and install with pip:

.. code-block:: bash

   git clone https://github.com/vogt-geisse/beep.git
   cd beep
   pip install .

This pulls in the core dependencies: ``qcportal``, ``qcelemental``, ``pydantic``,
``numpy``, ``pandas``, ``scipy``, ``matplotlib``, ``seaborn``, and ``tqdm``.

To install the test dependencies as well:

.. code-block:: bash

   pip install .[test]

**Optional:** The ``pre_exp`` workflow can use ``molsym`` for symmetry number
detection. Install it separately if needed.

Basic Usage
-----------

All workflows are driven through a single CLI entry point with a JSON
configuration file:

.. code-block:: bash

   beep --config input.json

The JSON file must contain a ``"workflow"`` key that selects which workflow to
run. Example configuration files are provided in the ``examples/`` directory.

Workflows
---------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Workflow
     - Description
   * - ``sampling``
     - Generate binding site candidates via random molecular placement, RMSD filtering, and geometry optimization
   * - ``be_hess``
     - Submit binding energy and Hessian computations for optimized structures
   * - ``extract``
     - Extract binding energies, apply ZPVE corrections, and generate summary tables and plots
   * - ``pre_exp``
     - Compute pre-exponential factors from vibrational analysis over a temperature range
   * - ``geom_benchmark``
     - Benchmark DFT geometry optimizations against high-level reference geometries
   * - ``energy_benchmark``
     - Benchmark DFT binding energies against high-level reference values (CBS extrapolation)

Output Directory Structure
--------------------------

Each workflow creates an organized output directory under the molecule name:

.. code-block:: text

   N2/
     sampling/       # binding site structures, site_finder debug data, log
     be_hess/        # log, config copy
     extract/        # CSV tables, SVG plots, log
     pre_exp/        # pre-exponential factor results, log
     geom_benchmark/ # geometry benchmark results, log
     energy_benchmark/ # energy benchmark results, log

Every output folder contains a log file and a copy of the input configuration
for reproducibility.

Running Tests
-------------

.. code-block:: bash

   pip install .[test]
   pytest

The test suite covers core logic, models, and workflow integration without
requiring a running QCFractal server or psi4.
