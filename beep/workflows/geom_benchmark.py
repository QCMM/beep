"""Geometry benchmark workflow — refactored from workflows/launch_geom_benchmark.py."""
import json
import time
import logging
import subprocess
import warnings
from typing import Dict, List, Tuple, Union
from collections import Counter

import numpy as np
import pandas as pd
from pathlib import Path
from qcelemental.models.molecule import Molecule

from ..models.geom_benchmark import GeomBenchmarkConfig
from ..models.base import safe_config_dump
from ..core.logging_utils import (
    padded_log, log_dataframe_averages, log_progress, dict_to_log, beep_banner,
)
from ..core.dft_functionals import (
    geom_hmgga_dz, geom_hmgga_tz, geom_gga_dz, geom_gga_tz, geom_sqm_mb,
)
from ..core.plotting_utils import rmsd_histograms
from ..core.benchmark_utils import create_benchmark_dataset_dict, compute_rmsd
from ..core.trajectory_workflow import run_trajectory_analysis
from ..adapters import qcfractal_adapter as qcf
from ..adapters.qcfractal_adapter import FractalClient, is_complete, is_incomplete, is_error, status_label

bcheck = "\u2714"
mia0911 = "\u2606"
gear = "\u2699"

welcome_msg = beep_banner(
    "Geometry Benchmark",
    tagline="Shine, Loom, Manifest.",
    authors="Stefan Vogt-Geisse",
)


def create_and_add_specification(client, odset, method, basis, program,
                                  qc_keyword, geom_keywords=None):
    logger = logging.getLogger("beep")
    spec_name = f"{method}_{basis}".lower()

    spec = {
        "name": spec_name,
        "description": f"Geometric {program}/{method}/{basis}",
        "optimization_spec": {"program": "geometric", "keywords": geom_keywords},
        "qc_spec": {
            "driver": "gradient",
            "method": method,
            "basis": basis,
            "keywords": qc_keyword if isinstance(qc_keyword, dict) else {},
            "program": program,
        },
    }
    qcf.add_opt_specification(odset, spec, overwrite=True)
    logger.debug(f"Created and added the specification {spec_name} to {odset.name}")
    return spec_name


def optimize_reference_molecule(odset, struct_name, geom_ref_opt_lot, mol_mult, opt_tag):
    if mol_mult in (1, 2):
        return qcf.submit_optimizations(odset, geom_ref_opt_lot, tag=opt_tag, subset={struct_name})
    else:
        raise RuntimeError(
            "Invalid value for molecular multiplicity. It has to be 1 (Singlet) or 2 (Doublet)"
        )


def optimize_dft_molecule(client, odset, struct_name, method, basis, program,
                           dft_keyword, opt_tag):
    spec_name = create_and_add_specification(client, odset, method, basis, program, dft_keyword)
    return qcf.submit_optimizations(odset, spec_name, tag=opt_tag, subset={struct_name})


def wait_for_completion(client, odset_dict, opt_lot, program,
                         wait_interval=600, check_errors=False,
                         ref_spec=None):
    logger = logging.getLogger("beep")
    if isinstance(opt_lot, str):
        opt_lot = [opt_lot]

    ref_spec_key = ref_spec.lower() if ref_spec else None

    logger.info("\nChecking if the computations have finished\n")
    while True:
        dft_complete = 0
        dft_incomplete = 0
        dft_error = 0
        ref_complete = 0
        ref_incomplete = 0
        ref_error = 0

        for lot in opt_lot:
            lot_key = lot.lower()
            for struct_name, odset in odset_dict.items():
                record = odset.get_record(struct_name, lot_key)
                if record is None:
                    continue
                if is_error(record.status) and check_errors:
                    raise RuntimeError(
                        f"Error encountered in computation for {struct_name} with spec '{lot_key}'"
                    )
                if is_complete(record.status):
                    dft_complete += 1
                elif is_incomplete(record.status):
                    dft_incomplete += 1
                elif is_error(record.status):
                    dft_error += 1

        if ref_spec_key:
            for struct_name, odset in odset_dict.items():
                record = odset.get_record(struct_name, ref_spec_key)
                if record is None:
                    continue
                if is_complete(record.status):
                    ref_complete += 1
                elif is_incomplete(record.status):
                    ref_incomplete += 1
                elif is_error(record.status):
                    ref_error += 1

        total_incomplete = dft_incomplete + ref_incomplete

        if total_incomplete == 0:
            if ref_spec:
                logger.info(
                    f"Reference [{ref_spec}]: Complete: {ref_complete}, "
                    f"Error: {ref_error}"
                )
            logger.info(
                f"DFT: Complete: {dft_complete}, Error: {dft_error}"
            )
            logger.info(f"\nAll entries have been processed. {bcheck}")
            return dft_complete + ref_complete

        status_parts = []
        if ref_spec:
            status_parts.append(
                f"Ref: {ref_complete} done, {ref_incomplete} running, {ref_error} err"
            )
        status_parts.append(
            f"DFT: {dft_complete} done, {dft_incomplete} running, {dft_error} err"
        )
        logger.info(
            f"  {' | '.join(status_parts)}"
        )
        logger.info(
            f"  Waiting {wait_interval}s before rechecking..."
        )
        time.sleep(wait_interval)


def compare_rmsd(dft_lot, odset_dict, ref_geom_fmols):
    logger = logging.getLogger("beep")
    logger.propagate = False
    rmsd_df = pd.DataFrame(index=odset_dict.keys(), columns=dft_lot)
    final_opt_lot = {}
    total_operations = len(dft_lot)

    errored_specs = []
    for i, opt_lot in enumerate(dft_lot):
        opt_lot_key = opt_lot.lower()
        rmsd_tot_dict = {}
        err = None
        for struct_name, odset in odset_dict.items():
            record = odset.get_record(struct_name, opt_lot_key)
            err = (is_error(record.status) or record.status.value == "cancelled") if record is not None else True
            if err:
                logger.warning(
                    f"WARNING: Calculation for {struct_name} at the {opt_lot} level of theory "
                    f"finished with error (record id: {record.id}). "
                    f"This level of theory will be excluded from the benchmark."
                )
                errored_specs.append((opt_lot, struct_name, record.id))
                break
            fmol = record.final_molecule
            rmsd = compute_rmsd(ref_geom_fmols[struct_name], fmol, rmsd_symm=True)
            rmsd_tot_dict[struct_name] = rmsd
            rmsd_df.at[struct_name, opt_lot] = rmsd

        if err:
            rmsd_df[opt_lot] = np.nan
            continue
        rmsd_tot = list(rmsd_tot_dict.values())
        final_opt_lot[opt_lot] = np.mean(rmsd_tot)
        log_progress(logger, i + 1, total_operations)

    if errored_specs:
        logger.warning(f"\nSummary of errored optimizations ({len(errored_specs)} total):")
        for spec, struct, rec_id in errored_specs:
            logger.warning(f"  {spec} / {struct}  (record id: {rec_id})")
        logger.warning("")

    rmsd_df = rmsd_df.dropna(axis=1, how="all")
    lowest_values = sorted(final_opt_lot.values())[:1]
    best_geom_lot = {k: v for k, v in final_opt_lot.items() if v in lowest_values}
    return best_geom_lot, final_opt_lot, rmsd_df


def compare_all_rmsd(functional_groups, odset_dict, ref_geom_fmols):
    logger = logging.getLogger("beep")
    best_opt_lot = {}
    combined_rmsd_df = pd.DataFrame()

    for func_group, functionals in functional_groups.items():
        logger.info(f"\nProcessing RMSD for {func_group} type methods:")
        group_best_opt_lot, final_opt_lot, rmsd_df = compare_rmsd(
            functionals, odset_dict, ref_geom_fmols
        )
        rmsd_df.columns = [f"{func_group}_{col}" for col in rmsd_df.columns]
        combined_rmsd_df = pd.concat([combined_rmsd_df, rmsd_df], axis=1)
        best_opt_lot[func_group] = group_best_opt_lot

    return best_opt_lot, combined_rmsd_df


def _write_cp_python_script(mol, n_adsorbate_atoms, method, basis, cores, memory, output_path):
    """Write a Python script that runs a CP-corrected optimization via Psi4's API.

    The adsorbate is always the last n_adsorbate_atoms atoms. Fragment 1 is
    the surface (first atoms), fragment 2 is the adsorbate (last atoms).
    """
    bohr_to_ang = 0.529177210903
    geom = mol.geometry * bohr_to_ang
    symbols = mol.symbols
    n_atoms = len(symbols)
    surface_atoms = list(range(n_atoms - n_adsorbate_atoms))
    adsorbate_atoms = list(range(n_atoms - n_adsorbate_atoms, n_atoms))

    # Build the geometry string with fragment separator
    geom_lines = []
    geom_lines.append("  0 1")
    for i in surface_atoms:
        x, y, z = geom[i]
        geom_lines.append(f"  {symbols[i]:2s}  {x:14.10f}  {y:14.10f}  {z:14.10f}")
    geom_lines.append("  --")
    geom_lines.append("  0 1")
    for i in adsorbate_atoms:
        x, y, z = geom[i]
        geom_lines.append(f"  {symbols[i]:2s}  {x:14.10f}  {y:14.10f}  {z:14.10f}")
    geom_lines.append("  symmetry c1")
    geom_lines.append("  no_reorient")
    geom_lines.append("  no_com")
    geom_str = "\n".join(geom_lines)

    result_file = output_path / "final_molecule.json"

    script = f'''#!/usr/bin/env python
"""CP-corrected geometry optimization using Psi4 Python API."""
import json
import os
import sys
import stat
import tempfile

# QCEngine spawns psi4 as a subprocess for CP/nbody calculations.
# The subprocess needs the conda env's bin on PATH to find dftd4, dftd3, etc.
# Also create a local bash wrapper for psi4 to avoid GlusterFS shebang issues.
_env_bin = os.path.dirname(sys.executable)
os.environ["PATH"] = _env_bin + ":" + os.environ.get("PATH", "")

_wrapper_dir = tempfile.mkdtemp(prefix="psi4_wrapper_")
_wrapper = os.path.join(_wrapper_dir, "psi4")
with open(_wrapper, "w") as _f:
    _f.write(f"#!/bin/bash\\nexec {{sys.executable}} {{os.path.join(_env_bin, 'psi4')}} \\"$@\\"\\n")
os.chmod(_wrapper, stat.S_IRWXU)
os.environ["PATH"] = _wrapper_dir + ":" + os.environ["PATH"]

import numpy as np
import psi4

psi4.set_memory("{memory}")
psi4.set_num_threads({cores})
psi4.set_output_file("{output_path}/output.dat", False)

mol = psi4.geometry("""
{geom_str}
""")

psi4.set_options({{
    "basis": "{basis}",
    "scf_type": "df",
    "freeze_core": True,
    "geom_maxiter": 200,
}})

E = psi4.optimize("{method}", bsse_type="cp", return_total_data=True)

# Save the final geometry (Psi4 geometry is in bohr)
final_geom = mol.geometry().np
symbols = [mol.symbol(i) for i in range(mol.natom())]
result = {{
    "symbols": symbols,
    "geometry": final_geom.tolist(),
    "energy": E,
}}
with open("{result_file}", "w") as f:
    json.dump(result, f, indent=2)

print(f"CP optimization complete. E = {{E:.10f}}")
print(f"Result saved to {result_file}")
'''
    script_path = output_path / "cp_optimize.py"
    script_path.write_text(script)
    return script_path


def _write_slurm_script(script_path, partition, cores, memory, walltime, output_path):
    """Write a Slurm submit script for a CP optimization Python script."""
    script = f"""#!/bin/bash
#SBATCH --job-name=cp_opt
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cores}
#SBATCH --mem={memory}
#SBATCH --time={walltime}
#SBATCH --output={output_path}/slurm_%j.out

SCRATCH=/scratch/$USER/$SLURM_JOB_ID
mkdir -p "$SCRATCH"
cp {script_path} "$SCRATCH/cp_optimize.py"
cd "$SCRATCH"

# Activate the full psi4 conda environment so subprocess psi4 finds all addons
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate psi4-1.10
python cp_optimize.py

rm -rf "$SCRATCH"
"""
    submit_path = output_path / "submit.sh"
    submit_path.write_text(script)
    return submit_path


def _submit_cp_jobs(config, client, bchmk_dset_names, smol_name, res_folder):
    """Submit CP-corrected optimization jobs via Slurm. Returns job info list."""
    logger = logging.getLogger("beep")
    bt = config.bsse_test
    if bt is None:
        return []

    padded_log(logger, "Counterpoise-corrected optimizations (BSSE test)")

    # Get adsorbate atom count from the small molecule collection
    smol_dset = qcf.get_collection(client, "OptimizationDataset", config.small_molecule_collection)
    smol_entry = smol_dset.get_entry(smol_name)
    if smol_entry is None:
        smol_entry = smol_dset.get_entry(smol_name.upper())
    n_adsorbate_atoms = len(smol_entry.initial_molecule.symbols)
    logger.info(f"  Adsorbate: {smol_name} ({n_adsorbate_atoms} atoms)")

    # Build CP method/basis combinations for all test functionals
    cp_disp = bt.cp_dispersion if bt.cp_dispersion is not None else bt.dispersion
    cp_methods = []
    for func in bt.functional:
        for basis in bt.basis_sets:
            for disp in cp_disp:
                method = f"{func}-{disp}" if disp else func
                cp_methods.append((method, basis))

    jobs = []
    for bchmk_struct_name, odset_name in bchmk_dset_names.items():
        odset = qcf.get_collection(client, "OptimizationDataset", odset_name)
        entry = odset.get_entry(bchmk_struct_name)
        mol = entry.initial_molecule
        # Try to get a better starting geometry from an existing optimization
        for spec in odset.specification_names:
            rec = odset.get_record(bchmk_struct_name, spec)
            if rec is not None and is_complete(rec.status):
                mol = rec.final_molecule
                break

        for method, basis in cp_methods:
            cp_label = f"{method}-CP_{basis}"

            job_dir = res_folder / "cp_jobs" / f"{bchmk_struct_name}_{cp_label}"
            job_dir.mkdir(parents=True, exist_ok=True)

            py_script = _write_cp_python_script(
                mol, n_adsorbate_atoms, method, basis,
                bt.cores, bt.memory, job_dir,
            )
            script_path = _write_slurm_script(
                py_script, bt.partition, bt.cores,
                bt.memory, bt.walltime, job_dir,
            )

            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True, text=True, cwd=str(job_dir),
            )
            job_id = result.stdout.strip().split()[-1]
            logger.info(f"  Submitted CP job {job_id}: {bchmk_struct_name} / {cp_label}")

            jobs.append({
                "job_id": job_id,
                "struct_name": bchmk_struct_name,
                "label": cp_label,
                "job_dir": job_dir,
            })

    return jobs


def _wait_for_cp_jobs(jobs, wait_interval=120, max_wait=14400):
    """Wait for all CP Slurm jobs to finish."""
    logger = logging.getLogger("beep")
    if not jobs:
        return

    job_ids = [j["job_id"] for j in jobs]
    logger.info(f"\n  Waiting for {len(job_ids)} CP-corrected jobs to finish...")

    elapsed = 0
    while elapsed < max_wait:
        result = subprocess.run(
            ["squeue", "--jobs", ",".join(job_ids), "--noheader"],
            capture_output=True, text=True,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if stderr:
            logger.debug(f"  squeue stderr: {stderr}")
        running = len([l for l in stdout.split("\n") if l.strip()])
        if running == 0:
            logger.info(f"  All CP jobs finished. {bcheck}")
            return
        logger.info(f"  {running} CP jobs still running, waiting {wait_interval}s...")
        time.sleep(wait_interval)
        elapsed += wait_interval
    logger.warning(f"  CP jobs timed out after {max_wait}s. Proceeding with available results.")


def _collect_cp_results(jobs, ref_geom_fmols):
    """Collect CP-optimized geometries and compute RMSD against reference."""
    logger = logging.getLogger("beep")
    cp_rmsd = {}

    for job in jobs:
        geom_file = job["job_dir"] / "final_molecule.json"
        if not geom_file.exists():
            logger.warning(f"  CP job {job['label']} / {job['struct_name']}: no output geometry found")
            continue

        with open(geom_file) as f:
            result = json.load(f)

        cp_mol = Molecule(
            symbols=result["symbols"],
            geometry=np.array(result["geometry"]),  # already in bohr
            fix_com=True, fix_orientation=True,
        )

        ref_mol = ref_geom_fmols.get(job["struct_name"])
        if ref_mol is None:
            continue

        rmsd = compute_rmsd(ref_mol, cp_mol, rmsd_symm=True)
        label = job["label"]
        if label not in cp_rmsd:
            cp_rmsd[label] = {}
        cp_rmsd[label][job["struct_name"]] = rmsd
        logger.info(f"  {job['struct_name']} / {label}: RMSD = {rmsd:.6f}")

    return cp_rmsd


def run(config: GeomBenchmarkConfig, client: FractalClient) -> None:
    logger = logging.getLogger("beep")

    smol_name = config.molecule

    # Create output folder: <cwd>/<molecule>/
    res_folder = Path.cwd() / smol_name
    res_folder.mkdir(parents=True, exist_ok=True)
    data_folder = res_folder / "data"
    data_folder.mkdir(exist_ok=True)

    # File logging inside the output folder
    log_file = res_folder / f"geom_benchmark_{smol_name}.log"
    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    # Save a copy of the input config
    config_path = res_folder / f"geom_benchmark_{smol_name}.json"
    config_path.write_text(safe_config_dump(config))

    logger.info(welcome_msg)

    hl_tag = config.tag_reference_geometry
    dft_tag = config.tag_dft_geometry
    gr_method, gr_basis, gr_program = config.reference_geometry_level_of_theory
    geom_ref_opt_lot = (gr_method + "_" + gr_basis).lower()

    bchmk_structs = config.benchmark_structures
    surf_dset_name = config.surface_model_collection
    smol_dset_name = config.small_molecule_collection

    padded_log(logger, "Starting BEEP geometry benchmark procedure", padding_char=gear)
    logger.info(f"Molecule: {smol_name}")
    logger.info(f"Surface Model: {smol_dset_name}")
    logger.info(f"Benchmark Structures: {bchmk_structs}")

    odset_dict = {}
    bchmk_dset_names = create_benchmark_dataset_dict(bchmk_structs)

    qcf.check_collection_existence(client, *bchmk_dset_names.values())
    qcf.check_collection_existence(client, smol_dset_name)
    qcf.check_collection_existence(client, surf_dset_name)

    smol_dset = qcf.get_collection(client, "OptimizationDataset", smol_dset_name)
    mol_mult = qcf.get_molecular_multiplicity(client, smol_dset, smol_name)
    logger.info(f"\n The molecular multiplicity of {smol_name} is {mol_mult}")

    odset_dict = {smol_name: smol_dset}
    for bchmk_struct_name, odset_name in bchmk_dset_names.items():
        odset_dict[bchmk_struct_name] = qcf.get_collection(client, "OptimizationDataset", odset_name)
        # Extract surface model: strip molecule prefix and binding site suffix
        # e.g. 'H2O_CD1_01_0001' with dataset 'H2O_CD1_01' → surf_mod 'CD1_01'
        surf_mod = odset_name.split(f"{smol_name}_", 1)[1]
        odset_dict[surf_mod] = qcf.get_collection(client, "OptimizationDataset", surf_dset_name)

    padded_log(logger, "Start of the geometry refrence processing")
    logger.info(f"Method: {gr_method}")
    logger.info(f"Basis: {gr_basis}")
    logger.info(f"Program: {gr_program}\n")
    gr_keywords = config.reference_geometry_keywords
    for odset in odset_dict.values():
        create_and_add_specification(
            client, odset, method=gr_method, basis=gr_basis,
            program=gr_program, qc_keyword=gr_keywords, geom_keywords=None,
        )

    ct = 0
    for struct_name, odset in odset_dict.items():
        meta = optimize_reference_molecule(odset, struct_name, geom_ref_opt_lot, mol_mult, hl_tag)
        ct += getattr(meta, 'n_inserted', 0) + getattr(meta, 'n_existing', 0)

    logger.info(
        f"\nSend a total of {ct} structures to compute at the "
        f"{geom_ref_opt_lot} level of theory to the tag {hl_tag}\n"
    )

    padded_log(logger, "Start of the DFT geometry computations")

    dft_program = config.dft_optimization_program
    dft_keyword = config.dft_optimization_keyword

    dft_geom_functionals = {
        "geom_hmgga_dz": geom_hmgga_dz(),
        "geom_hmgga_tz": geom_hmgga_tz(),
        "geom_gga_dz": geom_gga_dz(),
        "geom_gga_tz": geom_gga_tz(),
        "geom_sqm_mb": geom_sqm_mb(),
    }

    # Build BSSE test combinations (uncorrected, via QCFractal)
    if config.bsse_test:
        bt = config.bsse_test
        for func in bt.functional:
            bsse_test_functionals = []
            for basis in bt.basis_sets:
                for disp in bt.dispersion:
                    method = f"{func}-{disp}" if disp else func
                    label = f"{method}_{basis}"
                    bsse_test_functionals.append(label)
            dft_geom_functionals[f"bsse_test_{func}"] = bsse_test_functionals

    all_dft_functionals = [
        functional
        for functionals in dft_geom_functionals.values()
        for functional in functionals
    ]

    logger.info(f"Program: {dft_program}")
    logger.info(f"DFT and SQM geometry methods:")
    dict_to_log(logger, dft_geom_functionals)

    ct = 0
    c = 0
    padded_log(logger, "Start sending DFT optimizations")
    for struct_name, odset in odset_dict.items():
        logger.info(f"\nSending geometry optimizations for {struct_name}")
        cs = 0
        for functionals in dft_geom_functionals.values():
            for functional in functionals:
                method, basis = functional.split("_", 1)
                meta = optimize_dft_molecule(
                    client, odset, struct_name, method, basis,
                    dft_program, dft_keyword, dft_tag,
                )
                n = getattr(meta, 'n_inserted', 0) + getattr(meta, 'n_existing', 0)
                cs += n
                ct += n
                c += 1
        logger.info(f"Send {cs} geometry optimizations for structure {struct_name}")

    logger.info(f"\nSend {ct}/{c} to the tag {dft_tag}\n")

    # Submit CP-corrected jobs (if configured) while QCFractal jobs run
    cp_jobs = _submit_cp_jobs(config, client, bchmk_dset_names, smol_name, data_folder)

    wait_for_completion(
        client, odset_dict, all_dft_functionals, dft_program,
        wait_interval=200, check_errors=False,
        ref_spec=geom_ref_opt_lot,
    )

    # Wait for CP Slurm jobs if any
    _wait_for_cp_jobs(cp_jobs)

    ref_geom_fmols = {}
    for struct_name, odset in odset_dict.items():
        record = odset.get_record(struct_name, geom_ref_opt_lot)
        if config.use_initial_reference_geometry:
            ref_geom_fmols[struct_name] = record.initial_molecule
        else:
            ref_geom_fmols[struct_name] = record.final_molecule

    padded_log(
        logger,
        "Start of RMSD comparison between DFT and {} geometries",
        geom_ref_opt_lot,
    )

    best_opt_lot, rmsd_df = compare_all_rmsd(dft_geom_functionals, odset_dict, ref_geom_fmols)

    # BSSE test results: collect CP results and show dedicated comparison
    if cp_jobs:
        cp_rmsd = _collect_cp_results(cp_jobs, ref_geom_fmols)

        padded_log(logger, "BSSE / Dispersion Test Results")
        bt = config.bsse_test
        logger.info(f"  Test functionals: {', '.join(bt.functional)}\n")

        cp_disp = bt.cp_dispersion if bt.cp_dispersion is not None else bt.dispersion
        disp_methods = [d for d in bt.dispersion if d]

        # Collect RMSD values per functional: all_data[func][method][basis] = rmsd
        all_data = {}
        for func in bt.functional:
            data = {}
            for basis in bt.basis_sets:
                for disp in bt.dispersion:
                    method = f"{func}-{disp}" if disp else func
                    col = f"bsse_test_{func}_{method}_{basis}"
                    if col in rmsd_df.columns:
                        vals = rmsd_df[col].dropna()
                        if len(vals) > 0:
                            data.setdefault(method, {})[basis] = vals.mean()
                    cp_label = f"{method}-CP_{basis}"
                    if cp_label in cp_rmsd:
                        cp_vals = list(cp_rmsd[cp_label].values())
                        if cp_vals:
                            data.setdefault(f"{method}-CP", {})[basis] = np.mean(cp_vals)
            all_data[func] = data

        # --- Per-functional tables ---
        for func in bt.functional:
            data = all_data[func]
            logger.info(f"\n  === {func} ===\n")

            # Absolute RMSD
            all_methods = []
            for disp in bt.dispersion:
                method = f"{func}-{disp}" if disp else func
                all_methods.append(method)
                cp_method = f"{method}-CP"
                if cp_method in data:
                    all_methods.append(cp_method)

            logger.info("  Absolute RMSD\n")
            header = f"  {'Basis':<16s}" + "".join(f"{m:<16s}" for m in all_methods)
            logger.info(header)
            logger.info("  " + "-" * (len(header) - 2))
            for basis in bt.basis_sets:
                line = f"  {basis:<16s}"
                for m in all_methods:
                    val = data.get(m, {}).get(basis)
                    line += f"{val:<16.6f}" if val is not None else f"{'---':<16s}"
                logger.info(line)

            # Delta CP
            cp_methods_with_data = []
            for disp in cp_disp:
                method = f"{func}-{disp}" if disp else func
                cp_method = f"{method}-CP"
                if cp_method in data:
                    cp_methods_with_data.append((method, cp_method))

            if cp_methods_with_data:
                logger.info(f"\n  Delta CP\n")
                header = f"  {'Basis':<16s}" + "".join(f"{'d(' + cp + ')':<20s}" for _, cp in cp_methods_with_data)
                logger.info(header)
                logger.info("  " + "-" * (len(header) - 2))
                for basis in bt.basis_sets:
                    line = f"  {basis:<16s}"
                    for method, cp_method in cp_methods_with_data:
                        uncorr = data.get(method, {}).get(basis)
                        corr = data.get(cp_method, {}).get(basis)
                        if uncorr is not None and corr is not None:
                            line += f"{corr - uncorr:<+20.6f}"
                        else:
                            line += f"{'---':<20s}"
                    logger.info(line)

            # Delta Dispersion
            if disp_methods:
                bare = func
                logger.info(f"\n  Delta Dispersion\n")
                header = f"  {'Basis':<16s}" + "".join(f"{'d(' + func + '-' + d + ')':<20s}" for d in disp_methods)
                logger.info(header)
                logger.info("  " + "-" * (len(header) - 2))
                for basis in bt.basis_sets:
                    line = f"  {basis:<16s}"
                    bare_val = data.get(bare, {}).get(basis)
                    for d in disp_methods:
                        method = f"{func}-{d}"
                        disp_val = data.get(method, {}).get(basis)
                        if bare_val is not None and disp_val is not None:
                            line += f"{disp_val - bare_val:<+20.6f}"
                        else:
                            line += f"{'---':<20s}"
                    logger.info(line)

        # --- Average across all functionals ---
        if len(bt.functional) > 1:
            logger.info(f"\n  === Average across {', '.join(bt.functional)} ===\n")

            # Average Delta CP
            cp_methods_labels = []
            for disp in cp_disp:
                cp_methods_labels.append(disp if disp else "bare")
            has_cp = any(
                f"{(f'{func}-{disp}' if disp else func)}-CP" in all_data[func]
                for func in bt.functional for disp in cp_disp
            )
            if has_cp:
                logger.info("  Average |Delta CP| (absolute BSSE effect)\n")
                header = f"  {'Basis':<16s}" + "".join(f"{'|d(CP-' + l + ')|':<20s}" for l in cp_methods_labels)
                logger.info(header)
                logger.info("  " + "-" * (len(header) - 2))
                for basis in bt.basis_sets:
                    line = f"  {basis:<16s}"
                    for disp in cp_disp:
                        deltas = []
                        for func in bt.functional:
                            data = all_data[func]
                            method = f"{func}-{disp}" if disp else func
                            cp_method = f"{method}-CP"
                            uncorr = data.get(method, {}).get(basis)
                            corr = data.get(cp_method, {}).get(basis)
                            if uncorr is not None and corr is not None:
                                deltas.append(abs(corr - uncorr))
                        if deltas:
                            line += f"{np.mean(deltas):<20.6f}"
                        else:
                            line += f"{'---':<20s}"
                    logger.info(line)

            # Average Delta Dispersion
            if disp_methods:
                logger.info(f"\n  Average |Delta Dispersion| (absolute dispersion effect)\n")
                header = f"  {'Basis':<16s}" + "".join(f"{'|d(' + d + ')|':<20s}" for d in disp_methods)
                logger.info(header)
                logger.info("  " + "-" * (len(header) - 2))
                for basis in bt.basis_sets:
                    line = f"  {basis:<16s}"
                    for d in disp_methods:
                        deltas = []
                        for func in bt.functional:
                            data = all_data[func]
                            bare_val = data.get(func, {}).get(basis)
                            disp_val = data.get(f"{func}-{d}", {}).get(basis)
                            if bare_val is not None and disp_val is not None:
                                deltas.append(abs(disp_val - bare_val))
                        if deltas:
                            line += f"{np.mean(deltas):<20.6f}"
                        else:
                            line += f"{'---':<20s}"
                    logger.info(line)
        logger.info("")

    padded_log(logger, "BENCHMARK RESULTS")
    log_dataframe_averages(logger, rmsd_df)

    folder_path_json = data_folder / "json"
    folder_path_json.mkdir(parents=True, exist_ok=True)

    rmsd_df.to_json(str(folder_path_json / "results_geom_benchmark.json"))
    logger.info(f"\nDataFrame successfully saved to {folder_path_json}/results_geom_benchmark.json\n")

    if config.generate_plots:
        folder_path_plots = data_folder / "plots"
        folder_path_plots.mkdir(parents=True, exist_ok=True)
        rmsd_histograms(rmsd_df, smol_name, str(folder_path_plots))

    # Trajectory analysis: SP+gradient at every reference-trajectory geometry,
    # MAE/RMSE of E and forces vs reference, combined z-score ranking.
    # Runs AFTER BENCHMARK RESULTS so the eq-RMSD per-group output is the
    # first benchmark summary the user sees; the trajectory benchmark
    # appears immediately below in the same per-group style.
    if config.trajectory_analysis:
        run_trajectory_analysis(
            config=config, client=client, odset_dict=odset_dict,
            geom_ref_opt_lot=geom_ref_opt_lot,
            all_dft_functionals=all_dft_functionals,
            dft_geom_functionals=dft_geom_functionals,
            dft_program=dft_program, dft_keyword=dft_keyword,
            dft_tag=dft_tag, rmsd_df=rmsd_df,
            res_folder=data_folder, logger=logger,
        )

    padded_log(
        logger,
        "Geometry Benchmark finished successfully! Hasta pronto!",
        padding_char=mia0911,
    )

    logger.removeHandler(file_handler)
    file_handler.close()
