"""Normal-mode displacement benchmark workflow.

Entry point for ``beep --config <nm_sampling.json>``. Sets up logging
and the output folder, resolves the optimisation datasets for the
target molecule + surface model + benchmark binding sites, and hands
control to the orchestrator in
:mod:`beep.core.nm_sampling_workflow`.

The actual physics + orchestration lives in:
  - :mod:`beep.core.normal_mode_sampling` — classification, selection,
    displacement math.
  - :mod:`beep.core.nm_sampling_workflow` — chain Hessian → vibanal →
    displacement → SP+gradient → metrics.
  - :mod:`beep.adapters.qcfractal_adapter` — QCFractal I/O (``submit_hessians``,
    ``fetch_normal_modes``, the ``SinglepointDataset`` helpers).
"""
import logging
from pathlib import Path

from ..models.nm_sampling import NmSamplingConfig
from ..models.base import safe_config_dump
from ..core.logging_utils import padded_log, beep_banner
from ..core.dft_functionals import (
    geom_hmgga_dz, geom_hmgga_tz, geom_gga_dz, geom_gga_tz, geom_sqm_mb,
)
from ..core.benchmark_utils import create_benchmark_dataset_dict
from ..core.nm_sampling_workflow import run_nm_sampling
from ..adapters import qcfractal_adapter as qcf
from ..adapters.qcfractal_adapter import FractalClient


mia0911 = "☆"
gear = "⚙"

welcome_msg = beep_banner(
    "NM-Sampling",
    tagline="Move along the soft modes; let CCSD(T) keep score.",
    authors="Stefan Vogt-Geisse",
)


def run(config: NmSamplingConfig, client: FractalClient) -> None:
    logger = logging.getLogger("beep")

    smol_name = config.molecule

    res_folder = Path.cwd() / smol_name
    res_folder.mkdir(parents=True, exist_ok=True)
    data_folder = res_folder / "data"
    data_folder.mkdir(exist_ok=True)

    log_file = res_folder / "log"
    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    config_path = res_folder / f"nm_sampling_{smol_name}.json"
    config_path.write_text(safe_config_dump(config))

    logger.info(welcome_msg)

    bchmk_structs = config.benchmark_structures
    surf_dset_name = config.surface_model_collection
    smol_dset_name = config.small_molecule_collection

    padded_log(logger, "Starting BEEP NM-sampling benchmark", padding_char=gear)
    logger.info(f"Molecule: {smol_name}")
    logger.info(f"Surface model collection: {surf_dset_name}")
    logger.info(f"Small molecule collection: {smol_dset_name}")
    logger.info(f"Benchmark structures: {bchmk_structs}")
    logger.info(f"Geometry LOT: {config.geometry_opt_lot}")
    logger.info(f"Hessian LOT:  {config.hessian_lot}")
    logger.info(f"Reference grad LOT: {config.reference_grad_lot}\n")

    bchmk_dset_names = create_benchmark_dataset_dict(bchmk_structs)
    qcf.check_collection_existence(client, *bchmk_dset_names.values())
    qcf.check_collection_existence(client, smol_dset_name)
    qcf.check_collection_existence(client, surf_dset_name)

    smol_dset = qcf.get_collection(client, "OptimizationDataset", smol_dset_name)
    odset_dict = {}
    for bchmk_struct_name, odset_name in bchmk_dset_names.items():
        odset_dict[bchmk_struct_name] = qcf.get_collection(
            client, "OptimizationDataset", odset_name,
        )

    # Adsorbate atom count: pull from the small-molecule collection (same
    # pattern as geom_benchmark.py:347-353 for the BSSE-test CP path).
    smol_entry = smol_dset.get_entry(smol_name)
    if smol_entry is None:
        smol_entry = smol_dset.get_entry(smol_name.upper())
    n_adsorbate_atoms = len(smol_entry.initial_molecule.symbols)
    logger.info(f"Adsorbate atom count: {n_adsorbate_atoms}\n")

    # Same five functional groups as geom_benchmark — the displacements
    # are evaluated at every functional in this list.
    dft_geom_functionals = {
        "geom_hmgga_dz": geom_hmgga_dz(),
        "geom_hmgga_tz": geom_hmgga_tz(),
        "geom_gga_dz":   geom_gga_dz(),
        "geom_gga_tz":   geom_gga_tz(),
        "geom_sqm_mb":   geom_sqm_mb(),
    }
    all_dft_functionals = [
        f for group in dft_geom_functionals.values() for f in group
    ]

    run_nm_sampling(
        config=config, client=client, odset_dict=odset_dict,
        all_dft_functionals=all_dft_functionals,
        dft_geom_functionals=dft_geom_functionals,
        n_adsorbate_atoms=n_adsorbate_atoms,
        res_folder=data_folder, logger=logger,
    )

    padded_log(
        logger,
        "NM-sampling benchmark finished successfully! Hasta pronto!",
        padding_char=mia0911,
    )
    logger.removeHandler(file_handler)
    file_handler.close()
