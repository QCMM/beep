"""Sampling workflow — refactored from workflows/launch_sampling.py."""
import logging
from pathlib import Path

from qcfractal.interface.client import FractalClient

from ..models.sampling import SamplingConfig
from ..core.logging_utils import padded_log
from ..core.errors import DatasetNotFound, LevelOfTheoryNotFound
from ..adapters import qcfractal_adapter as qcf

bcheck = "\u2714"
gear = "\u2699"

welcome_msg = """
---------------------------------------------------------------------------------------
Welcome to the BEEP  Set-of-clusters Sampling Workflow
---------------------------------------------------------------------------------------

"Adopt the pace of nature: her secret is patience."

                              \u2013 Ralph Waldo Emerson

Seek Locate Map
---------------------------------------------------------------------------------------

                            By:  Stefan Vogt-Geisse and Giulia M. Bovolenta
"""


def sampling_model_msg(surface_model, target_mol, method, basis, program):
    return f"""
-----------------------------------------------------------------------------------------
Starting the sampling of the surface model {surface_model} with the molecule {target_mol}
The sampling level of theory is method: {method} basis: {basis} program: {program}
-----------------------------------------------------------------------------------------
    """


def sampling_round_msg(opt_smplg, opt_refine):
    return f"""
The OptimizationDataset for the sampling results: {opt_smplg}
The OptimizationDataset for the unique structures for refinement: {opt_refine}
    """


def check_collection_existence(client, *collections, collection_type="OptimizationDataset"):
    for collection in collections:
        if not qcf.check_collection_exists(client, collection_type, collection):
            raise DatasetNotFound(
                f"Collection {collection} does not exist. Please create it first. Exiting..."
            )


def check_optimized_molecule(ds, opt_lot, mol_names):
    for mol in list(mol_names):
        try:
            rr = qcf.fetch_opt_record(ds, mol, opt_lot)
        except KeyError:
            raise LevelOfTheoryNotFound(
                f"{opt_lot} level of theory for {mol} or the entry itself does not exist "
                f"in {ds.name} collection. Add the molecule and optimize it first\n"
            )
        if rr.status == "INCOMPLETE":
            raise ValueError(f" Optimization has status {rr.status} restart it or wait")
        elif rr.status == "ERROR":
            raise ValueError(f" Optimization has status {rr.status} restart it or wait")


def process_refinement(client, ropt_lot_name, rmethod, rbasis, program,
                       qc_keyword, ds_opt, logger, rtag="refinement"):
    spec = {
        "name": ropt_lot_name,
        "description": f"Geometric + {rmethod}/{rbasis}/{program}",
        "optimization_spec": {"program": "geometric", "keywords": None},
        "qc_spec": {
            "driver": "gradient",
            "method": rmethod,
            "basis": rbasis,
            "keywords": qc_keyword,
            "program": program,
        },
    }

    qcf.add_opt_specification(ds_opt, spec, overwrite=True)
    ds_opt.save()
    c = qcf.submit_optimizations(ds_opt, ropt_lot_name, tag=rtag)

    logger.info(
        f"\nRefinement optimization initiated with specification '{ropt_lot_name}' \n"
        f"using {rmethod}/{rbasis} in {program}. \n"
        f"Description: {spec['description']}. \n"
        f"Tag applied: '{rtag}'\n"
        f"Number of optimizations submitted: {c}. {bcheck} \n"
    )


def run(config: SamplingConfig, client: FractalClient) -> None:
    from beep.adapters.qcfractal_adapter import run_sampling

    logger = logging.getLogger("beep")
    logger.info(welcome_msg)

    smol_name = config.molecule
    method = config.sampling_level_of_theory.method
    basis = config.sampling_level_of_theory.basis
    program = config.sampling_level_of_theory.program
    rmethod = config.refinement_level_of_theory.method
    rbasis = config.refinement_level_of_theory.basis
    rprogram = config.refinement_level_of_theory.program

    qc_keyword = config.keyword_id

    if basis:
        opt_lot = method + "_" + basis
    else:
        opt_lot = method
    ropt_lot = rmethod + "_" + rbasis

    args_dict = {
        "method": method,
        "basis": basis,
        "program": program,
        "tag": config.sampling_tag,
        "kw_id": None,  # keyword_id is used for refinement spec, not sampling
        "rmsd_symm": config.rmsd_symmetry,
        "store_initial": config.store_initial_structures,
        "rmsd_val": config.rmsd_value,
        "sampling_shell": config.sampling_shell,
        "sampling_condition": config.sampling_condition,
        "opt_lot": opt_lot,
        "logger": logger,
    }

    # Check if the OptimizationDataSets exist
    check_collection_existence(
        client, config.surface_model_collection, config.small_molecule_collection
    )
    ds_sm = qcf.get_collection(client, "OptimizationDataset", config.small_molecule_collection)
    ds_wc = qcf.get_collection(client, "OptimizationDataset", config.surface_model_collection)

    # Check if all the molecules are optimized at the requested level of theory
    if len(qcf.fetch_initial_molecule(ds_sm, smol_name, opt_lot).symbols) == 1:
        check_optimized_molecule(ds_wc, opt_lot, ds_wc.data.records.keys())
        args_dict["target_mol"] = qcf.fetch_initial_molecule(ds_sm, smol_name, opt_lot)
    else:
        check_optimized_molecule(ds_sm, opt_lot, [smol_name])
        check_optimized_molecule(ds_wc, opt_lot, ds_wc.data.records.keys())
        args_dict["target_mol"] = qcf.fetch_final_molecule(ds_sm, smol_name, opt_lot)

    args_dict["client"] = client

    count = 0
    logger.info(
        sampling_model_msg(config.surface_model_collection, smol_name, method, basis, program)
    )

    for c, w in enumerate(ds_wc.data.records):
        args_dict["cluster"] = qcf.fetch_final_molecule(ds_wc, w, opt_lot)

        ref_opt_dset_name = smol_name + "_" + w
        smplg_opt_dset_name = "pre_" + ref_opt_dset_name

        ds_smplg = qcf.get_or_create_opt_dataset(client, smplg_opt_dset_name)
        ds_ref = qcf.get_or_create_opt_dataset(client, ref_opt_dset_name)
        args_dict["sampling_opt_dset"] = ds_smplg
        args_dict["refinement_opt_dset"] = ds_ref

        padded_log(logger, f"Processing cluster: {w}", padding_char=gear, total_length=80)
        logger.info(sampling_round_msg(smplg_opt_dset_name, ref_opt_dset_name))

        debug_path = Path("./site_finder/" + str(smol_name) + "_w/" + w)
        if not debug_path.exists() and config.store_initial_structures:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
        args_dict["debug_path"] = debug_path

        run_sampling(**args_dict)

        process_refinement(
            client, ropt_lot, rmethod, rbasis, rprogram,
            qc_keyword, ds_ref, logger, config.refinement_tag,
        )

        ds_ref = qcf.get_or_create_opt_dataset(client, ref_opt_dset_name)
        count += len(ds_ref.data.records)
        logger.info(f"\nFinished sampling of cluster {w} number of binding sites: {len(ds_ref.data.records)}")
        logger.info(f"\nTotal number of binding sites thus far: {count}\n\n")

        if count > config.total_binding_sites:
            logger.info(f"Thank you for using BEEP binding site sampler. Total binding sites : {count}\n")
            break

    logger.info(f"Thank you for using BEEP binding site sampler. Total binding sites : {count}\n")
