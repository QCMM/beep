import sys, time, argparse, logging
import numpy as np
from pathlib import Path
import qcelemental as qcel
from cbs_extrapolation import *
from typing import Dict, Union, List, Tuple
from collections import Counter
import qcfractal.interface as ptl
from qcfractal.interface.collections.optimization_dataset import OptimizationDataset
from qcfractal.interface.collections.dataset import Dataset
from qcfractal.interface.collections.reaction_dataset import ReactionDataset
from qcfractal.interface.client import FractalClient
from qcelemental.models.molecule import Molecule
from beep.sampling import run_sampling
from beep.errors import DatasetNotFound, LevelOfTheoryNotFound


welcome_msg = """       
                  Welcome to the BEEP binding sites sampler! 

Description: The BEEP binding sites sampler optimizes random initial configurations
of a small molecule around a set-of-clusters surface model,  until a default of 250 binding 
sites are found..

Author: svogt, gbovolenta
            """


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the script.

    Returns:
    - Namespace containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="""
A command line interface to sample the surface of a set of water clusters (stored in a 
QCFractal DataSet) with a small molecule or atom. This CLI is part
of the Binding Energy Evaluation Platform (BEEP).
    """
    )
    parser.add_argument(
        "--client_address",
        default="localhost:7777",
        help="The URL address and port of the QCFractal server (default: localhost:7777)",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="The username for the database client (Default = None)",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="The password for the database client (Default = None)",
    )
    parser.add_argument(
        "--benchmark-structures",
        type=str,
        nargs="+",
        help="The name of the structures to use for the benchmark",
    )
    parser.add_argument(
        "--basis-set",
        type=str,
        nargs="+",
        help="One or more basis sets of the benchmarked DFT methods",
    )
    parser.add_argument(
        "--small-molecule-collection",
        default="Small_molecules",
        help="The name of the collection containing molecules or radicals (default: Small_molecules)",
    )
    parser.add_argument(
        "--molecule",
        required=True,
        help="The name of the molecule to be sampled (from a QCFractal OptimizationDataSet collection)",
    )
    parser.add_argument(
        "--surface-model-collection",
        default="small water",
        help="The name of the collection with the set of water clusters (default: small_water)",
    )
    parser.add_argument(
        "--reference-geometry-level-of-theory",
        nargs=3,
        default=["df-ccsd(t)", "cc-pvdz", "molpro"],
        help="The level of theory in the format: method basis program (default: df-ccsd(t) cc-pvdz molpro)",
    )
    parser.add_argument(
        "--reference-energy-level-of-theory",
        nargs=3,
        default=["ccsd(t)", "cbs", "psi4"],
        help="The level of theory and program for the geometry refinement in the format: method_basis (default: df-ccsd(t) cc-pvdz molpro)",
    )
    parser.add_argument(
        "--rmsd-value",
        type=float,
        default=0.1,
        help="RMSD geometrical criteria, all structures below this value will not be considered for energy computation. (default: 0.10 angstrom)",
    )
    parser.add_argument(
        "--rmsd-symmetry",
        action="store_true",
        help="Consider the molecular symmetry for the RMSD calculation",
    )
    return parser.parse_args()


def get_or_create_collection(client: FractalClient, dset_name: str, collection_type):
    """
    Get or create an optimization dataset collection.

    Args:
    - client: Fractal client instance to use.
    - dset_name: Name of the OptimizationDataset.

    Returns:
    - An instance of the OptimizationDataset.
    """
    try:
        ds_opt = client.get_collection(collection_type.__name__, dset_name)
        out_string = f"OptimizationDataset {dset_name} already exists, new sampled structures will be saved here."
    except KeyError:
        ds_opt = collection_type(dset_name, client=client)
        ds_opt.save()
        ds_opt = client.get_collection(collection_type.__name__, dset_name)
        out_string = f"Creating new OptimizationDataset: {dset_name}."
    return ds_opt


def check_collection_existence(
    client: FractalClient,
    *collections: List,
    collection_type: str = "OptimizationDataset",
):
    """
    Check the existence of collections and raise DatasetNotFound error if not found.

    Args:
    - client: QCFractal client object
    - *collections: List of QCFractal Datasets.
    - collection_type: type of Optimization Dataset

    Raises:
    - DatasetNotFound: If any of the specified collections do not exist.
    """
    for collection in collections:
        try:
            client.get_collection(collection_type, collection)
        except KeyError:
            raise DatasetNotFound(
                f"Collection {collection} does not exist. Please create it first. Exiting..."
            )


def check_dataset_status(client, dataset, cbs_list):
    while True:
        status_counts = {
            method: {"COMPLETE": 0, "INCOMPLETE": 0, "ERROR": 0}
            for method in set(lot.split("_")[0] for lot in cbs_list)
        }

        for lot in cbs_list:
            method, basis = lot.split("_")
            df = dataset.get_records(method=method, basis=basis, program="psi4")

            # Count the statuses for the current method
            for index, row in df.iterrows():
                status = row["record"].status.upper()
                if status not in status_counts[method]:
                    continue  # If status is not one of the expected statuses, skip it

                status_counts[method][status] += 1

                if status == "ERROR":
                    raise Exception(
                        f"Error in record {index} with level of theory {lot}"
                    )

        # Print the summary counts for each method
        for method, counts in status_counts.items():
            print(
                f"{method}: {counts['INCOMPLETE']} INCOMPLETE, {counts['COMPLETE']} COMPLETE, {counts['ERROR']} ERROR"
            )

        # Check if all records are complete or any record has an error
        all_complete = all(
            counts["INCOMPLETE"] == 0 and counts["ERROR"] == 0
            for counts in status_counts.values()
        )
        if all_complete:
            print("All records are COMPLETE.")
            break
        if any(counts["ERROR"] > 0 for counts in status_counts.values()):
            print("There are records with ERROR.")
            break

        time.sleep(600)  # Wait for 600 seconds before the next check


def wait_for_completion(
    odset_dict: Dict[str, "OptimizationDataset"],
    opt_lot: Union[str, List[str]],
    wait_interval: int = 600,
    check_errors: bool = False,
) -> str:
    """
    Waits indefinitely for all entries in the given datasets to complete, with an optional check for errors.
    Returns a summary of job statuses upon completion. Can handle a single optimization lot or a list of them.

    Args:
        odset_dict (Dict[str, "OptimizationDataset"]): Dictionary with structure names as keys and OptimizationDatasets as values.
        opt_lot (Union[str, List[str]]): The specification (level of theory) to check. Can be a string or a list of strings.
        wait_interval (int): Interval in seconds between status checks.
        check_errors (bool): If True, raise an error if any entry has a status of 'ERROR'.

    Raises:
        RuntimeError: If check_errors is True and any entry has a status of 'ERROR'.

    Returns:
        str: Summary message indicating the count of each job status.
    """
    if isinstance(opt_lot, str):
        opt_lot = [opt_lot]

    while True:
        statuses = []
        for lot in opt_lot:
            for struct_name, odset in odset_dict.items():
                status = odset.get_record(struct_name, specification=lot).status
                statuses.append(status)

                if status == "ERROR" and check_errors:
                    raise RuntimeError(
                        f"Error encountered in computation for {struct_name} with spec '{lot}'"
                    )

        status_counts = Counter(statuses)

        if status_counts["INCOMPLETE"] == 0:
            status_message = ", ".join(
                [f"{status}: {count}" for status, count in status_counts.items()]
            )
            print("All entries have been processed.")
            return f"Job Status Summary: {status_message}"

        print(
            f"Waiting for {wait_interval} seconds before rechecking statuses... (Incomplete: {status_counts['INCOMPLETE']})"
        )
        time.sleep(wait_interval)


def compute_rmsd(
    mol1: Molecule, mol2: Molecule, rmsd_symm: bool
) -> Tuple[float, float]:
    rmsd_val_mirror = 10.0
    if rmsd_symm:
        align_mols_mirror = mol1.align(mol2, run_mirror=True)
        rmsd_val_mirror = align_mols_mirror[1]["rmsd"]
    align_mols = mol1.align(mol2, atoms_map=True)
    rmsd_val = align_mols[1]["rmsd"]
    return rmsd_val, rmsd_val_mirror


def create_be_stoichiometry(odset, bench_struct, lot_geom):
    mol_name, surf_name, _ = bench_struct.split("_")
    bench_mol = (
        odset[mol_name.upper()]
        .get_record(name=mol_name.upper(), specification=lot_geom)
        .get_final_molecule()
    )  ##  m_2

    bench_struc_mol = (
        odset[bench_struct]
        .get_record(name=bench_struct, specification=lot_geom)
        .get_final_molecule()
    )  ## mol, g, s
    bench_geom = bench_struc_mol.geometry.flatten()
    bench_symbols = bench_struc_mol.symbols

    surf_mod_mol = (
        odset[surf_name.upper()]
        .get_record(name=surf_name.upper(), specification=lot_geom)
        .get_final_molecule()
    )  ## m_1
    surf_symbols = surf_mod_mol.symbols

    f_bench_struc_mol = ptl.Molecule(
        symbols=bench_symbols,
        geometry=bench_geom,
        fragments=[
            list(range(0, len(surf_symbols))),
            list(range(len(surf_symbols), len(bench_symbols))),
        ],
    )

    j5 = f_bench_struc_mol.get_fragment(0)
    j4 = f_bench_struc_mol.get_fragment(1)
    j7 = f_bench_struc_mol.get_fragment(0, 1)
    j6 = f_bench_struc_mol.get_fragment(1, 0)

    be_stoic = {
        "default": [
            (f_bench_struc_mol, 1.0),
            (j4, 1.0),
            (j5, 1.0),
            (j7, -1.0),
            (j6, -1.0),
            (surf_mod_mol, -1.0),
            (bench_mol, -1.0),
        ],
        "be_nocp": [
            (f_bench_struc_mol, 1.0),
            (surf_mod_mol, -1.0),
            (bench_mol, -1.0),
        ],
        "ie": [(f_bench_struc_mol, 1.0), (j7, -1.0), (j6, -1.0)],
        "de": [(surf_mod_mol, 1.0), (bench_mol, 1.0), (j4, -1.0), (j5, -1.0)],
    }
    return be_stoic


def get_cbs_energy(ds: Dataset, struct, cbs_lot_list):
    cbs_lot_en = {}
    for lot in cbs_lot_list:
        method, basis = lot.split("_")
        if not "ccsd" in method:
            rec = ds.get_records(method=method, basis=basis, program="Psi4").loc[
                struct
            ][0]
            cbs_lot_en[lot] = rec.return_result
        else:
            rec = ds.get_records(method=method, basis=basis, program="Psi4").loc[
                struct
            ][0]
            cbs_lot_en["mp2_" + basis] = rec.dict()["extras"]["qcvars"][
                "MP2 TOTAL ENERGY"
            ]
            cbs_lot_en["ccsd_" + basis] = rec.dict()["extras"]["qcvars"][
                "CCSD TOTAL ENERGY"
            ]
            cbs_lot_en[lot] = rec.return_result

    ## Extrapolations using the dictionary:
    scf_dtq = scf_xtpl_helgaker_3(
        "scf_ext",
        2,
        cbs_lot_en["scf_aug-cc-pVDZ"],
        3,
        cbs_lot_en["scf_aug-cc-pVTZ"],
        4,
        cbs_lot_en["scf_aug-cc-pVQZ"],
    )
    scf_tq = corl_xtpl_helgaker_2(
        "scf_ext1", 3, cbs_lot_en["scf_aug-cc-pVTZ"], 4, cbs_lot_en["scf_aug-cc-pVQZ"]
    )
    mp2_dt = corl_xtpl_helgaker_2(
        "mp2_ext", 2, cbs_lot_en["mp2_aug-cc-pVDZ"], 3, cbs_lot_en["mp2_aug-cc-pVTZ"]
    )
    mp2_tq = corl_xtpl_helgaker_2(
        "mp2_ext", 3, cbs_lot_en["mp2_aug-cc-pVTZ"], 4, cbs_lot_en["mp2_aug-cc-pVQZ"]
    )
    ccsd_dt = corl_xtpl_helgaker_2(
        "xtpl", 2, cbs_lot_en["ccsd_aug-cc-pVDZ"], 3, cbs_lot_en["ccsd_aug-cc-pVTZ"]
    )
    ccsdt_dt = corl_xtpl_helgaker_2(
        "xtpl", 2, cbs_lot_en["ccsd(t)_aug-cc-pVDZ"], 3, cbs_lot_en["ccsd(t)_aug-cc-pVTZ"]
    )

    ## Contributions:

    mp2_cbs_corr = mp2_tq - scf_tq
    ccsd_cbs_corr = ccsd_dt - mp2_dt
    ccsdt_cbs_corr = ccsdt_dt - ccsd_dt

    return scf_dtq + mp2_cbs_corr + ccsd_cbs_corr + ccsdt_cbs_corr


def main():
    # Call the arguments
    args = parse_arguments()

    # Create a logger
    logger = logging.getLogger("beep_sampling")
    logger.setLevel(logging.INFO)

    # File handler for logging to a file
    log_file = "bchmk_" + args.molecule + ".log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    # Console handler for logging to stdout
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    logger.info(welcome_msg)

    client = ptl.FractalClient(
        address=args.client_address,
        verify=False,
        username=args.username,
        password=args.password,
    )

    # The name of the molecule to be sampled at level of theory opt_lot
    smol_name = args.molecule
    gr_method, gr_basis, gr_program = args.reference_geometry_level_of_theory
    geom_ref_opt_lot = gr_method + "_" + gr_basis

    bchmk_structs = args.benchmark_structures
    surf_dset_name = args.surface_model_collection
    smol_dset_name = args.small_molecule_collection

    # Check if the OptimizationDataSets exist
    w_list = []
    odset_dict = {}
    bchmk_dset_names = {}

    # Save name of collections with the Benchmark strcutrues in a list
    for bchmk_struc_name in bchmk_structs:
        mol, surf, _ = bchmk_struc_name.split("_")
        bchmk_dset_names[bchmk_struc_name] = mol + "_" + surf

    # check existance of the collections

    check_collection_existence(client, *bchmk_dset_names.values())
    check_collection_existence(client, smol_dset_name)
    check_collection_existence(client, surf_dset_name)

    # Save collections in a dictionary

    odset_dict[smol_name] = client.get_collection("OptimizationDataset", smol_dset_name)

    for bchmk_struct_name, odset_name in bchmk_dset_names.items():
        odset_dict[bchmk_struct_name] = client.get_collection(
            "OptimizationDataset", odset_name
        )
        surf_mod = bchmk_struct_name.split("_")[1].upper()

        odset_dict[surf_mod] = client.get_collection(
            "OptimizationDataset", surf_dset_name
        )

    cc_spec = {
        "name": geom_ref_opt_lot,
        "description": "Geometric + " + gr_program + "/" + gr_method + "/" + gr_basis,
        "optimization_spec": {"program": "geometric", "keywords": None},
        "qc_spec": {
            "driver": "gradient",
            "method": gr_method,
            "basis": gr_basis,
            "keywords": None,
            "program": gr_program,
        },
    }

    for odset in odset_dict.values():
        odset.add_specification(**cc_spec, overwrite=True)
        odset.save()

    # Optimize all three molecules at the reference benchmark level of theory

    ct = 0

    for struct_name, odset in odset_dict.items():
        c1 = odset.compute(geom_ref_opt_lot, tag="ccsd_opt", subset={struct_name})
        ct += c1

    print(f"Send a total of {ct} structures to compute")

    ## Wait for the Opimizations to finish
    try:
        wait_for_completion(
            odset_dict, geom_ref_opt_lot, wait_interval=600, check_errors=True
        )
        print("Continuing with the script...")
        # Continue with the rest of the script
    except RuntimeError as e:
        print(str(e))

    ## Optimize with DFT functionals
    geom_dft = [
        "M05-2X",
        "MPWB1K-D3BJ",
        "PWB6K-D3BJ",
        "WPBE-D3BJ",
        "CAM-B3LYP-D3BJ",
        "WB97X-D3BJ",
        "B3LYP-D3BJ",
        "PBE0-D3BJ",
    ]
    geom_sqm = ["HF3C", "PBEh3c"]

    all_geom_method = geom_dft  # + geom_sqm
    geom_basis = args.basis_set

    ct = 0
    dft_lot = []
    for struct_name, odset in odset_dict.items():
        for method in geom_dft:
            for basis in geom_basis:
                spec_name = method + "_" + basis
                dft_lot.append(spec_name)
                spec = {
                    "name": method + "_" + basis,
                    "description": "Geometric + Psi4/" + method + "/" + basis,
                    "optimization_spec": {"program": "geometric", "keywords": None},
                    "qc_spec": {
                        "driver": "gradient",
                        "method": method,
                        "basis": basis,
                        "keywords": None,
                        "program": "psi4",
                    },
                }

                # try:
                #    del odset.data.specs[spec_name]
                #    odset.save()
                # except KeyError as e:
                #    print(e)
                odset.add_specification(**spec, overwrite=True)
                odset.save()
                c = odset.compute(spec_name, tag="bench_dft", subset={struct_name})
                ct += c

    print(f"Computed {ct} DFT jobs")
    wait_for_completion(odset_dict, dft_lot, wait_interval=200, check_errors=False)
    print("Continuing with the script...")

    ## Get final molecules for completed records and compute the RMSD for all structures
    # Create a dictionary ref_geom_mols = {"name" : FinalMolecule}

    # Save optimized molecules of the reference structures
    ref_geom_fmols = {}
    for struct_name, odset in odset_dict.items():
        record = odset.get_record(struct_name, specification=geom_ref_opt_lot)
        ref_geom_fmols[struct_name] = record.get_final_molecule()

    # Compute RMSD
    dft_geom_fmols = {}
    final_opt_lot = {}
    for opt_lot in dft_lot:
        rmsd_tot_dict = {}
        rmsd_tot_mirror = []
        for struct_name, odset in odset_dict.items():
            record = odset.get_record(struct_name, specification=opt_lot)
            fmol = record.get_final_molecule()
            rmsd, rmsd_mirror = compute_rmsd(
                ref_geom_fmols[struct_name], fmol, rmsd_symm=True
            )
            rmsd_tot_dict[struct_name] = rmsd
            rmsd_tot_mirror.append(rmsd_mirror)
        rmsd_tot = list(rmsd_tot_dict.values())
        if np.mean(rmsd_tot) < 0.15:
            final_opt_lot[opt_lot] = np.mean(rmsd_tot)
        elif np.mean(rmsd_tot_mirror) < 0.15:
            final_opt_lot[opt_lot] = np.mean(rmsd_tot_mirror)
    print(final_opt_lot)
    print(rmsd_tot_dict)
    print(rmsd_tot_mirror)
    print(len(final_opt_lot.values()))

    # Compute the reference energy at CCSD(T)/CBS
    cbs_list = [
        "scf_aug-cc-pVDZ",
        "scf_aug-cc-pVTZ",
        "scf_aug-cc-pVQZ",
        "mp2_aug-cc-pVQZ",
        "ccsd(t)_aug-cc-pVDZ",
        "ccsd(t)_aug-cc-pVTZ",
    ]

    cbs_col = get_or_create_collection(
        client, "cbs" + "_" + smol_name + "_" + surf_dset_name, Dataset
    )
    print(cbs_col.name)
    print(ref_geom_fmols.items())

    for name, fmol in ref_geom_fmols.items():
        try:
            cbs_col.add_entry(name, fmol)
        except KeyError:
            print(f"Entry {name} already in Dataset {cbs_col.name}")
        cbs_col.save()

    ct = 0
    all_cbs_ids = []
    for lot in cbs_list:
        c = cbs_col.compute(
            lot.split("_")[0], lot.split("_")[1], tag="cbs_en", program="psi4"
        )
        all_cbs_ids.extend(c.ids)

    check_dataset_status(client, cbs_col, cbs_list)

    # Get reference energy dict:
    ref_be = {}
    for bench_struct in bchmk_structs:
        mol_name, surf_name, _ = bench_struct.split("_")
        mol_cbs_en = get_cbs_energy(cbs_col, mol_name.upper(), cbs_list)
        surf_cbs_en = get_cbs_energy(cbs_col, surf_name.upper(), cbs_list)
        struct_cbs_en = get_cbs_energy(cbs_col, bench_struct, cbs_list)
        ref_be[bench_struct] = (struct_cbs_en - (mol_cbs_en + surf_cbs_en))*qcel.constants.hartree2kcalmol
    print(ref_be)
    return None

    # Create or get bench_be dataset
    rdset_name = "bchmk_be_" + smol_name + "_" + surf_dset_name
    try:
        ds_be = ReactionDataset(
            rdset_name, ds_type="rxn", client=client, default_program="psi4"
        )
        ds_be.save()
    except KeyError:
        ds_be = client.get_collection("ReactionDataset", rdset_name)

    for bench_struct in bchmk_structs:
        for lot in final_opt_lot:
            be_stoich = create_be_stoichiometry(odset_dict, bench_struct, lot)
            bench_entry = bench_struct + "_" + lot
            try:
                ds_be.add_rxn(bench_entry, be_stoich)
            except KeyError:
                continue
    ds_be.save()

    # Compute BE for all functionals

    import dft_functionals

    hybrid_gga = dft_functionals.hybrid_gga()
    lrc = dft_functionals.lrc()
    meta_hybrid_gga = dft_functionals.meta_hybrid_gga()

    all_dft = hybrid_gga + lrc + meta_hybrid_gga
    stoich_list = ["default", "de", "ie", "be_nocp"]

    c_list = []
    for func in all_dft:
        for stoich in stoich_list:
            c = ds_be.compute(
                method=func,
                basis="def2-tzvp",
                program="psi4",
                stoich=stoich,
                tag="bench_dft",
            )
            c_list.extend(c)
    print(f"Sumbited a total of {len(c_list)} DFT computations")


if __name__ == "__main__":
    main()
