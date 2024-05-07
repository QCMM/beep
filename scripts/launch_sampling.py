import sys, time, argparse, logging
from pathlib import Path
from typing import Dict, Tuple, List
import qcfractal.interface as ptl
from qcfractal.interface.collections.optimization_dataset import OptimizationDataset
from qcfractal.interface.client import FractalClient
from beep.sampling import run_sampling
from beep.errors import DatasetNotFound, LevelOfTheoryNotFound

welcome_msg = """       
                  Welcome to the BEEP binding sites sampler! 

Description: The BEEP binding sites sampler optimizes random initial configurations
of a small molecule around a set-of-clusters surface model,  until a default of 250 binding 
sites are found..

Author: svogt, gbovolenta
            """


def sampling_model_msg(
    surface_model: str, target_mol: str, method: str, basis: str, program: str
) -> str:
    """
    Format a message for the start of sampling of a surface model.

    Args:
    - surface_model: Name of the surface model.
    - target_mol: Name of the target molecule.
    - method: Computational method used.
    - basis: Basis set used.
    - program: Program used for computation.

    Returns:
    - Formatted message string.
    """
    return f"""
-----------------------------------------------------------------------------------------
Starting the sampling of the surface model {surface_model} with the molecule {target_mol}
The sampling level of theory is method: {method} basis: {basis} program: {program}
-----------------------------------------------------------------------------------------
    """


def sampling_round_msg(name_cluster: str, opt_smplg: str, opt_refine: str) -> str:
    """
    Format a message for the processing of a cluster round.

    Args:
    - name_cluster: Name of the cluster being processed.
    - opt_smplg: Name of the OptimizationDataset for sampling results.
    - opt_refine: Name of the OptimizationDataset for unique structures for refinement.

    Returns:
    - Formatted message string.
    """
    return f"""##############################################################################
Processing cluster: {name_cluster}
The OptimizationDataset for the sampling results: {opt_smplg}
The OptimizationDataset for the unique structures for refinement: {opt_refine}
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
        "--molecule",
        required=True,
        help="The name of the molecule to be sampled (from a QCFractal OptimizationDataSet collection)",
    )
    parser.add_argument(
        "--surface-model-collection",
        default="Water_22",
        help="The name of the collection with the set of water clusters (default: Water_22)",
    )
    parser.add_argument(
        "--small-molecule-collection",
        default="Small_molecules",
        help="The name of the collection containing molecules or radicals (default: Small_molecules)",
    )
    parser.add_argument(
        "--sampling-shell",
        type=float,
        default=2.0,
        help="The shell size of the sampling space in Angstrom (Default = 2.0)",
    )
    parser.add_argument(
        "--total-binding-sites",
        type=int,
        default=220,
        help="The total binding sites for the distribution (Default = 250)",
    )
    parser.add_argument(
        "--sampling-condition",
        type=str,
        default="normal",
        help="How tight the sampling should be done for each surface. Options: sparse, normal, fine (Default: normal)",
    )
    parser.add_argument(
        "--sampling-level-of-theory",
        nargs=3,
        default=["blyp", "def2-svp", "terachem"],
        help="The level of theory in the format: method basis program (default: blyp def2-svp terachem)",
    )
    parser.add_argument(
        "--refinement-level-of-theory",
        nargs=3,
        default=["bhandhlyp", "def2-svp", "psi4"],
        help="The level of theory and program for the geometry refinement in the format: method_basis (default: bhandhlyp def2-svp psi4)",
    )
    parser.add_argument(
        "--rmsd-value",
        type=float,
        default=0.40,
        help="RMSD geometrical criteria, all structures below this value will not be considered as unique. (default: 0.40 angstrom)",
    )
    parser.add_argument(
        "--rmsd-symmetry",
        action="store_true",
        help="Consider the molecular symmetry for the RMSD calculation",
    )
    parser.add_argument(
        "--store-initial-structures",
        action="store_true",
        help="Save the initial structures of the sampling procedure in the site_finder folder for visualization",
    )
    parser.add_argument(
        "--sampling-tag",
        type=str,
        default="sampling",
        help="The tag to use to specify the qcfractal-manager for the sampling optimization (default: sampling)",
    )
    parser.add_argument(
        "--refinement-tag",
        type=str,
        default="refinement",
        help="The tag to use to specify the qcfractal-manager for the refinement optimization (default: refinement)",
    )
    parser.add_argument(
        "--keyword-id",
        type=int,
        default=None,
        help="ID of the QCFractal keyword for the OptimizationDataSet specification (default: None)",
    )

    return parser.parse_args()


def sampling_args(args: argparse.Namespace) -> Dict[str, any]:
    """
    Create a dictionary of sampling arguments.

    Args:
    - args: Parsed command line arguments.

    Returns:
    - Dictionary of arguments for sampling.
    """
    return {
        "method": args.sampling_level_of_theory[0],
        "basis": args.sampling_level_of_theory[1],
        "program": args.sampling_level_of_theory[2],
        "tag": args.sampling_tag,
        "kw_id": args.keyword_id,
        #"opt_lot": args.sampling_level_of_theory[0]
        #+ "_"
        #+ args.sampling_level_of_theory[1],
        "rmsd_symm": args.rmsd_symmetry,
        "store_initial": args.store_initial_structures,
        "rmsd_val": args.rmsd_value,
        "sampling_shell": args.sampling_shell,
        "sampling_condition": args.sampling_condition,
    }


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


def check_optimized_molecule(
    ds: OptimizationDataset, opt_lot: str, mol_names: List[str]
) -> None:
    """
    Check if all molecules are optimized at the requested level of theory.

    Args:
    - ds: OptimizationDataset containing the optimization records.
    - opt_lot: Level of theory string.
    - mol_names: List of molecule names to check.

    Raises:
    - LevelOfTheoryNotFound: If the level of theory for a molecule or the entry itself does not exist.
    - ValueError: If optimizations are incomplete or encountered an error.
    """
    for mol in list(mol_names):
        try:
            rr = ds.get_record(mol, opt_lot)
        except KeyError:
            raise LevelOfTheoryNotFound(
                f"{opt_lot} level of theory for {mol} or the entry itself does not exist. Add the molecule and optimize it first\n"
            )
        if rr.status == "INCOMPLETE":
            raise ValueError(f" Optimization has status {rr.status} restart it or wait")
        elif rr.status == "ERROR":
            raise ValueError(f" Optimization has status {rr.status} restart it or wait")


def get_or_create_opt_collection(
    client: FractalClient, dset_name: str
) -> OptimizationDataset:
    """
    Get or create an optimization dataset collection.
>>>>>>> c83f9f92b789c9a8ff37f1dd629119b6e00b8488

    Args:
    - client: Fractal client instance to use.
    - dset_name: Name of the OptimizationDataset.

    Returns:
    - An instance of the OptimizationDataset.
    """
    try:
        ds_opt = client.get_collection("OptimizationDataset", dset_name)
        out_string = f"OptimizationDataset {dset_name} already exists, new sampled structures will be saved here."
    except KeyError:
        ds_opt = ptl.collections.OptimizationDataset(dset_name, client=client)
        ds_opt.save()
        ds_opt = client.get_collection("OptimizationDataset", dset_name)
        out_string = f"Creating new OptimizationDataset: {dset_name}."
    return ds_opt


def process_refinement(
    client: FractalClient, ropt_lot_name: str, rmethod: str, rbasis: str, program: str , qc_keyword: int,  ds_opt: OptimizationDataset, rtag = "refinement", 
) -> None:
    """
    Process the refinement optimization at a given level of theory.

    Args:
    - client: Fractal client instance to use.
    - refine_lot: A tuple containing the refinement level of theory as (method, basis, program).
    - ds_opt: Optimizationdataset to contain the refinement optimizations.

    Other considerations:
    - Assumes all optimization entries in ds_opt are unique.

    Returns:
    - None, no return value, this function is expected to execute a procedure that does not need a return.
    """
    logger = logging.getLogger("beep_logger")
    #method, basis, program = refine_lot
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
    ds_opt.add_specification(**spec, overwrite=True)
    ds_opt.save()
    c = ds_opt.compute(ropt_lot_name, tag=rtag)
    logger.info(f"Sending {c} refinement optimizations at {refine_lot} level.")
    return None


def main():
    # Call the arguments
    args = parse_arguments()

    # Create a logger
    logger = logging.getLogger("beep_sampling")
    logger.setLevel(logging.INFO)

    # File handler for logging to a file
    log_file = (
        "beep_sampling_" + args.molecule + "_" + args.surface_model_collection + ".log"
    )
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
    method, basis, program = args.sampling_level_of_theory
    rmethod, rbasis, rprogram  = args.refinement_level_of_theory

    qc_keyword = args.keyword_id
    opt_lot = method + "_" + basis
    ropt_lot = rmethod + "_" + rbasis
    args_dict = sampling_args(args)

    try:
        if ("uks" or "uhf") in client.query_keywords()[qc_keyword-1].values.values() and program == "psi4":
            opt_lot = "U"+opt_lot
            ropt_lot = "U"+opt_lot
    except TypeError:
        pass

    args_dict["opt_lot"] = opt_lot

    # Check if the OptimizationDataSets exist
    check_collection_existence(
        client, args.surface_model_collection, args.small_molecule_collection
    )
    ds_sm = client.get_collection("OptimizationDataset", args.small_molecule_collection)
    ds_wc = client.get_collection("OptimizationDataset", args.surface_model_collection)

    # Check if all the molecules are optimized at the requested level of theory
    check_optimized_molecule(ds_sm, opt_lot, [smol_name])
    check_optimized_molecule(ds_wc, opt_lot, ds_wc.data.records.keys())

    args_dict["target_mol"] = ds_sm.get_record(smol_name, opt_lot).get_final_molecule()
    args_dict["client"] = client

    count = 0

    logger.info(
        sampling_model_msg(
            args.surface_model_collection, smol_name, method, basis, program
        )
    )

    for c, w in enumerate(ds_wc.data.records):
        # Getting cluster record
        args_dict["cluster"] = ds_wc.get_record(w, opt_lot).get_final_molecule()

        # Defining collection names and passing the to argunment dict
        smol_name = args.molecule
        ref_opt_dset_name = smol_name + "_" + w
        smplg_opt_dset_name = "pre_" + ref_opt_dset_name

        # Retrieve or create the Sampling collection
        ds_smplg = get_or_create_opt_collection(client, smplg_opt_dset_name)
        ds_ref = get_or_create_opt_collection(client, ref_opt_dset_name)
        args_dict["sampling_opt_dset"] = ds_smplg
        args_dict["refinement_opt_dset"] = ds_ref

        ## logging info for sampling cluster
        logger.info(sampling_round_msg(w, smplg_opt_dset_name, ref_opt_dset_name))

        # Path for debugging
        debug_path = Path("./site_finder/" + str(smol_name) + "_w/" + w)
        if not debug_path.exists() and args.store_initial_structures:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
        args_dict["debug_path"] = debug_path

        # Calling run sampling function
        run_sampling(**args_dict)

        # Do the geometry refinement optimizations
        process_refinement(client, ropt_lot, rmethod, rbasis, rprogram, qc_keyword ,ds_ref ,args.refinement_tag)

        # Count number of structures for this model
        ds_ref = get_or_create_opt_collection(client, ref_opt_dset_name)
        count += len(ds_ref.data.records)

        logger.info(
            f"\nFinished sampling of cluster {w} number of binding sites: {len(ds_ref.data.records)}"
        )
        logger.info(f"\nTotal number of binding sites thus far: {count}\n\n")

        if count > args.total_binding_sites:
            logger.info(
                f"Thank you for using BEEP binding site sampler. Total binding sites : {count}\n"
            )
            break

    logger.info(
        f"Thank you for using BEEP binding site sampler. Total binding sites : {count}\n"
    )


if __name__ == "__main__":
    main()
