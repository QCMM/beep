import sys, time, argparse, logging
from pathlib import Path
import qcfractal.interface as ptl
from beep.sampling import run_sampling
from beep.errors import DatasetNotFound, LevelOfTheoryNotFound

welcome_msg = """       
                  Welcome to the BEEP binding sites sampler! 

Description: The BEEP binding sites sampler optimizes random initial configurations
of a small molecule around a set-of-clusters surface model,  until a default of 250 binding 
sites are found..

Author: svogt, gbovolenta

            """


def sampling_model_msg(surface_model, target_mol, method, basis, program):
    return """
------------------------------------------------------------------------------
Starting the sampling of the surface model {0} with the molecule {1}
The sampling level of theory is method : {2} basis: {3} program {4}
------------------------------------------------------------------------------

    """.format(
        surface_model, target_mol, method, basis, program
    )


def sampling_round_msg(name_cluster, opt_smplg, opt_refine):
    return """
##############################################################################
Processing cluster: {}
The OptimizationDataset for the sampling results: {}
The OptimiztionDataset for the unique structures for refinement: {}

    """.format(
        name_cluster, opt_smplg, opt_refine
    )


def parse_arguments():
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
        "--sampling-tag",
        type=str,
        default="sampling",
        help="The tag to use to specify the qcfractal-manager for the sampling optimization (default: sampling)",
    )
    parser.add_argument(
        "--keyword-id",
        type=int,
        default=None,
        help="ID of the QC keywords for the OptimizationDataSet specification of the sampling (default: None)",
    )

    return parser.parse_args()


def sampling_args(args):
    return {
        "method": args.sampling_level_of_theory[0],
        "basis": args.sampling_level_of_theory[1],
        "program": args.sampling_level_of_theory[2],
        "tag": args.sampling_tag,
        "kw_id": args.keyword_id,
        "opt_lot": args.sampling_level_of_theory[0]
        + "_"
        + args.sampling_level_of_theory[1],
        "rmsd_symm": args.rmsd_symmetry,
        "rmsd_val": args.rmsd_value,
        "sampling_shell": args.sampling_shell,
        "sampling_condition": args.sampling_condition,
    }


def check_collection_existence(
    client, *collections, collection_type="OptimizationDataset"
):
    """
    Check the existence of collections and raise DatasetNotFound error if not found.

    Args:
    - client: The client used for collection retrieval.
    - *collections: Variable number of collection names to check for existence.

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


def check_optimized_molecule(ds, opt_lot, mol_names):
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


def get_or_create_opt_collection(client, dset_name):
    try:
        ds_opt = client.get_collection("OptimizationDataset", dset_name)
        out_string = f"OptimizationDataset {dset_name} already exists, new sampled structures will be saved here."
    except KeyError:
        ds_opt = ptl.collections.OptimizationDataset(dset_name, client=client)
        ds_opt.save()
        ds_opt = client.get_collection("OptimizationDataset", dset_name)
        out_string = f"Creating new OptimizationDataset: {dset_name}."
    return ds_opt


def process_refinement(client, refine_lot, ds_opt):
    logger = logging.getLogger("beep_logger")
    method, basis, program = refine_lot
    spec = {
        "name": method + "_" + basis,
        "description": f"Geometric + {method}/{basis}/{program}",
        "optimization_spec": {"program": "geometric", "keywords": None},
        "qc_spec": {
            "driver": "gradient",
            "method": method,
            "basis": basis,
            "keywords": None,
            "program": program,
        },
    }
    ds_opt.add_specification(**spec, overwrite=True)
    ds_opt.save()
    num_of_bs = len(ds_opt.data.records)
    logger.info(f"{num_of_bs} unique binding sites found in this round.")
    c = ds_opt.compute(method + "_" + basis, tag="refinement")
    logger.info(f"Sending {c} optimizations at {refine_lot} level.")


def main():
    # Call the arguments
    args = parse_arguments()

    # Create a logger
    logger = logging.getLogger("beep_logger")
    # logger.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)

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
    method = args.sampling_level_of_theory[0]
    basis = args.sampling_level_of_theory[1]
    program = args.sampling_level_of_theory[2]
    opt_lot = method + "_" + basis

    # Check if the OptimizationDataSets exist
    check_collection_existence(
        client, args.surface_model_collection, args.small_molecule_collection
    )
    ds_sm = client.get_collection("OptimizationDataset", args.small_molecule_collection)
    ds_wc = client.get_collection("OptimizationDataset", args.surface_model_collection)

    # Check if all the molecules are optimized at the requested level of theory
    check_optimized_molecule(ds_sm, opt_lot, [smol_name])
    check_optimized_molecule(ds_wc, opt_lot, ds_wc.data.records.keys())

    args_dict = sampling_args(args)
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

        try:
            old_opt_num = len(ds_ref.data.records)
            logger.debug(
                f"Number of existing binding sites for this cluster: {old_opt_num}"
            )
            count = count + old_opt_num
        except KeyError:
            pass

        # Logging
        out_file = Path(
            "./site_finder/" + str(smol_name) + "_w/" + w + "/out_sampl.dat"
        )
        if not out_file.is_file():
            out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            f.write(welcome_msg)
        args_dict["o_file"] = out_file

        run_sampling(**args_dict)

        logger.info(f"\n\nTotal number of binding sites so far: {count}")
        r_lot = args.refinement_level_of_theory
        process_refinement(client, r_lot, ds_ref)
        count = count + int(c)
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
