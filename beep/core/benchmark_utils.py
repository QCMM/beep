"""
Benchmark utilities — pure computation functions shared by geometry and energy benchmarks.

No QCFractal server dependencies.
"""
import pandas as pd
from qcelemental.models.molecule import Molecule


def create_benchmark_dataset_dict(benchmark_structs):
    """Build {struct_name: 'mol_surf'} dict from benchmark structure names like 'mol_surf_001'."""
    dataset_dict = {}
    for bchmk_struc_name in benchmark_structs:
        mol, surf, _ = bchmk_struc_name.split("_")
        dataset_dict[bchmk_struc_name] = f"{mol}_{surf}"
    return dataset_dict


def create_molecular_fragments(mol, len_f1):
    """Split a molecule into two fragments at the len_f1 boundary."""
    geom = mol.geometry.flatten()
    symbols = mol.symbols
    f_mol = Molecule(
        symbols=symbols,
        geometry=geom,
        fragments=[
            list(range(0, len_f1)),
            list(range(len_f1, len(symbols))),
        ],
    )
    f1_mol = f_mol.get_fragment(0)
    f2_mol = f_mol.get_fragment(1)
    return f1_mol, f2_mol


def get_errors_dataframe(df, ref_en_dict):
    """Compute absolute and relative error DataFrames against reference energies."""
    def construct_key(index):
        return "_".join(index.split("_")[:3])

    df = df[df.index.map(construct_key).isin(ref_en_dict.keys())]

    abs_error_df = pd.DataFrame(index=df.index, columns=df.columns)
    rel_error_df = pd.DataFrame(index=df.index, columns=df.columns)

    for row_index in df.index:
        ref_value = ref_en_dict["_".join(row_index.split("_")[:3])]
        for col in df.columns:
            abs_error = df.at[row_index, col] - ref_value
            abs_error_df.at[row_index, col] = abs_error
            rel_error_df.at[row_index, col] = abs_error / ref_value

    return abs_error_df, rel_error_df


def compute_rmsd(mol1, mol2, rmsd_symm):
    """Compute RMSD between two molecules, optionally considering mirror symmetry."""
    rmsd_val_mirror = 10.0
    if rmsd_symm:
        align_mols_mirror = mol1.align(mol2, run_mirror=True)
        rmsd_val_mirror = align_mols_mirror[1]["rmsd"]
    align_mols = mol1.align(mol2, atoms_map=True)
    rmsd_val = align_mols[1]["rmsd"]

    if rmsd_val < rmsd_val_mirror:
        return rmsd_val
    else:
        return rmsd_val_mirror
