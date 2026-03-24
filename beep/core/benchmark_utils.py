"""
Benchmark utilities — pure computation functions shared by geometry and energy benchmarks.

No QCFractal server dependencies.
"""
import pandas as pd
from qcelemental.models.molecule import Molecule


def create_benchmark_dataset_dict(benchmark_structs):
    """Build {struct_name: dataset_name} dict from benchmark structure names.

    The last '_NNNN' segment is the binding site number; everything before it
    is the dataset name (e.g. 'H2O_CD1_01_0001' → 'H2O_CD1_01').
    """
    dataset_dict = {}
    for bchmk_struc_name in benchmark_structs:
        dataset_name = bchmk_struc_name.rsplit("_", 1)[0]
        dataset_dict[bchmk_struc_name] = dataset_name
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
    """Compute absolute and relative error DataFrames against reference energies.

    df: DataFrame where either rows or columns contain structure names that
        can be matched to ref_en_dict keys by longest prefix match.
    ref_en_dict: {structure_name: reference_energy} e.g. {'H2O_CD1_01_0002': -2.5}
    """
    def _match_ref_key(name):
        for key in sorted(ref_en_dict.keys(), key=len, reverse=True):
            if name.startswith(key):
                return key
        return None

    abs_error_df = pd.DataFrame(index=df.index, columns=df.columns)
    rel_error_df = pd.DataFrame(index=df.index, columns=df.columns)

    # Try matching on index (rows = structures, columns = functionals)
    for row_index in df.index:
        ref_key = _match_ref_key(str(row_index))
        if ref_key is None:
            continue
        ref_value = ref_en_dict[ref_key]
        for col in df.columns:
            val = df.at[row_index, col]
            if pd.notna(val):
                abs_error = val - ref_value
                abs_error_df.at[row_index, col] = abs_error
                rel_error_df.at[row_index, col] = abs_error / ref_value if ref_value != 0 else None

    result_ae = abs_error_df.dropna(how='all')
    result_re = rel_error_df.dropna(how='all')

    # If nothing matched on index, try columns (rows = functionals, columns = structures)
    if result_ae.empty:
        abs_error_df = pd.DataFrame(index=df.index, columns=df.columns)
        rel_error_df = pd.DataFrame(index=df.index, columns=df.columns)
        for col in df.columns:
            ref_key = _match_ref_key(str(col))
            if ref_key is None:
                continue
            ref_value = ref_en_dict[ref_key]
            for row_index in df.index:
                val = df.at[row_index, col]
                if pd.notna(val):
                    abs_error = val - ref_value
                    abs_error_df.at[row_index, col] = abs_error
                    rel_error_df.at[row_index, col] = abs_error / ref_value if ref_value != 0 else None
        result_ae = abs_error_df.dropna(how='all')
        result_re = rel_error_df.dropna(how='all')

    return result_ae, result_re


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
