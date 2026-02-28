"""Tests for beep.core.benchmark_utils — pure benchmark utility functions."""
import numpy as np
import pandas as pd
import pytest
from qcelemental.models.molecule import Molecule

from beep.core.benchmark_utils import (
    create_benchmark_dataset_dict,
    create_molecular_fragments,
    get_errors_dataframe,
    compute_rmsd,
)


class TestCreateBenchmarkDatasetDict:
    def test_basic(self):
        structs = ["CO_W2_001", "CO_W3_002"]
        result = create_benchmark_dataset_dict(structs)
        assert result == {"CO_W2_001": "CO_W2", "CO_W3_002": "CO_W3"}

    def test_single(self):
        result = create_benchmark_dataset_dict(["H2O_ICE_003"])
        assert result == {"H2O_ICE_003": "H2O_ICE"}

    def test_empty(self):
        result = create_benchmark_dataset_dict([])
        assert result == {}


class TestCreateMolecularFragments:
    def test_water_dimer(self):
        # Create a "dimer" of two waters (6 atoms total)
        mol = Molecule(
            symbols=["O", "H", "H", "O", "H", "H"],
            geometry=[
                0.0, 0.0, 0.0,
                1.5, 0.0, 0.0,
                -1.5, 0.0, 0.0,
                6.0, 0.0, 0.0,
                7.5, 0.0, 0.0,
                4.5, 0.0, 0.0,
            ],
        )
        f1, f2 = create_molecular_fragments(mol, len_f1=3)
        assert len(f1.symbols) == 3
        assert len(f2.symbols) == 3
        assert tuple(f1.symbols) == ("O", "H", "H")
        assert tuple(f2.symbols) == ("O", "H", "H")

    def test_unequal_split(self):
        mol = Molecule(
            symbols=["O", "H", "H", "C"],
            geometry=[
                0.0, 0.0, 0.0,
                1.5, 0.0, 0.0,
                -1.5, 0.0, 0.0,
                10.0, 0.0, 0.0,
            ],
        )
        f1, f2 = create_molecular_fragments(mol, len_f1=3)
        assert len(f1.symbols) == 3
        assert len(f2.symbols) == 1


class TestGetErrorsDataframe:
    def test_basic_errors(self):
        df = pd.DataFrame(
            {"method1": [-5.0, -3.0], "method2": [-4.5, -2.5]},
            index=["CO_W2_001_lot1", "CO_W3_002_lot2"],
        )
        ref = {"CO_W2_001": -5.5, "CO_W3_002": -3.5}
        abs_err, rel_err = get_errors_dataframe(df, ref)

        # abs error = value - ref
        assert abs_err.at["CO_W2_001_lot1", "method1"] == pytest.approx(0.5)
        assert abs_err.at["CO_W3_002_lot2", "method1"] == pytest.approx(0.5)

        # rel error = abs_error / ref
        assert rel_err.at["CO_W2_001_lot1", "method1"] == pytest.approx(0.5 / -5.5)

    def test_filters_by_ref_keys(self):
        df = pd.DataFrame(
            {"m1": [-5.0, -3.0, -1.0]},
            index=["CO_W2_001_a", "CO_W3_002_b", "XX_YY_003_c"],
        )
        ref = {"CO_W2_001": -5.5}
        abs_err, rel_err = get_errors_dataframe(df, ref)
        assert len(abs_err) == 1
        assert "CO_W2_001_a" in abs_err.index


class TestComputeRmsd:
    def test_identical_molecules(self):
        mol = Molecule(
            symbols=["O", "H", "H"],
            geometry=[
                0.0, 0.0, 0.2217,
                0.0, 1.4309, -0.8869,
                0.0, -1.4309, -0.8869,
            ],
        )
        rmsd = compute_rmsd(mol, mol, rmsd_symm=False)
        assert rmsd == pytest.approx(0.0, abs=1e-10)

    def test_with_symmetry(self):
        mol = Molecule(
            symbols=["O", "H", "H"],
            geometry=[
                0.0, 0.0, 0.2217,
                0.0, 1.4309, -0.8869,
                0.0, -1.4309, -0.8869,
            ],
        )
        rmsd = compute_rmsd(mol, mol, rmsd_symm=True)
        assert rmsd == pytest.approx(0.0, abs=1e-10)
