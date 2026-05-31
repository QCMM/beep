"""Tests for beep/workflows/extract.py."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_be_df(entries, be_methods, basis):
    """Build a fake df_be DataFrame as returned by fetch_reaction_values."""
    rng = np.random.default_rng(42)
    columns = [f"{bm}/{basis}" for bm in be_methods]
    # Use a tight, deep range so that adding a +ZPVE correction (~6 kcal/mol
    # in this mocked setup) doesn't push values past the default
    # ``be_range`` filter at -0.1 kcal/mol.
    data = {col: rng.uniform(-18.0, -12.0, len(entries)) for col in columns}
    return pd.DataFrame(data, index=entries)


def _make_nocp_df(entries):
    """Build a fake df_nocp DataFrame with atom+cluster fragments per entry."""
    rows = []
    for i, e in enumerate(entries):
        rows.append({"name": e, "stoichiometry": "be_nocp",
                     "molecule": 1000 + i, "coefficient": 1})    # dimer
        rows.append({"name": e, "stoichiometry": "be_nocp",
                     "molecule": 2000 + i, "coefficient": -1})   # atom fragment
        rows.append({"name": e, "stoichiometry": "be_nocp",
                     "molecule": 3000 + i, "coefficient": -1})   # cluster fragment
    return pd.DataFrame(rows)


def _fake_get_zpve_mol(client, mol_id, lot_opt, **kw):
    """Simulate the adapter's get_zpve_mol return values.

    - Atoms (IDs in 2000s) return (0.0, True) — valid atomic ZPVE.
    - Cluster fragments (3000s) return a valid non-zero ZPVE.
    - Dimers (1000s) return a valid non-zero ZPVE.
    """
    if 2000 <= mol_id < 3000:
        return (0.0, True)
    if 3000 <= mol_id < 4000:
        return (0.04, True)
    return (0.05, True)


def test_zpve_correction_handles_atom_fragments():
    """Regression test: zpve_correction must not raise RuntimeError
    when one of the two fragments is a single atom (ZPVE 0.0).

    Pre-fix bug: the falsy check ``if not m1`` conflated 0.0 (valid
    atomic ZPVE) with None (missing hessian record), so atom-on-cluster
    entries crashed with 'Missing hessian for fragment molecule(s) ...'.
    """
    from beep.workflows.extract import zpve_correction

    entries = [f"C_W5_01_{i:04d}" for i in range(1, 6)]  # 5 to pass the guard
    be_methods = ["wb97x-v", "m06-hf"]
    basis = "def2-tzvp"

    mock_ds = MagicMock()
    mock_ds.entry_names = entries

    with patch("beep.workflows.extract.qcf.get_collection", return_value=mock_ds), \
         patch("beep.workflows.extract.qcf.fetch_reaction_values",
               return_value=_make_be_df(entries, be_methods, basis)), \
         patch("beep.workflows.extract.qcf.fetch_reaction_entries",
               return_value=_make_nocp_df(entries)), \
         patch("beep.workflows.extract.qcf.get_zpve_mol",
               side_effect=_fake_get_zpve_mol):
        df_be, fit_params, todelete = zpve_correction(
            name_be=["be_C_W5_01_HF3C_MINIX"],
            be_methods=be_methods,
            lot_opt="hf3c_minix",
            basis=basis,
            client=MagicMock(),
            scale_factor=1.0,
            be_range=(-0.1, -25.0),
        )

    # All 5 entries flowed through; the atom branch produced a valid ZPVE
    # correction = E_ZPVE(dimer) - E_ZPVE(atom) - E_ZPVE(cluster)
    #            = 0.05 - 0.0 - 0.04 = 0.01 hartree
    assert len(todelete) == 0
    assert "Delta_ZPVE" in df_be.columns
    assert len(df_be) == 5


def test_zpve_correction_raises_for_truly_missing_hessian():
    """Sanity-check: when get_zpve_mol returns (None, True) — i.e. no
    hessian record on the server — zpve_correction must still raise
    RuntimeError. The fix only changed how 0.0 is interpreted, not how
    None is."""
    from beep.workflows.extract import zpve_correction

    entries = [f"M_W5_01_{i:04d}" for i in range(1, 6)]
    be_methods = ["wb97x-v"]
    basis = "def2-tzvp"

    def missing_for_one_fragment(client, mol_id, lot_opt, **kw):
        if 3000 <= mol_id < 4000:
            return (None, True)         # cluster fragment has no hessian
        if 2000 <= mol_id < 3000:
            return (0.02, True)
        return (0.05, True)

    mock_ds = MagicMock()
    mock_ds.entry_names = entries

    with patch("beep.workflows.extract.qcf.get_collection", return_value=mock_ds), \
         patch("beep.workflows.extract.qcf.fetch_reaction_values",
               return_value=_make_be_df(entries, be_methods, basis)), \
         patch("beep.workflows.extract.qcf.fetch_reaction_entries",
               return_value=_make_nocp_df(entries)), \
         patch("beep.workflows.extract.qcf.get_zpve_mol",
               side_effect=missing_for_one_fragment):
        with pytest.raises(RuntimeError, match="Missing hessian"):
            zpve_correction(
                name_be=["be_M_W5_01_HF3C_MINIX"],
                be_methods=be_methods,
                lot_opt="hf3c_minix",
                basis=basis,
                client=MagicMock(),
                scale_factor=1.0,
                be_range=(-0.1, -25.0),
            )
