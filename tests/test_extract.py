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


@patch("beep.workflows.extract.qcf.check_collection_exists", return_value=True)
@patch("beep.workflows.extract.qcf.fetch_reaction_values")
def test_concatenate_frames_keeps_mlp_composite(mock_fetch, mock_exists):
    """Range-separated MLP composite column (basis-less, ends in a dispersion
    suffix, e.g. ``lmft-co-v0-d3bj``) is the summed electronic + D3BJ BE and must
    NOT be dropped as a bare-dispersion piece. Regression: dropping it produced
    "No valid binding energies" for every MLP range-separated extraction."""
    from beep.workflows.extract import concatenate_frames
    entries = ["CO_W12_1_0001", "CO_W12_1_0002"]
    # fetch_reaction_values has already summed the electronic MLP into the
    # dispersion column, leaving a single basis-less composite column.
    mock_fetch.return_value = pd.DataFrame(
        {"lmft-co-v0-d3bj": [-1.7, -2.1]}, index=entries)
    ds_w = MagicMock()
    ds_w.entry_names = ["W12_1"]
    df, ok = concatenate_frames(
        MagicMock(), "CO", ds_w, "lmft-co-d-v0",
        be_range=(2.0, -20.0), stoichiometry="ie_nocp")
    assert ok
    assert "lmft-co-v0-d3bj" in df.columns          # composite survived
    assert df["Mean_Eb_all_dft"].notna().all()       # real BEs, not empty
    assert len(df) == 2


@patch("beep.workflows.extract.qcf.check_collection_exists", return_value=True)
@patch("beep.workflows.extract.qcf.fetch_reaction_values")
def test_concatenate_frames_drops_dft_bare_dispersion(mock_fetch, mock_exists):
    """DFT separated pair: the bare-dispersion column (no basis) whose composite
    ``<disp>/<basis>`` exists is still dropped, and the composite is kept. Guards
    that the MLP fix does not regress the DFT path."""
    from beep.workflows.extract import concatenate_frames
    entries = ["CO_W12_1_0001"]
    mock_fetch.return_value = pd.DataFrame({
        "mpwb1k/def2-tzvpd": [-1.5],          # bare electronic -> drop
        "mpwb1k-d3bj": [-0.3],                 # bare dispersion (no basis) -> drop (composite exists)
        "mpwb1k-d3bj/def2-tzvpd": [-1.8],      # composite -> keep
    }, index=entries)
    ds_w = MagicMock()
    ds_w.entry_names = ["W12_1"]
    df, ok = concatenate_frames(
        MagicMock(), "CO", ds_w, "mpwb1k_def2-tzvpd",
        be_range=(2.0, -20.0), stoichiometry="bsse")
    assert ok
    assert "mpwb1k-d3bj/def2-tzvpd" in df.columns   # composite kept
    assert "mpwb1k-d3bj" not in df.columns           # bare dispersion dropped
    assert "mpwb1k/def2-tzvpd" not in df.columns     # bare electronic dropped


@pytest.mark.parametrize("cluster_order", [
    ["W12_1", "W12_2"],   # method-poor cluster first (the pre-fix failure order)
    ["W12_2", "W12_1"],   # method-rich cluster first
])
@patch("beep.workflows.extract.qcf.check_collection_exists", return_value=True)
@patch("beep.workflows.extract.qcf.fetch_reaction_values")
def test_concatenate_frames_unions_method_columns_across_clusters(
    mock_fetch, mock_exists, cluster_order,
):
    """Regression: methods present only in later clusters must survive.

    Pre-fix, each subsequent cluster's frame was reindexed onto the columns
    accumulated from the first cluster, so a method computed only for a
    later cluster (e.g. B3LYP added mid-campaign) vanished from the report,
    and the outcome depended on cluster iteration order. pd.concat's outer
    join takes the column union; clusters lacking a method carry NaN rows.
    """
    from beep.workflows.extract import concatenate_frames

    frames = {
        # W12_1: PBE only
        "be_CO_W12_1_PBE-D3BJ_DEF2-SVP": pd.DataFrame(
            {"pbe-d3bj/def2-svp": [-10.0, -11.0]},
            index=["CO_W12_1_0001", "CO_W12_1_0002"]),
        # W12_2: PBE + B3LYP
        "be_CO_W12_2_PBE-D3BJ_DEF2-SVP": pd.DataFrame(
            {"pbe-d3bj/def2-svp": [-12.0],
             "b3lyp-d3bj/def2-svp": [-13.0]},
            index=["CO_W12_2_0001"]),
    }
    mock_fetch.side_effect = lambda client, name, stoich: frames[name]

    ds_w = MagicMock()
    ds_w.entry_names = cluster_order
    df, ok = concatenate_frames(
        MagicMock(), "CO", ds_w, "pbe-d3bj_def2-svp",
        be_range=(-0.1, -25.0), stoichiometry="bsse")

    assert ok
    # The union must hold regardless of iteration order
    assert "pbe-d3bj/def2-svp" in df.columns
    assert "b3lyp-d3bj/def2-svp" in df.columns, (
        f"B3LYP column lost with cluster order {cluster_order}"
    )
    # All three sites survive; the PBE-only cluster carries NaN for B3LYP
    assert len(df) == 3
    assert df.loc["CO_W12_2_0001", "b3lyp-d3bj/def2-svp"] == pytest.approx(-13.0)
    assert df.loc[["CO_W12_1_0001", "CO_W12_1_0002"],
                  "b3lyp-d3bj/def2-svp"].isna().all()
