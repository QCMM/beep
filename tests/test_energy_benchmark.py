"""Tests for beep/workflows/energy_benchmark.py."""
import logging

import pandas as pd
import pytest

from types import SimpleNamespace

from beep.workflows.energy_benchmark import (
    get_cc_keywords, log_mae_per_geometry, scf_state_jumps, check_scf_states,
)


def _capture_logger():
    logger = logging.getLogger("beep")
    logger.setLevel(logging.INFO)
    records = []

    class _Grab(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Grab()
    logger.addHandler(handler)
    return logger, handler, records


def test_log_mae_per_geometry_one_labeled_section_per_lot():
    """Regression: with N opt LOTs the results summary must print N sections,
    each labeled with its geometry LOT and computed only from that LOT's
    entries. Pre-fix, a single unlabeled table pooled every geometry's
    errors into one MAE per functional."""
    lot_a = "mpwb1k-d3bj_def2-tzvpd"
    lot_b = "gfn2-xtb"
    # Signed errors: lot_a rows → MAE 2.0; lot_b row → MAE 5.0. The pooled
    # (pre-fix) value would be 3.0 and must NOT be what either section shows.
    df_ae = pd.DataFrame(
        {"pbe/def2-tzvp": [1.0, -3.0, 5.0]},
        index=[
            f"CO_W12_1_0001_{lot_a}",
            f"CO_W12_1_0002_{lot_a}",
            f"CO_W12_1_0001_{lot_b}",
        ],
    )
    logger, handler, records = _capture_logger()
    try:
        log_mae_per_geometry(
            logger, df_ae, {"GGA": ["pbe"]}, [lot_a, lot_b], "BE MAE",
        )
    finally:
        logger.removeHandler(handler)

    text = "\n".join(records)
    assert f"BE MAE — {lot_a} geometries" in text
    assert f"BE MAE — {lot_b} geometries" in text
    # Per-section MAEs, not the pooled 3.0
    assert "2.000000" in text
    assert "5.000000" in text
    assert "3.000000" not in text
    # Section order matches the config order and each MAE sits in its section
    a_pos = text.index(f"{lot_a} geometries")
    b_pos = text.index(f"{lot_b} geometries")
    assert a_pos < text.index("2.000000") < b_pos < text.index("5.000000")


def test_log_mae_per_geometry_skips_lot_without_entries():
    """A LOT with no matching entries warns and is skipped, without touching
    the other sections."""
    df_ae = pd.DataFrame(
        {"pbe/def2-tzvp": [1.0]}, index=["CO_W12_1_0001_gfn2-xtb"],
    )
    logger, handler, records = _capture_logger()
    logger.setLevel(logging.WARNING)
    try:
        log_mae_per_geometry(
            logger, df_ae, {"GGA": ["pbe"]}, ["hf3c_minix", "gfn2-xtb"], "BE MAE",
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(logging.INFO)

    text = "\n".join(records)
    assert "No entries for geometry hf3c_minix" in text


def test_get_cc_keywords_open_shell_has_iteration_cap():
    """Open-shell DF-CCSD needs a raised cc_maxiter: DIIS oscillates around
    the residual criterion with the energy already converged (CN, CH3O)."""
    kw = get_cc_keywords(2)
    assert kw["qc_module"] == "OCC"
    assert kw["reference"] == "uhf"
    assert kw["cc_maxiter"] == 200
    # Closed shell unchanged: no UHF machinery, no cap needed
    kw1 = get_cc_keywords(1)
    assert "cc_maxiter" not in kw1
    assert "reference" not in kw1


# --- open-shell SCF state check -------------------------------------------

_H2K = 627.5094740631
_BASES = ["aug-cc-pvdz", "aug-cc-pvtz", "aug-cc-pvqz"]
_SCF_LOTS = [f"scf_{b}" for b in _BASES]


def test_scf_state_jumps_smooth_passes():
    ie = dict(zip(_BASES, [-3.10, -2.85, -2.78]))
    be = dict(zip(_BASES, [-2.60, -2.30, -2.20]))
    assert scf_state_jumps(ie, be) == []
    assert scf_state_jumps(ie, be, threshold=0.2) != []


def test_scf_state_jumps_oh_w3_0005():
    """OH_W3_01_0005: aVTZ/aVQZ complex in a different state than aVDZ."""
    ie = dict(zip(_BASES, [-2.50, -0.01, 0.01]))
    be = dict(zip(_BASES, [-2.00, 0.40, 0.50]))
    jumps = scf_state_jumps(ie, be)
    assert len(jumps) == 2
    assert jumps[0].startswith("IE aug-cc-pvdz->aug-cc-pvtz")


class _FakeDS:
    def __init__(self, energies):
        self.energies = energies

    def get_record(self, entry, spec):
        e = self.energies.get((entry, spec))
        return None if e is None else SimpleNamespace(return_result=e)


def _site(struct, ie_scf):
    """SCF records of one site with the given IE ladder (BE follows the IE)."""
    mol, surf = struct.split("_")[0], "_".join(struct.split("_")[1:3])
    out = {}
    for b, ie in zip(_BASES, ie_scf):
        f1, f2 = -100.0, -50.0
        cx = f1 + f2 + ie / _H2K
        for ent, e in [(struct, cx), (struct + "_f1", f1), (struct + "_f2", f2),
                       (mol, f2), (surf, f1)]:
            out[(ent, f"scf_{b}")] = e
    return out


def test_check_scf_states_passes_and_fails():
    ok = _site("OH_W3_01_0002", [-3.10, -2.85, -2.78])
    check_scf_states(_FakeDS(ok), ["OH_W3_01_0002"], _SCF_LOTS)

    bad = dict(ok)
    bad.update(_site("OH_W3_01_0005", [-2.50, -0.01, 0.01]))
    with pytest.raises(RuntimeError, match="OH_W3_01_0005"):
        check_scf_states(_FakeDS(bad), ["OH_W3_01_0002", "OH_W3_01_0005"], _SCF_LOTS)
