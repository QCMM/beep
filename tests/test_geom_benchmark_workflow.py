"""Tests for beep/workflows/geom_benchmark.py (server calls mocked)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qcportal.record_models import RecordStatusEnum

from beep.workflows.geom_benchmark import compare_rmsd, create_and_add_specification


def test_compare_rmsd_excludes_lot_with_missing_record():
    """qcportal 0.64 get_record returns None for a missing record; the
    warning used to dereference record.id and crash. The LOT must be
    excluded with a clean warning instead."""
    odset = MagicMock()
    ok_record = MagicMock()
    ok_record.status = RecordStatusEnum.complete
    ok_record.final_molecule = MagicMock()

    def get_record(struct, lot):
        return None if lot == "pbe_def2-svp" else ok_record
    odset.get_record.side_effect = get_record

    logger = MagicMock()
    with patch("beep.workflows.geom_benchmark.logging.getLogger", return_value=logger), \
         patch("beep.workflows.geom_benchmark.compute_rmsd", return_value=0.1):
        best, final, df = compare_rmsd(
            ["pbe_def2-svp", "b3lyp_def2-svp"], {"W22_01": odset}, {"W22_01": MagicMock()},
        )

    assert "pbe_def2-svp" not in final
    assert "pbe_def2-svp" not in df.columns
    assert final == {"b3lyp_def2-svp": pytest.approx(0.1)}
    assert best == {"b3lyp_def2-svp": pytest.approx(0.1)}
    warns = " ".join(c.args[0] for c in logger.warning.call_args_list)
    assert "No record for W22_01 at the pbe_def2-svp" in warns
    assert "no record" in warns


@patch("beep.workflows.geom_benchmark.qcf")
def test_create_and_add_specification_passes_keyword_dict(mock_qcf):
    create_and_add_specification(
        MagicMock(), MagicMock(), "pbe", "def2-svp", "psi4", {"scf_type": "df"},
    )
    spec = mock_qcf.add_opt_specification.call_args.args[1]
    assert spec["qc_spec"]["keywords"] == {"scf_type": "df"}
    assert spec["name"] == "pbe_def2-svp"


@patch("beep.workflows.geom_benchmark.qcf")
def test_create_and_add_specification_none_keyword_is_empty_dict(mock_qcf):
    create_and_add_specification(MagicMock(), MagicMock(), "pbe", "def2-svp", "psi4", None)
    spec = mock_qcf.add_opt_specification.call_args.args[1]
    assert spec["qc_spec"]["keywords"] == {}
