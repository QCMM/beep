"""Shared fixtures for the BEEP test suite."""
import logging
from pathlib import Path

import pytest
from qcelemental.models.molecule import Molecule

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Session-scoped molecule fixtures (loaded once, immutable)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def h2_mol():
    return Molecule.from_file(str(DATA_DIR / "sm_h2.xyz"))


@pytest.fixture(scope="session")
def ch3oh_mol():
    return Molecule.from_file(str(DATA_DIR / "sm_ch3oh.xyz"))


@pytest.fixture(scope="session")
def nh2ch2cn_mol():
    return Molecule.from_file(str(DATA_DIR / "sm_nh2ch2cn.xyz"))


@pytest.fixture(scope="session")
def hco_mol():
    return Molecule.from_file(str(DATA_DIR / "sr_hco.xyz"))


@pytest.fixture(scope="session")
def ws3_cluster():
    return Molecule.from_file(str(DATA_DIR / "ws3.xyz"))


@pytest.fixture(scope="session")
def ws5_cluster():
    return Molecule.from_file(str(DATA_DIR / "ws5.xyz"))


@pytest.fixture(scope="session")
def w22_cluster():
    return Molecule.from_file(str(DATA_DIR / "w22_01.xyz"))


# ---------------------------------------------------------------------------
# Function-scoped utility fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_logger():
    logger = logging.getLogger("beep_test")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


@pytest.fixture
def test_output_dir():
    out = Path(__file__).resolve().parent / "test_data_output"
    out.mkdir(exist_ok=True)
    return out
