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


# Real optimized binding-site structures fetched from the QCFractal server
# (CO on water clusters, B3LYP-D3BJ/def2-TZVP)

@pytest.fixture(scope="session")
def co_w2_0001():
    return Molecule.from_file(str(DATA_DIR / "co_w2_0001.xyz"))


@pytest.fixture(scope="session")
def co_w2_0007():
    return Molecule.from_file(str(DATA_DIR / "co_w2_0007.xyz"))


@pytest.fixture(scope="session")
def co_w3_0001():
    return Molecule.from_file(str(DATA_DIR / "co_w3_0001.xyz"))


@pytest.fixture(scope="session")
def co_w3_0004():
    return Molecule.from_file(str(DATA_DIR / "co_w3_0004.xyz"))


@pytest.fixture(scope="session")
def co_w5_0001():
    return Molecule.from_file(str(DATA_DIR / "co_w5_0001.xyz"))


@pytest.fixture(scope="session")
def co_w5_0002():
    return Molecule.from_file(str(DATA_DIR / "co_w5_0002.xyz"))


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
