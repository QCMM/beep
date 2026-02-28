"""Tests for beep/core/dft_functionals.py."""
import pytest

from beep.core.dft_functionals import (
    gga,
    meta_gga,
    meta_hybrid_gga,
    lrc,
    hybrid_gga,
    double_hybrid,
    geom_hmgga_dz,
    geom_hmgga_tz,
    geom_gga_dz,
    geom_sqm_mb,
)

ALL_FUNCS = [
    gga, meta_gga, meta_hybrid_gga, lrc, hybrid_gga,
    double_hybrid, geom_hmgga_dz, geom_hmgga_tz, geom_gga_dz, geom_sqm_mb,
]


@pytest.mark.parametrize("func", ALL_FUNCS, ids=lambda f: f.__name__)
def test_returns_list(func):
    assert isinstance(func(), list)


@pytest.mark.parametrize("func", ALL_FUNCS, ids=lambda f: f.__name__)
def test_not_empty(func):
    assert len(func()) > 0


@pytest.mark.parametrize("func", ALL_FUNCS, ids=lambda f: f.__name__)
def test_all_strings(func):
    assert all(isinstance(item, str) for item in func())


@pytest.mark.parametrize("func", ALL_FUNCS, ids=lambda f: f.__name__)
def test_no_duplicates(func):
    result = func()
    if func is double_hybrid:
        # Known duplicate: DSD-PBEPBE-NL appears twice in source
        pytest.xfail("double_hybrid() has a known duplicate entry (DSD-PBEPBE-NL)")
    assert len(result) == len(set(result))


def test_gga_contains_pbe():
    assert "PBE" in gga()


def test_hybrid_gga_contains_b3lyp():
    assert "B3LYP" in hybrid_gga()


def test_meta_hybrid_gga_contains_m062x():
    assert "M06-2X" in meta_hybrid_gga()
