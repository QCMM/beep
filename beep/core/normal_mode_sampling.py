"""Normal-mode displacement sampling — pure math.

Used by the ``nm_sampling`` workflow to:
  1. Run vibrational analysis on a Hessian record (wraps ``core.zpve``).
  2. Classify each mode as intermolecular / bending / stretching via
     fragment-COM projection.
  3. Pick the lowest-frequency modes from each band, up to a configured cap.
  4. Generate ± displaced Cartesian geometries at a target RMS amplitude.

No QCFractal dependencies — all functions take plain numpy arrays plus a
qcelemental Molecule. The workflow layer is responsible for fetching the
Hessian record off the server and unpacking it into the array shape used here.
"""
from typing import Dict, List, Literal, Tuple

import numpy as np

from .zpve import _vibanal_wfn


# Bohr ↔ Å (CODATA 2018). qcelemental has these too but inlining keeps the
# module dependency-light + matches the constants used in trajectory_metrics.
_BOHR_TO_A = 0.529177210903


ModeBand = Literal["intermolecular", "bending", "stretching"]


# ---------------------------------------------------------------------------
# Hessian → normal modes (wrap zpve._vibanal_wfn)
# ---------------------------------------------------------------------------

def extract_normal_modes_from_hessian_record(
    hess: np.ndarray, molecule, energy: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run vibrational analysis on a Hessian.

    Parameters
    ----------
    hess : (3N, 3N) ndarray
        Non-mass-weighted Cartesian Hessian, units ``Eh / bohr²``. Shape
        matches the raw ``record.return_result`` of a hessian SP after
        reshape.
    molecule : qcelemental.models.Molecule
        Geometry + atomic masses container.
    energy : float, optional
        Electronic energy (Eh). Only used by ``_vibanal_wfn`` to populate
        the thermodynamic info dict; the modes themselves do not depend
        on it.

    Returns
    -------
    frequencies_cm : (n_vib,) complex ndarray
        Vibrational frequencies in cm⁻¹. Imaginary frequencies appear as
        purely imaginary complex values (real part 0); real frequencies
        appear with imaginary part 0. Translation/rotation modes are
        stripped from the array.
    modes_cart : (n_vib, n_atoms, 3) ndarray
        Cartesian (un-mass-weighted) displacement pattern per mode.
        Direction only — the absolute scale follows numpy's eigh
        normalisation and gets renormalised by ``displace_along_mode``.
    """
    vibinfo, _therminfo = _vibanal_wfn(hess=hess, molecule=molecule, energy=energy)
    trv = vibinfo["TRV"].data           # list of "TR" or "V" per mode
    omega = vibinfo["omega"].data       # (ndof,) complex
    modes = vibinfo["modes_cart"].data  # (ndof, n_atoms, 3)

    keep = [i for i, label in enumerate(trv) if label == "V"]
    if not keep:
        # All TR — shouldn't happen for a bound molecule but be safe.
        return np.empty(0, dtype=complex), np.empty((0, *modes.shape[1:]))
    return omega[keep], modes[keep]


# ---------------------------------------------------------------------------
# Mode classification
# ---------------------------------------------------------------------------

def classify_mode(
    mode_cart: np.ndarray,
    masses: np.ndarray,
    n_adsorbate_atoms: int,
    frequency_cm: float,
    inter_threshold: float = 0.5,
    bend_max_cm: float = 1500.0,
) -> ModeBand:
    """Classify a single normal mode.

    Uses a two-step rule:

      1. Compute ``f_inter`` — the fraction of the mode's kinetic energy
         that goes into rigid fragment-COM translation (adsorbate vs
         cluster). If ``f_inter > inter_threshold`` the mode is labelled
         *intermolecular* — it primarily moves the adsorbate against the
         cluster.

      2. Otherwise the mode is intramolecular; split by frequency:
         ``frequency_cm < bend_max_cm`` → *bending*, else *stretching*.
         (Frequency is a robust separator for intramolecular modes;
         only for mixed-character modes does it fail, which is exactly
         what step 1 catches.)

    Fragment convention: the **last** ``n_adsorbate_atoms`` rows are the
    adsorbate, the rest are the cluster — same convention BEEP uses
    everywhere else (see ``reference_beep_last_molecule_adsorbate``).

    Parameters
    ----------
    mode_cart : (n_atoms, 3) ndarray
        Cartesian displacement pattern of the mode (units irrelevant —
        only the relative atomic motion is used).
    masses : (n_atoms,) ndarray
        Atomic masses (amu).
    n_adsorbate_atoms : int
        Number of trailing atoms that constitute the adsorbate.
    frequency_cm : float
        Mode frequency (real part if complex) in cm⁻¹.
    inter_threshold : float, default 0.5
        ``f_inter`` above this → intermolecular.
    bend_max_cm : float, default 1500.0
        Intramolecular cutoff between bending and stretching.
    """
    n_atoms = mode_cart.shape[0]
    if n_adsorbate_atoms <= 0 or n_adsorbate_atoms >= n_atoms:
        # Degenerate fragmenting (whole system or empty) — call everything
        # intramolecular and let the frequency cut decide.
        f_inter = 0.0
    else:
        cluster_idx = slice(0, n_atoms - n_adsorbate_atoms)
        ads_idx = slice(n_atoms - n_adsorbate_atoms, n_atoms)

        m_cluster = masses[cluster_idx]
        m_ads = masses[ads_idx]
        v_cluster = mode_cart[cluster_idx]
        v_ads = mode_cart[ads_idx]

        com_cluster = (m_cluster[:, None] * v_cluster).sum(axis=0) / m_cluster.sum()
        com_ads = (m_ads[:, None] * v_ads).sum(axis=0) / m_ads.sum()

        # Kinetic energy carried by fragment-COM motion (relative to the
        # whole-system COM, which is zero for a TR-projected mode).
        k_inter = (
            m_cluster.sum() * float(np.dot(com_cluster, com_cluster))
            + m_ads.sum() * float(np.dot(com_ads, com_ads))
        )
        k_total = float(np.einsum("a,ai,ai->", masses, mode_cart, mode_cart))
        f_inter = k_inter / k_total if k_total > 0.0 else 0.0

    if f_inter > inter_threshold:
        return "intermolecular"
    if frequency_cm < bend_max_cm:
        return "bending"
    return "stretching"


# ---------------------------------------------------------------------------
# Per-band selection
# ---------------------------------------------------------------------------

def select_modes(
    frequencies_cm: np.ndarray,
    classes: List[str],
    band_caps: Dict[str, int],
    band_amplitudes: Dict[str, float],
    freq_max_imag_cm: float = 50.0,
    extra_amplitudes_lowest_count: int = 1,
    extra_amplitude_factor: float = 2.0,
) -> List[Tuple[int, float, str]]:
    """Pick which modes to displace and at what amplitude.

    Per band, take the ``cap`` lowest-frequency modes whose imaginary
    component is below the threshold. The N lowest-frequency entries of
    the final picked set (across all bands) get a second amplitude
    appended at ``amplitude × extra_amplitude_factor``.

    Parameters
    ----------
    frequencies_cm : (n_modes,) complex ndarray
        Mode frequencies. Real frequencies have ``imag == 0``; imaginary
        frequencies have ``real == 0``.
    classes : list of str, length n_modes
        Each element is "intermolecular", "bending", or "stretching".
    band_caps, band_amplitudes : dict
        Per-band cap (int) and amplitude (Å, RMS Cartesian).
    freq_max_imag_cm : float, default 50.0
        Drop modes whose |imag| exceeds this magnitude.
    extra_amplitudes_lowest_count : int, default 1
        Number of lowest-frequency selected modes that get the extra
        amplitude.
    extra_amplitude_factor : float, default 2.0
        Multiplier for the extra-amplitude entries.

    Returns
    -------
    list of (mode_index, amplitude_A, band_label)
        Each entry represents a displacement pair (the caller submits
        both + and − signs separately). Sorted ascending by frequency
        within bands; extra-amplitude entries trail their primary entry.
    """
    selected: List[Tuple[int, float, str]] = []

    # Build per-band candidate lists (mode_index, real_freq), filtered.
    candidates: Dict[str, List[Tuple[int, float]]] = {b: [] for b in band_caps}
    for idx, (omega, band) in enumerate(zip(frequencies_cm, classes)):
        if band not in candidates:
            continue
        re = float(np.real(omega))
        im = float(np.imag(omega))
        if abs(im) > freq_max_imag_cm:
            continue
        candidates[band].append((idx, re))

    # Sort each band's candidates by ascending real frequency, take the cap.
    for band, cap in band_caps.items():
        picks = sorted(candidates[band], key=lambda x: x[1])[: int(cap)]
        amp = float(band_amplitudes[band])
        for idx, _freq in picks:
            selected.append((idx, amp, band))

    # Identify the N lowest-frequency entries across the whole selection
    # and append extra-amplitude duplicates.
    if extra_amplitudes_lowest_count > 0 and selected:
        # Pair each entry with its frequency for ranking.
        ranked = sorted(
            range(len(selected)),
            key=lambda i: float(np.real(frequencies_cm[selected[i][0]])),
        )
        extras = []
        for slot in ranked[: int(extra_amplitudes_lowest_count)]:
            idx, amp, band = selected[slot]
            extras.append((idx, amp * float(extra_amplitude_factor), band))
        selected.extend(extras)

    return selected


# ---------------------------------------------------------------------------
# Displacement geometry generation
# ---------------------------------------------------------------------------

def displace_along_mode(
    geometry_bohr: np.ndarray,
    mode_cart: np.ndarray,
    amplitude_A: float,
    sign: int = +1,
) -> np.ndarray:
    """Apply a ±-signed displacement along a mode to a geometry.

    The mode is scaled so its RMS Cartesian length (over all atoms) equals
    ``amplitude_A`` (in Å), converted to bohr, then added to the
    equilibrium geometry with the given sign.

    Parameters
    ----------
    geometry_bohr : (n_atoms, 3) ndarray
        Equilibrium Cartesian geometry in bohr.
    mode_cart : (n_atoms, 3) ndarray
        Mode direction in Cartesians (un-mass-weighted). Magnitude is
        irrelevant — only the direction is used.
    amplitude_A : float
        Target RMS Cartesian displacement of all atoms, in Å.
    sign : int, default +1
        ``+1`` or ``-1``.

    Returns
    -------
    (n_atoms, 3) ndarray — displaced geometry in bohr.
    """
    if sign not in (+1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    n_atoms = mode_cart.shape[0]
    if n_atoms == 0:
        return geometry_bohr.copy()

    # RMS Cartesian length of the mode vector, in whatever units the input
    # came with. We rescale to amplitude_A (in bohr) — RMS of a target
    # displacement field of RMS=A_bohr has Σ|v|² = 3 N A_bohr² (each of
    # 3 N components contributes A_bohr²/(3N) on average, and the RMS of
    # a vector with 3 N components is √(Σ|v_i|² / (3N))).
    sum_sq = float(np.einsum("ai,ai->", mode_cart, mode_cart))
    if sum_sq <= 0.0:
        return geometry_bohr.copy()
    rms_mode = np.sqrt(sum_sq / (3.0 * n_atoms))
    amp_bohr = amplitude_A / _BOHR_TO_A
    scale = amp_bohr / rms_mode

    return geometry_bohr + sign * scale * mode_cart
