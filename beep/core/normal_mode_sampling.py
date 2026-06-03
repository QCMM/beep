"""Normal-mode displacement sampling — pure math.

Used by the ``nm_sampling`` workflow to:
  1. Run vibrational analysis on a Hessian record (wraps ``core.zpve``).
  2. Classify each mode as intermolecular / bending / stretching via
     rigid-body kinetic-energy projection (fragment-COM translation +
     rotation about COM).
  3. Pick the lowest-frequency modes from each band, up to a configured cap.
  4. Generate ± displaced Cartesian geometries at a target RMS amplitude.
  5. Write a Molden-format file + a programmatic JSON for visualization.

No QCFractal dependencies — all functions take plain numpy arrays plus a
qcelemental Molecule. The workflow layer is responsible for fetching the
Hessian record off the server and unpacking it into the array shape used here.
"""
import json
from pathlib import Path
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

def _fragment_rigid_kinetic(
    positions: np.ndarray, masses: np.ndarray, mode_cart: np.ndarray,
) -> float:
    """Rigid-body kinetic energy (× 2) of one fragment in a normal mode.

    Returns ``K_trans + K_rot`` for the fragment, where

      * ``K_trans = M_f · |c_f|²``       (fragment-COM translation)
      * ``K_rot   = Lᵀ · I⁺ · L``        (rotation about fragment COM)

    ``c_f`` is the mass-weighted COM displacement, ``L`` the angular
    momentum about the COM, and ``I⁺`` the Moore–Penrose pseudo-inverse
    of the inertia tensor (which gracefully handles single-atom and
    collinear fragments — for those the null-space component of ``L``
    is projected out and contributes zero rotational KE).

    The constant ``½`` is dropped — only the ratio ``K_rigid / K_total``
    is used downstream, so the factor cancels.
    """
    if masses.size == 0:
        return 0.0
    M = float(masses.sum())
    if M <= 0.0:
        return 0.0
    com_pos = (masses[:, None] * positions).sum(axis=0) / M
    com_disp = (masses[:, None] * mode_cart).sum(axis=0) / M
    rho = positions - com_pos
    L = (masses[:, None] * np.cross(rho, mode_cart)).sum(axis=0)
    rho_sq = float((masses * (rho * rho).sum(axis=1)).sum())
    I = rho_sq * np.eye(3) - np.einsum("a,ai,aj->ij", masses, rho, rho)
    omega = np.linalg.pinv(I) @ L
    k_trans = M * float(com_disp @ com_disp)
    k_rot = float(L @ omega)
    return k_trans + k_rot


def classify_mode(
    mode_cart: np.ndarray,
    masses: np.ndarray,
    positions: np.ndarray,
    n_adsorbate_atoms: int,
    frequency_cm: float,
    inter_threshold: float = 0.5,
    bend_max_cm: float = 1500.0,
) -> ModeBand:
    """Classify a single normal mode.

    Uses a two-step rule:

      1. Compute ``f_inter`` — the fraction of the mode's kinetic energy
         carried by **rigid-body motion of each fragment** (translation
         of the fragment COM plus rotation about that COM). If
         ``f_inter > inter_threshold`` the mode is labelled
         *intermolecular* — it primarily moves the adsorbate against the
         cluster, whether by translation or by libration.

      2. Otherwise the mode is intramolecular; split by frequency:
         ``frequency_cm < bend_max_cm`` → *bending*, else *stretching*.
         (Frequency is a robust separator for intramolecular modes;
         only for mixed-character modes does it fail, which is exactly
         what step 1 catches.)

    Including the rotational term is essential for low-frequency
    librations: a libration is a pure intramolecular rotation of one
    fragment about its own COM, so the COM *displacement* is ~0 even
    though every atom in the fragment is moving relative to the rest of
    the complex. A COM-displacement-only ratio misses librations and
    labels them as internal bends.

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
    positions : (n_atoms, 3) ndarray
        Equilibrium atomic positions, same Cartesian frame as
        ``mode_cart``. Used to compute angular momenta + inertia
        tensors about each fragment COM. Units cancel in the ratio.
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
        n_cluster = n_atoms - n_adsorbate_atoms
        k_rigid = (
            _fragment_rigid_kinetic(
                positions[:n_cluster], masses[:n_cluster], mode_cart[:n_cluster],
            )
            + _fragment_rigid_kinetic(
                positions[n_cluster:], masses[n_cluster:], mode_cart[n_cluster:],
            )
        )
        k_total = float(np.einsum("a,ai,ai->", masses, mode_cart, mode_cart))
        f_inter = k_rigid / k_total if k_total > 0.0 else 0.0

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


# ---------------------------------------------------------------------------
# Molden + JSON writers (visualization / programmatic post-analysis)
# ---------------------------------------------------------------------------

def write_molden(
    filepath,
    symbols: List[str],
    geometry_bohr: np.ndarray,
    frequencies_cm: np.ndarray,
    modes_cart: np.ndarray,
    title: str = "BEEP nm_sampling normal modes",
) -> None:
    """Write Molden-format normal-mode file (visualisable in Avogadro, jmol, IboView).

    Imaginary frequencies are written with a negative sign (Molden's
    convention for saddle-direction modes).

    Parameters
    ----------
    filepath : str or Path
        Output ``.molden`` path.
    symbols : list of str, length n_atoms
        Atomic symbols.
    geometry_bohr : (n_atoms, 3) ndarray
        Atomic positions in Bohr.
    frequencies_cm : (n_vib,) complex ndarray
        Vibrational frequencies; real modes have ``imag == 0``, imaginary
        modes have ``real == 0`` (qcelemental convention).
    modes_cart : (n_vib, n_atoms, 3) ndarray
        Cartesian displacement patterns (un-mass-weighted).
    title : str
        Title line written into the ``[Title]`` block.
    """
    import qcelemental as qcel
    atomic_numbers = [qcel.periodictable.to_atomic_number(s) for s in symbols]
    n_atoms = len(symbols)

    lines = ["[Molden Format]", "[Title]", title, "[Atoms] AU"]
    for i, (sym, Z) in enumerate(zip(symbols, atomic_numbers), start=1):
        x, y, z = geometry_bohr[i - 1]
        lines.append(
            f"{sym:<3s} {i:>5d} {Z:>4d}  "
            f"{x:>16.10f} {y:>16.10f} {z:>16.10f}"
        )

    lines.append("[FREQ]")
    for omega in frequencies_cm:
        re = float(np.real(omega))
        im = float(np.imag(omega))
        value = -abs(im) if abs(im) > abs(re) else re
        lines.append(f"{value:>14.4f}")

    lines.append("[FR-COORD]")
    for sym, (x, y, z) in zip(symbols, geometry_bohr):
        lines.append(f"{sym:<3s} {x:>16.10f} {y:>16.10f} {z:>16.10f}")

    lines.append("[FR-NORM-COORD]")
    for i, mode in enumerate(modes_cart, start=1):
        lines.append(f"vibration {i}")
        for dx, dy, dz in mode:
            lines.append(f"  {dx:>14.10f} {dy:>14.10f} {dz:>14.10f}")

    Path(filepath).write_text("\n".join(lines) + "\n")


def write_modes_json(
    filepath,
    symbols: List[str],
    geometry_bohr: np.ndarray,
    frequencies_cm: np.ndarray,
    modes_cart: np.ndarray,
    classes: List[str],
    n_adsorbate_atoms: int,
    level_of_theory: str = "",
) -> None:
    """Write a per-system normal-mode JSON.

    Matches the structure of the test-case file BEEP was validated against
    (``h2s_h2o_modes.json``): symbols, geometry in Å, the fragment
    boundary (BEEP convention: last ``n_adsorbate_atoms`` rows are the
    adsorbate), and a list of ``{freq_cm, disp, class}`` entries per mode.

    Parameters
    ----------
    filepath : str or Path
        Output ``.json`` path.
    symbols : list of str, length n_atoms
    geometry_bohr : (n_atoms, 3) ndarray
        Positions in Bohr — converted to Å on write.
    frequencies_cm : (n_vib,) complex ndarray
    modes_cart : (n_vib, n_atoms, 3) ndarray
    classes : list of str, length n_vib
        Output of ``classify_mode`` per mode.
    n_adsorbate_atoms : int
        BEEP convention: last N atoms are the adsorbate.
    level_of_theory : str
        For the ``level_of_theory`` key — e.g. ``hf_def2-svp``.
    """
    n_atoms = len(symbols)
    geom_A = (np.asarray(geometry_bohr) * _BOHR_TO_A).tolist()
    adsorbate_index = list(range(n_atoms - n_adsorbate_atoms, n_atoms))

    modes = []
    for omega, disp, cls in zip(frequencies_cm, modes_cart, classes):
        re = float(np.real(omega))
        im = float(np.imag(omega))
        freq = re if abs(re) > abs(im) else im  # store imag with positive magnitude
        modes.append({
            "freq_cm": freq,
            "is_imaginary": bool(abs(im) > abs(re)),
            "class": cls,
            "disp": np.asarray(disp).tolist(),
        })

    payload = {
        "level_of_theory": level_of_theory,
        "symbols": list(symbols),
        "geometry_A": geom_A,
        "adsorbate_atom_index": adsorbate_index,
        "modes": modes,
    }
    Path(filepath).write_text(json.dumps(payload, indent=2))
