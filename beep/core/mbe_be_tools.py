"""Pure helpers for assembling MBE binding energies.

Ported from the standalone beep-mbe package (``models`` name helpers +
``assemble_be`` energy math). Everything here is server-free except
:func:`borrow_zpve_corrections`, which takes an already-connected client and
issues only read-only calls (it never submits or mutates datasets).

Sign / unit conventions (load-bearing, preserved from beep-mbe):
  * binding energy = negative of the (complex - fragments) energy difference
  * energies are converted Hartree -> kcal/mol
  * ``BE_1b/2b/3b`` are cumulative; ``*_contrib`` are per-body increments
"""
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import qcelemental

from ..adapters import qcfractal_adapter as qcf

logger = logging.getLogger("beep")

# qcmanybody property-name prefix per BSSE scheme.
BSSE_PREFIX = {
    "cp": "cp_corrected",
    "vmfc": "vmfc_corrected",
    "nocp": "nocp",
}


@dataclass
class MbeSubmissionResult:
    """Structured summary of an MBE submission (returned by the mbe workflow)."""
    dataset_name: str
    specification_names: List[str]
    entry_names: List[str]
    submitted_metadata: Optional[Any]
    fetched_statuses: Dict[Any, str]
    monitor_result: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Dataset naming (opt-LOT suffix convention)
# ---------------------------------------------------------------------------

def manybody_dataset_name(dataset: str, opt_level_of_theory: str) -> str:
    """ManybodyDataset server name = ``<dataset>_<opt_level_of_theory>``."""
    return f"{dataset}_{opt_level_of_theory}"


def monomer_dataset_name(small_molecule_collection: str, opt_level_of_theory: str) -> str:
    """Monomer SinglepointDataset name = ``<collection>_<opt_level_of_theory>``."""
    return f"{small_molecule_collection}_{opt_level_of_theory}"


def submitted_entry_names(
    surface_model: Optional[str],
    small_molecule: Optional[str],
    entries: Sequence[str],
) -> List[str]:
    """Ordered entry list submitted/monitored: surface first, then clusters.

    The small molecule is excluded (its monomer goes to the SinglepointDataset).
    Duplicates are removed preserving order.
    """
    ordered: List[str] = []
    if surface_model:
        ordered.append(surface_model)
    for name in entries:
        if small_molecule and name == small_molecule:
            continue
        ordered.append(name)

    seen = set()
    deduped: List[str] = []
    for name in ordered:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def dataset_entry_names(dataset) -> List[str]:
    """Best-effort list of entry names across qcportal dataset variants."""
    if getattr(dataset, "entry_names", None) is not None:
        return list(dataset.entry_names)
    if hasattr(dataset, "get_entries"):
        entries = dataset.get_entries()
        if isinstance(entries, dict):
            return list(entries.keys())
        names: List[str] = []
        for entry in entries:
            if isinstance(entry, str):
                names.append(entry)
            else:
                name = getattr(entry, "name", None) or getattr(entry, "entry_name", None)
                if name:
                    names.append(name)
        return names
    return []


def resolve_entries(dataset, entries: Optional[Sequence[str]], surface_entry: str) -> List[str]:
    """Explicit entry list, or all dataset entries minus the surface reference."""
    if entries:
        return list(entries)
    all_entries = dataset_entry_names(dataset)
    return [entry for entry in all_entries if entry != surface_entry]


# ---------------------------------------------------------------------------
# Energy extraction / BE math
# ---------------------------------------------------------------------------

def _convert_energy(value: Optional[float], factor: float) -> Optional[float]:
    if value is None:
        return None
    return value * factor


def extract_mbe_components(rec, pref: str) -> Dict[str, Optional[float]]:
    """Pull per-order energies from a ManybodyRecord's properties."""
    props = getattr(rec, "properties", None) or {}
    results = props.get("results", {}) or {}

    e1 = results.get(f"{pref}_total_energy_through_1_body")
    e2 = results.get(f"{pref}_2_body_contribution_to_energy")
    e3 = results.get(f"{pref}_3_body_contribution_to_energy")
    etot = results.get(f"{pref}_total_energy")
    if etot is None:
        etot = props.get("ret_energy")

    return {"e1": e1, "e2": e2, "e3": e3, "etot": etot}


def extract_monomer_energy(rec) -> float:
    """Pull the isolated-monomer reference energy from a SinglepointRecord."""
    value = getattr(rec, "return_result", None)
    props = getattr(rec, "properties", None) or {}
    if value is None:
        value = props.get("return_result")
    if value is None:
        results = props.get("results", {}) or {}
        value = results.get("return_energy")
    if value is None or not isinstance(value, (float, int)):
        from .exceptions import MbeExtractError
        raise MbeExtractError("Monomer record is missing a usable return_result energy.")
    return float(value)


def compute_be_values(
    super_components: Dict[str, Optional[float]],
    surface_components: Dict[str, Optional[float]],
    monomer_energy: float,
) -> Dict[str, Optional[float]]:
    """Binding energies (Hartree, BE-sign) from complex/surface MBE components."""
    e1_super = super_components.get("e1")
    e1_surf = surface_components.get("e1")
    e2_super = super_components.get("e2")
    e2_surf = surface_components.get("e2")
    e3_super = super_components.get("e3")
    e3_surf = surface_components.get("e3")
    etot_super = super_components.get("etot")
    etot_surf = surface_components.get("etot")

    be_le1 = None
    if e1_super is not None and e1_surf is not None:
        be_le1 = -(e1_super - (e1_surf + monomer_energy))

    be_2 = None
    if e2_super is not None and e2_surf is not None:
        be_2 = -(e2_super - e2_surf)

    be_3 = None
    if e3_super is not None and e3_surf is not None:
        be_3 = -(e3_super - e3_surf)

    be_total = None
    if etot_super is not None and etot_surf is not None:
        be_total = -(etot_super - (etot_surf + monomer_energy))

    return {"be_le1": be_le1, "be_2": be_2, "be_3": be_3, "be_total": be_total}


def build_cumulative(values: Dict[str, Optional[float]]) -> Dict[str, float]:
    """Cumulative n-body BEs: BE_1b, BE_1b+2b, BE_1b+2b+3b (NaN-propagating)."""
    c1 = values.get("be_le1")
    c2 = values.get("be_2")
    c3 = values.get("be_3")

    be_1b = c1
    if c1 is None:
        be_2b = None
        be_3b = None
    else:
        if c2 is None:
            be_2b = None
            be_3b = None
        else:
            be_2b = c1 + c2
            be_3b = None if c3 is None else c1 + c2 + c3

    return {
        "BE_1b": float("nan") if be_1b is None else float(be_1b),
        "BE_2b": float("nan") if be_2b is None else float(be_2b),
        "BE_3b": float("nan") if be_3b is None else float(be_3b),
    }


def build_contributions(values: Dict[str, Optional[float]]) -> Dict[str, float]:
    """Per-body BE increments (not cumulative)."""
    return {
        "BE_1b_contrib": float("nan") if values.get("be_le1") is None else float(values["be_le1"]),
        "BE_2b_contrib": float("nan") if values.get("be_2") is None else float(values["be_2"]),
        "BE_3b_contrib": float("nan") if values.get("be_3") is None else float(values["be_3"]),
    }


def compute_convergence(values: Dict[str, Optional[float]], tol: float = 0.05) -> Dict[str, Any]:
    """Estimate the MBE truncation error for one site/spec.

    The binding energy is the many-body sum ``BE = Δ₁ + Δ₂ + Δ₃ + …`` where
    ``Δ₁ = be_le1`` (monomer deformation), ``Δ₂ = be_2`` (2-body) and
    ``Δ₃ = be_3`` (3-body). The uncomputed tail (Δ₄ + …) is estimated by
    assuming the interaction increments keep shrinking geometrically at the
    rate seen between the two highest computed orders:

        r    = |Δₙ / Δₙ₋₁|                 (interaction increments only)
        bar  = |Δₙ| · r / (1 − r)          (magnitude of the geometric tail)

    ``bar`` is reported as a **symmetric** ± uncertainty on ``BE_total`` — never
    as a signed correction, because the *direction* of the next term is not
    predictable (the increment sign can flip, as it does for CO on cd5). Only
    the *magnitude* of the decay is inferable.

    Requirements / edge cases:
      * needs two interaction increments (Δ₂ and Δ₃), i.e. a 3-body run. A
        2-body-only run has no ratio → ``error_bar`` is ``None`` ("n/a"): a
        2-body calculation carries no information about its own convergence.
      * ``|r| ≥ 1`` (increment did not shrink) → not converging → ``error_bar``
        ``None`` and ``converged`` ``False``.
      * ``Δ₁`` (deformation) is excluded from the ratio.

    Returns a dict with ``n_body_max``, ``delta_last``, ``ratio_r``,
    ``error_bar`` (kcal/mol, or ``None``), ``rel_error`` (or ``None``) and
    ``converged`` (``bool`` or ``None`` when not estimable).
    """
    d2 = values.get("be_2")
    d3 = values.get("be_3")
    be_total = values.get("be_total")

    # Highest computed body order (1 = deformation only).
    if d3 is not None:
        n_body_max = 3
    elif d2 is not None:
        n_body_max = 2
    else:
        n_body_max = 1

    result: Dict[str, Any] = {
        "n_body_max": n_body_max,
        "delta_last": None,
        "ratio_r": None,
        "error_bar": None,
        "rel_error": None,
        "converged": None,
    }

    # Need two interaction increments (Δ₂, Δ₃) to form a decay ratio.
    if d2 is None or d3 is None or d2 == 0.0:
        result["delta_last"] = d3 if d3 is not None else d2
        return result

    r = abs(d3 / d2)
    result["delta_last"] = d3
    result["ratio_r"] = r

    if r >= 1.0:
        # Series is not shrinking — cannot bound the tail; flag non-converged.
        result["converged"] = False
        return result

    bar = abs(d3) * r / (1.0 - r)
    result["error_bar"] = bar
    if be_total is not None and be_total != 0.0:
        rel = bar / abs(be_total)
        result["rel_error"] = rel
        result["converged"] = rel < tol
    return result


def resolve_bsse(bsse: Sequence[str]):
    """Return (scheme, property-prefix) for the single configured BSSE scheme."""
    from .exceptions import MbeExtractError
    schemes = list(bsse or [])
    if len(schemes) != 1:
        raise MbeExtractError(
            "Config must specify exactly one BSSE scheme for reporting (e.g. bsse: ['vmfc'])."
        )
    scheme = schemes[0]
    pref = BSSE_PREFIX.get(scheme.lower())
    if pref is None:
        raise MbeExtractError(f"Unknown BSSE scheme '{scheme}'.")
    return scheme, pref


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def safe_filename(value: str) -> str:
    """Sanitize a spec name for use in a CSV filename."""
    safe = value
    for token in ("/", "\\", ":", " ", "__"):
        safe = safe.replace(token, "_")
    return safe


def _format_float(value) -> str:
    if value is None:
        return "NaN"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{float(value): .6f}"


def format_convergence_table(conv_by_entry: Dict[str, Dict[str, Any]]) -> str:
    """Render the per-site MBE truncation-error summary for the text report.

    One row per site: ``BE_total   ± error_bar   [n-body]   converged?``. Sites
    without a computable bar (2-body runs, or a non-shrinking series) show
    ``n/a``.
    """
    headers = ["entry", "BE_total", "error_bar", "n_body", "converged"]
    rows: List[List[str]] = []
    for entry, c in conv_by_entry.items():
        bar = c.get("error_bar")
        n = c.get("n_body_max")
        conv = c.get("converged")
        be = c.get("BE_total")
        if bar is None:
            bar_str = "n/a (2b)" if n == 2 else "n/a (not conv.)"
        else:
            bar_str = f"± {bar:.6f}"
        conv_str = {True: "yes", False: "no", None: "-"}[conv]
        be_str = "NaN" if be is None or (isinstance(be, float) and math.isnan(be)) else f"{be: .6f}"
        rows.append([str(entry), be_str, bar_str, f"{n}b", conv_str])

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)] if rows else [len(h) for h in headers]
    lines = ["  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))]
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        lines.append("  ".join(r[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def render_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a fixed-width ASCII table for the text report."""
    headers = ["entry", *[str(col) for col in df.columns]]
    rows: List[List[str]] = []
    for entry in df.index:
        row = [str(entry)]
        for col in df.columns:
            row.append(_format_float(df.at[entry, col]))
        rows.append(row)

    widths = []
    for idx, header in enumerate(headers):
        column_values = [header]
        column_values.extend(row[idx] for row in rows)
        widths.append(max(len(value) for value in column_values))

    lines = []
    lines.append("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ZPVE borrow (read-only) — populated in the ZPVE step
# ---------------------------------------------------------------------------

def _discover_hessian_clusters(client, molecule, opt_method, opt_basis, logger) -> List[str]:
    """Find be_hess cluster names for ``molecule`` at the given LOT (read-only).

    Matches reaction datasets named ``be_<MOL>_<CLUSTER>_<METHOD>_<BASIS>_be_nocp``
    (or the legacy method-only form) and extracts ``<CLUSTER>`` by stripping the
    known prefix/suffix, so clusters containing underscores (e.g. ``cd5_01``) are
    recovered intact.
    """
    prefix = f"be_{molecule.upper()}_"
    suffixes = [
        f"_{opt_method.upper()}_{opt_basis.upper()}_be_nocp",
        f"_{opt_method.upper()}_be_nocp",
    ]
    clusters = []
    for name in qcf.list_reaction_dataset_names(client):
        if not name.startswith(prefix):
            continue
        for suf in suffixes:
            if name.endswith(suf):
                clusters.append(name[len(prefix):-len(suf)].lower())
                break
    found = sorted(set(clusters))
    logger.info(f"Auto-discovered {len(found)} be_hess cluster(s) for {molecule}: {found}")
    return found


def borrow_zpve_corrections(
    client,
    molecule: str,
    hessian_clusters: Optional[Sequence[str]],
    opt_lot: str,
    entry_names: Sequence[str],
    scale_factor: float = 0.958,
    imag_threshold: float = 50.0,
    logger=logger,
) -> "pd.Series":
    """Per-site ZPVE correction (kcal/mol) borrowed read-only from be_hess.

    Locates the ``be_<MOL>_<CLUSTER>_<METHOD>_<BASIS>`` reaction datasets a prior
    ``be_hess`` run produced, reads the counterpoise (``be_nocp``) stoichiometry
    rows to get the dimer + two fragment molecule IDs per site, and computes the
    ZPVE correction from the corresponding Hessians. Never submits anything;
    sites without a usable Hessian are returned as NaN.

    When ``hessian_clusters`` is ``None`` the cluster list is auto-discovered from
    the server (all ``be_<MOL>_*`` reaction datasets at this LOT).

    Returns a Series indexed by site label (``Delta_ZPVE``, kcal/mol).
    """
    hartree_to_kcal = qcelemental.constants.hartree2kcalmol
    opt_method, _, opt_basis = opt_lot.partition("_")

    if hessian_clusters is None:
        hessian_clusters = _discover_hessian_clusters(
            client, molecule, opt_method, opt_basis, logger
        )

    # Gather the be_nocp stoichiometry rows across the requested clusters.
    frames = []
    for cluster in hessian_clusters:
        base = f"be_{molecule.upper()}_{cluster.upper()}_{opt_method.upper()}_{opt_basis.upper()}"
        if not qcf.check_collection_exists(client, "ReactionDataset", f"{base}_be_nocp"):
            # Legacy fallback: method-only dataset name (matches extract.py).
            base = f"be_{molecule.upper()}_{cluster.upper()}_{opt_method.upper()}"
            if not qcf.check_collection_exists(client, "ReactionDataset", f"{base}_be_nocp"):
                logger.info(
                    f"No be_nocp reaction dataset for cluster {cluster}; skipping (run be_hess to fill it)."
                )
                continue
        try:
            frames.append(qcf.fetch_reaction_entries(client, base, stoich="be_nocp"))
        except Exception as exc:
            logger.info(f"Could not read be_nocp entries for {base}: {exc}")

    if not frames:
        logger.info("No be_hess Hessian datasets found; ZPVE corrections are all NaN.")
        return pd.Series({entry: float("nan") for entry in entry_names}, name="Delta_ZPVE")

    df_rxn = pd.concat(frames)

    corrections: Dict[str, float] = {}
    for entry in entry_names:
        rows = df_rxn[df_rxn["name"] == entry] if "name" in df_rxn.columns else df_rxn.loc[df_rxn.index == entry]
        if len(rows) != 3:
            logger.info(
                f"Site {entry}: expected 3 be_nocp rows (dimer + 2 fragments), found {len(rows)}; NaN."
            )
            corrections[entry] = float("nan")
            continue

        dimer_ids = list(rows.loc[rows["coefficient"] > 0, "molecule"])
        frag_ids = list(rows.loc[rows["coefficient"] < 0, "molecule"])
        if len(dimer_ids) != 1 or len(frag_ids) != 2:
            logger.info(
                f"Site {entry}: unexpected stoichiometry (dimer={len(dimer_ids)}, frags={len(frag_ids)}); NaN."
            )
            corrections[entry] = float("nan")
            continue

        zpves = []
        ok = True
        for mol_id in [dimer_ids[0], *frag_ids]:
            zpve, real = qcf.get_zpve_mol(
                client, mol_id, opt_lot, on_imaginary="return", imag_threshold=imag_threshold
            )
            if zpve is None or not real:
                ok = False
                break
            zpves.append(zpve)

        if not ok:
            logger.info(f"Site {entry}: no usable Hessian yet; NaN (run be_hess to fill it).")
            corrections[entry] = float("nan")
            continue

        d, m1, m2 = zpves
        corrections[entry] = (d - m1 - m2) * hartree_to_kcal * scale_factor

    return pd.Series(corrections, name="Delta_ZPVE").reindex(entry_names)
