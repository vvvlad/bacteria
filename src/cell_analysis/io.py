"""Loading and saving image stacks and results."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile


def load_stack(path: str | Path) -> np.ndarray:
    """Load a TIFF image stack.

    Parameters
    ----------
    path : str or Path
        Path to a .tif/.tiff file. Expected shapes:
        - Single channel time-lapse: (T, Y, X)
        - Multi-channel time-lapse: (T, C, Y, X)

    Returns
    -------
    np.ndarray
        The image stack.
    """
    return tifffile.imread(str(path))


def load_paired_stacks(
    phase_path: str | Path,
    fluor_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load paired phase-contrast and fluorescence stacks.

    Validates that both stacks have matching T, Y, X dimensions.

    Returns
    -------
    tuple of (phase_stack, fluor_stack)
        Both as np.ndarray with shape (T, Y, X).
    """
    phase = load_stack(phase_path)
    fluor = load_stack(fluor_path)

    # If multi-channel, take first channel
    if phase.ndim == 4:
        phase = phase[:, 0]
    if fluor.ndim == 4:
        fluor = fluor[:, 0]

    if phase.shape != fluor.shape:
        raise ValueError(
            f"Shape mismatch: phase {phase.shape} vs fluorescence {fluor.shape}. "
            "Both stacks must have identical (T, Y, X) dimensions."
        )
    return phase, fluor


def save_results(df, path: str | Path) -> None:
    """Save a pandas DataFrame of results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def export_notebook_html(
    notebook_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Export an executed notebook to a self-contained HTML file."""
    import nbformat
    from nbconvert import HTMLExporter

    exporter = HTMLExporter(
        exclude_input=True,
        exclude_input_prompt=True,
        exclude_output_prompt=True,
    )
    nb = nbformat.read(notebook_path, as_version=4)
    body, _ = exporter.from_notebook_node(nb)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body, encoding="utf-8")
    return output_path


def save_summary(data: dict, path: str | Path) -> None:
    """Flatten a nested summary dict and save as a single-row CSV.

    Recursively flattens nested dicts into underscore-joined column names.
    Skips array-like values (e.g. null distributions) that don't fit a
    single-row tabular format.
    """
    import pandas as pd

    def _flatten(d, prefix=""):
        flat = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(_flatten(v, prefix=f"{key}_"))
            elif hasattr(v, "__len__") and not isinstance(v, str):
                continue
            else:
                flat[key] = v
        return flat

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_flatten(data)]).to_csv(path, index=False)


def _sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_path(path: str | Path, repo_root: Path | None) -> str:
    p = Path(path).resolve()
    if repo_root is None:
        return str(p)
    repo_root_resolved = Path(repo_root).resolve()
    # Only return relative path if repo_root is the direct parent
    # Otherwise, always return absolute (path is outside or too nested)
    p_parent = p.parent
    if p_parent == repo_root_resolved:
        return str(p.name)
    # If repo_root is not the direct parent, return absolute
    return str(p)


def compute_provenance(
    extraction_params: dict,
    stack_path: str | Path,
    fluor_path: str | Path,
    *,
    repo_root: Path | None = None,
    package_version: str | None = None,
) -> dict:
    """Build a provenance dict for an extraction run.

    ``extract_hash`` = sha256 of (canonicalized params JSON + stack sha256
    + fluor sha256). ``cell_analysis_version`` is recorded for audit but
    is NOT part of ``extract_hash``.
    """
    stack_sha = _sha256_file(stack_path)
    fluor_sha = _sha256_file(fluor_path)
    params_canonical = json.dumps(extraction_params, sort_keys=True,
                                  separators=(",", ":"))
    h = hashlib.sha256()
    h.update(params_canonical.encode("utf-8"))
    h.update(stack_sha.encode("utf-8"))
    h.update(fluor_sha.encode("utf-8"))
    if package_version is None:
        try:
            from importlib.metadata import version
            package_version = version("cell-analysis")
        except Exception:
            package_version = "unknown"
    return {
        "extract_hash": h.hexdigest(),
        "stack_path": _canonical_path(stack_path, repo_root),
        "fluor_path": _canonical_path(fluor_path, repo_root),
        "stack_sha256": stack_sha,
        "fluor_sha256": fluor_sha,
        "params": extraction_params,
        "cell_analysis_version": package_version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
    }


def provenance_matches(existing: dict, candidate: dict) -> tuple[bool, list[str]]:
    """Compare two provenance dicts; return (match, drifted_fields).

    ``drifted_fields`` lists which top-level fields changed. For
    ``params``, entries look like ``params.KEY`` for each differing key.
    """
    if existing.get("extract_hash") == candidate.get("extract_hash"):
        return True, []
    drift: list[str] = []
    for field in ("stack_sha256", "fluor_sha256"):
        if existing.get(field) != candidate.get(field):
            drift.append(field)
    p_old = existing.get("params", {})
    p_new = candidate.get("params", {})
    for key in sorted(set(p_old) | set(p_new)):
        if p_old.get(key) != p_new.get(key):
            drift.append(f"params.{key}")
    return False, drift
