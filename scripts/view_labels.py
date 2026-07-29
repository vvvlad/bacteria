"""Open a run's label stacks (and raw images) in napari for inspection.

Usage:
    uv run python scripts/view_labels.py results/control_experiment/
    uv run python scripts/view_labels.py results/control_experiment/extraction/
    uv run python scripts/view_labels.py results/control_experiment/ --no-raw

The path can point to a run directory (containing `extraction/`) or
directly to the `extraction/` folder. With `--no-raw`, the phase and
fluorescence TIFFs are skipped — useful when the raw stacks live on a
slow or unmounted drive.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def resolve_extraction_dir(user_path: Path) -> Path:
    p = user_path.resolve()
    if (p / "provenance.json").exists():
        return p
    if (p / "extraction" / "provenance.json").exists():
        return p / "extraction"
    raise FileNotFoundError(
        f"No extraction bundle at {p} or {p / 'extraction'}. "
        "Pass a run directory (e.g. results/control_experiment/) or "
        "its extraction subfolder."
    )


# (uses resolve_provenance_path from cell_analysis.io)


def main():
    parser = argparse.ArgumentParser(
        description="Open a run's label stacks (and raw images) in napari.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Run directory or its extraction/ subfolder",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip loading raw phase/fluor TIFFs (labels only)",
    )
    args = parser.parse_args()

    extraction_dir = resolve_extraction_dir(args.path)
    print(f"Loading from {extraction_dir}")

    label_stack = np.load(extraction_dir / "label_stack.npz")["label_stack"]
    nucleus_label_stack = np.load(
        extraction_dir / "nucleus_label_stack.npz"
    )["nucleus_label_stack"]
    print(
        f"  label_stack: {label_stack.shape} {label_stack.dtype}, "
        f"max_id={int(label_stack.max())}"
    )
    print(
        f"  nucleus_label_stack: {nucleus_label_stack.shape} "
        f"{nucleus_label_stack.dtype}, max_id={int(nucleus_label_stack.max())}"
    )

    phase_stack = fluor_stack = None
    if not args.no_raw:
        from cell_analysis.io import load_stack, resolve_provenance_path

        provenance = json.loads(
            (extraction_dir / "provenance.json").read_text(encoding="utf-8")
        )
        stack_path = resolve_provenance_path(provenance["stack_path"], repo_root=REPO_ROOT)
        fluor_path = resolve_provenance_path(provenance["fluor_path"], repo_root=REPO_ROOT)
        for name, p in [("phase", stack_path), ("fluor", fluor_path)]:
            if not p.exists():
                print(f"  WARNING: {name} stack not found at {p}; skipping")
                continue
        if stack_path.exists():
            phase_stack = load_stack(stack_path)
            if phase_stack.ndim == 4:
                phase_stack = phase_stack[:, 0]
            print(f"  phase_stack: {phase_stack.shape} {phase_stack.dtype}")
        if fluor_path.exists():
            fluor_stack = load_stack(fluor_path)
            if fluor_stack.ndim == 4:
                fluor_stack = fluor_stack[:, 0]
            print(f"  fluor_stack: {fluor_stack.shape} {fluor_stack.dtype}")

    import napari

    viewer = napari.Viewer()
    if phase_stack is not None:
        viewer.add_image(phase_stack, name="phase", colormap="gray")
    if fluor_stack is not None:
        viewer.add_image(fluor_stack, name="fluor", colormap="green",
                         blending="additive")
    viewer.add_labels(label_stack, name="cells (phase masks)")
    viewer.add_labels(
        nucleus_label_stack, name="nuclei (fluor masks)", visible=False
    )
    napari.run()


if __name__ == "__main__":
    main()
