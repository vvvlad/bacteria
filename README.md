# Cell Analysis Pipeline

Napari-based pipeline for bacterial cell tracking and fluorescence analysis in time-lapse phase-contrast microscopy.

The pipeline detects individual cells using [Cellpose](https://github.com/MouseLand/cellpose), links them across frames with [trackpy](https://github.com/soft-matter/trackpy), and measures fluorescence intensity per cell. Results are exported as CSV files for downstream analysis.

## Overview

| Step | Module | Description |
|------|--------|-------------|
| 1 | `cell_analysis.io` | Load single- or multi-channel TIFF stacks |
| 2 | `cell_analysis.segmentation` | Detect cells per frame (Cellpose or classical fallback) |
| 3 | `cell_analysis.tracking` | Link detections into tracks across time |
| 4 | `cell_analysis.matching` | Match phase-contrast cells to fluorescence nuclei |
| 5 | `cell_analysis.io` | Export tracked data and statistics to CSV |

The main entry point is the Jupyter notebook at `notebooks/analysis.ipynb`.

## Prerequisites

- **Python 3.11 or newer**
- **uv** (Python package manager)
- **VS Code** with the Jupyter extension (recommended) or JupyterLab
- **Git**

---

## Environment Setup

<details>
<summary><strong>macOS</strong></summary>

### 1. Install Homebrew (if not already installed)

Open **Terminal** (Cmd+Space, type "Terminal") and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen instructions. After installation, restart your terminal or run the commands shown at the end of the installer to add Homebrew to your PATH.

### 2. Install Git

macOS includes Git via Xcode Command Line Tools. If you don't have it:

```bash
xcode-select --install
```

Or install via Homebrew:

```bash
brew install git
```

Verify:

```bash
git --version
```

### 3. Install Python 3.11+

```bash
brew install python@3.11
```

Verify:

```bash
python3 --version
```

### 4. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal, then verify:

```bash
uv --version
```

### 5. Install VS Code

Download from https://code.visualstudio.com/download and drag the `.app` to your Applications folder.

To launch VS Code from the terminal, open VS Code, press **Cmd+Shift+P**, type "Shell Command: Install 'code' command in PATH", and select it. Then you can run:

```bash
code .
```

### 6. Install VS Code Extensions

Open VS Code and install the following extensions (Cmd+Shift+X to open the Extensions panel):

- **Python** (`ms-python.python`) — Python language support and interpreter selection
- **Jupyter** (`ms-toolsai.jupyter`) — Run `.ipynb` notebooks inside VS Code

Or install from the terminal:

```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
```

### 7. Clone the Repository

```bash
cd ~/Projects  # or wherever you keep your repos
git clone https://github.com/vvvlad/bacteria.git
cd bacteria
```

### 8. Create the Virtual Environment and Install Dependencies

```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock`, creates a `.venv/` directory, and installs all pinned dependencies into it.

### 9. Select the Python Interpreter in VS Code

1. Open the project folder in VS Code: `code .`
2. Press **Cmd+Shift+P** and type **"Python: Select Interpreter"**
3. Choose the interpreter at `./.venv/bin/python`

This ensures both the editor and Jupyter notebooks use the project's virtual environment.

### 10. Run an Experiment

Place your TIFF stacks in a folder under `data/` (e.g. `data/my_experiment/phase.tif` and `fluorescence.tif`), then copy and edit a config:

```bash
cp configs/run_01.yaml configs/my_experiment.yaml
# Edit configs/my_experiment.yaml: update RUN_NAME, STACK_PATH, FLUOR_PATH
```

Run the experiment:

```bash
uv run python scripts/run_experiment.py configs/my_experiment.yaml
```

Results (HTML report + CSVs) are saved to `results/<run_name>/`.

### Notes

- **Apple Silicon (M1/M2/M3):** Cellpose uses MPS GPU acceleration on Apple Silicon when `gpu: true`. First run downloads the Cellpose model (~1 GB).
- **Large files:** Raw TIFF stacks can be 50-100+ MB per stack. They are excluded from git via `.gitignore`.

</details>

<details>
<summary><strong>Windows</strong></summary>

### 1. Install Git

Download the installer from https://git-scm.com/download/win and run it.

During installation:
- Keep the default options
- When asked about adjusting your PATH, select **"Git from the command line and also from 3rd-party software"**
- When asked about line endings, select **"Checkout as-is, commit Unix-style line endings"**

Open a new **PowerShell** window and verify:

```powershell
git --version
```

### 2. Install Python 3.11+

Download the installer from https://www.python.org/downloads/ (choose 3.11 or newer).

During installation:
- **Check "Add python.exe to PATH"** at the bottom of the first screen
- Click **"Install Now"**

Open a new PowerShell window and verify:

```powershell
python --version
```

### 3. Install uv

In PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell, then verify:

```powershell
uv --version
```

### 4. Install VS Code

Download the installer from https://code.visualstudio.com/download and run it.

During installation:
- **Check "Add to PATH"** so you can launch VS Code from the terminal
- **Check "Register Code as an editor for supported file types"** (optional)

Restart PowerShell, then verify:

```powershell
code --version
```

### 5. Install VS Code Extensions

Open VS Code and install the following extensions (Ctrl+Shift+X to open the Extensions panel):

- **Python** (`ms-python.python`) — Python language support and interpreter selection
- **Jupyter** (`ms-toolsai.jupyter`) — Run `.ipynb` notebooks inside VS Code

Or install from PowerShell:

```powershell
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
```

### 6. Clone the Repository

```powershell
cd ~\Projects  # or wherever you keep your repos
git clone https://github.com/vvvlad/bacteria.git
cd bacteria
```

### 7. Create the Virtual Environment and Install Dependencies

```powershell
uv sync
```

This reads `pyproject.toml` and `uv.lock`, creates a `.venv\` directory, and installs all pinned dependencies into it.

### 8. Select the Python Interpreter in VS Code

1. Open the project folder in VS Code: `code .`
2. Press **Ctrl+Shift+P** and type **"Python: Select Interpreter"**
3. Choose the interpreter at `.\.venv\Scripts\python.exe`

This ensures both the editor and Jupyter notebooks use the project's virtual environment.

### 9. Run an Experiment

Place your TIFF stacks in a folder under `data\` (e.g. `data\my_experiment\phase.tif` and `fluorescence.tif`), then copy and edit a config:

```powershell
Copy-Item configs\run_01.yaml configs\my_experiment.yaml
# Edit configs\my_experiment.yaml: update RUN_NAME, STACK_PATH, FLUOR_PATH
```

Run the experiment:

```powershell
uv run python scripts\run_experiment.py configs\my_experiment.yaml
```

Results (HTML report + CSVs) are saved to `results\<run_name>\`.

### Notes

- **NVIDIA GPU (optional):** If you have an NVIDIA GPU with CUDA drivers, set `gpu: true` in the config for faster Cellpose inference. CPU mode works fine without it.
- **First run:** Cellpose downloads its pretrained model (~1 GB) on first use. Make sure you have a stable internet connection.
- **Large files:** Raw TIFF stacks can be 50-100+ MB per stack. They are excluded from git via `.gitignore`.
- **Long paths:** If you encounter path-length errors, enable long paths in Windows:
  ```powershell
  git config --global core.longpaths true
  ```

</details>

---

## Project Structure

```
bacteria/
├── configs/
│   └── run_01.yaml           # One YAML config per experiment
├── data/
│   └── gradient_0011/        # One folder per dataset
│       ├── phase.tif         # Phase-contrast channel
│       └── fluorescence.tif  # Fluorescence channel (background-subtracted)
├── notebooks/
│   └── analysis.ipynb        # Analysis notebook (also papermill template)
├── results/
│   └── run_01/                       # One folder per run (see "Output artifacts")
│       ├── config.yaml               # Frozen copy of config used
│       ├── extraction/               # Image → identities (expensive, cached)
│       │   ├── provenance.json
│       │   ├── label_stack.npz       # Phase-contrast Cellpose masks (T,Y,X) int32
│       │   ├── nucleus_label_stack.npz  # Fluor Cellpose masks
│       │   └── *.csv                 # tracked_cells, track_statistics, ...
│       └── analysis/                 # Identities → metrics + plots (fast, iterated)
│           ├── report.html           # Self-contained HTML report
│           └── *.csv                 # enriched tracked, fate predictions, ...
├── docs/                             # Real docs (Markdown, images)
│   └── reports/                      # Publishing view over results/ (git-trackable)
│       ├── index.html                # Landing page listing all runs
│       └── <run>/report.html         # Report copy with a "back to index" link
├── scripts/
│   ├── run_experiment.py             # Two-phase runner (extraction + analysis)
│   ├── view_labels.py                # Open a run's label stacks in napari
│   └── diagnostic_overlay.py         # Visual debugging of detection filters
├── src/
│   └── cell_analysis/
│       ├── __init__.py
│       ├── io.py             # TIFF loading and CSV export
│       ├── segmentation.py   # Cell detection (Cellpose + classical)
│       ├── tracking.py       # Temporal linking with trackpy
│       ├── matching.py       # Phase-to-fluorescence cell matching
│       ├── pipeline.py       # High-level pipeline orchestration
│       └── plotting.py       # All visualization functions
├── tests/
├── .gitignore
├── pyproject.toml
└── uv.lock
```

## Adding a New Dataset

Create a folder under `data/` with a descriptive name and place the two TIFF stacks inside:

```
data/
  my_new_experiment/
    phase.tif            # Phase-contrast channel
    fluorescence.tif     # Fluorescence channel (background-subtracted)
```

Expected TIFF formats:

- **Shape:** `(T, Y, X)` — e.g. 25 frames of 1040x1388 pixels
- **Multi-channel:** `(T, C, Y, X)` is also supported (first channel is used)
- **Bit depth:** uint8 or uint16

Then create a config file (copy `configs/run_01.yaml` and update the paths and run name).

## Running Experiments

### Via CLI (recommended)

The runner drives two notebooks per config — `notebooks/extract.ipynb`
(image → identities, expensive Cellpose + tracking) and
`notebooks/analysis.ipynb` (identities → derived metrics + plots, fast).
It skips extraction when nothing that would change it has changed,
so re-running to iterate on analysis takes seconds instead of minutes.

**Single config:**

```bash
uv run python scripts/run_experiment.py configs/run_01.yaml
```

**Multiple chained configs** (runs sequentially, in the order given):

```bash
uv run python scripts/run_experiment.py configs/control_experiment.yaml configs/protein_synthesis_arrested.yaml
```

**All configs in `configs/`** (shell glob):

```bash
uv run python scripts/run_experiment.py configs/*.yaml
```

Each run writes to its own `results/<RUN_NAME>/{extraction,analysis}/`
folders (see [Output artifacts](#output-artifacts) for the layout).
When multiple configs are passed, the script prints a summary table at
the end and exits with code 0 if all runs succeeded, or 1 if any failed.
Failures do **not** stop the remaining runs — each config is attempted
independently and a failed run leaves `report_failed.ipynb` alongside
`report.html` for debugging.

### Extraction reuse

Every extraction run writes a `provenance.json` alongside its outputs.
It contains an `extract_hash` — sha256 of the extraction params (from
the YAML `extraction:` section, excluding path strings) combined with
the sha256 of both source TIFFs. On the next run the runner compares:

- If `extract_hash` matches → **reuse**: prints
  `reusing extraction at <path>` and jumps straight to analysis.
- If artifacts are missing or the hash differs → **re-extract**: prints
  `extracting: <reason>` (e.g. `missing artifacts (label_stack.npz)` or
  `param drift: params.GATING_Z_THRESHOLD`) and re-runs Cellpose.

Analysis always runs (it's cheap). This means iterating on
`analysis:` params (`PIXEL_SIZE_UM`, `PERI_CORE_RINGS`, plot styling,
etc.) never triggers Cellpose.

### CLI flags

| Flag | Effect |
|------|--------|
| `--force-extract` | Re-run extraction even when the provenance hash matches. Use after changing extraction *code* (which the hash doesn't track). |
| `--skip-analysis` | Run extraction only. Useful for pre-baking extractions across many datasets. |
| `--analysis-only` | Skip extraction; error if artifacts are missing. If provenance hash drifted from the current config, prints a loud warning listing the drifted fields but proceeds. |
| `--results-root PATH` | Override the results root for this invocation (wins over YAML `RESULTS_ROOT`; falls back to `<repo>/results/`). |

### Via Jupyter (interactive)

For development or one-off analysis, open either notebook directly:

```bash
uv run jupyter lab notebooks/extract.ipynb   # to re-extract manually
uv run jupyter lab notebooks/analysis.ipynb  # to play with plots + metrics
```

Both notebooks have a `parameters`-tagged cell with defaults for
`control_experiment` so opening them and running-all "just works"
against the existing extraction on disk. Edit that cell to switch
runs. When executed via the CLI runner, papermill overwrites those
defaults with the YAML config's values.

### How the notebooks and CLI runner relate

The CLI runner is **not** a separate reimplementation — it executes the
two notebooks:

1. `run_experiment.py` reads the YAML config and calls
   `validate_config`, which enforces the two-section schema (`extraction:` +
   `analysis:`) and per-section allowlists.
2. For extraction: checks `provenance.json` against a freshly computed
   hash of the current `extraction:` section and source-file sha256s.
   If stale (or `--force-extract`), papermill executes
   `notebooks/extract.ipynb` with the `extraction:` section injected as
   parameters, plus the resolved absolute `STACK_PATH`/`FLUOR_PATH` and
   `RESULTS_ROOT`.
3. For analysis: papermill executes `notebooks/analysis.ipynb` with the
   `analysis:` section injected as parameters, reading the extraction
   bundle from disk via `load_extraction(...)`. Executed notebook is
   rendered to `results/<RUN_NAME>/analysis/report.html`.
4. `publish_reports()` copies each `report.html` to `docs/reports/<RUN>/`
   and rebuilds `docs/reports/index.html` — real Markdown docs live
   directly under `docs/`, undisturbed.

Consequences:

- **New cells you add to `analysis.ipynb` show up in `report.html`.**
- **New config knobs must be registered.** Add the key to the
  notebook's `parameters` cell and to the appropriate allowlist
  (`EXTRACTION_ALLOWED` / `EXTRACTION_TYPES` or `ANALYSIS_ALLOWED` /
  `ANALYSIS_TYPES`) in `scripts/run_experiment.py`.
- **Cell failures abort the run.** Papermill stops at the first
  exception; the partially-executed notebook is saved as
  `report_failed.ipynb` in the run's `analysis/` folder.
- **The notebooks are the single source of truth** for pipeline logic;
  the CLI just parameterizes them.

## Config Parameters

Configs use a two-section schema with `RUN_NAME` and an optional
`RESULTS_ROOT` at the top level. See `configs/control_experiment.yaml`
for a complete example.

```yaml
RUN_NAME: my_experiment
# RESULTS_ROOT: /path/to/cloud-synced/folder   # optional override

extraction:
  STACK_PATH: "../data/my_experiment/phase.tif"
  FLUOR_PATH: "../data/my_experiment/fluorescence.tif"
  # ... extraction params (see table) ...

analysis:
  # ... analysis params (see table) ...
```

**Which section owns which knob:** anything that affects the produced
`label_stack.npz` / tracked identities goes in `extraction:` and
triggers a Cellpose re-run when changed. Anything that only reshapes
already-extracted data (pixel-size calibration, plot styling, feature
selection) goes in `analysis:` and re-runs in seconds.

### Top-level keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `RUN_NAME` | str | *(required)* | Unique per-experiment name. Results go to `<results_root>/<RUN_NAME>/`. No `..`, `/`, or `\`. |
| `RESULTS_ROOT` | str | `<repo>/results` | Override root for this experiment. `--results-root` on the CLI overrides both. |
| `extraction` | dict | *(required)* | Extraction-phase params (below). |
| `analysis` | dict | `{}` | Analysis-phase params (below). Empty section is allowed. |

### `extraction:` — Cellpose + tracking + gating

Changing any of these keys invalidates cached extractions.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `STACK_PATH` | str | *(required)* | Path to phase-contrast TIFF, relative to `notebooks/` |
| `FLUOR_PATH` | str | *(required)* | Path to fluorescence TIFF, relative to `notebooks/` |
| `MODEL_TYPE` | str | `cyto3` | Cellpose model name |
| `DETECT_PARAMS.diameter` | int | 32 | Median cell diameter in pixels |
| `DETECT_PARAMS.min_area` | int | 300 | Minimum cell area (rejects debris) |
| `DETECT_PARAMS.min_circularity` | float | 0.7 | Minimum circularity (1.0 = perfect circle) |
| `DETECT_PARAMS.min_contrast` | int | 1250 | Minimum intensity contrast (rejects faded cells) |
| `DETECT_PARAMS.exclude_edges` | bool | true | Drop cells touching the frame border |
| `DETECT_PARAMS.gpu` | bool | true | GPU acceleration (MPS on Apple Silicon, CUDA on NVIDIA) |
| `DETECT_PARAMS.resample` | bool | false | Resample masks (slower, more precise boundaries) |
| `GATING_Z_THRESHOLD` | float | 3.5 | MAD-based Z-score threshold for flagging bad frames |
| `SEARCH_RANGE` | float | 30.0 | Max cell displacement between frames (pixels) |
| `MEMORY` | int | 3 | Frames a cell can disappear before breaking the track |
| `MERGE_MAX_DISTANCE` | float | 15.0 | Max pixels between track end/start to merge fragments |
| `MERGE_MAX_GAP` | int | 18 | Max frame gap for fragment merging |
| `MIN_TRACK_DETECTIONS` | int | 4 | Minimum frames a track must span to be kept. Lives here (not in analysis) because the filter drops identities before persistence. |
| `NUCLEUS_DIAMETER` | int | 25 | Cellpose diameter for the fluorescence-channel nucleus segmentation |
| `NUCLEUS_MIN_AREA` | int | 100 | Minimum nucleus area |

### `analysis:` — geometry + metrics + fate prediction

Change these freely; only the analysis phase re-runs.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `PIXEL_SIZE_UM` | float | 0.0645 | µm-per-pixel calibration. Scales `area` to µm², derives `radius`, `volume`, `surface_area`. Pass `1.0` to keep pixel units. |
| `BASELINE_FRAMES` | int | 3 | Number of initial frames used for the §6.3 area-histogram baseline (before medium changes drive swelling) |
| `PERI_CORE_RINGS` | list of 3 floats | `[0.15, 0.35, 0.55]` | `(core_max, peri_min, peri_max)` as fractions of the cell's equivalent radius R. Defines the inner disk and outer annulus for the peri/core asymmetry metric. |
| `FLUOR_ALIGN_WINDOW` | int | 5 | Frames on each side of disappearance for the fluorescence-alignment tables (`fluorescence_alignment.csv`, `peri_core_alignment.csv`) |
| `FATE_FEATURES_FULL` | list of str | `[area, cv, nnrm, mean_edge_distance_norm, gaussian_sigma_norm]` | Frame-0 columns fed to the logistic-regression fate predictor |
| `FATE_FEATURES_NO_AREA` | list of str | `[cv, nnrm, mean_edge_distance_norm, gaussian_sigma_norm]` | Same, without `area` — for the ablation comparison |

## Output artifacts

Each run is split across two subfolders under `results/<run_name>/`, matching
the two phases of the pipeline. Extraction is expensive (Cellpose + tracking)
and is cached; analysis is fast and re-runs freely as you iterate on plots
and metrics.

### `results/<run_name>/extraction/` — image → identities

Written by `notebooks/extract.ipynb`. Skipped on subsequent runs when the
config's extraction section and the source-file hashes still match
`provenance.json`.

| File | Format | Contents |
|---|---|---|
| `provenance.json` | JSON | `{extract_hash, stack_path, fluor_path, stack_sha256, fluor_sha256, params, cell_analysis_version, timestamp_utc}`. `extract_hash` = sha256 of extraction params (excluding `STACK_PATH`/`FLUOR_PATH` strings) + both file sha256s. Runner compares against this to decide reuse vs. re-extract. |
| `label_stack.npz` | `np.savez_compressed`, key `label_stack` | `(T, Y, X) int32` Cellpose masks on the phase-contrast channel. `0` = background, `N` = per-frame mask id. Bad frames (from gating) are zeroed. |
| `nucleus_label_stack.npz` | key `nucleus_label_stack` | Same shape, Cellpose masks on the fluorescence channel. |
| `tracked_cells.csv` | CSV | `frame, track_id, label, centroid_y, centroid_x, area` — post-merge track ids joined with the per-frame `label` (which indexes into `label_stack`). |
| `track_statistics.csv` | CSV | `track_id, first_frame, last_frame, mean_area, num_detections, lifetime, disappeared` |
| `frame_diagnostics.csv` | CSV | Per-frame z-score gating stats |
| `merge_log.csv` | CSV | trackpy → merged track_id remapping |
| `dropped_frames.csv` | CSV | Bad-frame audit (empty when nothing was dropped) |

### `results/<run_name>/analysis/` — identities → metrics + plots

Written by `notebooks/analysis.ipynb`. Rebuilt every run.

| File | Description |
|---|---|
| `report.html` | Self-contained HTML report with all plots and statistics |
| `tracked_cells.csv` | Enriched per-cell per-frame — adds fluorescence, nucleoid metrics (`cv`, `nnrm`, `mean_edge_distance_norm`, `gaussian_sigma_norm`, `peri_core_asymmetry`), geometry (`radius`, `volume`, `surface_area`, `volume_rel`), `fluor_concentration`, `sav_ratio`. |
| `track_statistics.csv` | Enriched per-track summary — adds means (`mean_fluor_intensity`, `mean_cv`, etc.) plus pre-burst columns |
| `fluorescence_alignment.csv` | Long-form table of F(offset)/F(-window) around each disappearing cell |
| `peri_core_alignment.csv` | Same shape, but for peri/core asymmetry |
| `nucleus_persistence.csv` | Per-frame phase-cell vs. fluorescence-nucleus counts |
| `nucleus_persistence_summary.csv` | Endpoint / trajectory test verdict (parallel vs. divergent) |
| `frame0_fate_comparison.csv` | Frame-0 Mann-Whitney U per feature (survived vs. died) |
| `fate_predictions.csv` | Per-cell logistic-regression predictions (LOO CV) |
| `fate_prediction_summary.csv` | AUC, accuracy, feature importances |
| `fate_predictions_no_area.csv`, `fate_prediction_summary_no_area.csv` | Same LR but with `area` dropped |

`results/<run_name>/config.yaml` is a frozen copy of the YAML config used for the run.

### Overriding the results root

By default, both subfolders live at `<repo>/results/<run_name>/`. To keep
big `.npz` files out of the repo (e.g. in a cloud-synced folder), set
`RESULTS_ROOT` at the top level of the config YAML, or pass
`--results-root PATH` to `run_experiment.py` (CLI wins over YAML).

### Viewing label stacks

The `.npz` files store integer label arrays — one number per pixel
indicating which cell/nucleus the pixel belongs to. Options for
inspecting them:

**napari (recommended, one-liner):**

```bash
uv run python scripts/view_labels.py results/control_experiment/
```

The script auto-detects the `extraction/` subfolder, loads the raw
phase + fluor TIFFs (via paths recorded in `provenance.json`), and opens
a napari viewer with four layers: `phase`, `fluor` (green additive),
`cells (phase masks)`, and `nuclei (fluor masks)` (hidden by default —
toggle in the layer list). Scrub the time slider to page through frames.

Skip loading raw TIFFs (faster, useful if the source stacks are on a
slow/unmounted drive):

```bash
uv run python scripts/view_labels.py results/control_experiment/ --no-raw
```

**Python one-liner (matplotlib):**

```python
import matplotlib.pyplot as plt, numpy as np
labels = np.load("results/control_experiment/extraction/label_stack.npz")["label_stack"]
plt.imshow(labels[0], cmap="tab20"); plt.title(f"frame 0 — {labels[0].max()} cells"); plt.show()
```

**Fiji / ImageJ:** Fiji can't open `.npz` directly, but a one-line
convert-to-TIFF makes them openable:

```bash
uv run python -c "
import numpy as np, tifffile
labels = np.load('results/control_experiment/extraction/label_stack.npz')['label_stack']
tifffile.imwrite('/tmp/labels.tif', labels.astype('uint16'))
print('wrote /tmp/labels.tif —', labels.shape, 'uint16')
"
```

Open `/tmp/labels.tif` in Fiji, then **Image → Lookup Tables → glasbey_on_dark**
(or any categorical LUT) for a colored view. **Image → Adjust →
Brightness/Contrast → Auto** stretches the range to cover all label ids.
For overlays on the raw stack, load both and use **Image → Overlay → Add
Image…** or **Image → Color → Merge Channels…** (labels as a colored
channel over the phase gray).

## Diagnostic Overlay

The `scripts/diagnostic_overlay.py` script generates visual overlays that show which cells were **accepted** and which were **rejected** by the detection filters, along with the reason each cell was rejected. This is the main tool for tuning detection parameters.

### What it produces

Two images saved to `results/`:

| File | Description |
|------|-------------|
| `diagnostic_full.png` | Full frame — accepted cells marked with **red X**, rejected cells marked with **cyan O** and labeled with the rejection reason |
| `diagnostic_crops.png` | Six zoomed crop regions for close inspection of individual cells and their rejection labels |

### Basic usage

Run with the current default parameters:

```bash
uv run python scripts/diagnostic_overlay.py
```

### Overriding detection parameters

Pass any detection parameter as a CLI flag to test different thresholds without editing code:

```bash
# Relax contrast filter to accept more faded cells
uv run python scripts/diagnostic_overlay.py --min_contrast 1400

# Lower area threshold and relax circularity
uv run python scripts/diagnostic_overlay.py --min_area 200 --min_circularity 0.5

# Combine multiple overrides
uv run python scripts/diagnostic_overlay.py --min_area 200 --min_contrast 1400 --min_circularity 0.5
```

### Inspecting a specific frame

By default the script analyses frame 0. Use `--frame` to pick another:

```bash
uv run python scripts/diagnostic_overlay.py --frame 12
```

### Custom crop regions

Zoom into specific areas of interest by passing `--crops` with `Y0:Y1:X0:X1` coordinates (up to 6 regions):

```bash
uv run python scripts/diagnostic_overlay.py --crops 150:400:250:550 400:650:700:1000
```

If omitted, six crops are auto-generated spread across the frame.

### Using a different stack

```bash
uv run python scripts/diagnostic_overlay.py --stack path/to/other_stack.tif
```

### Debugging detection with the overlay

The overlay is designed for an iterative tuning workflow:

1. **Run the script** with current defaults and open `results/diagnostic_full.png`.

2. **Look at the cyan circles.** Each rejected cell has a label explaining why it was rejected:
   - `edge` — cell touches the frame border (filtered by `--exclude_edges`)
   - `area=N` — cell area is below `--min_area`
   - `circ=N.NN` — circularity is below `--min_circularity`
   - `c=N` — intensity contrast (std-dev) is below `--min_contrast`
   - A cell can have **multiple reasons** (e.g. `area=180, circ=0.52`)

3. **Decide if rejected cells should be accepted.** Open `results/diagnostic_crops.png` for a closer look. If you see real cells being rejected, relax the corresponding threshold:
   - Too many real cells rejected for contrast? Lower `--min_contrast`
   - Small but valid cells being dropped? Lower `--min_area`
   - Slightly elongated cells being rejected? Lower `--min_circularity`

4. **Re-run with adjusted parameters** and compare the new overlay:
   ```bash
   uv run python scripts/diagnostic_overlay.py --min_contrast 1400
   ```

5. **Check for false positives.** If relaxing a threshold lets in debris or halos (red X on non-cells), tighten the threshold back.

6. **Once satisfied**, update `DETECT_PARAMS` in your config YAML (e.g. `configs/run_01.yaml`) with the tuned values and re-run.

The script also prints a rejection reason summary to the terminal:

```
Accepted: 349, Rejected: 72

Rejection reasons (cells can have multiple):
  edge: 38
  c: 22
  area: 15
  circ: 8
```

This tells you at a glance which filter is rejecting the most cells, helping you prioritise which threshold to adjust first.

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--stack` | `data/gradient_0011/phase.tif` | Path to the TIFF stack |
| `--frame` | `0` | Frame index to analyse |
| `--outdir` | `results` | Output directory for PNG files |
| `--diameter` | `32` | Cellpose cell diameter |
| `--min_area` | `300` | Minimum cell area in pixels |
| `--min_circularity` | `0.7` | Minimum circularity (0-1) |
| `--min_contrast` | `1250` | Minimum intensity std-dev |
| `--exclude_edges` / `--no_exclude_edges` | `True` | Include/exclude border cells |
| `--gpu` / `--no_gpu` | `True` | Enable/disable GPU |
| `--resample` | `False` | Resample masks (slower, more precise boundaries) |
| `--crops` | auto | Custom crop regions as `Y0:Y1:X0:X1` (up to 6) |

## Alternative: Running with JupyterLab

If you prefer JupyterLab over VS Code:

```bash
uv run jupyter lab
```

This starts a local Jupyter server and opens it in your browser. Navigate to `notebooks/analysis.ipynb`.

## Development

Install dev dependencies:

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check src/
```
