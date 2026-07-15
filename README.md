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
│   └── run_01/               # One folder per run
│       ├── report.html       # Self-contained HTML report with all plots
│       ├── config.yaml       # Frozen copy of config used
│       ├── tracked_cells.csv # Per-cell per-frame data
│       └── ...               # Other CSV outputs
├── scripts/
│   ├── run_experiment.py     # CLI runner (papermill + HTML export)
│   └── diagnostic_overlay.py # Visual debugging of detection filters
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

Run an experiment by passing its YAML config to the runner script:

```bash
uv run python scripts/run_experiment.py configs/run_01.yaml
```

This executes the notebook via papermill, generates an HTML report with all plots, and saves CSV results to `results/<run_name>/`.

To re-run all experiments (e.g. after a pipeline update):

```bash
uv run python scripts/run_experiment.py configs/*.yaml
```

The script exits with code 0 if all runs succeed, or 1 if any fail.

### Via Jupyter (interactive)

For development or one-off analysis, open the notebook directly:

```bash
uv run jupyter lab notebooks/analysis.ipynb
```

Edit the parameters cell at the top and run all cells. The notebook is also the template used by the CLI runner.

### How the notebook and CLI runner relate

The CLI runner is **not** a separate reimplementation of the pipeline — it executes the notebook itself:

1. `run_experiment.py` reads `notebooks/analysis.ipynb`.
2. Papermill injects the YAML config as parameters into the notebook's parameters cell.
3. Every cell runs top-to-bottom into a temporary `.ipynb`.
4. The executed notebook is rendered to `results/<RUN_NAME>/report.html`.

Consequences:

- **Any new executable cell you add to the notebook will run under the CLI**, and its output (plots, prints, tables) will appear in `report.html`.
- **New config knobs must be registered.** If a new cell reads a value that should vary per run, add the key to the notebook's parameters cell **and** to `ALLOWED_KEYS` / `TYPE_RULES` in `scripts/run_experiment.py`. Otherwise the value is fixed to the notebook's default.
- **Cell failures abort the run.** Papermill stops at the first exception; the partially-executed notebook is saved as `report_failed.ipynb` in the results folder and later cells are skipped.
- **The notebook is the single source of truth** for analysis logic. The CLI is just a batch driver that parameterizes and captures it.

## Config Parameters

All parameters are set in the YAML config file (or the notebook's parameters cell). See `configs/run_01.yaml` for a complete example.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RUN_NAME` | `run_01` | Unique name; results saved to `results/<RUN_NAME>/` |
| `STACK_PATH` | — | Path to phase-contrast TIFF (relative to `notebooks/`) |
| `FLUOR_PATH` | — | Path to fluorescence TIFF (relative to `notebooks/`) |
| `DETECT_PARAMS.diameter` | 32 | Median cell diameter in pixels |
| `DETECT_PARAMS.min_area` | 300 | Minimum cell area (rejects debris) |
| `DETECT_PARAMS.min_circularity` | 0.7 | Minimum circularity (1.0 = perfect circle) |
| `DETECT_PARAMS.min_contrast` | 1250 | Minimum intensity contrast (rejects faded cells) |
| `DETECT_PARAMS.gpu` | true | GPU acceleration (MPS on Apple Silicon, CUDA on NVIDIA) |
| `SEARCH_RANGE` | 30.0 | Max cell displacement between frames (pixels) |
| `MEMORY` | 3 | Frames a cell can disappear before breaking the track |
| `MERGE_MAX_DISTANCE` | 15.0 | Max pixels between track end/start to merge fragments |
| `MERGE_MAX_GAP` | 18 | Max frame gap for fragment merging |
| `MIN_TRACK_DETECTIONS` | 4 | Minimum frames a track must span to be kept |
| `GATING_Z_THRESHOLD` | 3.5 | MAD-based Z-score threshold for flagging bad frames |
| `FLUOR_DROP_THRESHOLD` | -0.3 | Fluorescence drop threshold for disappearance detection |
| `FLUOR_DROP_WINDOW` | 2 | Frames over which to measure cumulative fluorescence drop |

## Output

Each run produces results in `results/<run_name>/`:

| File | Description |
|------|-------------|
| `report.html` | Self-contained HTML report with all plots and statistics |
| `config.yaml` | Frozen copy of the YAML config used for this run |
| `tracked_cells.csv` | Per-cell per-frame: morphology, fluorescence, speed, SA:V ratio |
| `track_statistics.csv` | Per-track summary: lifetime, growth, fluorescence, migration |
| `frame_diagnostics.csv` | Per-frame detection statistics and quality gating flags |
| `merge_log.csv` | Track merging audit log |
| `fate_predictions.csv` | Per-cell predicted death probability (frame-0 cohort) |
| `fate_prediction_summary.csv` | Model AUC, accuracy, per-feature coefficients |
| `spatial_gradient.csv` | Per-cell position with fate and gradient quartile |
| `spatial_gradient_summary.csv` | Per-axis gradient statistics |
| `clustering_summary.csv` | Spatial clustering test results |
| `nucleus_persistence.csv` | Per-frame phase cell vs fluorescence nucleus counts |
| `nucleus_persistence_summary.csv` | Nucleus persistence conclusion |

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
