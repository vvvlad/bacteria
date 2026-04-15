# Review 1 Feedback

## 1. Track Identity Fragmentation

**Observation:** Many cells are detected for the first time in frames beyond frame 0 and assigned new track IDs. It is unclear whether these "new" cells have coordinates close to previously detected (but temporarily disappeared) cells. This is especially pronounced around frame 1, which is out of focus. Cells appearing "de novo" late (e.g., frame 16) likely survived from the beginning, meaning their true lifetime is much longer than what the pipeline reports.

**Impact:** Lifetime calculations, disappearance-per-frame statistics, and any analysis comparing functional states across conditions are unreliable until this is resolved.

**Analysis:** The pipeline tracks cells via centroid proximity between consecutive frames (`trackpy.link()` with `memory=3`). When a cell fails detection for more than 3 consecutive frames (e.g., due to an out-of-focus frame dropping contrast below `min_contrast`), trackpy creates a new track ID when the cell reappears. The same physical cell is now split across two or more tracks, each with artificially short lifetimes and false disappearance/appearance events.

Data analysis confirmed:
- 404 total tracks, 55 starting after frame 0
- 16 late-starting tracks are fragments of earlier tracks (<15px, gaps 5-18 frames)
- Some form chains (e.g., track 98 -> 388 -> 399, same cell across frames 0, 9-11, 17-23)
- 28 tracks starting at frame 1 are genuinely new (>100px away), not fragments

**Implemented solutions:**

- [x] **(a) Frame trimming:** `TRIM_FRAMES = 2` in the notebook slices off the first two frames before detection. `FRAME_OFFSET` converts analysis frame indices back to raw frame numbers.
- [x] **(b) Post-hoc track merging:** `merge_fragmented_tracks()` in `tracking.py` finds tracks whose start is spatially close to another track's end, merges them via Union-Find. Parameters: `MERGE_MAX_DISTANCE = 15.0`, `MERGE_MAX_GAP = 18`. Returns an auditable merge log.

Verified: 404 -> 388 tracks (16 fragments absorbed), all 28 genuinely new frame-1 tracks untouched.

---

## 2. Analysis Additions

### 2a. Cell count per frame

- [x] Plot cell count vs. frame.
- [x] Compute 50% disappearance frame (time point where half the initial population is gone).
- [x] Compute total fraction of disappeared cells.

### 2b. Lifetime distribution and disappearance per frame

- [x] Plot lifetime distribution histogram (with median).
- [x] Plot disappearance count per frame (with peak annotation).
- [x] Report frame of maximum disappearance.
- [x] Report fraction of survivors (cells present in the final frame).

### 2c. Cell volume, surface area, and swelling dynamics

Time-averaged area is not meaningful because cells actively swell as external osmotic pressure decreases. Averaging across time smears out the signal of interest.

Assuming approximate spherical geometry (reasonable for round bacteria):
- **Volume:** V = (4/3) * pi * r^3, where r = sqrt(area / pi)
- **Surface area:** S = 4 * pi * r^2

Volume changes more strongly during swelling than area. Surface area increase allows estimation of critical membrane elastic stretch before lysis.

Per-cell area data is already stored per frame (used to compute the time-average), so deriving these quantities is straightforward.

**Implemented:** `radius`, `volume`, `surface_area` columns added to `tracked` DataFrame; `mean_volume`, `mean_surface_area` added to `track_stats`. Area histogram has secondary volume axis. V(t)/V(0) and S(t)/S(0) population-averaged swelling curves with SEM bands and individual traces.

- [x] Add volume and surface area columns derived from measured area.
- [x] Plot V(t)/V(0) per cell (relative swelling), then average across the population.
- [x] Same for surface area.
- [x] Area histogram annotated with equivalent volume axis.

### 2d. Extended correlations

- [x] Dependence of swelling extent on initial cell size (scatter + linear fit).
- [ ] Dependence of swelling rate and extent on DNA content (requires fluorescence channel; only phase-contrast Ch0 is available).
- [x] Compare swelling dynamics for disappeared vs. surviving cells (split V(t)/V(0) curves).
