# Perf check report

- **Host:** `naamasim-mobl.ger.corp.intel.com`
- **Date:** 2026-06-06 10:46
- **Platform:** Darwin arm64
- **CPU:** 16 physical / 16 logical
- **RAM:** 137.4 GB total (74.6 GB free at start)
- **Cellpose:** 4.1.1
- **GPU requested / resolved:** True / True
- **Stack:** (25, 1040, 1388) uint16
- **Process RSS after Cellpose:** 1.7 GB (1% of total)

| model | phase s/f | nucleus s/f | total min | cells (frame 0) | delta vs baseline |
|-------|-----------|-------------|-----------|-----------------|-------------------|
| `default` | 5.7 | 7.1 | 5.3 | 371 | +0 |
| `cyto3` | 5.6 | 7.2 | 5.3 | 371 | +0 |
