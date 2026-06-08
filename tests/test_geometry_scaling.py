import numpy as np
import pandas as pd

from cell_analysis import add_geometry


def _toy_inputs():
    """Two tracks, two frames each. Areas chosen so V0 != V1."""
    tracked = pd.DataFrame({
        "track_id": [1, 1, 2, 2],
        "frame":    [0, 1, 0, 1],
        "label":    [1, 1, 2, 2],
        "area":     [100.0, 400.0, 200.0, 200.0],  # px²
    })
    track_stats = pd.DataFrame({"track_id": [1, 2]})
    return tracked, track_stats


def test_pixel_scaling_doubles_radius_when_pixel_size_doubles():
    tracked, track_stats = _toy_inputs()
    t1, _ = add_geometry(tracked.copy(), track_stats.copy(), pixel_size_um=1.0)
    t2, _ = add_geometry(tracked.copy(), track_stats.copy(), pixel_size_um=2.0)

    # radius is proportional to pixel_size_um
    np.testing.assert_allclose(t2["radius"], 2.0 * t1["radius"])

    # area is proportional to pixel_size_um**2
    np.testing.assert_allclose(t2["area"], 4.0 * t1["area"])

    # volume is proportional to pixel_size_um**3
    np.testing.assert_allclose(t2["volume"], 8.0 * t1["volume"])

    # surface_area is proportional to pixel_size_um**2
    np.testing.assert_allclose(t2["surface_area"], 4.0 * t1["surface_area"])


def test_pixel_scaling_default_is_one_unit():
    """Defaulting pixel_size_um leaves area unchanged from input (px²)."""
    tracked, track_stats = _toy_inputs()
    t, _ = add_geometry(tracked.copy(), track_stats.copy())  # default = 1.0
    np.testing.assert_allclose(t["area"], [100.0, 400.0, 200.0, 200.0])


def test_volume_rel_first_frame_is_one():
    tracked, track_stats = _toy_inputs()
    t, _ = add_geometry(tracked.copy(), track_stats.copy(), pixel_size_um=1.0)

    first_frame = t[t["frame"] == 0]
    np.testing.assert_allclose(first_frame["volume_rel"], 1.0)


def test_volume_rel_matches_v_over_v0():
    tracked, track_stats = _toy_inputs()
    t, _ = add_geometry(tracked.copy(), track_stats.copy(), pixel_size_um=1.0)

    # Track 1: area 100 → 400 ⇒ radius 2x, volume 8x ⇒ V/V0 = 8
    track1 = t[t["track_id"] == 1].sort_values("frame")
    np.testing.assert_allclose(track1["volume_rel"].values, [1.0, 8.0])

    # Track 2: area 200 → 200 ⇒ V/V0 = 1
    track2 = t[t["track_id"] == 2].sort_values("frame")
    np.testing.assert_allclose(track2["volume_rel"].values, [1.0, 1.0])
