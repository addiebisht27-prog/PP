"""Grid tracker: measure displacement across a whole image, not just one window.

    from src.tracker import track_grid, coregister, consistency_map
"""

import numpy as np
from skimage.registration import phase_cross_correlation


def _taper(a):
    """Fade the window edges to zero before correlating.

    Without this the sharp window edge leaks across the frequency spectrum and
    small shifts come back roughly half their true size: a true 0.5 px reads as
    about 0.24 untapered, about 0.46 tapered.
    """
    h = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    m = np.nanmean(a)
    return (np.nan_to_num(a, nan=m) - m) * h


def measure_window(before, after, upsample=100):
    """Displacement between two equally sized patches, in pixels.

    Returns (dx, dy). Positive dx is east, positive dy is south.
    """
    est, _, _ = phase_cross_correlation(
        _taper(before), _taper(after), upsample_factor=upsample, normalization=None
    )
    return -est[1], -est[0]


def track_grid(before, after, window=32, step=16, min_valid=0.7, upsample=100):
    """Measure the shift at a grid of positions across the image.

    window  side of each measuring window. Bigger is more reliable but smears
            the edges of whatever is moving. The Ronti block is about 50 px
            across, so 32 fits inside it and 64 does not.
    step    distance between positions. Half the window gives a smooth map, but
            neighbouring measurements then overlap and are not independent.
    min_valid  skip a window if less than this fraction of pixels are usable.
            Cloud is NaN, so this is the cloud cut-off.

    Returns dict: dx, dy (maps, NaN where skipped), rows, cols (centres in
    image pixels), window, step.
    """
    H, W = before.shape
    row_starts = range(0, H - window + 1, step)
    col_starts = range(0, W - window + 1, step)

    dx = np.full((len(row_starts), len(col_starts)), np.nan)
    dy = np.full_like(dx, np.nan)

    for i, r in enumerate(row_starts):
        for j, c in enumerate(col_starts):
            b = before[r:r + window, c:c + window]
            a = after[r:r + window, c:c + window]
            valid = np.mean(np.isfinite(b) & np.isfinite(a))
            if valid < min_valid or np.nanstd(b) < 1e-6:
                continue
            dx[i, j], dy[i, j] = measure_window(b, a, upsample)

    return {
        "dx": dx, "dy": dy,
        "rows": np.array([r + window // 2 for r in row_starts]),
        "cols": np.array([c + window // 2 for c in col_starts]),
        "window": window, "step": step,
    }


def select(result, box):
    """The part of a displacement map inside an image-pixel box (r0, r1, c0, c1)."""
    r0, r1, c0, c1 = box
    rsel = (result["rows"] >= r0) & (result["rows"] < r1)
    csel = (result["cols"] >= c0) & (result["cols"] < c1)
    idx = np.ix_(rsel, csel)
    return result["dx"][idx], result["dy"][idx]


def coregister(result, stable_box):
    """Remove the misalignment between the two scenes.

    Sentinel-2 georeferencing is good to about 10 m, which is one pixel, while
    the slope moves about a fifth of a pixel a year. The misalignment is larger
    than the signal, so it has to come out first.

    Ground that did not move still reads non-zero, and that reading IS the
    misalignment. Subtract it from everything.

    stable_box is (r0, r1, c0, c1) in image pixels. Choose bare rock. Do not
    include the glacier: ice flows on its own, and subtracting real glacier
    motion from the slope would be wrong.

    Median, not mean, so one bad window cannot drag the correction around.
    """
    patch_dx, patch_dy = select(result, stable_box)
    n = int(np.isfinite(patch_dx).sum())
    if n < 10:
        raise ValueError(
            f"only {n} usable windows in the stable region, probably cloud. "
            "This image cannot be co-registered and should be marked unusable."
        )

    off_dx = float(np.nanmedian(patch_dx))
    off_dy = float(np.nanmedian(patch_dy))

    out = dict(result)
    out["dx"] = result["dx"] - off_dx
    out["dy"] = result["dy"] - off_dy
    out["offset"] = (off_dx, off_dy)
    out["n_stable_windows"] = n
    return out


def consistency_map(results):
    """How much each grid position disagrees across several date pairs.

    Ground that never moved should read the same every time. A position whose
    answer jumps around is being driven by something other than ground motion:
    changing snow, moving shadow, or ice that flows on its own.

    Low values are candidates for the stable reference region.
    """
    DX = np.stack([r["dx"] for r in results])
    DY = np.stack([r["dy"] for r in results])
    return np.nanstd(DX, axis=0) + np.nanstd(DY, axis=0)
