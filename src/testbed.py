"""Synthetic image pairs with a known displacement.

Real satellite images never tell you how far the ground moved, so the
only way to measure a method's error is to manufacture pairs where you imposed
the motion yourself.

    from src.testbed import Testbed
    tb = Testbed(base_image, box=(57, 107, 233, 283))
    before, after, mask, true_shift = tb.pair(dx=0.5, dy=0.0, noise=0.1, cloud=0.2, seed=0)
"""

import numpy as np
from scipy import ndimage


def move_patch(img, dx, dy, box, order=3):
    """Build a before/after pair where `box` has moved by (dx, dy) pixels.

    The displacement is SPLIT between the two images - before gets -d/2, after
    gets +d/2 - so both receive the same amount of interpolation smoothing.

    Shifting only the "after" image would leave it slightly blurrier than the
    "before", and a tracker can lock onto that difference instead of the motion.
    Passing "before" through the same function with a zero shift does not fix
    this: cubic splines are exact at integer positions, so a zero shift returns
    the image untouched (difference about 1e-16).
    """
    r0, r1, c0, c1 = box
    before = ndimage.shift(img, (-dy / 2, -dx / 2), order=order, mode="nearest")
    moved = ndimage.shift(img, (dy / 2, dx / 2), order=order, mode="nearest")
    after = before.copy()
    after[r0:r1, c0:c1] = moved[r0:r1, c0:c1]
    return before, after


def add_noise(img, level, rng, base_std):
    """Gaussian noise at `level` times the image's own contrast."""
    return img + rng.normal(0, level * base_std, img.shape)


def add_clouds(img, fraction, rng, smooth=15):
    """Blank out `fraction` of pixels as cloud blobs. Returns (image, mask).

    Blobs, not speckle: a lone missing pixel is trivial to fill from its
    neighbours, a 50-pixel hole genuinely destroys information. Smoothing a
    random field and cutting it at a quantile gives connected holes and the
    exact fraction asked for.

    Missing pixels are NaN rather than a brightness value, so later steps have
    to handle the gap deliberately instead of treating cloud as dark ground.
    """
    if fraction <= 0:
        return img.copy(), np.zeros(img.shape, bool)
    field = ndimage.gaussian_filter(rng.random(img.shape), smooth)
    mask = field > np.quantile(field, 1 - fraction)
    out = img.copy()
    out[mask] = np.nan
    return out, mask


class Testbed:
    """Holds a base image and the block that moves, so pairs are one call."""

    def __init__(self, base, box):
        self.base = base
        self.box = box
        self.std = float(np.nanstd(base))

    def pair(self, dx, dy=0.0, noise=0.1, cloud=0.0, seed=0):
        """One synthetic pair. Returns (before, after, mask, (dx, dy)).

        Order matches reality: the ground moves, the sensor adds noise, the
        atmosphere blocks the view.

        One seed drives everything. This matters for the week 6 comparison:
        the three methods must see identical inputs, or part of the measured
        difference is luck in the random draw.
        """
        rng = np.random.default_rng(seed)
        before, after = move_patch(self.base, dx, dy, self.box)
        before = add_noise(before, noise, rng, self.std)
        after = add_noise(after, noise, rng, self.std)
        after, mask = add_clouds(after, cloud, rng)
        return before, after, mask, (dx, dy)


def misalign(before, after, dx, dy=0.0, order=3):
    """Shift the two scenes apart by (dx, dy) px, faking georeferencing error.

    The satellite never points at exactly the same spot twice. Sentinel-2 places
    a pixel to about 10 m, which is one whole pixel, while the slope creeps
    roughly a fifth of a pixel a year. The pointing error is several times the
    signal, so it has to be removed before any real measurement means anything.

    Unlike move_patch, this moves EVERYTHING. That is what makes it a
    misalignment rather than ground motion: real movement is local, pointing
    error is global. Telling those two apart is the whole job of coregister.

    Split between the two images for the same reason as move_patch - both must
    receive the same amount of interpolation smoothing, or the tracker can lock
    onto the difference in blur instead of the difference in position.
    """
    b = ndimage.shift(before, (-dy / 2, -dx / 2), order=order, mode="nearest")
    a = ndimage.shift(after, (dy / 2, dx / 2), order=order, mode="nearest")
    return b, a
