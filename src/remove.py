"""Module C: region-matched relighting.

For every connected shadow region we estimate a per-channel multiplicative
gain that maps the shadow core to the lit pixels of the SAME surface (a thin
band just outside the shadow). This is the "region-matched" part of our
contribution. The global version (one gain for the whole image) is kept as a
baseline.
"""

import cv2
import numpy as np


def _ellipse(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def remove_global(img, mask, max_gain=4.0):
    """BASELINE: one gain per channel using all shadow vs all lit pixels."""
    binary = mask > 0.5
    if binary.sum() < 10 or (~binary).sum() < 10:
        return img.copy()
    f = img.astype(np.float32)
    gain = f[~binary].mean(0) / np.maximum(f[binary].mean(0), 1.0)
    gain = np.clip(gain, 1.0, max_gain)
    out = f.copy()
    out[binary] *= gain
    return np.clip(out, 0, 255).astype(np.uint8)


def remove(img, mask, gap=None, band=None, max_gain=4.0, min_pixels=30):
    """Region-matched relighting. Returns an RGB uint8 image.

    The result is "hard": the gain is applied inside the (slightly dilated)
    shadow region. Edge softness is handled afterwards by blend.blend().

    Args:
        gap: pixels skipped next to the boundary (penumbra) when sampling.
        band: width in pixels of the lit reference band.
    """
    h, w = img.shape[:2]
    scale = min(h, w)
    gap = gap if gap is not None else max(2, int(0.01 * scale))
    band = band if band is not None else max(6, int(0.04 * scale))

    f = img.astype(np.float32)
    binary = (mask > 0.5).astype(np.uint8)
    n, labels = cv2.connectedComponents(binary, connectivity=8)
    any_shadow_far = cv2.dilate(binary, _ellipse(gap + 1))  # exclusion zone

    out = f.copy()
    for i in range(1, n):
        region = (labels == i).astype(np.uint8)
        core = cv2.erode(region, _ellipse(gap))
        ring = cv2.dilate(region, _ellipse(gap + band)) & (1 - any_shadow_far)
        if core.sum() < min_pixels or ring.sum() < min_pixels:
            # fall back to the whole region if erosion removed it
            core = region
            if core.sum() < min_pixels or ring.sum() < min_pixels:
                continue
        s_mean = f[core > 0].mean(0)
        l_mean = f[ring > 0].mean(0)
        gain = np.clip(l_mean / np.maximum(s_mean, 1.0), 1.0, max_gain)

        # Apply over the region plus a small margin so the penumbra is covered.
        apply = region.astype(bool)
        out[apply] = f[apply] * gain
    return np.clip(out, 0, 255).astype(np.uint8)
