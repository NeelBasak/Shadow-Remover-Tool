"""Module A: automatic shadow detection (classical, LAB-based + hierarchical Otsu + multi-cue chroma/hue check).

Combines:
1. Otsu thresholding on Gaussian-smoothed L channel.
2. Hierarchical Otsu re-splitting if the dark class is too large (separating
   dark materials vs actual cast shadows).
3. Multi-cue material consistency verification:
   - Normalized RGB chromaticity distance.
   - HSV circular hue consistency (shadow preserves surface hue).
   - RGB channel gain dispersion (shadow scales all color channels proportionally;
     dark objects exhibit high cross-channel ratio dispersion).
   - Minimum luminance drop ratio.
4. Morphological filtering, hole filling, and Gaussian softening.
"""
import cv2
import numpy as np

from .utils import rgb_to_lab


def detect_otsu(img):
    """BASELINE: plain Otsu threshold on L, no cleanup. Returns binary float32 mask."""
    L = rgb_to_lab(img)[..., 0]
    L8 = np.clip(L * 255.0 / 100.0, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(L8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (binary > 0).astype(np.float32)


def detect(
    img,
    min_area_frac=0.002,
    min_separation=10.0,
    max_chroma_diff=0.030,
    max_hue_diff=15.0,
    max_gain_disp=0.030,
    min_l_ratio=1.15,
    soft_sigma=2.0,
    max_area_frac=0.5,
    max_splits=2,
):
    """Detect shadows with hierarchical Otsu and multi-cue dark object rejection.

    Args:
        img: RGB uint8 image.
        min_area_frac: connected components smaller than this fraction of the
            image are removed as noise.
        min_separation: if the mean L of the lit and shadow classes differ by
            less than this (L units, 0-100), we assume there is no shadow.
        max_chroma_diff: maximum allowed normalized chromaticity Euclidean
            distance between candidate shadow core and its surrounding lit ring.
        max_hue_diff: maximum allowed circular hue difference in degrees (in HSV).
        max_gain_disp: maximum allowed coefficient of variation (std / mean)
            of the per-channel gains. Shadows have uniform scale across RGB channels,
            while dark materials exhibit high cross-channel dispersion.
        min_l_ratio: minimum ratio (L_lit / L_shadow) required for a valid shadow.
        soft_sigma: Gaussian sigma used to soften the final mask.
        max_area_frac: if Otsu labels more than this fraction of the image as shadow,
            re-split inside the dark class (hierarchical Otsu).
        max_splits: maximum number of hierarchical Otsu splits.

    Returns:
        float32 soft mask in [0, 1] (1 = shadow).
    """
    h, w = img.shape[:2]
    lab = rgb_to_lab(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    L = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.5)

    L8 = np.clip(L * 255.0 / 100.0, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(L8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    shadow = binary > 0

    # Hierarchical Otsu: if first split labels a huge area as shadow, it may be
    # separating broad materials rather than lighting. Re-threshold inside dark class.
    for _ in range(max_splits):
        if shadow.mean() <= max_area_frac or shadow.sum() < 100:
            break
        t2, _ = cv2.threshold(
            L8[shadow].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        shadow = shadow & (L8 <= t2)

    # No clear bimodality -> treat as shadow-free.
    if shadow.sum() == 0 or (~shadow).sum() == 0:
        return np.zeros((h, w), np.float32)
    if float(L[~shadow].mean() - L[shadow].mean()) < min_separation:
        return np.zeros((h, w), np.float32)

    # Initial morphological cleanup
    k = max(3, int(round(min(h, w) * 0.01)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = shadow.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    # Normalized RGB for material chromaticity comparison: r = R/(R+G+B), g = G/(R+G+B)
    img_f = img.astype(np.float32)
    rgb_sum = img_f.sum(axis=2, keepdims=True) + 1e-5
    norm_rgb = img_f / rgb_sum

    # Component-level multi-cue consistency check
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_area = min_area_frac * h * w
    ring_radius = max(5, int(round(min(h, w) * 0.025)) | 1)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_radius, ring_radius))

    keep = np.zeros_like(m)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        comp = (labels == i).astype(np.uint8)
        # Extract surrounding lit ring (outside all candidate shadows)
        dilated = cv2.dilate(comp, ring_kernel)
        ring = dilated & (1 - m)

        # If ring is too small (e.g. shadow touches image borders), fall back to erosion check
        if ring.sum() < max(15, int(0.05 * area)):
            ring = dilated & (1 - comp)

        if ring.sum() >= 10:
            c_in = norm_rgb[comp > 0].mean(axis=0)
            c_out = norm_rgb[ring > 0].mean(axis=0)
            chroma_diff = float(np.linalg.norm(c_in - c_out))

            L_in = float(L[comp > 0].mean())
            L_out = float(L[ring > 0].mean())
            l_ratio = L_out / max(L_in, 1e-3)

            # Circular Hue difference in degrees [0, 180] in OpenCV
            H_in = float(hsv[comp > 0, 0].mean())
            H_out = float(hsv[ring > 0, 0].mean())
            hue_diff = min(abs(H_in - H_out), 180.0 - abs(H_in - H_out))
            sat_ring = float(hsv[ring > 0, 1].mean())

            # Channel gain dispersion (coefficient of variation)
            g = img_f[ring > 0].mean(axis=0) / np.maximum(img_f[comp > 0].mean(axis=0), 1.0)
            gain_disp = float(g.std() / (g.mean() + 1e-5))

            # Multi-cue validation:
            # 1. Luminance must drop
            # 2. Chromaticity distance must not be excessive
            # 3. If surface has chroma/color, hue must match
            # 4. Gain must be proportional across all RGB channels
            valid_chroma = chroma_diff <= max_chroma_diff
            valid_hue = (sat_ring < 12.0) or (hue_diff <= max_hue_diff)
            valid_disp = gain_disp <= max_gain_disp

            if (valid_chroma or (valid_hue and valid_disp)) and l_ratio >= min_l_ratio:
                # Additional check: if hue difference or gain dispersion is very high, it is a material shift
                if not (hue_diff > 20.0 and gain_disp > 0.04):
                    keep[comp > 0] = 1
        else:
            # Insufficient border context; preserve if area is significant
            keep[comp > 0] = 1

    # Fill small holes within kept shadow regions
    inv = (1 - keep).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n2):
        if st2[i, cv2.CC_STAT_AREA] < min_area:
            keep[lab2 == i] = 1

    soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), soft_sigma)
    return np.clip(soft, 0.0, 1.0).astype(np.float32)
