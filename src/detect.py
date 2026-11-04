"""Module A: automatic shadow detection (classical, LAB-based + hierarchical Otsu + chroma consistency).

Combines:
1. Otsu thresholding on Gaussian-smoothed L channel.
2. Hierarchical Otsu re-splitting if the dark class is too large (separating
   dark materials vs actual cast shadows).
3. Localized normalized chromaticity consistency check and luminance drop ratio
   against the adjacent lit ring to reject dark-but-not-shadow objects.
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
    max_chroma_diff=0.035,
    min_l_ratio=1.15,
    soft_sigma=2.0,
    max_area_frac=0.5,
    max_splits=2,
):
    """Detect shadows with hierarchical Otsu and chroma-consistency filtering.

    Args:
        img: RGB uint8 image.
        min_area_frac: connected components smaller than this fraction of the
            image are removed as noise.
        min_separation: if the mean L of the lit and shadow classes differ by
            less than this (L units, 0-100), we assume there is no shadow.
        max_chroma_diff: maximum allowed normalized chromaticity Euclidean
            distance between candidate shadow core and its surrounding lit ring.
            Higher differences indicate inherently dark objects of different materials.
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

    # Component-level chroma & luminance consistency check
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

            # Accept only if chromaticity is consistent and luminance actually drops
            if chroma_diff <= max_chroma_diff and l_ratio >= min_l_ratio:
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
