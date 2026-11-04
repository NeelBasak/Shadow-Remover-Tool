"""Module A: automatic shadow detection (classical, LAB-based).

Idea: shadows are darker in the L channel but keep roughly the same
chromaticity as the surface they fall on. We threshold L (Otsu), reject
the result if the two classes are not clearly separated, then clean the
mask with morphology and return a soft mask.
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
    min_separation=12.0,
    soft_sigma=2.0,
    max_area_frac=0.5,
    max_splits=2,
):
    """Detect shadows. Returns a soft float32 mask in [0, 1] (1 = shadow).

    Args:
        img: RGB uint8 image.
        min_area_frac: connected components smaller than this fraction of the
            image are removed as noise.
        min_separation: if the mean L of the lit and shadow classes differ by
            less than this (L units, 0-100), we assume there is no shadow.
        soft_sigma: Gaussian sigma used to soften the final mask.
        max_area_frac: if the first Otsu split labels more than this fraction
            of the image as shadow, the split is probably separating two
            MATERIALS (e.g. dark bricks vs light stone) rather than lit vs
            shadow. We then re-split inside the dark class (hierarchical Otsu)
            and keep only its darker part.
        max_splits: maximum number of such re-splits.
    """
    h, w = img.shape[:2]
    lab = rgb_to_lab(img)
    L = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.5)

    L8 = np.clip(L * 255.0 / 100.0, 0, 255).astype(np.uint8)
    thr, binary = cv2.threshold(L8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    shadow = binary > 0

    # Hierarchical Otsu: a huge "shadow" class means we split materials, not
    # lighting. Re-threshold inside the dark class and keep its darker part.
    for _ in range(max_splits):
        if shadow.mean() <= max_area_frac or shadow.sum() < 100:
            break
        t2, _ = cv2.threshold(
            L8[shadow].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        shadow = shadow & (L8 <= t2)

    # No clear bimodality -> probably no shadow in the image.
    if shadow.sum() == 0 or (~shadow).sum() == 0:
        return np.zeros((h, w), np.float32)
    if L[~shadow].mean() - L[shadow].mean() < min_separation:
        return np.zeros((h, w), np.float32)

    # Morphological cleanup.
    k = max(3, int(round(min(h, w) * 0.01)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = shadow.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    # Drop tiny components and fill small holes.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_area = min_area_frac * h * w
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    inv = (1 - keep).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n2):
        if st2[i, cv2.CC_STAT_AREA] < min_area:
            keep[lab2 == i] = 1

    soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), soft_sigma)
    return np.clip(soft, 0, 1).astype(np.float32)
