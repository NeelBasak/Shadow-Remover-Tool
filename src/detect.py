"""Module A: automatic shadow detection (classical, LAB-based + hierarchical Otsu + daylight physics).

Combines:
1. Otsu thresholding on Gaussian-smoothed L channel.
2. Hierarchical Otsu re-splitting if the dark class is too large (separating
   broad materials vs actual cast shadows).
3. Morphological opening (noise suppression) WITHOUT premature closing, preventing
   shadows from bridging into adjacent dark objects or occluders.
4. Physics-based daylight chromaticity verification:
   - Green-Red axis invariance: |a_shadow - a_lit| <= 3.5.
   - Blue skylight shift consistency: shadows may shift blue (negative b*),
     but cannot become yellower than sunlit surroundings (b_shadow - b_lit <= 2.5).
   - Luminance drop validation: L_lit / L_shadow >= 1.15.
   This cleanly rejects dark-colored materials (shoes, asphalt, dark brown objects)
   while preserving real outdoor shadows with blue ambient skylight.
5. Post-validation morphological closing, hole filling, and Gaussian softening.
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
    max_delta_a=3.5,
    max_delta_b_pos=2.5,
    min_l_ratio=1.15,
    soft_sigma=1.0,
    max_area_frac=0.5,
    max_splits=2,
):
    """Detect shadows with hierarchical Otsu and daylight physics validation.

    Args:
        img: RGB uint8 image.
        min_area_frac: connected components smaller than this fraction of the
            image are removed as noise.
        min_separation: if the mean L of the lit and shadow classes differ by
            less than this (L units, 0-100), we assume there is no shadow.
        max_delta_a: maximum allowed absolute shift in the Green-Red 'a' channel.
            In daylight, direct sun and skylight share the same green-red balance on a surface.
        max_delta_b_pos: maximum allowed positive shift in the Blue-Yellow 'b' channel.
            Shadows can shift blue (negative delta_b), but cannot be yellower than lit ground.
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

    # Initial morphological cleanup: open ONLY (suppresses noise without bridging occluders)
    k = max(3, int(round(min(h, w) * 0.01)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = shadow.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)

    # Component-level daylight physics check
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_area = min_area_frac * h * w
    ring_radius = max(5, int(round(min(h, w) * 0.025)) | 1)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_radius, ring_radius))

    a_ch = lab[..., 1]
    b_ch = lab[..., 2]
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
            delta_a = abs(float(a_ch[comp > 0].mean() - a_ch[ring > 0].mean()))
            delta_b_pos = float(b_ch[comp > 0].mean() - b_ch[ring > 0].mean())
            L_in = float(L[comp > 0].mean())
            L_out = float(L[ring > 0].mean())
            l_ratio = L_out / max(L_in, 1e-3)

            # Accept if a* is invariant, b* does not turn yellower, and luminance drops
            if delta_a <= max_delta_a and delta_b_pos <= max_delta_b_pos and l_ratio >= min_l_ratio:
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

    # Apply morphological close ONLY after components are verified
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel)

    soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), soft_sigma)
    return np.clip(soft, 0.0, 1.0).astype(np.float32)
