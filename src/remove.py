"""Module C: region-matched and surface-aware relighting.

For every connected shadow region, we estimate a per-channel multiplicative
gain mapping the shadow core to the lit pixels of the SAME surface (a reference
band just outside the shadow). If a shadow spans multiple surface types (e.g.
grass and concrete), we partition the region into color-consistent clusters
and match each cluster to its corresponding lit surface to avoid color tinting.
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
    gain = np.median(f[~binary], axis=0) / np.maximum(np.median(f[binary], axis=0), 1.0)
    gain = np.clip(gain, 1.0, max_gain)
    out = f.copy()
    out[binary] = f[binary] * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def remove(
    img,
    mask,
    gap=None,
    band=None,
    max_gain=4.0,
    min_pixels=30,
    enable_clustering=True,
):
    """Region-matched, surface-aware relighting. Returns RGB uint8.

    Args:
        img: RGB uint8 image.
        mask: float32 shadow mask in [0, 1].
        gap: margin in pixels skipped at the shadow boundary (penumbra exclusion).
        band: width in pixels of the adjacent lit reference band.
        max_gain: upper bound on channel amplification.
        min_pixels: minimum pixels to compute reliable statistics.
        enable_clustering: whether to partition multi-surface shadows by chromaticity.
    """
    h, w = img.shape[:2]
    scale = min(h, w)
    gap = gap if gap is not None else max(2, int(0.01 * scale))
    band = band if band is not None else max(6, int(0.04 * scale))

    f = img.astype(np.float32)
    binary = (mask > 0.5).astype(np.uint8)
    if binary.sum() < min_pixels:
        return img.copy()

    n, labels = cv2.connectedComponents(binary, connectivity=8)
    any_shadow_far = cv2.dilate(binary, _ellipse(gap + 1))

    # Normalized RGB for chromaticity similarity
    rgb_sum = f.sum(axis=2, keepdims=True) + 1e-5
    norm_rgb = f / rgb_sum

    out = f.copy()

    for i in range(1, n):
        region = (labels == i).astype(np.uint8)
        reg_pixels = region.sum()
        if reg_pixels < min_pixels:
            continue

        # Lit reference ring for this component
        ring = cv2.dilate(region, _ellipse(gap + band)) & (1 - any_shadow_far)
        if ring.sum() < min_pixels:
            ring = cv2.dilate(region, _ellipse(gap + band)) & (1 - binary)
        if ring.sum() < 5:
            continue

        # Sub-surface clustering for multi-surface shadows
        sub_regions = [region]
        if enable_clustering and reg_pixels >= 400:
            coords = np.where(region > 0)
            sample_chroma = norm_rgb[coords]
            chroma_std = np.std(sample_chroma, axis=0).sum()
            # If color variance is high, partition into 2 surface clusters
            if chroma_std > 0.05:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, clabels, _ = cv2.kmeans(
                    sample_chroma, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
                )
                sub0 = np.zeros((h, w), dtype=np.uint8)
                sub1 = np.zeros((h, w), dtype=np.uint8)
                sub0[coords[0][clabels.flatten() == 0], coords[1][clabels.flatten() == 0]] = 1
                sub1[coords[0][clabels.flatten() == 1], coords[1][clabels.flatten() == 1]] = 1
                if sub0.sum() >= min_pixels and sub1.sum() >= min_pixels:
                    sub_regions = [sub0, sub1]

        for sub in sub_regions:
            sub_core = cv2.erode(sub, _ellipse(gap))
            if sub_core.sum() < min_pixels:
                sub_core = sub

            # Match sub-surface chromaticity with corresponding lit ring pixels
            sub_chroma = norm_rgb[sub_core > 0].mean(axis=0)
            ring_coords = np.where(ring > 0)
            ring_chroma = norm_rgb[ring_coords]
            chroma_dists = np.linalg.norm(ring_chroma - sub_chroma, axis=1)

            # Keep lit ring pixels that are close in chromaticity
            matched_indices = np.where(chroma_dists < 0.10)[0]
            if len(matched_indices) >= min_pixels:
                matched_ring = (
                    ring_coords[0][matched_indices],
                    ring_coords[1][matched_indices],
                )
                l_stat = np.median(f[matched_ring], axis=0)
            else:
                l_stat = np.median(f[ring > 0], axis=0)

            s_stat = np.median(f[sub_core > 0], axis=0)
            gain = np.clip(l_stat / np.maximum(s_stat, 1.0), 1.0, max_gain)

            apply_mask = cv2.dilate(sub, _ellipse(gap)).astype(bool)
            out[apply_mask] = f[apply_mask] * gain

    return np.clip(out, 0, 255).astype(np.uint8)
