"""Module C: region-matched, surface-aware relighting using boundary-strip ratios.

Physics of shadow formation:
    I_lit = rho * (L_direct + L_ambient)
    I_shadow = rho * L_ambient

Across the shadow boundary on the SAME surface, the material reflectance rho is
identical and cancels out in the log-domain illumination ratio:
    log(I_lit) - log(I_shadow) = log((L_direct + L_ambient) / L_ambient)

For every shadow component:
  1. We optionally cluster multi-surface shadows by chromaticity.
  2. For each sub-surface, we extract a paired inner strip (inside shadow) and
     outer strip (in lit band) along the boundary.
  3. We estimate the per-channel illumination ratio in linear radiometric space:
     gain = clip(exp(median(log(outer)) - median(log(inner))), 1.0, max_gain).
  4. Only pixels strictly within the shadow mask are relit (no dilated over-exposure).
  5. This provides surface-consistent, artifact-free relighting.
"""
import cv2
import numpy as np

from .utils import linear_to_srgb, normalized_rgb, srgb_to_linear


def _ellipse(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def remove_global(img, mask, max_gain=6.0, linear=True):
    """BASELINE: one gain per channel using all shadow vs all lit pixels."""
    binary = mask > 0.5
    if binary.sum() < 10 or (~binary).sum() < 10:
        return img.copy()

    f = srgb_to_linear(img) if linear else img.astype(np.float32)
    gain = np.median(f[~binary], axis=0) / np.maximum(np.median(f[binary], axis=0), 1e-4)
    gain = np.clip(gain, 1.0, max_gain)

    out = f.copy()
    out[binary] = f[binary] * gain

    if linear:
        return linear_to_srgb(out, as_uint8=True)
    return np.clip(out, 0, 255).astype(np.uint8)


def remove(
    img,
    mask,
    gap=None,
    band=None,
    max_gain=6.0,
    min_pixels=20,
    enable_clustering=True,
    linear=True,
):
    """Surface-aware log-boundary illumination ratio relighting in linear radiometric space.

    Args:
        img: RGB uint8 image.
        mask: float32 shadow mask in [0, 1].
        gap: margin in pixels skipped at the immediate boundary (penumbra exclusion).
        band: width in pixels of the sampling boundary strip.
        max_gain: upper bound on channel amplification.
        min_pixels: minimum pixels to compute reliable statistics.
        enable_clustering: whether to partition multi-surface shadows by chromaticity.
        linear: whether to perform ratio estimation and relighting in linear radiometric space.

    Returns:
        RGB uint8 relit image.
    """
    h, w = img.shape[:2]
    scale = min(h, w)
    gap = gap if gap is not None else max(2, int(0.01 * scale))
    band = band if band is not None else max(5, int(0.03 * scale))

    binary = (mask > 0.5).astype(np.uint8)
    if binary.sum() < min_pixels:
        return img.copy()

    f = srgb_to_linear(img) if linear else img.astype(np.float32)
    norm_rgb = normalized_rgb(f)
    log_f = np.log(np.maximum(f, 1e-4))

    n, labels = cv2.connectedComponents(binary, connectivity=8)
    any_shadow_far = cv2.dilate(binary, _ellipse(gap + 1))
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

        # Sub-surface clustering for multi-surface shadows
        sub_regions = [region]
        if enable_clustering and reg_pixels >= 250:
            coords = np.where(region > 0)
            sample_chroma = norm_rgb[coords]
            chroma_std = np.std(sample_chroma, axis=0).sum()
            # If color variance indicates multiple surface materials, cluster into 2
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
            # Paired inner strip: just inside shadow boundary
            inner_strip = cv2.erode(sub, _ellipse(gap)) & (1 - cv2.erode(sub, _ellipse(gap + band)))
            if inner_strip.sum() < min_pixels:
                inner_strip = cv2.erode(sub, _ellipse(gap))
            if inner_strip.sum() < min_pixels:
                inner_strip = sub

            # Paired outer strip: lit pixels outside boundary matching sub-surface chromaticity
            outer_strip = cv2.dilate(sub, _ellipse(gap + band)) & (1 - any_shadow_far)
            if outer_strip.sum() < 5:
                outer_strip = ring

            sub_chroma = norm_rgb[sub > 0].mean(axis=0)
            outer_coords = np.where(outer_strip > 0)
            outer_chroma = norm_rgb[outer_coords]
            chroma_dists = np.linalg.norm(outer_chroma - sub_chroma, axis=1)

            # Filter outer strip to pixels belonging to the same surface
            matched_indices = np.where(chroma_dists < 0.12)[0]
            if len(matched_indices) >= min_pixels:
                matched_outer = (
                    outer_coords[0][matched_indices],
                    outer_coords[1][matched_indices],
                )
            else:
                matched_outer = outer_coords

            # Log-domain ratio estimation
            if len(matched_outer[0]) > 0 and inner_strip.sum() > 0:
                log_outer_median = np.median(log_f[matched_outer], axis=0)
                log_inner_median = np.median(log_f[inner_strip > 0], axis=0)
                log_ratio = log_outer_median - log_inner_median
                gain = np.clip(np.exp(log_ratio), 1.0, max_gain)
            elif ring.sum() > 0 and sub.sum() > 0:
                gain = np.clip(
                    np.median(f[ring > 0], axis=0) / np.maximum(np.median(f[sub > 0], axis=0), 1e-4),
                    1.0,
                    max_gain,
                )
            else:
                gain = np.ones(3, dtype=np.float32)

            # Apply gain strictly to the shadow sub-region (no dilated halo artifact)
            out[sub > 0] = f[sub > 0] * gain

    if linear:
        return linear_to_srgb(out, as_uint8=True)
    return np.clip(out, 0, 255).astype(np.uint8)
