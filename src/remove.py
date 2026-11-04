"""Module C: region-matched, surface-aware relighting using boundary-strip ratios.

Physics of shadow formation:
    I_lit = rho * (L_direct + L_ambient)
    I_shadow = rho * L_ambient

Across the shadow boundary on the SAME surface, the material reflectance rho is
identical and cancels out in the log-domain illumination ratio:
    log(I_lit) - log(I_shadow) = log((L_direct + L_ambient) / L_ambient)

For every shadow component:
  1. We optionally cluster multi-surface shadows by chromaticity when distinct physical
     materials are detected (large chromatic separation with spatial coherence).
  2. For each sub-surface, we extract a paired inner strip (inside shadow) and
     outer strip (in lit band) along the boundary.
  3. We estimate the per-channel illumination ratio in linear radiometric space:
     gain = clip(exp(median(log(outer)) - median(log(inner))), 1.0, max_gain).
  4. Adaptive fallback: if local boundary pixels are scarce or noisy, gain is smoothly
     anchored to the robust global scene illumination ratio.
  5. Additive color temperature compensation: compensates the residual blue ambient
     skylight cast in CIE LAB space.
  6. Only pixels strictly within the shadow mask are relit (no dilated over-exposure).
  7. This provides surface-consistent, artifact-free relighting.
"""
import cv2
import numpy as np

from .utils import linear_to_srgb, normalized_rgb, srgb_to_linear


def _ellipse(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def remove_global(img, mask, max_gain=8.0, linear=True):
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
    max_gain=8.0,
    min_pixels=20,
    enable_clustering=True,
    linear=True,
    b_weight=0.5,
):
    """Surface-aware log-boundary illumination ratio relighting in linear radiometric space.

    Args:
        img: RGB uint8 image.
        mask: float32 shadow mask in [0, 1].
        gap: margin in pixels skipped at the immediate boundary (penumbra exclusion).
        band: width in pixels of the sampling boundary strip.
        max_gain: upper bound on channel amplification (default 8.0 for deep outdoor shadows).
        min_pixels: minimum pixels to compute reliable statistics.
        enable_clustering: whether to partition multi-surface shadows by chromaticity.
        linear: whether to perform ratio estimation and relighting in linear radiometric space.
        b_weight: weight for additive color temperature compensation in LAB b* channel
            to neutralize blue skylight illumination cast.

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

    # Global gain anchor: used when local boundary strip has insufficient lit context
    if (~(binary > 0)).sum() >= min_pixels:
        gain_global = np.median(f[binary == 0], axis=0) / np.maximum(np.median(f[binary > 0], axis=0), 1e-4)
        gain_global = np.clip(gain_global, 1.0, max_gain)
    else:
        gain_global = np.ones(3, dtype=np.float32)

    n, labels = cv2.connectedComponents(binary, connectivity=8)
    any_shadow_far = cv2.dilate(binary, _ellipse(gap + 1))
    out = f.copy()

    k_clean = max(3, int(0.015 * scale) | 1)
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_clean, k_clean))

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
        if enable_clustering and reg_pixels >= 400:
            coords = np.where(region > 0)
            sample_chroma = norm_rgb[coords]
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, clabels, centers = cv2.kmeans(
                sample_chroma, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
            )
            center_dist = np.linalg.norm(centers[0] - centers[1])
            # Only split if distinct physical surface materials exist (e.g. grass vs concrete)
            if center_dist >= 0.15:
                sub0 = np.zeros((h, w), dtype=np.uint8)
                sub1 = np.zeros((h, w), dtype=np.uint8)
                sub0[coords[0][clabels.flatten() == 0], coords[1][clabels.flatten() == 0]] = 1
                sub1[coords[0][clabels.flatten() == 1], coords[1][clabels.flatten() == 1]] = 1
                # Enforce spatial coherence to prevent speckled noise splitting
                sub0 = cv2.morphologyEx(sub0, cv2.MORPH_OPEN, clean_kernel)
                sub1 = cv2.morphologyEx(sub1, cv2.MORPH_OPEN, clean_kernel)
                if sub0.sum() >= 0.15 * reg_pixels and sub1.sum() >= 0.15 * reg_pixels:
                    sub_regions = [sub0, sub1]

        for sub in sub_regions:
            sub_px = sub.sum()
            if sub_px < min_pixels:
                continue

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
            chroma_dists = np.linalg.norm(outer_chroma - sub_chroma, axis=1) if len(outer_coords[0]) > 0 else np.array([])

            # Filter outer strip to pixels belonging to the same surface
            matched_indices = np.where(chroma_dists < 0.15)[0] if len(chroma_dists) > 0 else []
            if len(matched_indices) >= min_pixels:
                matched_outer = (
                    outer_coords[0][matched_indices],
                    outer_coords[1][matched_indices],
                )
            else:
                matched_outer = outer_coords

            # Log-domain ratio estimation
            if len(matched_outer[0]) >= min_pixels and inner_strip.sum() >= min_pixels:
                log_outer_median = np.median(log_f[matched_outer], axis=0)
                log_inner_median = np.median(log_f[inner_strip > 0], axis=0)
                log_ratio = log_outer_median - log_inner_median
                gain = np.clip(np.exp(log_ratio), 1.0, max_gain)
            elif ring.sum() >= min_pixels and sub.sum() >= min_pixels:
                gain = np.clip(
                    np.median(f[ring > 0], axis=0) / np.maximum(np.median(f[sub > 0], axis=0), 1e-4),
                    1.0,
                    max_gain,
                )
            else:
                gain = gain_global

            # Adaptive anchor: smoothly blend with global gain if matched pixels are few
            if len(matched_outer[0]) < 100:
                weight = len(matched_outer[0]) / 100.0
                gain = weight * gain + (1.0 - weight) * gain_global

            # Apply gain strictly to the shadow sub-region (no dilated halo artifact)
            out[sub > 0] = f[sub > 0] * gain

    relit_srgb = linear_to_srgb(out, as_uint8=True) if linear else np.clip(out, 0, 255).astype(np.uint8)

    if b_weight > 0 and binary.sum() >= min_pixels:
        # Additive color temperature compensation in LAB space (compensating blue ambient skylight)
        lab_relit = cv2.cvtColor(relit_srgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_orig = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        b_ch = lab_relit[..., 2]
        b_lit = np.median(lab_orig[binary == 0, 2])
        b_shadow = np.median(b_ch[binary > 0])
        delta_b = b_lit - b_shadow
        if delta_b > 0:
            b_ch[binary > 0] = np.clip(b_ch[binary > 0] + b_weight * delta_b, 0, 255)
            lab_relit[..., 2] = b_ch
            relit_srgb = cv2.cvtColor(np.clip(lab_relit, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    return relit_srgb
