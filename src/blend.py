"""Module D: penumbra-aware blending.

final = alpha * relit + (1 - alpha) * original

alpha is derived from the soft shadow mask, snapped to image edges using a
guided filter (He et al., ECCV 2010) with the grayscale input as guide. This
ensures the illumination transition conforms tightly to actual scene contours,
preventing halo and fringe artifacts at the boundary.
"""
import cv2
import numpy as np

from .utils import guided_filter


def blend(img, relit, mask, penumbra=None, use_guided=True, eps=1e-3):
    """Blend relit image into original with penumbra-aware edge smoothing.

    Args:
        img: original RGB uint8 image.
        relit: relit RGB uint8 image from remove.py.
        mask: float32 mask in [0, 1].
        penumbra: radius for transition band.
        use_guided: whether to apply guided filter edge alignment.
        eps: regularization parameter for guided filter.

    Returns:
        RGB uint8 shadow-free blended image.
    """
    h, w = img.shape[:2]
    penumbra = penumbra if penumbra is not None else max(2, int(0.008 * min(h, w)))

    alpha = mask.astype(np.float32)

    # Widen transition band slightly to encompass the full penumbra zone
    k = 2 * penumbra + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    alpha_dilated = cv2.dilate(alpha, kernel)
    alpha_blurred = cv2.GaussianBlur(alpha_dilated, (0, 0), penumbra)

    if use_guided:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        alpha_final = guided_filter(gray, alpha_blurred, r=max(4, 2 * penumbra), eps=eps)
    else:
        alpha_final = alpha_blurred

    alpha_final = np.clip(alpha_final, 0.0, 1.0)[..., None]

    # Seamless convex combination
    out = alpha_final * relit.astype(np.float32) + (1.0 - alpha_final) * img.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
