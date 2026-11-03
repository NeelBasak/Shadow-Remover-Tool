"""Module D: penumbra-aware blending.

final = alpha * relit + (1 - alpha) * original

alpha is the soft shadow mask, widened to cover the penumbra and snapped to
image edges with a guided filter, so that the transition follows the real
shadow boundary instead of leaving a dark or bright halo.
"""
import cv2
import numpy as np

from .utils import guided_filter


def blend(img, relit, mask, penumbra=None, use_guided=True, eps=1e-3):
    """Blend relit image into the original. Returns RGB uint8."""
    h, w = img.shape[:2]
    penumbra = penumbra if penumbra is not None else max(2, int(0.008 * min(h, w)))

    alpha = mask.astype(np.float32)
    # Widen slightly so the relit region covers the penumbra.
    k = 2 * penumbra + 1
    alpha = cv2.dilate(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    alpha = cv2.GaussianBlur(alpha, (0, 0), penumbra)

    if use_guided:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        alpha = guided_filter(gray, alpha, r=max(4, 2 * penumbra), eps=eps)
    alpha = np.clip(alpha, 0, 1)[..., None]

    out = alpha * relit.astype(np.float32) + (1 - alpha) * img.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
