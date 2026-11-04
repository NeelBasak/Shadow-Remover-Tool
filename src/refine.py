"""Module B: interactive mask refinement with user scribbles (GrabCut).

Scribble map convention: uint8 HxW with:
    0 = untouched / unconstrained
    1 = definite shadow (user marks shadow)
    2 = definite lit (user marks non-shadow)

The automatic mask initializes GrabCut; user scribbles override it as hard
constraints.
"""
import cv2
import numpy as np

SCRIB_NONE, SCRIB_SHADOW, SCRIB_LIT = 0, 1, 2


def refine(img, mask, scribbles=None, iters=3, soft_sigma=2.0):
    """Return a refined soft mask (float32 in [0, 1]).

    Args:
        img: RGB uint8 image.
        mask: initial float32 soft or binary mask in [0, 1].
        scribbles: uint8 array matching image size with values 0, 1, 2.
        iters: GrabCut iterations (3 is fast and sufficient for interactive UI).
        soft_sigma: Gaussian sigma to soften boundary.
    """
    if scribbles is None or not np.any(scribbles):
        return mask

    h, w = img.shape[:2]
    gc = np.where(mask > 0.5, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)

    # Hard constraints from user scribbles
    has_shadow_scribble = np.any(scribbles == SCRIB_SHADOW)
    has_lit_scribble = np.any(scribbles == SCRIB_LIT)

    if has_shadow_scribble:
        gc[scribbles == SCRIB_SHADOW] = cv2.GC_FGD
    if has_lit_scribble:
        gc[scribbles == SCRIB_LIT] = cv2.GC_BGD

    # If initial mask was empty and user scribbled shadow, provide background context
    if not np.any((gc == cv2.GC_BGD) | (gc == cv2.GC_PR_BGD)):
        gc[scribbles != SCRIB_SHADOW] = cv2.GC_PR_BGD

    # If mask covered everything and user scribbled lit, provide foreground context
    if not np.any((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)):
        gc[scribbles != SCRIB_LIT] = cv2.GC_PR_FGD

    # GrabCut requires at least one FG and one BG sample
    has_fg = np.any((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))
    has_bg = np.any((gc == cv2.GC_BGD) | (gc == cv2.GC_PR_BGD))
    if not (has_fg and has_bg):
        return mask

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(bgr, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask

    binary = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.float32)

    # Smooth boundary
    soft = cv2.GaussianBlur(binary, (0, 0), soft_sigma)

    # Enforce hard constraints so user scribbles aren't washed out
    if has_lit_scribble:
        soft[scribbles == SCRIB_LIT] = 0.0
    if has_shadow_scribble:
        soft[scribbles == SCRIB_SHADOW] = 1.0

    return np.clip(soft, 0.0, 1.0).astype(np.float32)
