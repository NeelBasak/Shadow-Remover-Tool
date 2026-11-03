"""Module B: interactive mask refinement with user scribbles (GrabCut).

Scribble map convention: uint8 HxW with 0 = untouched, 1 = "this is shadow",
2 = "this is NOT shadow". The automatic mask initialises GrabCut; scribbles
override it as hard constraints.
"""
import cv2
import numpy as np

SCRIB_NONE, SCRIB_SHADOW, SCRIB_LIT = 0, 1, 2


def refine(img, mask, scribbles=None, iters=4, soft_sigma=2.0):
    """Return a refined soft mask (float32 in [0,1])."""
    if scribbles is None or not np.any(scribbles):
        return mask

    gc = np.where(mask > 0.5, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gc[scribbles == SCRIB_SHADOW] = cv2.GC_FGD
    gc[scribbles == SCRIB_LIT] = cv2.GC_BGD

    # GrabCut needs at least one definite/probable sample of each class.
    if not np.any((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)) or \
       not np.any((gc == cv2.GC_BGD) | (gc == cv2.GC_PR_BGD)):
        return mask

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask

    binary = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.float32)
    return np.clip(cv2.GaussianBlur(binary, (0, 0), soft_sigma), 0, 1)
