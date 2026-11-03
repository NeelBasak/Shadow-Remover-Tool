"""Shared helpers: image I/O, colour conversion, box filter, guided filter.

Conventions (see DESIGN.md section 5):
  * images are RGB uint8, shape HxWx3
  * masks are float32 in [0, 1], shape HxW (1 = shadow)
"""
import cv2
import numpy as np


def load_image(path):
    """Read an image from disk as RGB uint8."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image(path, img):
    """Write an RGB uint8 image to disk."""
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_mask(path):
    """Read a mask image as float32 in [0, 1]."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (m.astype(np.float32) / 255.0 > 0.5).astype(np.float32)


def rgb_to_lab(img):
    """RGB uint8 -> float32 LAB (L in [0,100], a/b roughly [-127,127])."""
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def box_filter(x, r):
    """Mean filter with radius r (window 2r+1)."""
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REPLICATE)


def guided_filter(guide, src, r=8, eps=1e-3):
    """Edge-preserving smoothing of `src` using a grayscale `guide`.

    He et al., "Guided Image Filtering", ECCV 2010 (gray-guide version).
    Both inputs are float32 in [0, 1].
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    mean_g = box_filter(guide, r)
    mean_s = box_filter(src, r)
    cov_gs = box_filter(guide * src, r) - mean_g * mean_s
    var_g = box_filter(guide * guide, r) - mean_g * mean_g
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return box_filter(a, r) * guide + box_filter(b, r)


def overlay_mask(img, mask, color=(255, 0, 0), alpha=0.5):
    """Visualise a mask on top of an image (for figures and the demo)."""
    out = img.astype(np.float32).copy()
    m = mask[..., None]
    out = out * (1 - alpha * m) + np.array(color, np.float32) * alpha * m
    return np.clip(out, 0, 255).astype(np.uint8)
