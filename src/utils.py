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


def srgb_to_linear(img):
    """Convert sRGB image to linear radiometric RGB (IEC 61966-2-1).

    Accepts uint8 [0, 255] or float32. Returns float32 in [0, 1].
    Optical radiance (I = rho * L) is linear with photon flux; camera images
    are non-linear (~2.2 gamma). Relighting must occur in linear space.
    """
    x = img.astype(np.float32) / 255.0 if (img.dtype == np.uint8 or img.max() > 1.0) else img.astype(np.float32)
    x = np.clip(x, 0.0, 1.0)
    linear = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return linear.astype(np.float32)


def linear_to_srgb(linear, as_uint8=True):
    """Convert linear radiometric RGB back to display sRGB (IEC 61966-2-1).

    Args:
        linear: float32 linear RGB in [0, 1].
        as_uint8: if True, returns uint8 in [0, 255]; otherwise float32 in [0, 1].
    """
    x = np.clip(linear.astype(np.float32), 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)
    srgb_255 = np.clip(np.round(srgb * 255.0), 0.0, 255.0)
    return srgb_255.astype(np.uint8) if as_uint8 else (srgb_255 / 255.0).astype(np.float32)


def normalized_rgb(img, eps=1e-5):
    """Compute normalized chromaticity coordinates: (R, G, B) / (R + G + B + eps)."""
    f = img.astype(np.float32)
    s = f.sum(axis=-1, keepdims=True) + eps
    return f / s


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
