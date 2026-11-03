"""Evaluation metrics (DESIGN.md section 7)."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.utils import rgb_to_lab


def rmse_lab(pred, gt, region=None):
    """RMSE in LAB space (L, a, b scaled as in OpenCV), optionally on a region mask."""
    d = rgb_to_lab(pred) - rgb_to_lab(gt)
    if region is not None:
        sel = region > 0.5
        if sel.sum() == 0:
            return float("nan")
        d = d[sel]
    return float(np.sqrt((d ** 2).mean()))


def psnr(pred, gt):
    return float(peak_signal_noise_ratio(gt, pred, data_range=255))


def ssim(pred, gt):
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=255))


def iou(pred_mask, gt_mask):
    p, g = pred_mask > 0.5, gt_mask > 0.5
    union = (p | g).sum()
    return float((p & g).sum() / union) if union else 1.0


def ber(pred_mask, gt_mask):
    """Balanced error rate in percent (lower is better)."""
    p, g = pred_mask > 0.5, gt_mask > 0.5
    pos, neg = g.sum(), (~g).sum()
    tpr = (p & g).sum() / pos if pos else 1.0
    tnr = (~p & ~g).sum() / neg if neg else 1.0
    return float(100 * (1 - 0.5 * (tpr + tnr)))
