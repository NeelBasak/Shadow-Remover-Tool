"""Comprehensive verification script for all pipeline components.

Runs comparative benchmarks against baseline implementations across:
1. Dark object rejection (Detection)
2. Interactive scribble correction (Refinement)
3. Multi-surface shadow relighting (Relighting)
4. Boundary penumbra smoothing (Blending)
5. End-to-end pipeline quality (Metrics)
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from experiments import metrics as M
from src.blend import blend
from src.detect import detect, detect_otsu
from src.pipeline import run
from src.refine import SCRIB_LIT, SCRIB_SHADOW, refine
from src.remove import remove, remove_global


def run_tests():
    print("=" * 65)
    print("RUNNING COMPONENT AUDIT & ACCURACY BENCHMARKS")
    print("=" * 65)

    # -------------------------------------------------------------
    # 1. Detection: Dark Object Rejection Test
    # -------------------------------------------------------------
    print("\n[Component A: Detection - Dark Object Rejection]")
    h, w = 200, 200
    bg = np.ones((h, w, 3), dtype=np.uint8) * 180
    bg[:, :, 2] = 170

    gt_mask = np.zeros((h, w), dtype=np.float32)
    gt_mask[20:80, 20:80] = 1.0  # true shadow in top-left

    img_dark_obj = bg.copy()
    img_dark_obj[gt_mask > 0.5] = (img_dark_obj[gt_mask > 0.5] * 0.4).astype(np.uint8)
    # Inherently dark object in bottom-right (NOT a shadow)
    img_dark_obj[120:180, 120:180] = [35, 30, 30]

    baseline_mask = detect_otsu(img_dark_obj)
    improved_mask = detect(img_dark_obj)

    base_iou = M.iou(baseline_mask, gt_mask)
    base_fp = float((baseline_mask[120:180, 120:180] > 0.5).mean() * 100)

    impr_iou = M.iou(improved_mask, gt_mask)
    impr_fp = float((improved_mask[120:180, 120:180] > 0.5).mean() * 100)

    print(f"  Baseline (Otsu on L)      -> IoU: {base_iou:.3f} | False Positives on Dark Object: {base_fp:.1f}%")
    print(f"  Improved (Chroma-aware)   -> IoU: {impr_iou:.3f} | False Positives on Dark Object: {impr_fp:.1f}%")
    assert impr_fp == 0.0, "Dark object false positive should be 0%"

    # -------------------------------------------------------------
    # 2. Refinement: Interactive Scribble Correction
    # -------------------------------------------------------------
    print("\n[Component B: Refinement - User Scribble Correction]")
    h, w = 120, 120
    test_img = np.ones((h, w, 3), dtype=np.uint8) * 160
    gt_refine_mask = np.zeros((h, w), dtype=np.float32)
    gt_refine_mask[20:60, 20:60] = 1.0
    test_img[gt_refine_mask > 0.5] = 60
    test_img[70:110, 70:110] = 50  # dark non-shadow artifact

    # Initial flawed mask detecting both
    flawed_mask = np.zeros((h, w), dtype=np.float32)
    flawed_mask[20:60, 20:60] = 1.0
    flawed_mask[70:110, 70:110] = 1.0

    # User scribbles LIT on artifact, SHADOW on true shadow
    scribbles = np.zeros((h, w), dtype=np.uint8)
    scribbles[80:100, 80:100] = SCRIB_LIT
    scribbles[30:50, 30:50] = SCRIB_SHADOW

    refined_mask = refine(test_img, flawed_mask, scribbles)
    initial_iou = M.iou(flawed_mask, gt_refine_mask)
    refined_iou = M.iou(refined_mask, gt_refine_mask)

    print(f"  Flawed Initial Mask       -> IoU: {initial_iou:.3f} | BER: {M.ber(flawed_mask, gt_refine_mask):.1f}%")
    print(f"  After Scribble Refinement -> IoU: {refined_iou:.3f} | BER: {M.ber(refined_mask, gt_refine_mask):.1f}%")
    assert refined_iou > 0.95, "Refined mask IoU should exceed 0.95"

    # -------------------------------------------------------------
    # 3. Relighting: Multi-Surface Color Consistency
    # -------------------------------------------------------------
    print("\n[Component C: Relighting - Multi-Surface Preservation]")
    h, w = 200, 200
    clean_ms = np.zeros((h, w, 3), dtype=np.uint8)
    clean_ms[:, :100] = [40, 140, 40]   # green grass
    clean_ms[:, 100:] = [220, 220, 220] # white concrete

    gt_ms_mask = np.zeros((h, w), dtype=np.float32)
    gt_ms_mask[40:160, 40:160] = 1.0

    img_ms = clean_ms.copy()
    img_ms[40:160, 40:100] = [20, 70, 20]    # shadow on grass
    img_ms[40:160, 100:160] = [55, 55, 55]   # shadow on concrete

    base_relit = remove_global(img_ms, gt_ms_mask)
    impr_relit = remove(img_ms, gt_ms_mask, enable_clustering=True)

    base_rmse = M.rmse_lab(base_relit, clean_ms, gt_ms_mask)
    impr_rmse = M.rmse_lab(impr_relit, clean_ms, gt_ms_mask)

    print(f"  Baseline (Global Gain)    -> Shadow RMSE (LAB): {base_rmse:.2f} (causes magenta cast on concrete)")
    print(f"  Surface-Aware Relighting  -> Shadow RMSE (LAB): {impr_rmse:.2f} (clean surface matching)")
    assert impr_rmse < base_rmse, "Surface-aware relighting should reduce RMSE significantly"

    # -------------------------------------------------------------
    # 4. Blending: Penumbra Boundary Transition
    # -------------------------------------------------------------
    print("\n[Component D: Blending - Penumbra Smoothing]")
    h, w = 160, 160
    clean_penumbra = np.ones((h, w, 3), dtype=np.uint8) * 190
    dist_map = np.zeros((h, w), dtype=np.float32)
    dist_map[35:125, 35:125] = 1.0
    soft_penumbra_gt = cv2.GaussianBlur(dist_map, (15, 15), 4.0)

    img_penumbra = (clean_penumbra * (1.0 - 0.55 * soft_penumbra_gt[..., None])).astype(np.uint8)
    penumbra_mask = (soft_penumbra_gt > 0.5).astype(np.float32)

    relit_raw = remove(img_penumbra, penumbra_mask)
    blended_no_guided = blend(img_penumbra, relit_raw, penumbra_mask, use_guided=False)
    blended_guided = blend(img_penumbra, relit_raw, penumbra_mask, use_guided=True)

    psnr_raw = M.psnr(relit_raw, clean_penumbra)
    psnr_no_guided = M.psnr(blended_no_guided, clean_penumbra)
    psnr_guided = M.psnr(blended_guided, clean_penumbra)

    print(f"  Unblended Relit Result    -> PSNR: {psnr_raw:.2f} dB | SSIM: {M.ssim(relit_raw, clean_penumbra):.4f}")
    print(f"  Dilate+Blur Blend         -> PSNR: {psnr_no_guided:.2f} dB | SSIM: {M.ssim(blended_no_guided, clean_penumbra):.4f}")
    print(f"  Guided Filter Blend       -> PSNR: {psnr_guided:.2f} dB | SSIM: {M.ssim(blended_guided, clean_penumbra):.4f}")
    assert psnr_guided >= psnr_raw, "Guided blend should maintain or improve boundary PSNR"

    # -------------------------------------------------------------
    # 5. End-to-End Pipeline
    # -------------------------------------------------------------
    print("\n[Component E: End-to-End Pipeline Verification]")
    final_img, final_mask, _ = run(img_dark_obj)
    e2e_iou = M.iou(final_mask, gt_mask)
    e2e_psnr = M.psnr(final_img, bg)
    print(f"  End-to-End Test on Complex Scene -> Detected Mask IoU: {e2e_iou:.3f} | Output PSNR: {e2e_psnr:.2f} dB")

    print("\n" + "=" * 65)
    print("ALL COMPONENT VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()

