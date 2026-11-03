"""Evaluate baselines, our pipeline and ablations on ISTD / SRD-style folders.

Expected layout (ISTD):
    <root>/test_A  shadow images
    <root>/test_B  shadow masks
    <root>/test_C  shadow-free ground truth
Files in the three folders share the same name.

Usage:
    python -m experiments.run_eval --root data/ISTD/test --limit 50 --out experiments/results
"""
import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

from experiments import metrics as M
from src.blend import blend
from src.detect import detect, detect_otsu
from src.remove import remove, remove_global
from src.utils import load_image, load_mask, save_image


def methods(img, gt_mask):
    """Return {name: (result_image, predicted_mask)} for every variant."""
    auto = detect(img)
    otsu = detect_otsu(img)
    out = {}
    out["baseline: Otsu + global gain"] = (remove_global(img, otsu), otsu)
    out["ours (auto mask), no blend"] = (remove(img, auto), auto)
    out["ours (auto mask), no guided filter"] = (blend(img, remove(img, auto), auto, use_guided=False), auto)
    out["ours (auto mask), full"] = (blend(img, remove(img, auto), auto), auto)
    out["GT mask + global gain"] = (remove_global(img, gt_mask), gt_mask)
    out["GT mask + region relight, no blend"] = (remove(img, gt_mask), gt_mask)
    out["GT mask + region relight + blend"] = (blend(img, remove(img, gt_mask), gt_mask), gt_mask)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--save-images", type=int, default=5, help="save qualitative results for first N")
    args = ap.parse_args()

    root = Path(args.root)
    names = sorted(os.listdir(root / "test_A"))[: args.limit]
    os.makedirs(args.out, exist_ok=True)

    rows = {}
    for idx, name in enumerate(names):
        img = load_image(root / "test_A" / name)
        gt_mask = load_mask(root / "test_B" / name)
        gt_img = load_image(root / "test_C" / name)
        if gt_img.shape != img.shape:
            gt_img = cv2.resize(gt_img, (img.shape[1], img.shape[0]))
            gt_mask = cv2.resize(gt_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        for mname, (res, pmask) in methods(img, gt_mask).items():
            rows.setdefault(mname, []).append([
                M.rmse_lab(res, gt_img, gt_mask), M.rmse_lab(res, gt_img),
                M.psnr(res, gt_img), M.ssim(res, gt_img),
                M.iou(pmask, gt_mask), M.ber(pmask, gt_mask),
            ])
            if idx < args.save_images:
                tag = "".join(c if c.isalnum() else "_" for c in mname)
                save_image(Path(args.out) / f"{Path(name).stem}__{tag}.png", res)
        print(f"[{idx + 1}/{len(names)}] {name}")

    header = ["method", "RMSE_LAB_shadow", "RMSE_LAB_all", "PSNR", "SSIM", "mask_IoU", "mask_BER"]
    with open(Path(args.out) / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for mname, vals in rows.items():
            w.writerow([mname] + [f"{v:.3f}" for v in np.nanmean(np.array(vals), axis=0)])

    print("\n" + " | ".join(header))
    for mname, vals in rows.items():
        print(mname, "|", " | ".join(f"{v:.3f}" for v in np.nanmean(np.array(vals), axis=0)))


if __name__ == "__main__":
    main()
