"""Sweep detector hyper-parameters on mask quality only (fast: no relighting).

Usage:
    python -m experiments.sweep_detect --root data/ISTD_Dataset/test --limit 100
Prints mean IoU / BER and the worst-case IoU for each max_area_frac value.
Use an evenly spaced sample (default) so the sweep is not dominated by a few scenes.
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from experiments import metrics as M
from src.detect import detect
from src.utils import load_image, load_mask

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--limit", type=int, default=100)
ap.add_argument("--values", type=float, nargs="+", default=[0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1])
args = ap.parse_args()

root = Path(args.root)
all_names = sorted(os.listdir(root / "test_A"))
step = max(1, len(all_names) // args.limit)
names = all_names[::step][: args.limit]
data = []
for n in names:
    img, gt = load_image(root / "test_A" / n), load_mask(root / "test_B" / n)
    if gt.shape != img.shape[:2]:
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    data.append((img, gt))

gt_area = np.mean([(g > 0.5).mean() for _, g in data])
print(f"{len(names)} images, mean GT shadow area {100 * gt_area:.1f}%\n")
print("max_area_frac | mean IoU | mean BER | worst IoU | images with IoU<0.3")
for v in args.values:
    ious, bers = [], []
    for img, gt in data:
        m = detect(img, max_area_frac=v)
        ious.append(M.iou(m, gt))
        bers.append(M.ber(m, gt))
    ious = np.array(ious)
    print(f"{v:13.2f} | {ious.mean():8.3f} | {np.mean(bers):8.2f} | {ious.min():9.3f} | {(ious < 0.3).sum()}")
