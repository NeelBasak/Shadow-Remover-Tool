"""Evaluate baselines, our pipeline and ablations on ISTD / SRD-style folders.

Expected layout (ISTD):
    <root>/test_A  shadow images
    <root>/test_B  shadow masks
    <root>/test_C  shadow-free ground truth
Files in the three folders share the same name.

Usage:
    python -m experiments.run_eval --root data/ISTD_Dataset/test --limit 50 --out experiments/results

Rows in the output table:
  * baseline          : Otsu detector + one global gain
  * auto mask ablations: remove / blend variants using our detector
  * auto + scribbles  : our detector, then GrabCut refinement with SIMULATED
                        scribbles sampled from the detector's errors (stand-in
                        for a real user; see simulate_scribbles)
  * GT mask variants  : isolate removal quality from detection quality

Outputs (in --out):
  results.csv        mean/std per method
  per_image.csv      every metric for every image and method (long format)
  paired.csv         paired comparisons: mean difference, 95% bootstrap CI, win-rate
  failures.csv       images sorted by auto-mask IoU (worst first)
  worst/             comparison strips for the --save-worst hardest images
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
from src.refine import refine, SCRIB_SHADOW, SCRIB_LIT
from src.remove import remove, remove_global
from src.utils import load_image, load_mask, overlay_mask, save_image


def simulate_scribbles(pred_mask, gt_mask, n_points=6, radius=3, seed=0):
    """Fake a user: a few short strokes placed on the detector's mistakes.

    False negatives (missed shadow) get a 'shadow' stroke, false positives
    (wrongly marked shadow) get a 'lit' stroke. n_points strokes per type.
    """
    rng = np.random.default_rng(seed)
    scrib = np.zeros(gt_mask.shape, np.uint8)
    p, g = pred_mask > 0.5, gt_mask > 0.5
    for err, label in (((~p) & g, SCRIB_SHADOW), (p & (~g), SCRIB_LIT)):
        pts = np.argwhere(err)
        if len(pts) == 0:
            continue
        pick = pts[rng.choice(len(pts), min(n_points, len(pts)), replace=False)]
        for y, x in pick:
            cv2.circle(scrib, (int(x), int(y)), radius, int(label), -1)
    return scrib


def methods(img, gt_mask, seed=0):
    """Return {name: (result_image, predicted_mask)} for every variant."""
    auto = detect(img)
    otsu = detect_otsu(img)
    scrib = simulate_scribbles(auto, gt_mask, seed=seed)
    refined = refine(img, auto, scrib)

    out = {}
    out["baseline: Otsu + global gain"] = (remove_global(img, otsu), otsu)
    out["auto mask: global gain"] = (remove_global(img, auto), auto)
    out["auto mask: global gain + blend"] = (blend(img, remove_global(img, auto), auto), auto)
    out["auto mask: region relight, no blend"] = (remove(img, auto), auto)
    out["auto mask: region relight, blend w/o guided filter"] = (blend(img, remove(img, auto), auto, use_guided=False), auto)
    out["auto mask: region relight + blend (full)"] = (blend(img, remove(img, auto), auto), auto)
    out["auto + simulated scribbles: full"] = (blend(img, remove(img, refined), refined), refined)
    out["GT mask: global gain"] = (remove_global(img, gt_mask), gt_mask)
    out["GT mask: global gain + blend"] = (blend(img, remove_global(img, gt_mask), gt_mask), gt_mask)
    out["GT mask: region relight, no blend"] = (remove(img, gt_mask), gt_mask)
    out["GT mask: region relight + blend"] = (blend(img, remove(img, gt_mask), gt_mask), gt_mask)
    return out


# (name, A, B): is A better than B?  Compared on shadow-region RMSE (lower is better)
# and PSNR (higher is better).
PAIRS = [
    ("full pipeline vs baseline", "auto mask: region relight + blend (full)", "baseline: Otsu + global gain"),
    ("region relight vs global gain (both + blend, auto mask)", "auto mask: region relight + blend (full)", "auto mask: global gain + blend"),
    ("region relight vs global gain (both + blend, GT mask)", "GT mask: region relight + blend", "GT mask: global gain + blend"),
    ("blend vs no blend (region relight, auto mask)", "auto mask: region relight + blend (full)", "auto mask: region relight, no blend"),
    ("guided filter vs none (region relight + blend, auto mask)", "auto mask: region relight + blend (full)", "auto mask: region relight, blend w/o guided filter"),
    ("scribble refinement vs none", "auto + simulated scribbles: full", "auto mask: region relight + blend (full)"),
]
METRIC_IDX = {"RMSE_LAB_shadow": 0, "RMSE_LAB_all": 1, "PSNR": 2, "SSIM": 3, "mask_IoU": 4, "mask_BER": 5}
LOWER_IS_BETTER = {"RMSE_LAB_shadow", "RMSE_LAB_all", "mask_BER"}


def paired_stats(a, b, lower_better, n_boot=2000, seed=0):
    """Paired difference A - B, flipped so that positive always means 'A better'.

    Returns (mean_improvement, ci_low, ci_high, win_rate_percent).
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    if lower_better:
        d = -d
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (n_boot, len(d)), replace=True).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float(100 * (d > 0).mean())


def save_comparison(path, img, auto_mask, gt_mask, result, gt_img):
    """Strip: input | auto mask | GT mask | our result | ground truth."""
    panels = [img, overlay_mask(img, auto_mask), overlay_mask(img, gt_mask, color=(0, 255, 0)), result, gt_img]
    labels = ["input", "auto mask", "GT mask", "ours", "GT"]
    panels = [p.copy() for p in panels]
    for p, t in zip(panels, labels):
        cv2.putText(p, t, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    save_image(path, np.hstack(panels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--save-images", type=int, default=5, help="save qualitative results for first N")
    ap.add_argument("--save-worst", type=int, default=8, help="save comparison strips for the N images with lowest auto-mask IoU")
    args = ap.parse_args()

    root = Path(args.root)
    names = sorted(os.listdir(root / "test_A"))[: args.limit]
    os.makedirs(args.out, exist_ok=True)

    rows = {}
    per_image = []   # (image, method, metrics...)
    keep = {}        # image -> (img, auto mask, gt mask, full result, gt img) for failure strips
    for idx, name in enumerate(names):
        img = load_image(root / "test_A" / name)
        gt_mask = load_mask(root / "test_B" / name)
        gt_img = load_image(root / "test_C" / name)
        if gt_img.shape != img.shape:
            gt_img = cv2.resize(gt_img, (img.shape[1], img.shape[0]))
            gt_mask = cv2.resize(gt_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        for mname, (res, pmask) in methods(img, gt_mask, seed=idx).items():
            vals = [
                M.rmse_lab(res, gt_img, gt_mask), M.rmse_lab(res, gt_img),
                M.psnr(res, gt_img), M.ssim(res, gt_img),
                M.iou(pmask, gt_mask), M.ber(pmask, gt_mask),
            ]
            rows.setdefault(mname, []).append(vals)
            per_image.append([name, mname] + vals)
            if mname == "auto mask: region relight + blend (full)":
                keep[name] = (img, pmask, gt_mask, res, gt_img)
            if idx < args.save_images:
                tag = "".join(c if c.isalnum() else "_" for c in mname)
                save_image(Path(args.out) / f"{Path(name).stem}__{tag}.png", res)
        print(f"[{idx + 1}/{len(names)}] {name}")

    header = ["method", "RMSE_LAB_shadow", "RMSE_LAB_all", "PSNR", "SSIM", "mask_IoU", "mask_BER"]
    n = len(names)
    with open(Path(args.out) / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header + [f"std_{h}" for h in header[1:]] + ["n_images"])
        for mname, vals in rows.items():
            a = np.array(vals)
            w.writerow([mname] + [f"{v:.3f}" for v in np.nanmean(a, 0)]
                       + [f"{v:.3f}" for v in np.nanstd(a, 0)] + [n])

    # ---- per-image CSV (long format) ----
    with open(Path(args.out) / "per_image.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "method"] + header[1:])
        for r in per_image:
            w.writerow([r[0], r[1]] + [f"{v:.4f}" for v in r[2:]])

    # ---- paired comparisons ----
    print("\nPaired comparisons (positive = A better; 95% bootstrap CI of the mean; win-rate = % images A better)")
    with open(Path(args.out) / "paired.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "metric", "mean_improvement", "ci_low", "ci_high", "win_rate_percent", "n_images"])
        for label, A, B in PAIRS:
            if A not in rows or B not in rows:
                continue
            a_arr, b_arr = np.array(rows[A]), np.array(rows[B])
            for metric in ("RMSE_LAB_shadow", "PSNR"):
                i = METRIC_IDX[metric]
                m, lo, hi, win = paired_stats(a_arr[:, i], b_arr[:, i], metric in LOWER_IS_BETTER)
                w.writerow([label, metric, f"{m:.3f}", f"{lo:.3f}", f"{hi:.3f}", f"{win:.1f}", n])
                flag = "" if (lo > 0 or hi < 0) else "  (CI includes 0: not clearly different)"
                print(f"  {label} | {metric}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}], wins {win:.0f}%{flag}")

    # ---- failure analysis ----
    full_name = "auto mask: region relight + blend (full)"
    order = sorted(range(n), key=lambda k: rows[full_name][k][4])  # by auto-mask IoU, worst first
    with open(Path(args.out) / "failures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "auto_mask_IoU", "auto_mask_BER", "full_RMSE_shadow", "full_PSNR"])
        for k in order:
            r = rows[full_name][k]
            w.writerow([names[k], f"{r[4]:.3f}", f"{r[5]:.2f}", f"{r[0]:.3f}", f"{r[2]:.2f}"])
    worst_dir = Path(args.out) / "worst"
    os.makedirs(worst_dir, exist_ok=True)
    for rank, k in enumerate(order[: args.save_worst]):
        save_comparison(worst_dir / f"{rank + 1:02d}_{Path(names[k]).stem}.png", *keep[names[k]])
    print(f"\nWorst {min(args.save_worst, n)} images by auto-mask IoU (strips saved to {worst_dir}):")
    for k in order[: min(5, n)]:
        print(f"  {names[k]}: IoU {rows[full_name][k][4]:.3f}, BER {rows[full_name][k][5]:.1f}")

    print(f"\nMean over {n} images (std in brackets)")
    print(" | ".join(header))
    for mname, vals in rows.items():
        a = np.array(vals)
        m, s = np.nanmean(a, 0), np.nanstd(a, 0)
        print(mname, "|", " | ".join(f"{mv:.3f} ({sv:.2f})" for mv, sv in zip(m, s)))


if __name__ == "__main__":
    main()
