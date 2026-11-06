import sys
import os
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
os.environ["MPLCONFIGDIR"] = str(Path(__file__).parent)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import load_image, load_mask, overlay_mask

fig_dir = Path("report/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Pipeline Overview Diagram
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.8), dpi=300)
ax.axis("off")

boxes = [
    ("Input Image\nRGB (sRGB)", (0.05, 0.5), "#E8F0FE", "#1A73E8"),
    ("Module A: Detect\nDaylight Physics\n(|Δa|≤3.5, Δb≤2.5)", (0.24, 0.72), "#FEF7E0", "#F9AB00"),
    ("Module B: Refine\nInteractive GrabCut\n(Hard Constraints)", (0.45, 0.5), "#E6F4EA", "#1E8E3E"),
    ("Module C: Relight\nLinear Radiometric\nBoundary Ratio + Δb*", (0.68, 0.5), "#FCE8E6", "#D93025"),
    ("Module D: Blend\nEdge-Guided Filter\nPenumbra Blending", (0.88, 0.5), "#F3E8FD", "#9334E6"),
]

for label, (x, y), facecolor, edgecolor in boxes:
    ax.text(
        x, y, label, ha="center", va="center", fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", facecolor=facecolor, edgecolor=edgecolor, lw=2)
    )

ax.text(
    0.24, 0.28, "User Scribbles\n(Lit / Shadow Seeds)", ha="center", va="center", fontsize=8.5,
    style="italic", bbox=dict(boxstyle="round,pad=0.5", facecolor="#F1F3F4", edgecolor="#5F6368", lw=1.5, ls="--")
)

arrowprops = dict(arrowstyle="->", lw=2, color="#3C4043")
ax.annotate("", xy=(0.14, 0.65), xytext=(0.09, 0.55), arrowprops=arrowprops)
ax.annotate("", xy=(0.35, 0.72), xytext=(0.33, 0.72), arrowprops=arrowprops)
ax.annotate("", xy=(0.39, 0.56), xytext=(0.35, 0.70), arrowprops=arrowprops)
ax.annotate("", xy=(0.39, 0.44), xytext=(0.32, 0.32), arrowprops=arrowprops)
ax.annotate("", xy=(0.58, 0.5), xytext=(0.52, 0.5), arrowprops=arrowprops)
ax.annotate("", xy=(0.79, 0.5), xytext=(0.76, 0.5), arrowprops=arrowprops)

ax.annotate("", xy=(0.60, 0.62), xytext=(0.10, 0.62),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#5F6368", ls=":", connectionstyle="arc3,rad=-0.2"))
ax.text(0.35, 0.94, "Optional: Ground-Truth Mask (Evaluation Bypass)", ha="center", fontsize=7.5, color="#5F6368")

plt.tight_layout()
plt.savefig(fig_dir / "overview.png", bbox_inches="tight")
plt.close()
print("Saved report/figures/overview.png")

# -------------------------------------------------------------
# 2. Qualitative Teaser / Comparison Figure
# -------------------------------------------------------------
img_root = Path("experiments/eval_istd20_v2")
test_a = Path("data/ISTD_Dataset/test/test_A")
test_b = Path("data/ISTD_Dataset/test/test_B")
test_c = Path("data/ISTD_Dataset/test/test_C")

def build_comparison_row(name):
    orig = load_image(test_a / name)
    gt_m = load_mask(test_b / name)
    clean = load_image(test_c / name)
    if clean.shape != orig.shape:
        clean = cv2.resize(clean, (orig.shape[1], orig.shape[0]))
    base = load_image(img_root / f"{Path(name).stem}__baseline__Otsu___global_gain.png")
    auto_rel = load_image(img_root / f"{Path(name).stem}__auto_mask__region_relight___blend__full_.png")
    scrib_rel = load_image(img_root / f"{Path(name).stem}__auto___simulated_scribbles__full.png")
    return [orig, base, auto_rel, scrib_rel, clean]

row1 = build_comparison_row("100-1.png")
row2 = build_comparison_row("100-4.png")

fig, axes = plt.subplots(2, 5, figsize=(11, 4.4), dpi=300)
titles = ["(a) Input Image", "(b) Baseline (Otsu+Global)", "(c) Ours (Automated)", "(d) Ours (+ Scribbles)", "(e) Ground Truth"]

for col in range(5):
    axes[0, col].imshow(row1[col])
    axes[0, col].set_title(titles[col], fontsize=8.5, fontweight="bold" if "Ours" in titles[col] else "normal")
    axes[0, col].axis("off")
    axes[1, col].imshow(row2[col])
    axes[1, col].axis("off")

plt.tight_layout()
plt.savefig(fig_dir / "teaser.png", bbox_inches="tight")
plt.close()
print("Saved report/figures/teaser.png")

# -------------------------------------------------------------
# 3. Failure Mode & Interactive Scribble Correction Figure (Scene 100-6)
# -------------------------------------------------------------
from src.detect import detect
from src.refine import refine, SCRIB_SHADOW
from src.remove import remove
from src.blend import blend

img_100_6 = load_image(test_a / "100-6.png")
gt_100_6 = load_mask(test_b / "100-6.png")
auto_m_6 = detect(img_100_6)

# Unrefined result (leaves dark un-relighted patch)
unrefined_relit = remove(img_100_6, auto_m_6)
unrefined_final = blend(img_100_6, unrefined_relit, auto_m_6)

# User scribble on the missed shadow area
missed_6 = (gt_100_6 > 0.5) & (auto_m_6 <= 0.5)
pts_6 = np.argwhere(missed_6)
center_6 = pts_6[len(pts_6) // 2]

h_6, w_6 = img_100_6.shape[:2]
scrib_6 = np.zeros((h_6, w_6), dtype=np.uint8)
cv2.circle(scrib_6, (int(center_6[1]), int(center_6[0])), 22, int(SCRIB_SHADOW), -1)

# Refined result
refined_m_6 = refine(img_100_6, auto_m_6, scrib_6)
refined_relit = remove(img_100_6, refined_m_6)
refined_final = blend(img_100_6, refined_relit, refined_m_6)

# Visualize user stroke
stroke_vis_6 = img_100_6.copy()
stroke_mask_6 = scrib_6 == SCRIB_SHADOW
stroke_vis_6[stroke_mask_6] = [255, 30, 30]
border_6 = cv2.dilate(stroke_mask_6.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) - stroke_mask_6.astype(np.uint8)
stroke_vis_6[border_6 > 0] = [255, 255, 255]

fig, axes = plt.subplots(1, 5, figsize=(12, 2.5), dpi=300)
panels_6 = [
    img_100_6,
    overlay_mask(img_100_6, auto_m_6, color=(255, 0, 0), alpha=0.5),
    unrefined_final,
    stroke_vis_6,
    refined_final
]
f_titles_6 = [
    "(a) Input Photo",
    "(b) Auto-Mask (Missed Patch)",
    "(c) Auto Result (Dark Patch Left)",
    "(d) User Stroke (Red = Shadow)",
    "(e) Rescued Final Result"
]

for i in range(5):
    axes[i].imshow(panels_6[i])
    axes[i].set_title(f_titles_6[i], fontsize=8, fontweight="bold")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig(fig_dir / "scribble_refine.png", bbox_inches="tight")
plt.close()
print("Saved report/figures/scribble_refine.png")
plt.close()
print("Saved report/figures/scribble_refine.png")
