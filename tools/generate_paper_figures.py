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
# 3. Failure Mode & Interactive Scribble Correction Figure
# -------------------------------------------------------------
orig_102 = load_image(test_a / "102-1.png")
gt_m_102 = load_mask(test_b / "102-1.png")
clean_102 = load_image(test_c / "102-1.png")
if clean_102.shape != orig_102.shape:
    clean_102 = cv2.resize(clean_102, (orig_102.shape[1], orig_102.shape[0]))

from src.detect import detect
from src.refine import refine, SCRIB_LIT
from src.remove import remove
from src.blend import blend

auto_m_102 = detect(orig_102)

h, w = orig_102.shape[:2]
scrib_102 = np.zeros((h, w), dtype=np.uint8)
cv2.circle(scrib_102, (int(w * 0.75), int(h * 0.8)), 25, int(SCRIB_LIT), -1)
refined_m_102 = refine(orig_102, auto_m_102, scrib_102)

res_auto_102 = blend(orig_102, remove(orig_102, auto_m_102), auto_m_102)
res_scrib_102 = blend(orig_102, remove(orig_102, refined_m_102), refined_m_102)

stroke_vis = orig_102.copy()
stroke_vis[scrib_102 == SCRIB_LIT] = [0, 255, 0]

fig, axes = plt.subplots(1, 5, figsize=(11, 2.3), dpi=300)
f_titles = [
    "(a) Input (Scene 102)",
    "(b) Auto Mask (FP on Road)",
    "(c) User Scribble (Green)",
    "(d) Refined Mask",
    "(e) Final Shadow-Free"
]

axes[0].imshow(orig_102)
axes[1].imshow(overlay_mask(orig_102, auto_m_102, color=(255, 0, 0), alpha=0.5))
axes[2].imshow(stroke_vis)
axes[3].imshow(overlay_mask(orig_102, refined_m_102, color=(0, 255, 255), alpha=0.5))
axes[4].imshow(res_scrib_102)

for i in range(5):
    axes[i].set_title(f_titles[i], fontsize=8)
    axes[i].axis("off")

plt.tight_layout()
plt.savefig(fig_dir / "scribble_refine.png", bbox_inches="tight")
plt.close()
print("Saved report/figures/scribble_refine.png")
