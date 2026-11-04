# Shadow Remover Tool: Design Document

EE604 Course Project (group of 5). Write-up due **Nov 6, 2026**; presentation and demo **Nov 7-8, 2026**.

> Status of this document: design v2.1, updated 2026-11-04 following component overhaul, ISTD dataset evaluation, and interactive demo completion.

---

## 1. Problem Statement

- **Input**: an RGB image containing cast shadows (plus, optionally, user scribbles marking shadow / non-shadow regions).
- **Output**: (a) a binary/soft shadow mask, (b) a shadow-free image with colour and texture preserved and no halos at shadow edges.
- **Constraints**:
  - Single image, no multi-view or flash/no-flash pairs.
  - Must run on a laptop CPU in a few seconds per image (for live demo).
  - Built in about 2 days, so classical methods plus light learned components only.
  - Existing code may be reused, but there must be a defensible novelty.

## 2. Proposed Novelty (to defend in the presentation)

1. **Interactive scribble-guided mask refinement**: automatic detection gives an initial mask; user scribbles (shadow / lit) correct it via GrabCut-style refinement. Most classical removal pipelines are fully automatic and fail silently on bad masks.
2. **Surface-aware region-matched relighting**: paired log-domain boundary ratios and chromaticity clustering match each shadow region exclusively to adjacent lit pixels of the same surface, cancelling out surface reflectance $\rho$ and preventing multi-surface color bleeding.
3. **Penumbra-aware edge-guided blending**: edge-preserving grayscale guided filtering with penumbra dilation ensures that soft shadow boundaries are relit smoothly without dark/bright halos.

---

## 3. System Overview

```
Input image
    |
    v
[A] Shadow detection (Otsu + Chroma check + Hierarchical split) ---> initial soft mask M0
    |
    v
[B] Interactive refinement (user scribbles / GrabCut) ---> refined mask M
    |
    v
[C] Surface-aware log-boundary relighting ---> relit image R
    |
    v
[D] Penumbra-aware guided blending (uses soft M) ---> final shadow-free image
```

---

## 4. Module Design

### A. Shadow Detection (`detect.py`)
- Convert to CIE LAB; shadows exhibit low $L$ while retaining similar chromaticity to the surface they fall on.
- Candidate mask obtained by Otsu thresholding on Gaussian-blurred $L$ channel.
- **Hierarchical Otsu**: If the initial split labels $> 50\%$ of the image as shadow (indicating broad material separation rather than lighting), re-threshold inside the dark class to extract the true shadow core.
- **Chroma-Consistency Verification**: To reject dark-but-not-shadow objects, candidate components are verified against an adjacent lit boundary ring in normalized RGB ($r = R / \sum, g = G / \sum$):
  - Euclidean chromaticity distance: $\Delta c = \|\mathbf{r}_{\text{core}} - \mathbf{r}_{\text{ring}}\|_2 \le \tau_{\text{chroma}}$ (default 0.035).
  - Relative luminance drop: $L_{\text{ring}} / L_{\text{core}} \ge \tau_{\text{ratio}}$ (default 1.15).
- Morphological opening and closing with ellipse structuring elements, removal of small noise components ($< 0.2\%$ image area), filling holes, and Gaussian softening.
- **Baseline for comparison**: Plain Otsu on $L$ (`detect_otsu`).

### B. Interactive Refinement (`refine.py`)
- Input: image, $M_0$, scribbles (0 = untouched, 1 = definite shadow, 2 = definite lit).
- Initializes `cv2.grabCut` with probabilistic foreground/background from $M_0$, overridden by user scribbles as hard constraints (`GC_FGD` / `GC_BGD`).
- Handles empty/edge-case initial masks by synthesizing fallback background/foreground priors.
- Preserves user hard scribbles explicitly after boundary Gaussian smoothing.
- Output: refined soft mask $M$.

### C. Relighting (`remove.py`)
- Splits mask $M$ into connected shadow regions.
- For each region, samples paired inner and outer boundary strips along the contour:
  - Material reflectance $\rho$ cancels out in the log-domain illumination ratio:
    $$\log(\gamma_c) = \text{median}(\log I_{\text{outer}, c}) - \text{median}(\log I_{\text{inner}, c})$$
  - Per-channel gain: $g_c = \text{clip}(\exp(\log(\gamma_c)), 1.0, \text{max\_gain})$.
- **Surface-Aware Partitioning**: If a single shadow spans multiple surface types, $K$-means clustering ($K=2$) partitions the region, matching each sub-surface to lit reference pixels of the same chromaticity profile ($\Delta c < 0.12$).
- **Baseline for comparison**: Single global gain correction (`remove_global`).

### D. Penumbra-Aware Blending (`blend.py`)
- Derives alpha transition matte from the mask: dilates by penumbra radius $k = 2 \cdot \text{penumbra} + 1$, blurs with Gaussian filter, then snaps to scene edges using a Grayscale Guided Filter (He et al., ECCV 2010) with the input image as guide.
- Convex combination: $\mathbf{I}_{\text{out}} = \alpha \cdot \mathbf{I}_{\text{relit}} + (1 - \alpha) \cdot \mathbf{I}_{\text{original}}$.
- Ablations: with vs. without guided filter; with vs. without blending.

### E. Interface / Demo (`app.py`)
- Streamlit interactive web application supporting:
  - Preset gallery (dark object rejection, multi-surface shadow, soft penumbra, specular reflection failure case).
  - Image upload with live parameter tuning.
  - Interactive GrabCut scribble/box refinement.
  - Side-by-side comparison (Input, Mask, Relit, Final).
  - Quantitative metric calculator (PSNR, SSIM, LAB RMSE, IoU, BER) when Ground Truth is provided.
  - One-click PNG exports of shadow-free output and predicted mask.

---

## 5. Interfaces

```python
detect(img: np.ndarray) -> np.ndarray              # float32 soft mask in [0,1], HxW
refine(img, mask, scribbles) -> np.ndarray         # float32 soft mask in [0,1], HxW
remove(img, mask) -> np.ndarray                    # relit image, uint8 HxWx3
blend(img, relit, mask) -> np.ndarray              # final image, uint8 HxWx3
pipeline.run(img, mask=None, scribbles=None) -> (final, mask, relit)
```

---

## 6. Repository Layout

```
Shadow-Remover-Tool/
├── README.md                 # Project documentation and CLI/app instructions
├── DESIGN.md                 # System architecture, formulas & experiment tables
├── PROGRESS.md               # Task tracker and contribution log
├── SUGGESTIONS.md            # In-depth architectural analysis and recommendations
├── requirements.txt          # Python dependencies
├── src/
│   ├── detect.py             # Module A: shadow detection (Otsu + chroma filtering)
│   ├── refine.py             # Module B: GrabCut interactive scribble refinement
│   ├── remove.py             # Module C: surface-aware log-boundary relighting
│   ├── blend.py              # Module D: guided-filter penumbra blending
│   ├── pipeline.py           # End-to-end pipeline wrapper & CLI
│   └── utils.py              # Guided filter, color conversion, I/O helpers
├── experiments/
│   ├── metrics.py            # Evaluation metrics (RMSE in LAB, PSNR, SSIM, IoU, BER)
│   ├── test_components.py    # Synthetic component audit & benchmark test suite
│   ├── run_eval.py           # Comprehensive evaluation script with paired bootstrap CIs
│   ├── sweep_detect.py       # Parameter sweep script for detector thresholds
│   └── results_v2/           # 100-image benchmark results (results.csv, paired.csv, per_image.csv)
├── data/
│   └── demo_samples/         # Preset test scenes for the live demo
├── app.py                    # Streamlit interactive web application
└── try_it.py                 # Quick visual strip generator
```

---

## 7. Experiments & Quantitative Evaluation

### 7.1 Synthetic Component Audit Suite (`experiments/test_components.py`)

| Component | Test Case | Baseline Result | Proposed Pipeline | Improvement |
|---|---|---|---|---|
| A. Detection | Dark Object Rejection | IoU: 0.500 \| FP: 100.0% | **IoU: 0.994 \| FP: 0.0%** | Zero false positives on dark objects |
| B. Refinement | User Scribble Correction | IoU: 0.500 \| BER: 6.2% | **IoU: 0.993 \| BER: 0.4%** | 93% reduction in error |
| C. Relighting | Multi-Surface Preservation | LAB RMSE: 15.78 (Magenta cast) | **LAB RMSE: 3.99 (Clean match)** | 75% error reduction |
| D. Blending | Penumbra Boundary Smoothing | PSNR: 21.65 dB \| SSIM: 0.8045 | **PSNR: 23.16 dB \| SSIM: 0.8161** | Smooth halo-free edge transition |

### 7.2 Benchmark Dataset Evaluation (ISTD Test Set — 100 Images, `experiments/results_v2/`)

| Method / Configuration | RMSE (Shadow Region) ↓ | RMSE (Full Image) ↓ | PSNR ↑ | SSIM ↑ | Mask IoU ↑ | Mask BER % ↓ |
|---|---|---|---|---|---|---|
| Baseline: Otsu + Global Gain | 8.01 ± 3.27 | 7.90 ± 3.31 | 18.53 ± 3.99 | 0.691 ± 0.151 | 0.572 | 9.95% |
| Auto Mask: Global Gain | 7.92 ± 3.16 | 5.85 ± 2.69 | 21.52 ± 4.67 | 0.881 ± 0.076 | 0.693 | 5.75% |
| Auto Mask: Region Relight, No Blend | 7.42 ± 2.75 | 5.61 ± 2.63 | 21.93 ± 4.75 | 0.885 ± 0.075 | 0.693 | 5.75% |
| Auto Mask: Region Relight + Blend (w/o Guided) | 7.17 ± 2.84 | 5.41 ± 2.56 | 22.28 ± 4.81 | 0.890 ± 0.074 | 0.693 | 5.75% |
| **Auto Mask: Region Relight + Blend (Full Ours)** | **7.05 ± 2.87** | **5.29 ± 2.52** | **22.51 ± 4.83** | **0.892 ± 0.073** | **0.693** | **5.75%** |
| **Auto + Simulated Scribbles: Full** | **6.80 ± 3.02** | **4.73 ± 2.27** | **23.43 ± 4.68** | **0.905 ± 0.072** | **0.793** | **2.84%** |
| GT Mask: Region Relight + Blend (Upper Bound) | 6.33 ± 2.46 | 3.86 ± 2.44 | 25.80 ± 5.36 | 0.916 ± 0.069 | 1.000 | 0.00% |

### 7.3 Paired Statistical Comparisons (95% Bootstrap Confidence Interval, $N=100$)

| Comparison Hypothesis | Metric | Mean Improvement | 95% Bootstrap CI | Win Rate |
|---|---|---|---|---|
| **Full Pipeline vs Baseline** | PSNR (dB) | **+3.98 dB** | **[+3.19, +4.84]** | **94.0%** |
| Full Pipeline vs Baseline | Shadow RMSE | **-0.96** | **[-1.41, -0.56]** | **75.0%** |
| **Region Relight vs Global Gain** | Shadow RMSE | **-0.57** | **[-0.82, -0.33]** | **59.0%** |
| **Guided Filter vs No Guided** | PSNR (dB) | **+0.23 dB** | **[+0.20, +0.26]** | **100.0%** |
| **Blend vs No Blend** | PSNR (dB) | **+0.59 dB** | **[+0.52, +0.66]** | **100.0%** |
| **Scribble Refinement vs Auto Alone** | PSNR (dB) | **+0.92 dB** | **[+0.63, +1.23]** | **69.0%** |

---

## 8. Deliverables Mapping

| Deliverable | Weight | Status | Contents |
|---|---|---|---|
| Write-up (CVPR template) | 5% | Pending | Abstract, intro with novelty, method + overview figure, benchmark tables 7.2 & 7.3, appendix with GitHub link |
| Presentation & Demo | 5% | Ready | Slides, live demo on Streamlit (`app.py`), preset demo scenes (`data/demo_samples/`) |
| Code & Implementation | 5% | Complete | Clean modular Python code (`src/`), verification suite (`experiments/test_components.py`), full eval suite (`experiments/run_eval.py`) |

---

## 9. Team Roles (suggested; **[DECIDE]** names)

| # | Owner | Responsibility | Status |
|---|---|---|---|
| 1 | _name_ | Detection module (A) | Complete |
| 2 | _name_ | Relighting + penumbra blending (C, D) | Complete |
| 3 | _name_ | Interface + scribble refinement (B, E) | Complete (`app.py`) |
| 4 | _name_ | Datasets, metrics, experiments, ablations | Complete (ISTD 100 images) |
| 5 | _name_ | Literature review, overview figure, LaTeX, slides | In progress |
