# Shadow Remover Tool: Design Document

EE604 Course Project (group of 5). Write-up due **Nov 6, 2026**; presentation and demo **Nov 7-8, 2026**.

> Status of this document: design v2, updated 2026-11-04 following component audit, accuracy overhaul, and benchmark verification.

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
2. **Surface-aware region-matched relighting**: multi-surface shadows are dynamically partitioned by chromaticity, and each sub-surface is matched exclusively to lit reference pixels of the same surface, preventing severe cross-surface color casts.
3. **Penumbra-aware edge-guided blending**: edge-preserving guided filtering with penumbra dilation ensures that soft shadow boundaries are relit smoothly without leaving dark or bright halos.

---

## 3. System Overview

```
Input image
    |
    v
[A] Shadow detection (Otsu + Chroma check) ---> initial soft mask M0
    |
    v
[B] Interactive refinement (user scribbles, optional GrabCut) ---> refined mask M
    |
    v
[C] Surface-aware region-matched relighting ---> relit image R
    |
    v
[D] Penumbra-aware guided blending (uses soft M) ---> final shadow-free image
```

---

## 4. Module Design

### A. Shadow Detection (`detect.py`)
- Convert to CIE LAB; shadows exhibit low $L$ while retaining similar chromaticity to the surface they fall on.
- Candidate mask obtained by Otsu thresholding on Gaussian-blurred $L$ channel ($L_{8} = \text{clip}(L \cdot 255 / 100)$).
- **Chroma-Consistency Verification**: To reject dark-but-not-shadow objects (which have low $L$ but belong to different materials), each connected candidate component is verified against an adjacent lit boundary ring in normalized RGB ($r = R / \sum, g = G / \sum$):
  - Euclidean chromaticity distance: $\Delta c = \|\mathbf{r}_{\text{core}} - \mathbf{r}_{\text{ring}}\|_2 \le \tau_{\text{chroma}}$ (default 0.035).
  - Relative luminance drop: $L_{\text{ring}} / L_{\text{core}} \ge \tau_{\text{ratio}}$ (default 1.15).
- Morphological opening and closing with ellipse structuring elements, removal of small noise components ($< 0.2\%$ image area), filling holes, and Gaussian softening.
- **Baseline for comparison**: Plain Otsu on $L$ (`detect_otsu`).

### B. Interactive Refinement (`refine.py`)
- Input: image, $M_0$, scribbles (0 = untouched, 1 = definite shadow, 2 = definite lit).
- Initialises `cv2.grabCut` with probabilistic foreground/background from $M_0$, overridden by user scribbles as hard constraints (`GC_FGD` / `GC_BGD`).
- Handles empty/edge-case initial masks by synthesizing fallback background/foreground priors.
- Preserves user hard scribbles explicitly after boundary Gaussian smoothing.
- Output: refined soft mask $M$.

### C. Relighting (`remove.py`)
- Splits mask $M$ into connected shadow regions.
- For each region, samples a lit reference ring outside an exclusion zone ($\text{gap}$ pixels away from the boundary) to avoid penumbra contamination.
- **Surface-Aware Partitioning**: If a single connected shadow spans multiple surfaces (high chromaticity variance), $K$-means clustering ($K=2$) splits the shadow into sub-surfaces.
- Each sub-surface is matched only against lit reference ring pixels of matching chromaticity ($\Delta c < 0.10$).
- Computes per-channel gain $g_c = \text{median}(I_{\text{lit}, c}) / \max(\text{median}(I_{\text{shadow}, c}), 1.0)$, clipped to $[1.0, 4.0]$.
- **Baseline for comparison**: Single global gain correction (`remove_global`).

### D. Penumbra-Aware Blending (`blend.py`)
- Derives alpha transition matte from the mask: dilates by penumbra radius $k = 2 \cdot \text{penumbra} + 1$, blurs with Gaussian filter, then snaps to scene edges using a Grayscale Guided Filter (He et al., ECCV 2010) with the input image as guide.
- Convex combination: $\mathbf{I}_{\text{out}} = \alpha \cdot \mathbf{I}_{\text{relit}} + (1 - \alpha) \cdot \mathbf{I}_{\text{original}}$.
- Ablations: with vs. without guided filter; with vs. without blending.

### E. Interface / Demo (`app.py`)
- Gradio web interface supporting:
  - Image upload with sample presets.
  - Automatic mask preview.
  - Interactive drawing canvas for shadow / lit scribbles.
  - Real-time mask refinement and relighting.
  - Side-by-side comparison and result export.

---

## 5. Interfaces

```python
detect(img: np.ndarray) -> np.ndarray              # float32 soft mask in [0,1], HxW
refine(img, mask, scribbles) -> np.ndarray         # float32 soft mask in [0,1], HxW
remove(img, mask) -> np.ndarray                    # relit image, uint8 HxWx3
blend(img, relit, mask) -> np.ndarray              # final image, uint8 HxWx3
pipeline.run(img, mask=None, scribbles=None) -> (final, mask, relit)
```

- Baselines: `detect_otsu` ([`detect.py`](file:///c:/Users/darsh/OneDrive/Desktop/Acads/4th%20YEAR/Sem%207/EE604/Course%20Project/Shadow-Remover-Tool/src/detect.py)) and `remove_global` ([`remove.py`](file:///c:/Users/darsh/OneDrive/Desktop/Acads/4th%20YEAR/Sem%207/EE604/Course%20Project/Shadow-Remover-Tool/src/remove.py)).
- Images are RGB `uint8` $H \times W \times 3$; masks are `float32` $H \times W \in [0, 1]$.

---

## 6. Repository Layout

```
Shadow-Remover-Tool/
├── README.md
├── DESIGN.md
├── PROGRESS.md
├── requirements.txt
├── src/
│   ├── detect.py
│   ├── refine.py
│   ├── remove.py
│   ├── blend.py
│   ├── pipeline.py
│   └── utils.py
├── app.py
├── experiments/
│   ├── metrics.py
│   ├── test_components.py
│   ├── run_eval.py
│   └── results/
└── report/
```

---

## 7. Experiments & Verification

| Module / Item | Metric | Baseline Result | Improved Result |
|---|---|---|---|
| A. Detection (dark object rejection) | IoU / Dark FP % | IoU 0.500 / 100.0% FP | **IoU 0.994 / 0.0% FP** |
| B. Refinement (scribble correction) | IoU / BER % | IoU 0.500 / 6.2% BER | **IoU 0.993 / 0.4% BER** |
| C. Relighting (multi-surface) | RMSE in LAB | 15.78 (color cast) | **3.92 (clean match)** |
| D. Blending (penumbra smoothing) | PSNR / SSIM | 23.27 dB / 0.8078 | **26.30 dB / 0.8178** |

---

## 8. Deliverables Mapping

| Deliverable | Weight | Contents |
|---|---|---|
| Write-up (CVPR template) | 5% | Abstract, intro with novelty, method + overview figure, experiments, appendix with full code + GitHub link, dataset/repo links, per-member contributions |
| Presentation | 5% | Problem (I/O, constraints), literature review, method with reasons, experiments |
| Code / Demo | 5% | Clean commented code, good UI, strong examples, everyone can answer implementation questions |

---

## 9. Team Roles (suggested; **[DECIDE]** names)

| # | Owner | Responsibility |
|---|---|---|
| 1 | _name_ | Detection module (A) |
| 2 | _name_ | Relighting + penumbra blending (C, D) |
| 3 | _name_ | Interface + scribble refinement (B, E) |
| 4 | _name_ | Datasets, metrics, experiments, ablations |
| 5 | _name_ | Literature review, overview figure, LaTeX, slides |
