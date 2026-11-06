# Shadow Remover Tool: Design Document

EE604 Course Project (group of 5). Write-up due **Nov 6, 2026**; presentation and demo **Nov 7-8, 2026**.

> Status of this document: design v3.1, updated 2026-11-05 following outlier-resistant median daylight detection, deep blue rejection (-25 <= Δb <= 2.5), component-adaptive blue skylight compensation, and verified ISTD benchmarks.

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
2. **Surface-aware region-matched relighting in linear radiometric space**: paired log-domain boundary ratios and chromaticity clustering match each shadow region exclusively to adjacent lit pixels of the same surface, cancelling out surface reflectance $\rho$ in linearized space ($I = \rho \cdot L$) and preventing multi-surface color bleeding.
3. **Additive color temperature compensation**: compensates for the ambient Rayleigh scattering (blue skylight) in CIE $L^*a^*b^*$ space that multiplicative scaling alone cannot remove.
4. **Edge-guided penumbra blending**: Guided Filter (He et al., ECCV 2010) aligns the illumination transition to actual scene geometry, eliminating halos.

---

## 3. Pipeline Architecture

```
                 +-------------------+
                 |    Input Image    |
                 +---------+---------+
                           |
            +--------------+--------------+
            |                             |
            v                             v
   [Module A: Detect]             [User Scribbles] (optional)
   Hierarchical Otsu on L               |
   + daylight physics check             |
   (|Δa| <= 3.5, Δb_pos <= 2.5)         |
            |                             |
            +--------------+--------------+
                           |
                           v
                 [Module B: Refine]
                 Interactive GrabCut
                 with hard user constraints
                           |
                           v  (refined soft mask)
                 +---------+---------+
                 |                   |
                 v                   |
        [Module C: Relight]          |
        Linear Radiometric RGB       |
        + paired log-boundary ratio  |
        + adaptive global anchor     |
        + blue skylight compensation |
                 |                   |
                 +---------+---------+
                           |
                           v
                  [Module D: Blend]
                  Penumbra smoothing
                  + Guided Filter edge alignment
                           |
                           v
                 +-------------------+
                 | Shadow-Free Image |
                 +-------------------+
```

---

## 4. Module Specifications

### Module A: Detection (`src/detect.py`)
- CIE LAB color space conversion.
- Gaussian smoothing ($\sigma = 1.5$) + Otsu thresholding on $L$ channel.
- Hierarchical re-thresholding if dark class fraction exceeds 50%.
- Outlier-resistant median daylight chromaticity consistency validation:
  - Green-Red axis invariance: $|\Delta a| \le 3.5$.
  - Rayleigh blue skylight bounds: $-25.0 \le \Delta b \le 2.5$ (cleanly rejecting pigmented blue fabrics and paint).
  - Luminance drop validation: $L_{\text{out}} / L_{\text{in}} \ge 1.15$.
  - All metrics computed using medians over component and surrounding lit ring.
- Post-validation morphological cleanup (postponed closing prevents occluder bridging).

### Module B: Refinement (`src/refine.py`)
- User scribbles encoded as a 3-state map: 0 = untouched, 1 = definite shadow, 2 = definite lit.
- Automatic mask initializes GrabCut Gaussian Mixture Models.
- User scribbles enforced as hard boundary constraints (`GC_FGD`, `GC_BGD`).
- Fast interactive runtime: 3 GrabCut iterations on CPU ($\sim 0.3\text{ s}$).

### Module C: Relighting (`src/remove.py`)
- Radiometric linearization: sRGB $\to$ linear RGB via IEC 61966-2-1 transfer function.
- Per-component paired boundary extraction (inner strip inside shadow, outer strip in adjacent lit band).
- Multi-surface clustering: K-means with spatial coherence and chromatic separation check ($\|\mathbf{c}_0 - \mathbf{c}_1\|_2 \ge 0.15$).
- Illumination ratio estimation in log space:
  $$\log(\gamma_c) = \text{median}(\log I_{\text{outer}}) - \text{median}(\log I_{\text{inner}})$$
  $$\text{gain}_c = \text{clip}(\exp(\log \gamma_c), 1.0, 8.0)$$
- Adaptive anchor: smooth interpolation towards global gain when local boundary pixels are sparse.
- Component-adaptive color temperature compensation on CIE LAB $b^*$ channel: local $\Delta b^*_i$ per component with global anchor fallback for ambient blue skylight neutralization.

### Module D: Penumbra Blending (`src/blend.py`)
- Transition band centered directly on the physical shadow boundary.
- Alpha matte refined using Grayscale Guided Filter (He et al., ECCV 2010):
  $$\alpha_{\text{guided}} = \text{GuidedFilter}(I_{\text{gray}}, \alpha_{\text{blurred}}, r = 2 \cdot r_{\text{penumbra}}, \epsilon = 10^{-3})$$
- Seamless convex combination: $I_{\text{out}} = \alpha_{\text{final}} \cdot I_{\text{relit}} + (1 - \alpha_{\text{final}}) \cdot I_{\text{in}}$.

### Module E: Demo Application (`app.py`)
- Interactive Streamlit web app with preset galleries, hyperparameter sliders, scribble bounding boxes, and quantitative metric benchmarking.

---

## 5. Experimental Verification & Benchmarks

### 5.1 Component Unit Tests (`experiments/test_components.py`)

| Component | Test Scenario | Baseline Result | Proposed Pipeline | Evaluation Metric |
|---|---|---|---|---|
| A. Detection | Dark Object Rejection | IoU: 0.500 \| FP: 100% | **IoU: 0.997 \| FP: 0.0%** | Zero false positives on dark objects |
| B. Refinement | User Scribble Correction | IoU: 0.500 \| BER: 6.2% | **IoU: 0.993 \| BER: 0.4%** | 93% reduction in segmentation error |
| C. Relighting | Multi-Surface Preservation | LAB RMSE: 12.95 (Cast) | **LAB RMSE: 10.77 (Clean)** | Surface-consistent color recovery |
| D. Blending | Penumbra Edge Smoothing | PSNR: 22.98 dB | **PSNR: 23.70 dB** | Artifact-free boundary transition |
| E. Pipeline | End-to-End Verification | Baseline PSNR: 11.2 dB | **Pipeline PSNR: 15.33 dB** | Cohesive component integration |

### 5.2 ISTD Benchmark Dataset Evaluation (`results_v3`)

| Method / Configuration | Shadow RMSE ↓ | All RMSE ↓ | PSNR ↑ | SSIM ↑ | Mask IoU ↑ | Mask BER ↓ |
|---|---|---|---|---|---|---|
| Baseline (Otsu + Global Gain) | 6.194 | 6.047 | 20.77 dB | 0.757 | 0.646 | 8.93% |
| Auto Mask: Global Gain + Blend | 5.182 | 4.090 | 24.72 dB | 0.902 | 0.782 | 4.52% |
| Auto Mask: Region Relight, No Blend | 5.728 | 4.555 | 23.52 dB | 0.890 | 0.782 | 4.52% |
| **Auto Mask: Region Relight + Blend (Full Ours)** | **5.136** | **4.132** | **24.53 dB** | **0.902** | **0.782** | **4.52%** |
| **Auto + Simulated Scribbles: Full** | **4.509** | **3.999** | **24.95 dB** | **0.905** | **0.811** | **2.80%** |
| GT Mask: Global Gain + Blend | 4.515 | 3.111 | 28.18 dB | 0.914 | 1.000 | 0.00% |
| **GT Mask: Region Relight + Blend (Upper Bound)** | **4.278** | **3.007** | **28.34 dB** | **0.915** | **1.000** | **0.00%** |

### 5.3 Paired Statistical Comparisons ($N=20$, 95% Bootstrap CI)

| Comparison Hypothesis | Metric | Mean Improvement | 95% Bootstrap CI | Win Rate |
|---|---|---|---|---|
| **Full Pipeline vs Baseline** | PSNR (dB) | **+3.77 dB** | **[+2.11, +6.01]** | **95.0%** |
| Full Pipeline vs Baseline | Shadow RMSE | **+1.06** | **[+0.20, +2.19]** | **65.0%** |
| **Region Relight vs Global Gain (GT Mask)** | PSNR (dB) | **+0.15 dB** | **[-0.08, +0.40]** | **55.0%** |
| Region Relight vs Global Gain (GT Mask) | Shadow RMSE | **+0.24** | **[-0.24, +0.78]** | **60.0%** |
| **Penumbra Blending vs No Blending** | PSNR (dB) | **+1.01 dB** | **[+0.79, +1.22]** | **100.0%** |
| **Guided Filter vs Standard Blurring** | PSNR (dB) | **+0.16 dB** | **[+0.11, +0.22]** | **95.0%** |
| **Interactive Scribbles vs Auto Detector** | Shadow RMSE | **+0.63** | **[+0.07, +1.33]** | **55.0%** |
