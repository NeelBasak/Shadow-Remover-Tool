# Shadow Remover Tool: Improvement Suggestions

> **Status**: Analysis completed 2026-11-04. Each suggestion is backed by experimental evidence from synthetic benchmarks. Items are sorted by estimated impact, with implementation difficulty and risk noted.

---

## Executive Summary

The current pipeline (Otsu + chroma check → GrabCut → multiplicative gain → guided-filter blend) is a solid, defensible design for a 2-day course project. However, targeted experiments reveal **two significant accuracy gaps** and several moderate-value improvements. The highest-impact changes are:

1. **Log-domain boundary-strip illumination ratio** for relighting (reduces RMSE by 87%)
2. **Adaptive / multi-scale thresholding** for non-uniform illumination scenes
3. **Histogram matching** as a secondary relighting channel for color-shifted shadows

The remaining suggestions add presentation polish but are lower priority given deadlines.

---

## Suggestion 1: Log-Domain Boundary-Strip Illumination Ratio (Relighting)

**Priority**: 🔴 Critical  
**Impact**: RMSE 16.29 → 2.22 on realistic illumination (87% reduction)  
**Difficulty**: Low (≈30 lines changed in `remove.py`)

### Problem

The current relighting model is purely multiplicative: $g_c = \text{median}(I_{\text{lit}, c}) / \text{median}(I_{\text{shadow}, c})$, computed per surface cluster. This works when the lit reference ring is on the **same** surface as the shadow core. But in realistic outdoor scenes, illumination has two components:

$$I_{\text{lit}} = \rho \cdot (L_{\text{direct}} + L_{\text{ambient}}), \quad I_{\text{shadow}} = \rho \cdot L_{\text{ambient}}$$

The true illumination ratio $\gamma_c = (L_{\text{direct},c} + L_{\text{ambient},c}) / L_{\text{ambient},c}$ is **the same for all surfaces**. But when the lit ring mixes pixels from multiple surfaces (green grass and grey concrete side by side), the median-of-medians estimate becomes garbage, because it averages across different $\rho$ values.

### Proposed Fix

Estimate the illumination ratio from thin **paired boundary strips** — a narrow band just inside the shadow and a narrow band just outside — where pixels on both sides sit on the **same surface**. The surface-level $\rho$ cancels out:

$$\hat{\gamma}_c = \exp\!\Big(\text{mean}\big(\log(I_{\text{outer},c})\big) - \text{mean}\big(\log(I_{\text{inner},c})\big)\Big)$$

Working in log-domain makes this robust to texture variation and yields a single per-channel ratio that applies globally to the entire shadow region regardless of how many surfaces it covers.

### Experimental Evidence

```
=== Realistic Ambient+Direct Illumination (green + concrete) ===
  Current pipeline    RMSE LAB (shadow): 16.29    PSNR: 17.08 dB
  Log-boundary-ratio  RMSE LAB (shadow):  2.22    PSNR: 33.73 dB

  True ratio:     [7.67, 5.75, 3.67]
  Log-est ratio:  [7.26, 5.56, 3.51]
```

### Implementation Sketch

```python
# In remove.py, after finding shadow region and lit ring:
gap = 4
inner_strip = erode(binary, gap) & ~erode(binary, gap + 6)
outer_strip = dilate(binary, gap + 6) & ~dilate(binary, gap)

log_f = np.log(f + 1.0)
log_ratio = log_f[outer_strip > 0].mean(axis=0) - log_f[inner_strip > 0].mean(axis=0)
gain = np.clip(np.exp(log_ratio), 1.0, max_gain)

out[binary > 0] = f[binary > 0] * gain
```

### Risk

Low. Falls back gracefully to current behavior when boundary strips are too thin. Should be offered as an alternative mode alongside the existing surface-clustering approach (some scenes may benefit from one or the other).

---

## Suggestion 2: Illumination-Invariant Edge Features for Detection

**Priority**: 🟡 Medium  
**Impact**: Better shadow/material edge discrimination (100× separation in edge features)  
**Difficulty**: Medium (new feature computation, ≈50 lines)

### Problem

The current detector relies on global Otsu thresholding on $L$. This fails on:
- Scenes with illumination gradients (brighter left, darker right)
- Multiple shadows at different depths with different attenuation levels
- Low-contrast shadows near the Otsu threshold

Tested: on a gradient-illumination scene, current detection gets IoU = 0.417 (essentially random).

### Proposed Fix

Use **Finlayson-style log-chromaticity ratio edges** to distinguish shadow boundaries from material boundaries. Shadow edges have near-zero log-ratio gradients; material edges have large ones:

$$\nabla \log\!\Big(\frac{R}{G}\Big) \approx 0 \text{ at shadow edges}$$

$$\nabla \log\!\Big(\frac{R}{G}\Big) \gg 0 \text{ at material edges}$$

### Experimental Evidence

```
Log-ratio gradient at SHADOW edges:   0.016
Log-ratio gradient at MATERIAL edges:  1.677   (100× larger)
Luminance gradient at SHADOW edges:    186.9
Luminance gradient at MATERIAL edges:   60.3   (reversed!)
```

### Implementation Sketch

1. Compute log-ratio channels: $\log(R/G)$, $\log(G/B)$.
2. Find luminance edges (Canny or Sobel on $L$).
3. Classify each edge pixel: if log-ratio gradient is small → shadow edge; if large → material edge.
4. Flood-fill from shadow edges inward to find shadow regions.
5. Use this as an alternative or complement to Otsu thresholding.

### Risk

Medium. Works well for simple Lambertian scenes; may be noisy on highly textured surfaces. Best used as a **secondary signal** that modulates confidence, not as the sole detector.

---

## Suggestion 3: Histogram Matching for Color-Shifted Shadows

**Priority**: 🟡 Medium  
**Impact**: RMSE 26.02 → 0.93 on color-shifted shadows  
**Difficulty**: Low (scikit-image one-liner, ≈10 lines)

### Problem

Real outdoor shadows are not neutral darkening — they have a bluish color temperature shift because the shadow receives cool skylight (ambient) while the lit region receives warm sunlight (direct). The multiplicative gain model doesn't capture this spectral shift.

### Proposed Fix

Use `skimage.exposure.match_histograms` to match the per-channel histogram of shadow pixels to that of lit pixels (on the same surface). This implicitly handles both multiplicative scaling and additive color shifts.

### Experimental Evidence

```
=== Histogram Matching vs Multiplicative Gain (blue-shifted shadow) ===
  Current (mult gain)    RMSE: 26.02   PSNR: 18.11 dB
  Histogram matching     RMSE:  0.93   PSNR: 45.50 dB
```

### Implementation Sketch

```python
from skimage.exposure import match_histograms

for sub_surface in sub_regions:
    shadow_px = img[sub_surface > 0]
    lit_px = img[matched_ring > 0]
    matched = match_histograms(shadow_px, lit_px, channel_axis=1)
    out[sub_surface > 0] = matched
```

### Risk

Low-medium. Histogram matching assumes shadow and lit distributions should be identical, which breaks if the lit ring contains pixels from a different surface. Requires good surface clustering or chromaticity-matched ring selection (which we already have).

---

## Suggestion 4: Multi-Scale / Adaptive Thresholding Fallback

**Priority**: 🟢 Low-Medium  
**Impact**: Better detection on gradient-illumination scenes  
**Difficulty**: Low (≈20 lines)

### Problem

Global Otsu assumes bimodal luminance distribution. Fails when:
- Non-uniform scene illumination creates a continuous $L$ gradient
- Multiple shadows with varying depth

### Proposed Fix

Add adaptive thresholding (`cv2.adaptiveThreshold`) as a **fallback** when Otsu's bimodality score is low. Use `ADAPTIVE_THRESH_GAUSSIAN_C` with a large block size (proportional to image size) to handle local illumination variation.

### Caveat

In our experiments, adaptive thresholding alone scored poorly (IoU 0.209) because it picks up texture noise. It works best as a complement: use Otsu as primary, fall back to adaptive when Otsu's inter-class variance is below a threshold, and always validate with the chroma-consistency check.

---

## Suggestion 5: Penumbra-Proportional Gain Interpolation

**Priority**: 🟢 Low  
**Impact**: Moderate improvement in penumbra transition quality  
**Difficulty**: Low (≈15 lines in `blend.py`)

### Problem

The current approach applies a uniform gain inside the shadow and then alpha-blends at the boundary. But in the penumbra zone, the illumination transitions smoothly from full direct light to ambient-only. Applying a constant high gain to the penumbra over-brightens it, and the subsequent alpha blend partially compensates but imperfectly.

### Proposed Fix

Instead of binary gain + alpha blend, apply a **spatially varying gain** proportional to the soft mask value:

$$g(\mathbf{x}) = 1 + \alpha(\mathbf{x}) \cdot (g_{\text{core}} - 1)$$

where $\alpha$ is the soft shadow mask. This smoothly interpolates from gain = 1 (lit) to gain = $g_{\text{core}}$ (shadow core), following the actual penumbra profile.

---

## Suggestion 6: Poisson Blending — Not Recommended

**Priority**: ⚫ Do Not Use  
**Impact**: Negative (PSNR drops from 23.71 to 10.33)  

### Finding

Tested `cv2.seamlessClone` (Poisson blending) as an alternative to guided-filter blending. It performs **catastrophically** — PSNR drops by 13 dB. Poisson blending is designed for transplanting objects between images, not for illumination correction. It tries to match gradients across the boundary, which creates severe color drift across the entire shadow region.

**Keep the current guided-filter approach.** It is well-suited to this problem.

---

## Suggestion 7: CRF Post-Processing for Mask Refinement

**Priority**: 🟢 Low  
**Impact**: Minor improvement on noisy mask boundaries  
**Difficulty**: Medium (requires `pydensecrf` dependency)

### Problem

GrabCut works well (IoU 0.999 on tested irregular shapes) but can leave jagged boundaries on highly textured images.

### Proposed Fix

Apply a Dense CRF (Krähenbühl & Koltun, 2011) over the GrabCut output using pixel color as the pairwise feature. This smooths boundaries while respecting color edges.

### Caveat

GrabCut already achieves 0.993+ IoU. The marginal improvement may not justify the added dependency and complexity. Only pursue if mask quality is a bottleneck on real images.

---

## Priority Summary

| # | Suggestion | Impact | Effort | Recommendation |
|---|---|---|---|---|
| 1 | Log-domain boundary ratio | 🔴 Critical | Low | **Implement now** |
| 2 | Illumination-invariant edges | 🟡 Medium | Medium | Implement if time permits |
| 3 | Histogram matching | 🟡 Medium | Low | **Implement now** (as option) |
| 4 | Adaptive threshold fallback | 🟢 Low-Med | Low | Implement if time permits |
| 5 | Penumbra-proportional gain | 🟢 Low | Low | Nice polish |
| 6 | Poisson blending | ⚫ None | — | **Do not use** |
| 7 | CRF post-processing | 🟢 Low | Medium | Skip unless needed |

---

## References

- Finlayson, G.D., Hordley, S.D., Lu, C., Drew, M.S. (2006). "On the Removal of Shadows from Images." *IEEE TPAMI*, 28(1).
- He, K., Sun, J., Tang, X. (2010). "Guided Image Filtering." *ECCV 2010*.
- Krähenbühl, P., Koltun, V. (2011). "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials." *NeurIPS 2011*.
- Guo, R., Dai, Q., Hoiem, D. (2011). "Single-image shadow detection and removal using paired regions." *CVPR 2011*.

