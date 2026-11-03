# Shadow Remover Tool: Design Document

EE604 Course Project (group of 5). Write-up due **Nov 6, 2026**; presentation and demo **Nov 7-8, 2026**.

> Status of this document: design v1, updated 2026-11-03 to match the starter implementation. Items marked **[DECIDE]** need a group decision or instructor confirmation.

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
2. **Penumbra-aware removal**: soft mask plus per-pixel blending (guided filter / Poisson-style) so the soft edge of the shadow is relit smoothly rather than leaving dark or bright halos.
3. **Region-matched relighting**: per-channel gain/offset estimated against the adjacent lit region of the same surface (not global statistics), reducing colour shifts.

**[DECIDE]** Confirm with the instructor that 1 + 2 together are enough novelty, and whether a pretrained deep model is allowed as a baseline (we plan to use it for comparison only).

## 3. System Overview

```
Input image
    |
    v
[A] Shadow detection  --->  initial soft mask M0
    |
    v
[B] Interactive refinement (user scribbles, optional)  --->  refined mask M
    |
    v
[C] Region-matched relighting  --->  relit image R
    |
    v
[D] Penumbra-aware blending (uses soft M)  --->  final shadow-free image
```

The overview figure for the report should be drawn from this diagram.

## 4. Module Design

### A. Shadow Detection (`detect.py`)
- Convert to LAB (and HSV); shadows have low L, with small change in chroma ratios.
- Candidate mask from thresholding on L (Otsu / adaptive) combined with a chroma-consistency check to reject dark-but-not-shadow objects.
- Otsu runs on the Gaussian-blurred L channel. If the lit and shadow classes differ by less than a set L separation, the image is treated as shadow-free.
- Morphological open/close, remove small components, fill small holes, then soften the binary mask with a Gaussian to get the soft mask.
- Planned improvement (Person 1): chroma-consistency check to reject dark objects.
- **Baseline for comparison**: plain Otsu on L; optional pretrained detector.

### B. Interactive Refinement (`refine.py`)
- Input: image, M0, scribbles (shadow = 1, lit = 0).
- Scribble map: uint8 HxW, 0 = untouched, 1 = shadow, 2 = not shadow.
- Use `cv2.grabCut` with mask initialisation (definite fg/bg from scribbles, probable fg/bg from M0).
- Output refined mask M.

### C. Relighting (`remove.py`)
- Split M into connected shadow regions.
- For each region, take a thin band of lit pixels just outside the shadow boundary (same surface) as reference.
- Skip a small gap next to the boundary (penumbra) when sampling the shadow core and the lit reference band.
- Estimate a per-channel multiplicative RGB gain (clipped) matching the shadow core mean to the lit band mean, and apply it over the region plus a small margin.
- **Baseline for comparison**: single global gain correction.

### D. Penumbra-Aware Blending (`blend.py`)
- Alpha comes from the soft mask: dilate slightly, Gaussian blur, then guided filter (grayscale image as guide) so the transition follows real edges.
- `out = alpha * relit + (1 - alpha) * original`.
- Optional (if time): Poisson blending at the boundary.
- Ablation: with vs. without this step.

### E. Interface / Demo (`app.py`)
- Gradio or Streamlit (**[DECIDE]**; Gradio is fastest for scribble input).
- Features: upload image, show auto mask, draw scribbles, re-run refinement, side-by-side before/after, download result.
- Preloaded example images chosen to show strengths and one honest failure case.

## 5. Interfaces (agree up front so people can work in parallel)

```python
detect(img: np.ndarray) -> np.ndarray              # float32 soft mask in [0,1], HxW
refine(img, mask, scribbles) -> np.ndarray         # float32 soft mask in [0,1], HxW
remove(img, mask) -> np.ndarray                    # relit image, uint8 HxWx3
blend(img, relit, mask) -> np.ndarray              # final image, uint8 HxWx3
pipeline.run(img, mask=None, scribbles=None) -> (final, mask, relit)
```
- Baselines live next to the modules: `detect_otsu` (detect.py) and `remove_global` (remove.py).
- Images are RGB `uint8` HxWx3 unless stated; masks are `float32` HxW.

## 6. Repository Layout

```
shadow-remover/
├── README.md
├── DESIGN.md
├── PROGRESS.md
├── requirements.txt
├── src/
│   ├── detect.py
│   ├── refine.py
│   ├── remove.py
│   ├── blend.py
│   ├── pipeline.py        # end-to-end run() + CLI
│   └── utils.py           # I/O, LAB conversion, guided filter
├── app.py                 # demo UI
├── experiments/
│   ├── metrics.py
│   ├── run_eval.py
│   └── results/           # tables, figures
├── data/                  # gitignored; download script in README
└── report/                # LaTeX (CVPR 2022 template)
```

## 7. Experiments

| Item | Plan |
|---|---|
| Removal datasets | ISTD, SRD (have ground truth) |
| Detection dataset | SBU (masks) |
| Metrics (removal) | RMSE in LAB on shadow region, PSNR, SSIM |
| Metrics (detection) | IoU, BER |
| Baselines | Otsu detector + global gain; ground-truth-mask variants (isolate removal quality from detection); optional pretrained model |
| Ablations | w/o penumbra blending; w/o region-matched relighting; w/o user refinement |
| Figures | Qualitative before/after grid; ablation strip; 2-3 failure cases |

Use a fixed subset (e.g. 50-100 test images) if time is short, and state it in the report.

## 8. Deliverables Mapping

| Deliverable | Weight | Contents |
|---|---|---|
| Write-up (CVPR template) | 5% | Abstract, intro with novelty, method + overview figure, experiments, appendix with full code + GitHub link, dataset/repo links, per-member contributions |
| Presentation | 5% | Problem (I/O, constraints), literature review, method with reasons, experiments |
| Code / Demo | 5% | Clean commented code, good UI, strong examples, everyone can answer implementation questions |

## 9. Team Roles (suggested; **[DECIDE]** names)

| # | Owner | Responsibility |
|---|---|---|
| 1 | _name_ | Detection module (A) |
| 2 | _name_ | Relighting + penumbra blending (C, D) |
| 3 | _name_ | Interface + scribble refinement (B, E) |
| 4 | _name_ | Datasets, metrics, experiments, ablations |
| 5 | _name_ | Literature review, overview figure, LaTeX, slides |

Every member must be able to explain all modules during the demo.

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Detector fails on dark objects / low-contrast shadows | Scribble refinement; show as failure case |
| Colour shift after relighting | Region-matched reference band; LAB statistics |
| Time overrun on experiments | Fixed small test subset; script experiments early (Nov 4) |
| Integration trouble at the end | Fixed interfaces from day 1; end-to-end rough pipeline by Nov 3 |
| Novelty questioned | Prepare ablation showing each component helps; confirm with instructor early |

## 11. Reuse and Citations

List every dataset and repository used here (also required in the write-up):

- ISTD: _link_
- SRD: _link_
- SBU: _link_
- Reused code: _link / what was reused / what we changed_
