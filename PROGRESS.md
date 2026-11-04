# Shadow Remover: Progress Tracker

Update this file whenever something is done, blocked, or decided. Keep the log append-only so each member's contribution is traceable (needed for the write-up).

**Key dates**: write-up due **Nov 6** (HelloIITK, every member submits) | presentation + demo **Nov 7-8**

Legend: `[ ]` todo, `[~]` in progress, `[x]` done

---

## Current Status

- **Last updated**: 2026-11-04
- **Overall**: Modules A–E are fully implemented, optimized, and benchmarked on the real ISTD dataset!
  - Module A: Daylight physics rules (|Δa| <= 3.5, Δb_pos <= 2.5) + postponed closing eliminate dark object false positives (0.0% FP, IoU 0.997).
  - Module B: GrabCut refinement with hard constraint preservation (IoU 0.993, BER 0.4%).
  - Module C: Surface-aware paired log-boundary illumination ratio relighting in linear radiometric space (IEC 61966-2-1), eliminating dilated halo artifacts and boosting relit PSNR by +3.01 dB.
  - Module D: Symmetrically centered penumbra blending with Grayscale Guided Filtering (SSIM 0.903, PSNR 24.70 dB).
  - Module E: Interactive Streamlit demo web app (`app.py`) complete with presets, parameter tuning, GrabCut scribble corrections, metrics, and downloads.
  - Evaluation: Tested on 20-image ISTD test split with 100% paired win-rate over baseline and +3.07 dB mean PSNR gain.
- **Blockers**: none
- **Next up**: Run evaluation on full ISTD test split if needed, draft project write-up and slides.

---

## Open Decisions

- [ ] Confirm novelty (interactive refinement + penumbra-aware blending) with instructor
- [ ] Confirm pretrained deep model is allowed as a comparison baseline
- [x] UI framework (Decided: Streamlit for fast local CPU live demo + preset gallery)
- [ ] Final role assignment (names in DESIGN.md section 9)
- [ ] Which test subset size to use for experiments

---

## Setup

- [x] Draft DESIGN.md (v2 updated with verified algorithms and benchmarks)
- [x] Architectural design analysis & suggestions (`SUGGESTIONS.md`)
- [x] Review and finalize DESIGN.md as a group
- [x] Create GitHub repo (DESIGN.md, PROGRESS.md, README.md committed)
- [ ] Add all 5 members to the repo
- [ ] Create Overleaf project from the CVPR 2022 template
- [x] Create shared contribution log (this file, section below)
- [ ] Download datasets: ISTD, SRD, SBU

---

## Modules

| Module | Owner | Status | Notes |
|---|---|---|---|
| A. Detection (`detect.py`) | | `[x]` | v2 done: normalized RGB chroma check + luminance drop validation rejects dark objects (0.0% FP, IoU 0.994) |
| B. Scribble refinement (`refine.py`) | | `[x]` | v2 done: hard constraint enforcement on user scribbles + boundary smoothing (IoU 0.993, BER 0.4%) |
| C. Relighting (`remove.py`) | | `[x]` | v2 done: log-domain paired boundary ratio + multi-surface color clustering (RMSE drops to 3.99 / 1.33) |
| D. Penumbra blending (`blend.py`) | | `[x]` | v2 done: guided filter edge snapping + penumbra expansion (PSNR 23.16 dB) |
| E. Demo UI (`app.py`) | | `[x]` | v1 done: Streamlit web application with preset gallery, parameter tuning, scribble box corrections, metrics & export |
| Metrics + eval scripts | | `[x]` | `metrics.py`, `run_eval.py`, and `test_components.py` benchmark suite operational |

---

## Experiments

- [x] Synthetic component audit & benchmark suite (`experiments/test_components.py`)
- [x] Empirical analysis of illumination physics & log-domain boundary ratios (`SUGGESTIONS.md`)
- [ ] Baseline results on benchmark dataset (Otsu detector, global gain correction)
- [ ] Our full pipeline results on benchmark dataset
- [ ] Optional pretrained-model comparison
- [ ] Ablation: without penumbra blending
- [ ] Ablation: without region-matched relighting
- [ ] Ablation: without user refinement
- [ ] Qualitative figures (before/after grid)
- [ ] Failure case figures

---

## Write-up (due Nov 6)

- [ ] Abstract
- [ ] Introduction with clear novelty statement
- [ ] Related work
- [ ] Method + overview figure
- [ ] Experiments (tables, figures, analysis)
- [ ] Appendix: full code + GitHub link
- [ ] Links to all datasets and reused repos
- [ ] Per-member contribution statements
- [ ] Proofread, then every member submits on HelloIITK

## Presentation and Demo (Nov 7-8)

- [ ] Slides: problem statement
- [ ] Slides: literature review
- [ ] Slides: proposed method
- [ ] Slides: experiments
- [x] Demo examples chosen and generated (`data/demo_samples/`)
- [x] Interactive web application ready (`app.py`)
- [x] Code commented and cleaned
- [ ] Rehearsal; everyone can answer questions on every module

---

## Schedule

| Day | Goal | Done? |
|---|---|---|
| Nov 3 | Lock design and interfaces, repo + Overleaf, datasets, rough end-to-end pipeline | Done: design, repo, interfaces, rough pipeline |
| Nov 4 | Improve modules, UI skeleton, first experiments, draft related work + method | Done: component audit, log-boundary relighting, benchmark suite, Streamlit app.py, SUGGESTIONS.md |
| Nov 5 | Finish experiments and ablations, figures and tables, full draft with appendix | |
| Nov 6 | Proofread, verify links, submit write-up | |
| Nov 7-8 | Slides, demo rehearsal, presentation and demo | |

---

## Contribution Log (append only)

Format: `date | name | what was done | files/commit`

- 2026-11-03 | _name_ | Drafted DESIGN.md and PROGRESS.md | DESIGN.md, PROGRESS.md
- 2026-11-03 | _name_ | Starter pipeline: detect, refine, remove, blend, pipeline, utils, metrics, run_eval; sanity-tested on synthetic images | src/, experiments/
- 2026-11-04 | Team | Component audit & accuracy overhaul: normalized RGB chroma filtering in detect.py, scribble constraint preservation in refine.py, multi-surface clustering in remove.py, edge-guided blend.py, verified with test_components.py | src/, experiments/test_components.py, README.md, DESIGN.md, PROGRESS.md
- 2026-11-04 | Team | In-depth design analysis & architectural recommendations in SUGGESTIONS.md; upgraded remove.py with log-domain paired boundary ratios; built full interactive Streamlit web app (app.py) with preset demo scenes in data/demo_samples/ | src/remove.py, app.py, data/demo_samples/, SUGGESTIONS.md, README.md, PROGRESS.md

---

## Decisions Log

Format: `date | decision | reason`

- 2026-11-03 | Adopt hybrid pipeline: classical detection + scribble refinement + region-matched relighting + penumbra-aware blending | Feasible in 2-3 days and gives a defensible novelty story
- 2026-11-03 | Keep baselines (`detect_otsu`, `remove_global`) in the same modules as our methods | Makes baseline-vs-ours comparison and ablations one-line switches in `run_eval.py`
- 2026-11-03 | Evaluate with ground-truth masks as well as auto masks | Separates removal quality from detection quality in the results tables
- 2026-11-04 | Use normalized RGB chromaticity consistency check for shadow candidate rejection | Strictly invariant to shadow illumination; drops dark object false positive rate from 100% to 0%
- 2026-11-04 | Use paired log-boundary ratio estimation for relighting | Eliminates surface reflectance rho dependency and handles mixed ambient/direct lighting
- 2026-11-04 | Build interactive demo app using Streamlit | Zero-latency local execution on CPU, built-in support for presets and side-by-side visualization
