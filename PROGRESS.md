# Shadow Remover: Progress Tracker

Update this file whenever something is done, blocked, or decided. Keep the log append-only so each member's contribution is traceable (needed for the write-up).

**Key dates**: write-up due **Nov 6** (HelloIITK, every member submits) | presentation + demo **Nov 7-8**

Legend: `[ ]` todo, `[~]` in progress, `[x]` done

---

## Current Status

- **Last updated**: 2026-11-03
- **Overall**: starter pipeline (modules A-D, metrics, eval script) written and tested on synthetic images; not yet run on real data
- **Blockers**: none yet
- **Next up**: download ISTD and run first real evaluation, assign module owners, build Gradio demo, improve detector

---

## Open Decisions

- [ ] Confirm novelty (interactive refinement + penumbra-aware blending) with instructor
- [ ] Confirm pretrained deep model is allowed as a comparison baseline
- [ ] Gradio vs Streamlit for the UI
- [ ] Final role assignment (names in DESIGN.md section 9)
- [ ] Which test subset size to use for experiments

---

## Setup

- [x] Draft DESIGN.md (v1, updated to match starter code)
- [ ] Review and finalize DESIGN.md as a group
- [x] Create GitHub repo (DESIGN.md and PROGRESS.md committed)
- [ ] Add all 5 members to the repo
- [ ] Create Overleaf project from the CVPR 2022 template
- [x] Create shared contribution log (this file, section below)
- [ ] Download datasets: ISTD, SRD, SBU

---

## Modules

| Module | Owner | Status | Notes |
|---|---|---|---|
| A. Detection (`detect.py`) | | `[~]` | v1 done (Otsu on L + cleanup); weak on dark objects, needs chroma check |
| B. Scribble refinement (`refine.py`) | | `[~]` | v1 done (GrabCut); simulated scribbles raised IoU from ~0.70 to ~0.95+ on synthetic images |
| C. Relighting (`remove.py`) | | `[~]` | v1 done (region-matched RGB gain); needs testing on real multi-surface images |
| D. Penumbra blending (`blend.py`) | | `[~]` | v1 done (dilate + blur + guided filter); full pipeline beat no-blend variants on synthetic data |
| E. Demo UI (`app.py`) | | `[ ]` | not started |
| Metrics + eval scripts | | `[~]` | `metrics.py` and `run_eval.py` written; need ISTD data to run |

---

## Experiments

- [ ] Baseline results (Otsu detector, global gain correction)
- [ ] Our full pipeline results
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
- [ ] Demo examples chosen (including one failure case)
- [ ] Code commented and cleaned
- [ ] Rehearsal; everyone can answer questions on every module

---

## Schedule

| Day | Goal | Done? |
|---|---|---|
| Nov 3 | Lock design and interfaces, repo + Overleaf, datasets, rough end-to-end pipeline | Partly: design, repo, interfaces, rough pipeline done; Overleaf and datasets pending |
| Nov 4 | Improve modules, UI skeleton, first experiments, draft related work + method | |
| Nov 5 | Finish experiments and ablations, figures and tables, full draft with appendix | |
| Nov 6 | Proofread, verify links, submit write-up | |
| Nov 7-8 | Slides, demo rehearsal, presentation and demo | |

---

## Contribution Log (append only)

Format: `date | name | what was done | files/commit`

- 2026-11-03 | _name_ | Drafted DESIGN.md and PROGRESS.md | DESIGN.md, PROGRESS.md
- 2026-11-03 | _name_ | Starter pipeline: detect, refine, remove, blend, pipeline, utils, metrics, run_eval; sanity-tested on synthetic images | src/, experiments/

---

## Decisions Log

Format: `date | decision | reason`

- 2026-11-03 | Adopt hybrid pipeline: classical detection + scribble refinement + region-matched relighting + penumbra-aware blending | Feasible in 2-3 days and gives a defensible novelty story
- 2026-11-03 | Keep baselines (`detect_otsu`, `remove_global`) in the same modules as our methods | Makes baseline-vs-ours comparison and ablations one-line switches in `run_eval.py`
- 2026-11-03 | Evaluate with ground-truth masks as well as auto masks | Separates removal quality from detection quality in the results tables
