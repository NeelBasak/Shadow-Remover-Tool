# Shadow Remover: Progress Tracker

Update this file whenever something is done, blocked, or decided. Keep the log append-only so each member's contribution is traceable (needed for the write-up).

**Key dates**: write-up due **Nov 6** (HelloIITK, every member submits) | presentation + demo **Nov 7-8**

Legend: `[ ]` todo, `[~]` in progress, `[x]` done

---

## Current Status

- **Last updated**: 2026-11-03
- **Overall**: design phase
- **Blockers**: none yet
- **Next up**: finalize design, assign owners, set up repo and Overleaf

---

## Open Decisions

- [ ] Confirm novelty (interactive refinement + penumbra-aware blending) with instructor
- [ ] Confirm pretrained deep model is allowed as a comparison baseline
- [ ] Gradio vs Streamlit for the UI
- [ ] Final role assignment (names in DESIGN.md section 9)
- [ ] Which test subset size to use for experiments

---

## Setup

- [x] Draft DESIGN.md
- [ ] Review and finalize DESIGN.md as a group
- [ ] Create GitHub repo and add all 5 members
- [ ] Create Overleaf project from the CVPR 2022 template
- [ ] Create shared contribution log (this file, section below)
- [ ] Download datasets: ISTD, SRD, SBU

---

## Modules

| Module | Owner | Status | Notes |
|---|---|---|---|
| A. Detection (`detect.py`) | | `[ ]` | |
| B. Scribble refinement (`refine.py`) | | `[ ]` | |
| C. Relighting (`remove.py`) | | `[ ]` | |
| D. Penumbra blending (`blend.py`) | | `[ ]` | |
| E. Demo UI (`app.py`) | | `[ ]` | |
| Metrics + eval scripts | | `[ ]` | |

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
| Nov 3 | Lock design and interfaces, repo + Overleaf, datasets, rough end-to-end pipeline | |
| Nov 4 | Improve modules, UI skeleton, first experiments, draft related work + method | |
| Nov 5 | Finish experiments and ablations, figures and tables, full draft with appendix | |
| Nov 6 | Proofread, verify links, submit write-up | |
| Nov 7-8 | Slides, demo rehearsal, presentation and demo | |

---

## Contribution Log (append only)

Format: `date | name | what was done | files/commit`

- 2026-11-03 | _name_ | Drafted DESIGN.md and PROGRESS.md | DESIGN.md, PROGRESS.md

---

## Decisions Log

Format: `date | decision | reason`

- 2026-11-03 | Adopt hybrid pipeline: classical detection + scribble refinement + region-matched relighting + penumbra-aware blending | Feasible in 2-3 days and gives a defensible novelty story
