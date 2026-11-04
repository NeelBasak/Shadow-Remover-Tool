# Shadow-Remover-Tool
Course Project for EE604: Digital Image Processing.

A single-image shadow detection and removal system featuring:
- **Chroma-Consistent Detection**: Classical LAB luminance thresholding coupled with localized normalized chromaticity consistency filtering to reject inherently dark non-shadow objects.
- **Interactive Scribble Refinement**: Graph cut (`cv2.grabCut`) guided by user scribbles (definite/probable shadow vs. lit) for rapid manual correction.
- **Surface-Aware Region-Matched Relighting**: Color-consistent clustering and local lit reference band sampling to relight multi-surface shadows without cross-surface color contamination.
- **Penumbra-Aware Blending**: Grayscale guided filter edge snapping across dilated penumbra regions to eliminate boundary seams and halo artifacts.

---

## Repository Structure

```
Shadow-Remover-Tool/
├── DESIGN.md                 # System architecture, module design, & mathematical formulation
├── PROGRESS.md               # Task tracker and contribution log
├── README.md                 # Project documentation and quickstart
├── requirements.txt          # Python dependencies
├── src/
│   ├── detect.py             # Module A: shadow detection (Otsu + chroma filtering)
│   ├── refine.py             # Module B: GrabCut interactive scribble refinement
│   ├── remove.py             # Module C: surface-aware region-matched relighting
│   ├── blend.py              # Module D: guided-filter penumbra blending
│   ├── pipeline.py           # End-to-end pipeline wrapper & CLI
│   └── utils.py              # Guided filter, color conversion, I/O helpers
├── experiments/
│   ├── metrics.py            # Evaluation metrics (RMSE in LAB, PSNR, SSIM, IoU, BER)
│   ├── test_components.py    # Synthetic component audit & benchmark test suite
│   ├── run_eval.py           # Evaluation script for ISTD/SRD datasets
│   └── results/              # Output tables and figures
└── app.py                    # Interactive demo web application
```

---

## Installation

Ensure you have Python 3.10+ installed:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Run End-to-End Pipeline on an Image
```bash
python -m src.pipeline path/to/input.jpg path/to/output.png [--mask path/to/gt_mask.png]
```

### 2. Run Component Audit and Benchmark Suite
To test all four pipeline modules against baseline implementations across dark object rejection, scribble refinement, multi-surface relighting, and penumbra blending:
```bash
python experiments/test_components.py
```

### 3. Evaluate on Benchmark Datasets (ISTD / SRD)
```bash
python -m experiments.run_eval --root data/ISTD/test --limit 50 --out experiments/results
```
