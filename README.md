# Shadow-Remover-Tool
Course Project for EE604: Digital Image Processing.

A single-image shadow detection and removal system featuring:
- **Chroma-Consistent Detection**: Classical LAB luminance thresholding coupled with localized normalized chromaticity consistency filtering to reject inherently dark non-shadow objects.
- **Interactive Scribble Refinement**: Graph cut (`cv2.grabCut`) guided by user scribbles (definite/probable shadow vs. lit) for rapid manual correction.
- **Surface-Aware Region-Matched Relighting**: Color-consistent clustering and paired boundary-strip log-domain illumination ratio estimation to relight multi-surface shadows without cross-surface color contamination.
- **Penumbra-Aware Blending**: Grayscale guided filter edge snapping across dilated penumbra regions to eliminate boundary seams and halo artifacts.

---

## Repository Structure

```
Shadow-Remover-Tool/
├── DESIGN.md                 # System architecture, module design, & mathematical formulation
├── PROGRESS.md               # Task tracker and contribution log
├── README.md                 # Project documentation and quickstart
├── SUGGESTIONS.md            # In-depth architectural analysis and experimental findings
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
│   ├── run_eval.py           # Evaluation script for ISTD/SRD datasets
│   └── results/              # Output tables and figures
├── data/
│   └── demo_samples/         # Preloaded preset demo images for live demo
└── app.py                    # Interactive demo web application (Streamlit)
```

---

## Installation

Ensure you have Python 3.10+ installed:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Launch Interactive Demo Web App (Recommended for Demo)
```bash
streamlit run app.py
```
The web application provides preset scenes (dark object rejection, multi-surface shadows, textured soft penumbra, specular reflection failure cases), interactive parameter tuning, GrabCut scribble correction, side-by-side comparison, and one-click PNG downloads.

### 2. Run End-to-End Pipeline CLI
```bash
python -m src.pipeline path/to/input.jpg path/to/output.png [--mask path/to/gt_mask.png]
```

### 3. Run Component Audit and Benchmark Suite
To test all four pipeline modules against baseline implementations across dark object rejection, scribble refinement, multi-surface relighting, and penumbra blending:
```bash
python experiments/test_components.py
```

### 4. Evaluate on Benchmark Datasets (ISTD / SRD)
```bash
python -m experiments.run_eval --root data/ISTD/test --limit 50 --out experiments/results
```
