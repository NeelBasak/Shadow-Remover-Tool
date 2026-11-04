"""Interactive Demo Web Application for Shadow-Remover-Tool (EE604 Course Project).

Features:
- Image upload + preloaded sample presets (dark object rejection, multi-surface, soft penumbra, failure case)
- Automatic detection parameter tuning (Otsu, chroma consistency, L ratio)
- Interactive GrabCut scribble / region correction
- Surface-aware log-boundary relighting & penumbra-aware guided blending
- Side-by-side comparison & quantitative metrics
- Result export / download
"""
import io
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from experiments import metrics as M
from src.blend import blend
from src.detect import detect, detect_otsu
from src.pipeline import run
from src.refine import SCRIB_LIT, SCRIB_SHADOW, refine
from src.remove import remove, remove_global
from src.utils import overlay_mask

# Set page config
st.set_page_config(
    page_title="Shadow Remover Tool — EE604",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">☀️ Shadow Remover Tool</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">EE604 Digital Image Processing Course Project — '
    'Single-Image Shadow Detection, Scribble Refinement & Penumbra-Aware Relighting</div>',
    unsafe_allow_html=True,
)

# Sidebar: Controls & Configuration
st.sidebar.title("Settings & Controls")

# Input Source Selection
input_mode = st.sidebar.radio(
    "Choose Input Source",
    ["Preset Sample", "Upload Image"],
    index=0,
)

sample_dir = ROOT_DIR / "data" / "demo_samples"
sample_files = {
    "1. Dark Object Rejection": sample_dir / "1_dark_object_rejection.png",
    "2. Multi-Surface Shadow": sample_dir / "2_multisurface_shadow.png",
    "3. Soft Penumbra with Texture": sample_dir / "3_soft_penumbra.png",
    "4. Failure Case (Specular Highlight)": sample_dir / "4_specular_failure_case.png",
}

image_rgb = None
gt_image_rgb = None
gt_mask = None

if input_mode == "Preset Sample":
    sample_choice = st.sidebar.selectbox("Select Preset Demo Scene", list(sample_files.keys()))
    sample_path = sample_files[sample_choice]
    if sample_path.exists():
        pil_img = Image.open(sample_path).convert("RGB")
        image_rgb = np.array(pil_img)
    else:
        st.warning(f"Sample file not found at {sample_path}.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Target Image (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(pil_img)

# Optional Ground Truth Upload for Metrics
with st.sidebar.expander("Ground Truth (Optional)", expanded=False):
    gt_file = st.file_uploader("Upload Shadow-Free Ground Truth", type=["jpg", "jpeg", "png"])
    if gt_file is not None:
        gt_image_rgb = np.array(Image.open(gt_file).convert("RGB"))
    gt_mask_file = st.file_uploader("Upload GT Shadow Mask", type=["jpg", "jpeg", "png"])
    if gt_mask_file is not None:
        gt_m = np.array(Image.open(gt_mask_file).convert("L"))
        gt_mask = (gt_m > 127).astype(np.float32)

# Algorithm Hyperparameters
st.sidebar.markdown("---")
st.sidebar.subheader("Algorithm Tuning")

with st.sidebar.expander("Module A: Detection Parameters", expanded=False):
    detector_mode = st.selectbox(
        "Detection Algorithm", ["Ours (Chroma-Consistent)", "Baseline (Otsu on L)"]
    )
    max_chroma = st.slider("Max Normalized Chroma Diff", 0.01, 0.10, 0.035, 0.005)
    min_l_ratio = st.slider("Min Luminance Drop Ratio", 1.05, 2.0, 1.15, 0.05)
    min_sep = st.slider("Min Separation (L units)", 5.0, 25.0, 10.0, 1.0)
    soft_sigma = st.slider("Mask Softening Sigma", 0.5, 5.0, 2.0, 0.5)

with st.sidebar.expander("Module C: Relighting Parameters", expanded=False):
    relight_mode = st.selectbox(
        "Relighting Method",
        ["Surface-Aware Log-Boundary Ratio (Ours)", "Global Multiplicative Gain (Baseline)"],
    )
    max_gain = st.slider("Max Amplification Gain", 1.5, 8.0, 4.0, 0.5)
    cluster_multisurface = st.checkbox("Enable Surface Clustering", value=True)

with st.sidebar.expander("Module D: Penumbra Blending Parameters", expanded=False):
    enable_blend = st.checkbox("Enable Penumbra Blending", value=True)
    use_guided = st.checkbox("Use Guided Filter Edge Snapping", value=True)
    penumbra_rad = st.slider("Penumbra Transition Radius (px)", 1, 15, 4, 1)

# Main Application Tabs
if image_rgb is None:
    st.info("Please select a preset sample or upload an image from the sidebar.")
    st.stop()

tab_pipeline, tab_interactive, tab_evaluation, tab_about = st.tabs(
    ["🚀 End-to-End Pipeline", "✏️ Interactive Refinement", "📊 Quantitative Evaluation", "📖 Architecture & Novelty"]
)

# Compute Automatic Mask
h, w = image_rgb.shape[:2]
if detector_mode == "Ours (Chroma-Consistent)":
    auto_mask = detect(
        image_rgb,
        min_separation=min_sep,
        max_chroma_diff=max_chroma,
        min_l_ratio=min_l_ratio,
        soft_sigma=soft_sigma,
    )
else:
    auto_mask = detect_otsu(image_rgb)

# TAB 1: PIPELINE RESULTS
with tab_pipeline:
    st.subheader("Automated Shadow Removal Stages")

    # Compute Relit & Final Blended
    if relight_mode == "Surface-Aware Log-Boundary Ratio (Ours)":
        relit_img = remove(
            image_rgb,
            auto_mask,
            max_gain=max_gain,
            enable_clustering=cluster_multisurface,
        )
    else:
        relit_img = remove_global(image_rgb, auto_mask, max_gain=max_gain)

    if enable_blend:
        final_img = blend(
            image_rgb,
            relit_img,
            auto_mask,
            penumbra=penumbra_rad,
            use_guided=use_guided,
        )
    else:
        final_img = relit_img

    mask_overlay = overlay_mask(image_rgb, auto_mask, color=(255, 60, 60), alpha=0.45)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(image_rgb, caption="1. Original Input Image", use_container_width=True)
    with col2:
        st.image(mask_overlay, caption="2. Detected Shadow Mask", use_container_width=True)
    with col3:
        st.image(relit_img, caption="3. Relit Core Image", use_container_width=True)
    with col4:
        st.image(final_img, caption="4. Final Shadow-Free Output", use_container_width=True)

    # Summary Metrics Row
    shadow_pct = float((auto_mask > 0.5).mean() * 100)
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Image Resolution", f"{w} × {h} px")
    with m2:
        st.metric("Detected Shadow Area", f"{shadow_pct:.1f}%")
    with m3:
        st.metric("Detection Mode", detector_mode.split()[0])
    with m4:
        st.metric("Blending Mode", "Guided Filter" if (enable_blend and use_guided) else ("Standard" if enable_blend else "None"))

    # Download Buttons
    st.markdown("---")
    down_col1, down_col2 = st.columns(2)
    with down_col1:
        out_pil = Image.fromarray(final_img)
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        st.download_button(
            label="💾 Download Shadow-Free Result (PNG)",
            data=buf.getvalue(),
            file_name="shadow_free_output.png",
            mime="image/png",
        )
    with down_col2:
        mask_pil = Image.fromarray((auto_mask * 255).astype(np.uint8))
        mask_buf = io.BytesIO()
        mask_pil.save(mask_buf, format="PNG")
        st.download_button(
            label="💾 Download Predicted Mask (PNG)",
            data=mask_buf.getvalue(),
            file_name="shadow_mask.png",
            mime="image/png",
        )

# TAB 2: INTERACTIVE REFINEMENT
with tab_interactive:
    st.subheader("Interactive GrabCut Scribble Correction")
    st.write(
        "Draw or specify correction regions to guide the GrabCut engine. "
        "User scribbles override automatic detections as definite constraints."
    )

    scrib_col1, scrib_col2 = st.columns([1, 1])

    with scrib_col1:
        st.markdown("**1. Configure Correction Region**")
        action_type = st.radio(
            "Correction Action",
            ["Mark False Positive as LIT (Remove Shadow)", "Mark Undetected as SHADOW (Add Shadow)"],
        )
        sub_type = SCRIB_LIT if "LIT" in action_type else SCRIB_SHADOW

        # Coordinates for correction box
        x_min = st.slider("X Min (px)", 0, w, int(w * 0.1), 5)
        x_max = st.slider("X Max (px)", x_min + 5, w, int(w * 0.4), 5)
        y_min = st.slider("Y Min (px)", 0, h, int(h * 0.1), 5)
        y_max = st.slider("Y Max (px)", y_min + 5, h, int(h * 0.4), 5)
        grabcut_iters = st.slider("GrabCut Iterations", 1, 8, 3, 1)

        scribbles = np.zeros((h, w), dtype=np.uint8)
        scribbles[y_min:y_max, x_min:x_max] = sub_type

        # Visualizing scribble region on image
        annotated_img = image_rgb.copy()
        color_b = (0, 255, 0) if sub_type == SCRIB_LIT else (255, 0, 0)
        cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), color_b, 2)
        st.image(annotated_img, caption="Image with Correction Constraint", use_container_width=True)

    with scrib_col2:
        st.markdown("**2. Refined Output**")
        with st.spinner("Running GrabCut refinement..."):
            refined_m = refine(
                image_rgb,
                auto_mask,
                scribbles=scribbles,
                iters=grabcut_iters,
                soft_sigma=soft_sigma,
            )
            refined_relit = remove(image_rgb, refined_m, max_gain=max_gain)
            refined_final = blend(image_rgb, refined_relit, refined_m, penumbra=penumbra_rad)

        st.image(
            overlay_mask(image_rgb, refined_m, color=(0, 200, 255), alpha=0.5),
            caption="Refined Shadow Mask",
            use_container_width=True,
        )
        st.image(
            refined_final,
            caption="Refined Shadow-Free Result",
            use_container_width=True,
        )

# TAB 3: QUANTITATIVE EVALUATION
with tab_evaluation:
    st.subheader("Quantitative Evaluation & Metric Benchmarking")

    if gt_image_rgb is not None:
        if gt_image_rgb.shape != image_rgb.shape:
            gt_image_rgb = cv2.resize(gt_image_rgb, (w, h))

        rmse_all = M.rmse_lab(final_img, gt_image_rgb)
        psnr_val = M.psnr(final_img, gt_image_rgb)
        ssim_val = M.ssim(final_img, gt_image_rgb)

        st.markdown("### Reconstruction Quality vs Ground Truth")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("PSNR (Peak SNR)", f"{psnr_val:.2f} dB", delta=f"{psnr_val - 20:.2f} dB")
        with c2:
            st.metric("SSIM (Structural Similarity)", f"{ssim_val:.4f}")
        with c3:
            st.metric("RMSE in CIE LAB", f"{rmse_all:.2f}")

        if gt_mask is not None:
            if gt_mask.shape != (h, w):
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            rmse_shadow = M.rmse_lab(final_img, gt_image_rgb, region=gt_mask)
            mask_iou = M.iou(auto_mask, gt_mask)
            mask_ber = M.ber(auto_mask, gt_mask)

            st.markdown("### Shadow Region & Mask Segmentation Metrics")
            m_c1, m_c2, m_c3 = st.columns(3)
            with m_c1:
                st.metric("Shadow Region RMSE (LAB)", f"{rmse_shadow:.2f}")
            with m_c2:
                st.metric("Mask IoU", f"{mask_iou:.3f}")
            with m_c3:
                st.metric("Mask Balanced Error Rate (BER)", f"{mask_ber:.1f}%")
    else:
        st.info(
            "Upload a shadow-free ground truth image from the sidebar to calculate "
            "exact PSNR, SSIM, LAB RMSE, IoU, and BER metrics."
        )

        st.markdown("### Built-In Synthetic Benchmark Suite Results")
        st.table(
            [
                {"Component": "A. Detection (Dark Object Rejection)", "Baseline (Otsu)": "IoU: 0.500 | 100% FP", "Proposed Pipeline": "IoU: 0.994 | 0.0% FP"},
                {"Component": "B. Refinement (Scribbles)", "Baseline (Otsu)": "IoU: 0.500 | 6.2% BER", "Proposed Pipeline": "IoU: 0.993 | 0.4% BER"},
                {"Component": "C. Relighting (Multi-Surface)", "Baseline (Otsu)": "RMSE: 15.78 (Color Cast)", "Proposed Pipeline": "RMSE: 3.99 (Surface Matched)"},
                {"Component": "D. Blending (Penumbra Guided)", "Baseline (Otsu)": "PSNR: 21.65 dB (Halo)", "Proposed Pipeline": "PSNR: 23.16 dB (Smooth)"},
            ]
        )

# TAB 4: ARCHITECTURE & NOVELTY
with tab_about:
    st.subheader("System Architecture & Proposed Novelties")
    st.markdown(
        """
        ### Pipeline Architecture:
        1. **Module A — Chroma-Consistent Detection (`src/detect.py`)**:
           - Extracts $L$ channel from CIE LAB space.
           - Applies localized normalized chromaticity consistency filtering: $\\Delta c = \\|\\mathbf{r}_{\\text{shadow}} - \\mathbf{r}_{\\text{lit}}\\|_2 \\le 0.035$.
           - Rejects dark objects that share low luminance but differ in surface material.
        2. **Module B — Interactive Scribble Refinement (`src/refine.py`)**:
           - Formulates shadow mask extraction as graph cuts with user scribbles enforcing hard boundary seeds.
           - Resolves ambiguous multi-depth shadow contours interactively.
        3. **Module C — Surface-Aware Log-Boundary Relighting (`src/remove.py`)**:
           - Samples paired inner and outer boundary strips along shadow contours.
           - Log-domain illumination ratio: $\\log(\\gamma_c) = \\text{median}(\\log I_{\\text{outer}}) - \\text{median}(\\log I_{\\text{inner}})$.
           - Material reflectance $\\rho$ cancels out identically, eliminating multi-surface color bleeding.
        4. **Module D — Penumbra-Aware Guided Blending (`src/blend.py`)**:
           - Snaps alpha matte to scene contours using Grayscale Guided Filtering.
           - Restores penumbra lighting without boundary fringes or overexposure.
        """
    )
