"""Interactive Streamlit Demo for Shadow-Remover-Tool.

Course project for EE604 (Image Processing), IIT Kanpur.
Features:
  * Select from preset demo images (single surface, multi-surface, soft penumbra).
  * Upload custom images and optional ground-truth for quantitative evaluation.
  * Adjust detection, relighting, and penumbra blending hyperparameters.
  * Interactive scribble refinement tab (simulate user marking lit/shadow).
  * Live comparison: Input vs Detected Mask vs Relit vs Final Blended.
  * Metric benchmarking against ground-truth (PSNR, SSIM, LAB RMSE, IoU, BER).
"""
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from experiments import metrics as M
from src.blend import blend
from src.detect import detect, detect_otsu
from src.refine import refine, SCRIB_LIT, SCRIB_SHADOW
from src.remove import remove, remove_global
from src.utils import overlay_mask

st.set_page_config(
    page_title="Shadow Remover Tool — EE604",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header & Project Information
st.title("☀️ Shadow-Remover-Tool: Interactive Visualizer")
st.markdown(
    """
    **EE604 Course Project** | Non-Deep Learning Physics-Based Shadow Detection, 
    Interactive Scribble Refinement, Surface-Aware Relighting, and Penumbra Blending.
    """
)

# Sidebar Configuration
st.sidebar.header("🕹️ Controls & Settings")

DATA_DIR = Path(__file__).parent / "data" / "demo_samples"
PRESETS = {
    "Select a preset...": None,
    "--- 🌟 Standard Benchmark Presets ---": None,
    "1. Single Surface Concrete Shadow (100-1)": "100-1.png",
    "2. High-Contrast Concrete Shadow (100-4)": "100-4.png",
    "3. Fine-Penumbra Pavement Shadow (100-5)": "100-5.png",
    "4. Challenging Occluder Scene (101-1)": "101-1.png",
    "--- 🛑 Dark Object Rejection (Ours vs. Otsu Failure) ---": None,
    "5. Dark Object Rejection: Black Shoe (Otsu fails 100%, Ours 0%)": "1_dark_object_rejection.png",
    "6. Dark Ground Material (103-5: Otsu false detects 173k px)": "103-5.png",
    "--- ✏️ Failure Cases & Human Scribble Rescue (Module B) ---": None,
    "7. Failure Case: Missed Branch Shadow (100-6: Auto misses 8.6k px, Scribble rescues)": "100-6.png",
    "8. Failure Case: Dark Asphalt Road (102-1: Auto over-detects road, Scribble rescues)": "102-1.png",
    "9. Failure Case: Specular Ambiguity (Demo 4: Auto detects 0%, Scribble recovers)": "4_specular_failure_case.png",
}

# Source Selection
preset_choice = st.sidebar.selectbox("Choose Demo Preset", list(PRESETS.keys()))

uploaded_file = st.sidebar.file_uploader(
    "Or upload custom image", type=["jpg", "jpeg", "png"]
)

image_rgb = None
gt_image_rgb = None
gt_mask = None

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(pil_img)
elif preset_choice and PRESETS[preset_choice] is not None:
    preset_file = PRESETS[preset_choice]
    img_path = DATA_DIR / preset_file
    if img_path.exists():
        pil_img = Image.open(img_path).convert("RGB")
        image_rgb = np.array(pil_img)

        # Check for matching ground truths in demo_samples or ISTD test folders
        base_stem = Path(preset_file).stem
        gt_img_path = DATA_DIR / f"{base_stem}_gt.png"
        gt_mask_path = DATA_DIR / f"{base_stem}_mask.png"
        if not gt_img_path.exists():
            gt_img_path = Path(__file__).parent / "data" / "ISTD_Dataset" / "test" / "test_C" / preset_file
        if not gt_mask_path.exists():
            gt_mask_path = Path(__file__).parent / "data" / "ISTD_Dataset" / "test" / "test_B" / preset_file

        if gt_img_path.exists():
            gt_image_rgb = np.array(Image.open(gt_img_path).convert("RGB"))
        if gt_mask_path.exists():
            gt_m = np.array(Image.open(gt_mask_path).convert("L"))
            gt_mask = (gt_m > 127).astype(np.float32)
else:
    # Default to first preset if available
    first_preset = DATA_DIR / "100-1.png"
    if first_preset.exists():
        pil_img = Image.open(first_preset).convert("RGB")
        image_rgb = np.array(pil_img)
        gt_img_path = DATA_DIR / "100-1_gt.png"
        gt_mask_path = DATA_DIR / "100-1_mask.png"
        if gt_img_path.exists():
            gt_image_rgb = np.array(Image.open(gt_img_path).convert("RGB"))
        if gt_mask_path.exists():
            gt_m = np.array(Image.open(gt_mask_path).convert("L"))
            gt_mask = (gt_m > 127).astype(np.float32)

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
        "Detection Algorithm", ["Ours (Daylight Physics)", "Baseline (Otsu on L)"]
    )
    max_delta_a = st.slider("Max a* Shift (|Δa|)", 1.0, 10.0, 3.5, 0.5)
    max_delta_b_pos = st.slider("Max b* Positive Shift (Δb_pos)", 0.5, 5.0, 2.5, 0.5)
    min_l_ratio = st.slider("Min Luminance Drop Ratio (L_out / L_in)", 1.05, 2.0, 1.15, 0.05)
    min_sep = st.slider("Min Separation (L units)", 5.0, 25.0, 10.0, 1.0)
    soft_sigma = st.slider("Mask Softening Sigma", 0.5, 5.0, 1.0, 0.5)

with st.sidebar.expander("Module C: Relighting Parameters", expanded=False):
    relight_mode = st.selectbox(
        "Relighting Method",
        ["Surface-Aware Log-Boundary Ratio (Ours)", "Global Multiplicative Gain (Baseline)"],
    )
    max_gain = st.slider("Max Amplification Gain", 1.5, 10.0, 8.0, 0.5)
    cluster_multisurface = st.checkbox("Enable Surface Clustering", value=True)
    b_weight = st.slider("Blue Skylight Compensation Weight", 0.0, 1.0, 0.5, 0.1)

with st.sidebar.expander("Module D: Penumbra Blending Parameters", expanded=False):
    enable_blend = st.checkbox("Enable Penumbra Blending", value=True)
    use_guided = st.checkbox("Use Guided Filter Edge Snapping", value=True)
    penumbra_rad = st.slider("Penumbra Transition Radius (px)", 1, 15, 4, 1)

# Main Application Tabs
if image_rgb is None:
    st.info("Please select a preset sample or upload an image from the sidebar.")
    st.stop()
    sys.exit(0)

tab_pipeline, tab_interactive, tab_evaluation, tab_about = st.tabs(
    ["🚀 End-to-End Pipeline", "✏️ Interactive Refinement", "📊 Quantitative Evaluation", "📖 Architecture & Novelty"]
)

# Compute Automatic Mask
h, w = image_rgb.shape[:2]
if detector_mode == "Ours (Daylight Physics)":
    auto_mask = detect(
        image_rgb,
        min_separation=min_sep,
        max_delta_a=max_delta_a,
        max_delta_b_pos=max_delta_b_pos,
        min_l_ratio=min_l_ratio,
        soft_sigma=soft_sigma,
    )
else:
    auto_mask = detect_otsu(image_rgb)

# TAB 1: PIPELINE RESULTS
with tab_pipeline:
    st.subheader("Automated Shadow Removal Stages")

    if preset_choice and ("1_dark_object" in preset_choice or "103-5" in preset_choice):
        st.info(
            "💡 **Live Demo Tip (Dark Object Rejection):** Open the sidebar under "
            "**Module A: Detection Parameters** and toggle the dropdown between "
            "**'Ours (Daylight Physics)'** and **'Baseline (Otsu on L)'** to demonstrate how Otsu flags dark non-shadow "
            "materials, whereas our daylight chromaticity check rejects them cleanly!"
        )

    # Compute Relit & Final Blended
    if relight_mode == "Surface-Aware Log-Boundary Ratio (Ours)":
        relit_img = remove(
            image_rgb,
            auto_mask,
            max_gain=max_gain,
            enable_clustering=cluster_multisurface,
            b_weight=b_weight,
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

        # Determine smart pre-aligned defaults based on the chosen preset
        default_action_idx = 0
        def_x_min, def_x_max = int(w * 0.1), int(w * 0.4)
        def_y_min, def_y_max = int(h * 0.1), int(h * 0.4)
        demo_hint = None

        if preset_choice and "100-6" in preset_choice:
            default_action_idx = 1  # Add Shadow
            def_x_min, def_x_max = 440, min(610, w)
            def_y_min, def_y_max = 30, min(295, h)
            demo_hint = "🎯 **Scene 100-6 Pre-aligned:** The red box is pre-positioned over the missed branch shadow. Notice how GrabCut immediately pulls in the entire missed shadow (~99.9% recovery)!"
        elif preset_choice and "102-1" in preset_choice:
            default_action_idx = 0  # Remove Shadow
            def_x_min, def_x_max = 50, min(590, w)
            def_y_min, def_y_max = 350, min(475, h)
            demo_hint = "🎯 **Scene 102-1 Pre-aligned:** The green box marks the dark asphalt road as LIT. Notice how GrabCut strips out the false detection on the street!"
        elif preset_choice and "4_specular" in preset_choice:
            default_action_idx = 1  # Add Shadow
            def_x_min, def_x_max = int(w * 0.2), int(w * 0.8)
            def_y_min, def_y_max = int(h * 0.25), int(h * 0.75)
            demo_hint = "🎯 **Specular Ambiguity Pre-aligned:** The red box guides GrabCut to recover the shadow band that was completely missed due to specular reflection!"

        if demo_hint:
            st.info(demo_hint)

        action_type = st.radio(
            "Correction Action",
            ["Mark False Positive as LIT (Remove Shadow)", "Mark Undetected as SHADOW (Add Shadow)"],
            index=default_action_idx,
        )
        sub_type = SCRIB_LIT if "LIT" in action_type else SCRIB_SHADOW

        # Coordinates for correction box
        x_min = st.slider("X Min (px)", 0, max(0, w - 10), min(def_x_min, max(0, w - 10)), 5)
        x_max = st.slider("X Max (px)", x_min + 5, w, min(max(x_min + 10, def_x_max), w), 5)
        y_min = st.slider("Y Min (px)", 0, max(0, h - 10), min(def_y_min, max(0, h - 10)), 5)
        y_max = st.slider("Y Max (px)", y_min + 5, h, min(max(y_min + 10, def_y_max), h), 5)
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
            refined_relit = remove(image_rgb, refined_m, max_gain=max_gain, b_weight=b_weight)
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
                {"Component": "A. Detection (Dark Object Rejection)", "Baseline (Otsu)": "IoU: 0.500 | 100% FP", "Proposed Pipeline": "IoU: 0.997 | 0.0% FP"},
                {"Component": "B. Refinement (Scribbles)", "Baseline (Otsu)": "IoU: 0.500 | 6.2% BER", "Proposed Pipeline": "IoU: 0.993 | 0.4% BER"},
                {"Component": "C. Relighting (Multi-Surface)", "Baseline (Global Gain)": "RMSE: 12.95 (Color Cast)", "Proposed Pipeline": "RMSE: 10.77 (Surface Matched)"},
                {"Component": "D. Blending (Penumbra Guided)", "Baseline (Unblended)": "PSNR: 22.98 dB", "Proposed Pipeline": "PSNR: 23.70 dB (Smooth)"},
            ]
        )

# TAB 4: ARCHITECTURE & NOVELTY
with tab_about:
    st.subheader("System Architecture & Proposed Novelties")
    st.markdown(
        """
        ### Pipeline Architecture:
        1. **Module A — Daylight Physics-Based Detection (`src/detect.py`)**:
           - Extracts $L$ channel from CIE LAB space.
           - Applies daylight chromaticity verification ($|\\Delta a| \\le 3.5$, $\\Delta b_{pos} \\le 2.5$) and luminance drop ($L_{out}/L_{in} \\ge 1.15$).
           - Postpones morphological closing to eliminate occluder bridging.
        2. **Module B — Interactive Scribble Refinement (`src/refine.py`)**:
           - Formulates shadow mask extraction as graph cuts with user scribbles enforcing hard boundary seeds.
           - Resolves ambiguous multi-depth shadow contours interactively.
        3. **Module C — Surface-Aware Radiometric Relighting (`src/remove.py`)**:
           - Relighting in linear radiometric RGB space (IEC 61966-2-1).
           - Paired inner and outer boundary strips along shadow contours: $\\log(\\gamma_c) = \\text{median}(\\log I_{\\text{outer}}) - \\text{median}(\\log I_{\\text{inner}})$.
           - Spatially coherent multi-surface clustering + adaptive global anchoring.
           - Additive LAB $b^*$ color temperature compensation for ambient blue skylight.
        4. **Module D — Penumbra-Aware Guided Blending (`src/blend.py`)**:
           - Symmetrically centered transition band with Grayscale Guided Filtering edge snapping.
           - Restores penumbra lighting without boundary fringes or overexposure (100% paired win rate).
        """
    )
