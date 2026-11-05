"""End-to-end shadow removal pipeline and command-line interface.

Usage:
    python -m src.pipeline input.jpg output.png [--mask gt_mask.png]
    python -m src.pipeline input.jpg output.png --save-strip
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from .blend import blend
from .detect import detect
from .refine import refine
from .remove import remove
from .utils import load_image, load_mask, overlay_mask, save_image


def run(
    img,
    mask=None,
    scribbles=None,
    use_blend=True,
    use_guided=True,
    max_gain=8.0,
    b_weight=0.5,
    enable_clustering=True,
    penumbra=None,
):
    """Full shadow detection and removal pipeline.

    Args:
        img: RGB uint8 input image.
        mask: optional float32 soft mask in [0, 1] to bypass automatic detection.
        scribbles: optional uint8 scribble map for interactive GrabCut correction.
        use_blend: whether to apply penumbra blending.
        use_guided: whether to use Grayscale Guided Filter edge snapping in blend.
        max_gain: upper bound on channel relighting amplification.
        b_weight: weight for additive blue ambient skylight compensation.
        enable_clustering: whether to perform multi-surface chromaticity clustering.
        penumbra: radius for the penumbra transition band.

    Returns:
        tuple of (final_image, mask, relit_image).
    """
    if mask is None:
        mask = detect(img)

    if scribbles is not None and np.any(scribbles):
        mask = refine(img, mask, scribbles)

    relit = remove(
        img,
        mask,
        max_gain=max_gain,
        enable_clustering=enable_clustering,
        b_weight=b_weight,
    )

    if use_blend:
        final = blend(img, relit, mask, penumbra=penumbra, use_guided=use_guided)
    else:
        final = relit

    return final, mask, relit


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EE604 Shadow Remover Tool")
    ap.add_argument("input", help="Path to input RGB image")
    ap.add_argument("output", help="Path to save shadow-free result")
    ap.add_argument("--mask", default=None, help="Optional path to ground-truth mask")
    ap.add_argument("--max-gain", type=float, default=8.0, help="Maximum relighting gain")
    ap.add_argument("--b-weight", type=float, default=0.5, help="Skylight color compensation weight")
    ap.add_argument("--no-blend", action="store_true", help="Disable penumbra blending")
    ap.add_argument("--no-guided", action="store_true", help="Disable guided filter edge alignment")
    ap.add_argument("--save-strip", action="store_true", help="Save 4-panel comparison strip")
    args = ap.parse_args()

    image = load_image(args.input)
    gt = load_mask(args.mask) if args.mask else None

    result, m, relit = run(
        image,
        mask=gt,
        use_blend=not args.no_blend,
        use_guided=not args.no_guided,
        max_gain=args.max_gain,
        b_weight=args.b_weight,
    )

    if args.save_strip:
        vis_mask = overlay_mask(image, m)
        panels = [image, vis_mask, relit, result]
        labels = ["input", "detected mask", "relit", "final"]
        for p, t in zip(panels, labels):
            cv2.putText(p, t, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        save_image(args.output, np.hstack(panels))
        print(f"Saved comparison strip to {args.output}")
    else:
        save_image(args.output, result)
        print(f"Saved {args.output}  (shadow area: {100 * (m > 0.5).mean():.1f}%)")
