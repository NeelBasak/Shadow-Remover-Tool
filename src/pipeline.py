"""End-to-end pipeline and a small CLI.

Usage:
    python -m src.pipeline input.jpg output.png [--mask gt_mask.png]
"""
import argparse

import numpy as np

from .blend import blend
from .detect import detect
from .refine import refine
from .remove import remove
from .utils import load_image, load_mask, save_image


def run(img, mask=None, scribbles=None, use_blend=True, use_guided=True):
    """Full pipeline. Returns (final_image, mask, relit_image).

    Pass `mask` to skip automatic detection (e.g. ground-truth masks in eval).
    """
    if mask is None:
        mask = detect(img)
    mask = refine(img, mask, scribbles)
    relit = remove(img, mask)
    final = blend(img, relit, mask, use_guided=use_guided) if use_blend else relit
    return final, mask, relit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--mask", default=None, help="optional ground-truth mask")
    args = ap.parse_args()

    image = load_image(args.input)
    gt = load_mask(args.mask) if args.mask else None
    result, m, _ = run(image, mask=gt)
    save_image(args.output, result)
    print(f"Saved {args.output}  (shadow area: {100 * (m > 0.5).mean():.1f}%)")
