"""Quick visual check of the pipeline on one image.

Usage (from the repo root):
    python try_it.py path/to/image.jpg
Saves <name>_result.png in the same folder as try_it.py: a 4-panel strip
[input | detected mask overlay | relit (no blend) | final].
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from src.pipeline import run
from src.utils import load_image, overlay_mask, save_image

if len(sys.argv) < 2:
    sys.exit("Usage: python try_it.py path/to/image.jpg")

img = load_image(sys.argv[1])
final, mask, relit = run(img)

print(f"Image size: {img.shape[1]}x{img.shape[0]}")
print(f"Detected shadow area: {100 * (mask > 0.5).mean():.1f}% of the image")

mask_vis = overlay_mask(img, mask)
panels = [img, mask_vis, relit, final]
labels = ["input", "detected mask", "relit (no blend)", "final"]
for p, t in zip(panels, labels):
    cv2.putText(p, t, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

out_dir = Path(__file__).parent / "outputs"
out_dir.mkdir(exist_ok=True)
out = out_dir / f"{Path(sys.argv[1]).stem}_result.png"
save_image(out, np.hstack(panels))
print(f"Saved {out}")
