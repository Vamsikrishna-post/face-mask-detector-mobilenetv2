"""
make_synthetic_dataset.py
--------------------------
Creates a small synthetic dataset of face-like images for quick pipeline
testing when the real dataset cannot be downloaded.

Generates 200 with_mask + 200 without_mask images (100x100 px each).

After running this script the folder layout will be:
  dataset/
    with_mask/       <- faces with a coloured rectangle over the lower half
    without_mask/    <- plain face-like circles
"""

import os
import numpy as np
from PIL import Image, ImageDraw

DEST     = "dataset"
N        = 200          # images per class
IMG_SIZE = (100, 100)

os.makedirs(os.path.join(DEST, "with_mask"),    exist_ok=True)
os.makedirs(os.path.join(DEST, "without_mask"), exist_ok=True)


def make_face(mask: bool, idx: int):
    rng = np.random.default_rng(seed=idx)

    # ── background ────────────────────────────────────────────────────────────
    bg = tuple(rng.integers(180, 255, 3).tolist())
    img  = Image.new("RGB", IMG_SIZE, bg)
    draw = ImageDraw.Draw(img)

    # ── skin colour ───────────────────────────────────────────────────────────
    skin = (
        rng.integers(180, 220),
        rng.integers(140, 180),
        rng.integers(100, 140),
    )

    # ── face oval ─────────────────────────────────────────────────────────────
    fx, fy = 20 + rng.integers(-5, 5), 15 + rng.integers(-5, 5)
    fw, fh = 60 + rng.integers(-5, 5), 70 + rng.integers(-5, 5)
    draw.ellipse([fx, fy, fx+fw, fy+fh], fill=tuple(int(c) for c in skin))

    # ── eyes ──────────────────────────────────────────────────────────────────
    ey = fy + int(fh * 0.35)
    for ex in [fx + int(fw*0.25), fx + int(fw*0.65)]:
        draw.ellipse([ex-4, ey-3, ex+4, ey+3], fill=(40, 40, 40))

    # ── nose ──────────────────────────────────────────────────────────────────
    nx = fx + fw//2
    ny = fy + int(fh * 0.55)
    draw.ellipse([nx-3, ny-2, nx+3, ny+3], fill=(int(skin[0])-30, int(skin[1])-30, int(skin[2])-30))

    if mask:
        # ── mask rectangle (covers nose + mouth) ──────────────────────────────
        mx1 = fx + int(fw*0.05)
        my1 = fy + int(fh*0.50)
        mx2 = fx + int(fw*0.95)
        my2 = fy + int(fh*0.92)
        mask_color = tuple(rng.integers(0, 255, 3).tolist())
        draw.rectangle([mx1, my1, mx2, my2], fill=tuple(int(c) for c in mask_color))
        # mask texture lines
        for row in range(my1+4, my2-2, 6):
            draw.line([(mx1+2, row), (mx2-2, row)],
                      fill=(min(255, mask_color[0]+40),
                            min(255, mask_color[1]+40),
                            min(255, mask_color[2]+40)), width=1)
    else:
        # ── mouth (smile) ─────────────────────────────────────────────────────
        smx = fx + int(fw*0.25)
        smy = fy + int(fh*0.70)
        draw.arc([smx, smy, fx+int(fw*0.75), smy+int(fh*0.18)],
                 start=10, end=170, fill=(180, 80, 80), width=2)

    # ── light noise ───────────────────────────────────────────────────────────
    arr  = np.array(img, dtype=np.float32)
    arr += rng.normal(0, 5, arr.shape)
    arr  = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


print("[INFO] Generating synthetic dataset …")

for i in range(N):
    img = make_face(mask=True, idx=i)
    img.save(os.path.join(DEST, "with_mask", f"mask_{i:04d}.png"))

for i in range(N):
    img = make_face(mask=False, idx=i + N)
    img.save(os.path.join(DEST, "without_mask", f"nomask_{i:04d}.png"))

print(f"[INFO] Created {N} with_mask + {N} without_mask images in '{DEST}/'")
print("[INFO] Done. You can now run:  python train_mask_detector.py")
