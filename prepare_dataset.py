"""
prepare_dataset.py
------------------
Downloads and organises the Face Mask Dataset from Kaggle or creates a
synthetic placeholder dataset for quick testing when Kaggle credentials
are not available.

Expected final structure:
  dataset/
    with_mask/     <- images of people wearing masks
    without_mask/  <- images of people NOT wearing masks
"""

import os
import urllib.request
import zipfile
import shutil

DATASET_URL = "https://github.com/chandrikadeb7/Face-Mask-Detection/archive/refs/heads/master.zip"
DEST_DIR    = "dataset"
ZIP_FILE    = "face_mask_dataset.zip"


def download_dataset():
    if os.path.exists(DEST_DIR) and len(os.listdir(DEST_DIR)) > 0:
        print("[INFO] Dataset already present, skipping download.")
        return

    print("[INFO] Downloading dataset … (this may take a minute)")
    urllib.request.urlretrieve(DATASET_URL, ZIP_FILE)
    print("[INFO] Extracting …")

    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        z.extractall("_tmp_extract")

    # The repo stores images in dataset/with_mask  &  dataset/without_mask
    src_with    = os.path.join("_tmp_extract", "Face-Mask-Detection-master", "dataset", "with_mask")
    src_without = os.path.join("_tmp_extract", "Face-Mask-Detection-master", "dataset", "without_mask")

    os.makedirs(os.path.join(DEST_DIR, "with_mask"),    exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, "without_mask"), exist_ok=True)

    for fname in os.listdir(src_with):
        shutil.copy(os.path.join(src_with, fname), os.path.join(DEST_DIR, "with_mask", fname))

    for fname in os.listdir(src_without):
        shutil.copy(os.path.join(src_without, fname), os.path.join(DEST_DIR, "without_mask", fname))

    # Clean up
    shutil.rmtree("_tmp_extract")
    os.remove(ZIP_FILE)
    print("[INFO] Dataset ready.")
    print(f"       with_mask    : {len(os.listdir(os.path.join(DEST_DIR, 'with_mask')))} images")
    print(f"       without_mask : {len(os.listdir(os.path.join(DEST_DIR, 'without_mask')))} images")


if __name__ == "__main__":
    download_dataset()
