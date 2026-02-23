# 😷 Face Mask Detection — MobileNetV2 Transfer Learning

A complete computer-vision pipeline that detects whether a person is wearing a face mask, built with **TensorFlow / Keras** and **MobileNetV2 transfer learning**.  
Includes a **real-time webcam demo**.

---

## Project Structure

```
Fake mask detection/
│
├── face_mask_detection.ipynb   ← Main notebook (run this!)
├── train_mask_detector.py      ← Standalone training script
├── webcam_demo.py              ← Real-time webcam detection
├── predict_image.py            ← Single-image prediction CLI
├── prepare_dataset.py          ← Download real dataset
├── make_synthetic_dataset.py   ← Generate synthetic dataset (offline)
│
├── dataset/
│   ├── with_mask/              ← Training images (mask worn)
│   └── without_mask/           ← Training images (no mask)
│
├── mask_detector.keras         ← Saved model (after training)
├── training_plot.png           ← Accuracy/Loss curves
└── confusion_matrix.png        ← Evaluation heatmap
```

---

## Quick Start

### 1 — Install dependencies
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn pillow seaborn
```

### 2 — Open the notebook
```bash
jupyter notebook face_mask_detection.ipynb
```
Run all cells top-to-bottom.  
The notebook auto-generates a synthetic dataset so it works **without internet access**.

---

## Using the Real Dataset

For higher accuracy, use the real dataset (~3,800 images):

1. Clone https://github.com/chandrikadeb7/Face-Mask-Detection  
2. Copy `dataset/with_mask/` and `dataset/without_mask/` into this folder  
3. Re-run the notebook

---

## Model Architecture

```
Input (224×224×3)
      │
MobileNetV2  ← ImageNet pre-trained, FROZEN
      │
AveragePooling2D (7×7)
      │
Flatten
      │
Dense (128, ReLU)
      │
Dropout (0.5)
      │
Dense (2, Softmax)   → with_mask  /  without_mask
```

| Setting | Value |
|---------|-------|
| Base model | MobileNetV2 (ImageNet) |
| Learning rate | 1e-4 (Adam) |
| Batch size | 32 |
| Epochs | Up to 20 (EarlyStopping) |
| Loss | Binary Cross-Entropy |

---

## Webcam Demo

After training:
```bash
python webcam_demo.py
```
Press **Q** to quit.

---

## Single Image Prediction

```bash
python predict_image.py --image path/to/photo.jpg
```
Output is saved as `predicted_output.jpg`.

---

## Windows Long-Path Note

TensorFlow has deeply-nested internal files. If installation fails with a path-length error on Windows, enable long paths:

```
Settings → System → About → Advanced system settings → 
Environment Variables → Enable Win32 long paths
```
Or run in PowerShell **as Administrator**:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
Then re-install TensorFlow.
