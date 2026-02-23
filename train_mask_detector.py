"""
train_mask_detector.py
-----------------------
Face-Mask Detector — PyTorch + MobileNetV2 Transfer Learning

Steps:
  1. Load dataset/with_mask + dataset/without_mask
  2. Augment training data
  3. Fine-tune MobileNetV2 head
  4. Evaluate and save model + plots

Outputs:
  mask_detector.pth    – trained model weights
  training_plot.png    – accuracy / loss curves
  confusion_matrix.png – confusion matrix heatmap
"""

import os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.metrics      import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data     import Dataset, DataLoader
from torch.optim          import Adam
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as T
from torchvision.models    import mobilenet_v2, MobileNet_V2_Weights

# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset"
IMG_SIZE    = 224
BATCH_SIZE  = 32
LR          = 1e-4
EPOCHS      = 20
PATIENCE    = 5
MODEL_PATH  = "mask_detector.pth"
CATEGORIES  = ["with_mask", "without_mask"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device : {DEVICE}")

# ─── Dataset ──────────────────────────────────────────────────────────────────
class MaskDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ─── 1. Load Image Paths ──────────────────────────────────────────────────────
print("[INFO] Loading image paths …")
paths, labels = [], []
for cls_idx, cat in enumerate(CATEGORIES):
    folder = os.path.join(DATASET_DIR, cat)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"'{folder}' not found. Run make_synthetic_dataset.py first.")
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath):
            paths.append(fpath)
            labels.append(cls_idx)

print(f"       Total images  : {len(paths)}")
for i, cat in enumerate(CATEGORIES):
    print(f"       {cat:15s}: {labels.count(i)}")

# ─── 2. Train / Test Split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    paths, labels, test_size=0.20, stratify=labels, random_state=42
)

# ─── 3. Transforms ────────────────────────────────────────────────────────────
train_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])
val_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

train_ds = MaskDataset(X_train, y_train, transform=train_tf)
val_ds   = MaskDataset(X_test,  y_test,  transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"[INFO] Train batches : {len(train_loader)}  |  Val batches : {len(val_loader)}")

# ─── 4. Build Model ───────────────────────────────────────────────────────────
print("[INFO] Building MobileNetV2 model …")

weights = MobileNet_V2_Weights.IMAGENET1K_V1
model   = mobilenet_v2(weights=weights)

# Freeze all base layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head for 2-class output
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(128, 2)
)

model = model.to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"       Total params     : {total:,}")
print(f"       Trainable params : {trainable:,}  (head only)")

# ─── 5. Loss, Optimiser, Scheduler ───────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# ─── 6. Train ─────────────────────────────────────────────────────────────────
print("[INFO] Training …")

history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
best_val_acc     = 0.0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ── Train phase ──────────────────────────────────────────────────────────
    model.train()
    running_loss, correct, total_samples = 0.0, 0, 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()

        running_loss   += loss.item() * imgs.size(0)
        _, preds        = torch.max(out, 1)
        correct        += (preds == lbls).sum().item()
        total_samples  += imgs.size(0)

    t_loss = running_loss / total_samples
    t_acc  = correct      / total_samples

    # ── Validation phase ─────────────────────────────────────────────────────
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, lbls)
            v_loss    += loss.item() * imgs.size(0)
            _, preds   = torch.max(out, 1)
            v_correct += (preds == lbls).sum().item()
            v_total   += imgs.size(0)

    v_loss /= v_total
    v_acc   = v_correct / v_total

    history["train_acc"].append(t_acc)
    history["val_acc"].append(v_acc)
    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)

    elapsed = time.time() - t0
    print(f"  Epoch {epoch:02d}/{EPOCHS}  "
          f"train_loss={t_loss:.4f}  train_acc={t_acc:.4f}  "
          f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}  "
          f"({elapsed:.1f}s)")

    # ── Save best ────────────────────────────────────────────────────────────
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"             ✅ New best val_acc={best_val_acc:.4f}  → saved {MODEL_PATH}")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    scheduler.step()

# ─── 7. Evaluate ──────────────────────────────────────────────────────────────
print("\n[INFO] Evaluating best model …")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for imgs, lbls in val_loader:
        out    = model(imgs.to(DEVICE))
        _, p   = torch.max(out, 1)
        all_preds.extend(p.cpu().numpy())
        all_true.extend(lbls.numpy())

print("\nClassification Report:")
print(classification_report(all_true, all_preds, target_names=CATEGORIES))

# ─── 8. Confusion Matrix ──────────────────────────────────────────────────────
cm = confusion_matrix(all_true, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax, annot_kws={"size": 14})
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix – Face Mask Detector", fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("[INFO] Saved → confusion_matrix.png")

# ─── 9. Training Curves ───────────────────────────────────────────────────────
epochs_ran = len(history["train_acc"])
x          = range(1, epochs_ran + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("MobileNetV2 Transfer Learning – Face Mask Detection", fontsize=14, fontweight="bold")

ax1.plot(x, history["train_acc"], label="Train", linewidth=2, color="#4C72B0")
ax1.plot(x, history["val_acc"],   label="Val",   linewidth=2, color="#DD8452", linestyle="--")
ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)
ax1.set_ylim([0, 1.05])

ax2.plot(x, history["train_loss"], label="Train", linewidth=2, color="#4C72B0")
ax2.plot(x, history["val_loss"],   label="Val",   linewidth=2, color="#DD8452", linestyle="--")
ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_plot.png", dpi=150)
plt.close()
print("[INFO] Saved → training_plot.png")
print(f"\n✅ Training complete!  Best val_acc = {best_val_acc:.4f}")
print(f"   Model saved → {MODEL_PATH}")
