"""
webcam_demo.py
--------------
Real-time Face-Mask Detection via webcam.

Prerequisites:
  • mask_detector.pth  (produced by train_mask_detector.py)
  • OpenCV (pip install opencv-python)

Controls:  Q = quit
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image

MODEL_PATH  = "mask_detector.pth"
IMG_SIZE    = 224
CATEGORIES  = ["with_mask", "without_mask"]
LABELS      = ["Mask ✅", "No Mask ❌"]
COLORS      = [(0, 200, 0), (0, 0, 220)]   # BGR: green / red
CONF_MIN    = 0.60
DEVICE      = torch.device("cpu")

# ─── Load Model ───────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"'{MODEL_PATH}' not found. Run train_mask_detector.py first.")

print("[INFO] Loading model …")
weights = MobileNet_V2_Weights.IMAGENET1K_V1
model   = mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(128, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ─── Webcam Loop ──────────────────────────────────────────────────────────────
print("[INFO] Starting webcam … Press Q to quit.")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(roi)
            inp = transform(pil).unsqueeze(0).to(DEVICE)

            out  = model(inp)
            prob = torch.softmax(out, dim=1)[0]
            idx  = torch.argmax(prob).item()
            conf = prob[idx].item()

            if conf < CONF_MIN:
                continue

            lbl   = f"{LABELS[idx]} ({conf*100:.1f}%)"
            color = COLORS[idx]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
            cv2.putText(frame, lbl, (x+4, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2)

        cv2.putText(frame, "Face Mask Detector  |  Q to quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
        cv2.imshow("Face Mask Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Done.")
