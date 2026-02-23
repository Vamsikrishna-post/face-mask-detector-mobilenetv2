"""
predict_image.py
----------------
Run face-mask detection on a single image.

Usage:
  python predict_image.py --image path/to/photo.jpg
"""

import argparse, os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.models                    import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image       import img_to_array, load_img

MODEL_PATH = "mask_detector.keras"
IMG_SIZE   = (224, 224)
LABELS     = ["With Mask ✅", "No Mask ❌"]
COLORS     = [(0, 200, 0), (0, 0, 220)]


def predict_single_image(image_path: str):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train first.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model        = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    frame = cv2.imread(image_path)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    print(f"[INFO] Detected {len(faces)} face(s).")

    for (x, y, w, h) in faces:
        roi   = frame[y:y+h, x:x+w]
        roi   = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi   = cv2.resize(roi, IMG_SIZE)
        roi   = img_to_array(roi)
        roi   = preprocess_input(roi)
        roi   = np.expand_dims(roi, axis=0)

        preds     = model.predict(roi, verbose=0)[0]
        idx       = np.argmax(preds)
        label     = f"{LABELS[idx]} ({preds[idx]*100:.1f}%)"
        color     = COLORS[idx]

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
        cv2.putText(frame, label, (x+4, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        print(f"       {label}")

    out_path = "predicted_output.jpg"
    cv2.imwrite(out_path, frame)
    print(f"[INFO] Result saved → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to input image")
    args = ap.parse_args()
    predict_single_image(args.image)
