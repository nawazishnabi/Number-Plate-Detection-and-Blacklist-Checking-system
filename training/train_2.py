"""
train.py
---------
Fine-tunes YOLOv8 on your custom Indian plate dataset.
Run AFTER prepare_custom_dataset.py
"""

from ultralytics import YOLO
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_YAML  = r"C:\Users\HP\Desktop\Projects\License_PD\combined_dataset\data.yaml"

PRETRAINED = r"C:\Users\HP\Desktop\Projects\License_PD\runs\detect\plate_detector\weights\best.pt"

FALLBACK   = "yolov8n.pt"

EPOCHS  = 50
BATCH   = 4
DEVICE  = "cpu"  
# ──────────────────────────────────────────────────────────────────────────────

if not os.path.exists(DATA_YAML):
    print("[ERROR] data.yaml not found. Run prepare_custom_dataset.py first.")
    exit(1)

# Pick best available starting weights
if os.path.exists(PRETRAINED):
    start_weights = PRETRAINED
    print(f"[INFO] Using your Roboflow-trained weights as base (transfer learning)")
else:
    start_weights = FALLBACK
    print(f"[INFO] Pretrained weights not found, starting from {FALLBACK}")

print(f"[INFO] Data   : {DATA_YAML}")
print(f"[INFO] Epochs : {EPOCHS}  Batch: {BATCH}  Device: {DEVICE}")
print()

model = YOLO(start_weights)

model.train(
    data     = DATA_YAML,
    epochs   = EPOCHS,
    imgsz    = 640,
    batch    = BATCH,
    device   = DEVICE,
    workers  = 0,
    name     = "custom_indian_plates",
    patience = 15,
    save     = True,
    plots    = True,
)

print("\n✅ Training complete!")
print("   Best model: runs/detect/custom_indian_plates/weights/best.pt")