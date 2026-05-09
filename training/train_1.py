from ultralytics import YOLO
import torch
import os

# -------------------------------
# 1. Dataset path
# -------------------------------
DATA_PATH = r"C:\Users\moomi\OneDrive\Desktop\Number-plate-detection-main\Vehicle-Registration-Plates-1\data.yaml"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset YAML not found at: {DATA_PATH}")

# -------------------------------
# 2. Check GPU
# -------------------------------
if torch.cuda.is_available():
    device = 0
    print("✅ Using GPU:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("⚠️ GPU not found, using CPU (will be slow)")

# -------------------------------
# 3. Load model
# -------------------------------
model = YOLO("yolov8n.pt")   # lightweight model (best for your system)

# -------------------------------
# 4. Train (optimized settings)
# -------------------------------
model.train(
    data=DATA_PATH,
    epochs=20,        # good balance
    imgsz=320,        # faster but still decent accuracy
    batch=2,          # safe for 6GB GPU
    workers=2,        # avoid RAM overload
    device=device,
    name="plate_detector"
)

# -------------------------------
# 5. Validate model
# -------------------------------
metrics = model.val()

# -------------------------------
# 6. Test prediction
# -------------------------------
# Replace 'test.jpg' with your image
test_image = r"C:\Users\moomi\OneDrive\Desktop\Number-plate-detection-main\test.jpeg"

if os.path.exists(test_image):
    model.predict(
        source=test_image,
        show=True,
        save=True
    )
else:
    print("⚠️ Test image not found, skipping prediction step")

# -------------------------------
# 7. Done
# -------------------------------
print("🎉 Training Completed!")
print("Check results in: runs/detect/number_plate_detector/")