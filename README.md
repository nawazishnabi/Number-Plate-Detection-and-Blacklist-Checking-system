# 🚗 License Plate Recognition & Blacklist Check System

A B.Tech Minor Project built with Python, YOLOv8, Tesseract OCR, and EasyOCR.  
Detects Indian vehicle license plates from images, reads the plate number using multiple OCR engines, and checks it against a blacklist database.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Blacklist Management](#blacklist-management)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)

---

## Overview

This system is designed to:
1. Accept an image of a vehicle
2. Detect the license plate region using a custom-trained YOLOv8 model
3. Preprocess the plate image for better OCR accuracy
4. Extract the plate number using Tesseract + EasyOCR
5. Cross-check the result against a blacklist CSV
6. Display results in a clean GUI with color-coded status

---

## ✨ Features

- **YOLOv8-based plate detection** — custom model trained on Indian vehicle plates
- **Dual OCR engine** — Tesseract (4 PSM modes) + EasyOCR for maximum accuracy
- **Smart preprocessing pipeline** — CLAHE → Sharpen → Bilateral → Threshold → Morphology
- **Auto dark-plate inversion** — handles blue/black background plates automatically
- **Position-aware OCR correction** — fixes common misreadings (O→0, I→1 etc.) only at digit positions
- **Indian plate format validation** — regex pattern `AA##AA####`
- **Exact + fuzzy blacklist matching** — 90% similarity threshold to catch OCR variants
- **Debug view** — shows original crop, processed image, OCR result, and image quality score
- **Contour fallback** — works even if YOLO misses the plate

---

## 📁 Project Structure

```
LICENSE_PD/
│
├── main.py                        # Entry point — launches the app
├── requirements.txt               # All Python dependencies
├── README.md                      # This file
│
├── config/
│   ├── __init__.py
│   └── settings.py                # All constants and tunable parameters
│
├── core/
│   ├── __init__.py
│   ├── detector.py                # YOLO detection + contour fallback
│   ├── preprocessor.py            # Image preprocessing pipeline
│   ├── ocr_engine.py              # OCR, error correction, format validation
│   └── blacklist.py               # Blacklist load, exact match, fuzzy match
│
├── ui/
│   ├── __init__.py
│   ├── app.py                     # Main Tkinter application window
│   └── debug_window.py            # Debug popup window
│
├── models/
│   └── custom.pt                  # Trained YOLOv8 model weights
│   └── best.pt  
│   └── yolov8.pt.pt   
├── data/
│   └── blacklist.csv              # Blacklisted plate numbers with reasons
│
├── logs/                          # Training logs
└── runs/                          # YOLO training run outputs
```

---

## 💻 System Requirements

| Component        | Minimum                        |
|------------------|-------------------------------|
| OS               | Windows 10 / Ubuntu 20.04+    |
| Python           | 3.9 or higher                 |
| RAM              | 4 GB (8 GB recommended)       |
| GPU              | Optional (CPU works fine)     |
| Tesseract OCR    | Version 5.x                   |
| Disk Space       | ~3 GB (for models + deps)     |

---

## ⚙️ Installation

### Step 1 — Install Tesseract OCR

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default path: `C:\Program Files\Tesseract-OCR\`
3. Verify: open CMD and run `tesseract --version`

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
tesseract --version
```

---

### Step 2 — Clone / Download the Project

```bash
# If using git
git clone <your-repo-url>
cd LICENSE_PD

# Or just extract the ZIP and open the folder
```

---

### Step 3 — Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

---

### Step 4 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` — image processing
- `pytesseract` — Tesseract OCR wrapper
- `ultralytics` — YOLOv8
- `easyocr` — deep learning OCR
- `pandas` — blacklist CSV handling
- `Pillow` — image display in Tkinter
- `numpy` — array operations

> ⚠️ First run will download EasyOCR models (~200MB). Make sure you have internet.

---

### Step 5 — Add Your Model

Place your trained YOLOv8 model at:
```
LICENSE_PD/models/custom.pt
```

If you want to use a different path, update `MODEL_PATH` in `config/settings.py`.

---

## ▶️ How to Run

```bash
python main.py
```

The GUI will open. Then:
1. Click **Open Image** → select a vehicle photo
2. Click **Detect Plates** → system detects and reads the plate
3. Results appear in the table (GREEN = clean, RED = blacklisted, ORANGE = unreadable)
4. Click **Debug View** → see the raw crop, processed image, and OCR details

---

## 🔍 How It Works

### 1. Plate Detection (`core/detector.py`)

```
Input Image
    │
    ▼
YOLOv8 Model (custom.pt)
    │
    ├── Detects bounding boxes with conf > 0.3
    ├── Filters by aspect ratio (1.5 to 6.0) and size
    │
    └── If nothing found → Contour Fallback
            │
            ├── Grayscale → Bilateral Filter → Adaptive Threshold
            ├── Find contours, filter by shape
            └── Check Sobel edge density > 0.1
```

### 2. Preprocessing (`core/preprocessor.py`)

The correct order is critical — wrong order causes noisy output:

```
BGR plate crop
    │
    ├── 1. Convert to Grayscale
    ├── 2. Upscale to min 400×120 px  (using max scale factor)
    ├── 3. Center crop 20%–80%        (remove grille/bumper lines)
    ├── 4. CLAHE                      (contrast enhancement)
    ├── 5. Sharpen kernel             (enhance character edges)
    ├── 6. Bilateral filter           (denoise, keep edges)
    ├── 7. Adaptive Threshold         (convert to binary)
    ├── 8. Morphological close+open   (clean up binary noise)
    └── 9. Auto-invert if dark bg     (blue/black plates)
```

### 3. OCR (`core/ocr_engine.py`)

Three approaches run in parallel, best confidence wins:

| Approach | Method | Notes |
|---|---|---|
| A | Tesseract × 4 PSM modes | PSM 7, 8, 11, 13 |
| B | EasyOCR | Deep learning OCR |
| C | Otsu threshold + Tesseract | Simple fallback |

After OCR, **position-aware correction** is applied:
- Positions 0–1 → kept as letters (state code)
- Positions 2–3 → digit corrections applied (O→0, I→1 etc.)
- Last 4 chars → digit corrections applied (serial number)

### 4. Blacklist Check (`core/blacklist.py`)

```
Plate text
    │
    ├── Clean: remove non-alphanumeric
    ├── Exact match in CSV → BLACKLISTED
    └── Fuzzy match (SequenceMatcher > 90%) → BLACKLISTED
```

---

## ⚙️ Configuration

All tunable parameters are in `config/settings.py`. No need to touch any other file.

| Parameter | Default | Description |
|---|---|---|
| `YOLO_CONFIDENCE` | `0.3` | Min YOLO detection confidence |
| `PLATE_PADDING` | `5` | Pixels added around YOLO box |
| `MIN_PLATE_ASPECT` | `1.5` | Min width:height ratio |
| `MAX_PLATE_ASPECT` | `6.0` | Max width:height ratio |
| `TARGET_OCR_WIDTH` | `400` | Min plate width before OCR |
| `TARGET_OCR_HEIGHT` | `120` | Min plate height before OCR |
| `CROP_TOP_RATIO` | `0.20` | Top strip to remove (grille noise) |
| `CROP_BOTTOM_RATIO` | `0.80` | Bottom limit of center crop |
| `DARK_BG_THRESHOLD` | `0.60` | Auto-invert if >60% pixels black |
| `TESSERACT_CONF_MIN` | `60` | Min Tesseract word confidence |
| `EASYOCR_CONF_MIN` | `0.6` | Min EasyOCR confidence |
| `FUZZY_MATCH_THRESHOLD` | `0.90` | Blacklist fuzzy match threshold |

---

## 📋 Blacklist Management

The blacklist is stored at `data/blacklist.csv` with two columns:

```csv
PlateNumber,Reason
JK08D4356,Traffic Violation
JK01AB1234,Stolen Vehicle
JK14SU3550,Duplicate Number Plate
JK08XP1434,Unregistered Vehicle
```

**To add a plate:**
- Open `data/blacklist.csv` in Excel or Notepad
- Add a new row: `JK09XY1234,Your Reason Here`
- Save the file — changes take effect on next app launch

**Format rules:**
- PlateNumber: uppercase, no spaces or hyphens (e.g. `JK08D4356` not `JK-08-D-4356`)
- Reason: free text, kept short

---

## ⚠️ Known Limitations

1. **Small plates (<100px wide)** — OCR accuracy drops significantly below 100px
2. **Tilted plates** — YOLO handles mild tilt but extreme angles reduce accuracy
3. **Night/low-light images** — preprocessing helps but very dark images may fail
4. **Two-line plates** — some old Indian plates with two-row format may not match the regex
5. **Non-Indian plates** — format validation is tuned for Indian plates only
6. **First run slow** — EasyOCR downloads ~200MB of models on first launch

---

## 🛠️ Troubleshooting

**App doesn't start / import errors**
```bash
pip install -r requirements.txt
```

**"Tesseract not found" error**
- Windows: Install from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt install tesseract-ocr`
- Verify path in `main.py` matches your installation

**Model not loading**
- Check that `models/custom.pt` exists
- Update `MODEL_PATH` in `config/settings.py` if stored elsewhere

**All plates showing UNREADABLE**
- Open Debug View and check "Processed for OCR" image
- If it looks like black noise → preprocessing pipeline issue
- If it looks clean but OCR fails → lower `TESSERACT_CONF_MIN` to `40` in settings

**Plate detected but wrong text**
- Check Debug View → Original Detected Region
- If wrong region detected → adjust `MIN_PLATE_ASPECT` / `MAX_PLATE_HEIGHT` in settings
- If right region but wrong text → the OCR correction maps may need tuning

---

## 🧰 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core language |
| OpenCV | 4.8+ | Image processing |
| Ultralytics YOLOv8 | 8.0+ | Plate detection |
| Tesseract / pytesseract | 5.x / 0.3+ | OCR engine 1 |
| EasyOCR | 1.7+ | OCR engine 2 |
| Pandas | 2.0+ | Blacklist CSV |
| Pillow | 10.0+ | Tkinter image display |
| Tkinter | built-in | GUI framework |

---

## 👨‍💻 Authors(Group Members)

**Kaiser Mohiuddin(Group Leader)**  
**Nawazish Nabi,** 
**Abdul Momin,**
**Auzair Yousf**     
B.Tech Computer Science Engineering — 6th Semester  
Minor Project — License Plate Recognition & Blacklist Check System

---

## 📄 License

This project is submitted as an academic minor project.  
For educational use only.
