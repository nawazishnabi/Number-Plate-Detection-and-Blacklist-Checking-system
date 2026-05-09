"""
main.py
-------
Entry point — just launches the app.
All logic lives in core/ and ui/.
"""

import os
import sys
import tkinter as tk

# ── Dependency check ──────────────────────────────────────────────────
REQUIRED = {
    "cv2":          "opencv-python",
    "pytesseract":  "pytesseract",
    "pandas":       "pandas",
    "ultralytics":  "ultralytics",
    "easyocr":      "easyocr",
}

missing = []
for module, package in REQUIRED.items():
    try:
        __import__(module)
    except ImportError:
        missing.append(package)

if missing:
    print("Missing dependencies — install with:\n")
    for pkg in missing:
        print(f"  pip install {pkg}")
    sys.exit(1)

# ── Tesseract path (Windows) ──────────────────────────────────────────
import pytesseract
_tess = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(_tess):
    pytesseract.pytesseract.tesseract_cmd = _tess

# ── Launch ────────────────────────────────────────────────────────────
from ui.app import LicensePlateApp

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app  = LicensePlateApp(root)

        # Centre window on screen
        root.update_idletasks()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        ww, wh = root.winfo_width(),       root.winfo_height()
        root.geometry(f"+{(sw-ww)//2}+{(sh-wh)//2}")

        print("License Plate Recognition System started.")
        root.mainloop()

    except Exception as e:
        print(f"Failed to start: {e}")
        input("Press Enter to exit…")
