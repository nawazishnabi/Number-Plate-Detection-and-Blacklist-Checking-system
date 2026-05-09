"""
detector.py
-----------
Handles all plate detection:
  - YOLO-based plate localisation  (detect_plates_yolo)
  - Contour-based fallback          (find_plate_regions)
  - Combined pipeline               (detect_plates)
"""

import cv2
import numpy as np
from ultralytics import YOLO

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MODEL_PATH, YOLO_CONFIDENCE, PLATE_PADDING,
    MIN_PLATE_ASPECT, MAX_PLATE_ASPECT, MAX_PLATE_WIDTH, MAX_PLATE_HEIGHT,
    CONTOUR_MIN_ASPECT, CONTOUR_MAX_ASPECT, CONTOUR_MIN_AREA,
    CONTOUR_MIN_W, CONTOUR_MIN_H, CONTOUR_MAX_H, SOBEL_DENSITY_MIN,
)


class PlateDetector:
    def __init__(self):
        try:
            self.model = YOLO(MODEL_PATH)
            print(f"[Detector] Model loaded. Classes: {self.model.names}")
        except Exception as e:
            print(f"[Detector] ERROR loading model: {e}")
            self.model = None

    # ── Public API ────────────────────────────────────────────────────

    def detect_plates(self, image_path: str) -> list:
        """
        Main entry point.
        Returns a list of BGR numpy arrays, each being one plate crop.
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Detector] Could not read image: {image_path}")
            return []

        # 1. Try YOLO first
        plates = self._detect_plates_yolo(img)

        # 2. Fallback: contour search on full image
        if not plates:
            print("[Detector] YOLO found nothing — trying contour fallback")
            plates = self._find_plate_regions(img)

        print(f"[Detector] Found {len(plates)} plate region(s)")
        return plates

    # ── Private helpers ───────────────────────────────────────────────

    def _detect_plates_yolo(self, img: np.ndarray) -> list:
        if self.model is None:
            return []

        try:
            results = self.model(img, conf=YOLO_CONFIDENCE)
        except Exception as e:
            print(f"[Detector] YOLO inference error: {e}")
            return []

        plates = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1 = max(0, x1 - PLATE_PADDING)
                y1 = max(0, y1 - PLATE_PADDING)
                x2 = min(img.shape[1], x2 + PLATE_PADDING)
                y2 = min(img.shape[0], y2 + PLATE_PADDING)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                reg_h, reg_w = crop.shape[:2]
                aspect = reg_w / float(reg_h) if reg_h > 0 else 0

                if (MIN_PLATE_ASPECT <= aspect <= MAX_PLATE_ASPECT
                        and reg_w < MAX_PLATE_WIDTH
                        and reg_h < MAX_PLATE_HEIGHT):
                    plates.append(crop)

        return plates

    def _find_plate_regions(self, img: np.ndarray) -> list:
        """Contour-based plate finder used as YOLO fallback."""
        try:
            gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
            thresh    = cv2.adaptiveThreshold(
                bilateral, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours     = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            candidates = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect = w / float(h)
                area   = w * h

                if not (CONTOUR_MIN_ASPECT <= aspect <= CONTOUR_MAX_ASPECT
                        and area   > CONTOUR_MIN_AREA
                        and w      > CONTOUR_MIN_W
                        and CONTOUR_MIN_H < h < CONTOUR_MAX_H):
                    continue

                region = img[y:y+h, x:x+w]
                gray_r = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                sobelx = cv2.Sobel(gray_r, cv2.CV_8U, 1, 0, ksize=3)
                if np.sum(sobelx > 0) / (w * h) > SOBEL_DENSITY_MIN:
                    candidates.append(region)

            return candidates

        except Exception as e:
            print(f"[Detector] Contour search error: {e}")
            return []
