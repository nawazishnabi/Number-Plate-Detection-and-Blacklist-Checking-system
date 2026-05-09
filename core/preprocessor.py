
"""
preprocessor.py
---------------
Image preprocessing pipeline that prepares a raw plate crop
for OCR. Enhanced with better algorithms for Indian plates.
"""

import cv2
import numpy as np
from skimage import restoration, filters
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    TARGET_OCR_WIDTH, TARGET_OCR_HEIGHT,
    CROP_TOP_RATIO, CROP_BOTTOM_RATIO,
)


class Preprocessor:

    def process(self, plate_img: np.ndarray) -> np.ndarray:
        """Returns the BEST single preprocessed image."""
        variants = self.process_all(plate_img)
        return variants[0]

    def process_all(self, plate_img: np.ndarray) -> list:
        """Returns multiple preprocessed variants optimized for Indian plates."""
        try:
            # ── Step 1: Grayscale ─────────────────────────────────────
            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_img.copy()

            # ── Step 2: Intelligent upscaling ─────────────────────────
            h, w = gray.shape[:2]
            # For very small plates, use super-resolution approach
            if h < 50 or w < 150:
                scale = max(4.0, 200.0 / h)  # Aggressive upscaling for tiny plates
            else:
                scale = max(TARGET_OCR_HEIGHT / h, TARGET_OCR_WIDTH / w, 2.5)
            
            scale = min(scale, 8.0)  # Cap to prevent memory issues
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_LANCZOS4)

            # ── Step 3: Enhanced deskew ───────────────────────────────
            gray = self._deskew(gray)

            # ── Step 4: Noise reduction ───────────────────────────────
            # Non-local means denoising for better edge preservation
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            # ── Step 5: Contrast enhancement ──────────────────────────
            # Multiple CLAHE variants for different lighting conditions
            clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            clahe3 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12, 12))
            
            enhanced1 = clahe1.apply(denoised)
            enhanced2 = clahe2.apply(denoised)
            enhanced3 = clahe3.apply(denoised)

            # ── Step 6: Gamma correction for dark plates ──────────────
            gamma_corrected = self._gamma_correction(denoised, gamma=1.5)

            # ── Step 7: Edge enhancement ──────────────────────────────
            # Sharpening with better kernel
            kernel_sharpen1 = np.array([[-1, -1, -1],
                                        [-1,  9, -1],
                                        [-1, -1, -1]]) / 1.0
            
            kernel_sharpen2 = np.array([[ 0, -1,  0],
                                        [-1,  5, -1],
                                        [ 0, -1,  0]]) / 1.0

            sharpened1 = cv2.filter2D(enhanced1, -1, kernel_sharpen1)
            sharpened2 = cv2.filter2D(enhanced2, -1, kernel_sharpen2)
            
            sharpened1 = np.clip(sharpened1, 0, 255).astype(np.uint8)
            sharpened2 = np.clip(sharpened2, 0, 255).astype(np.uint8)

            # ── Step 8: Build threshold variants ─────────────────────
            variants = []
            
            # Method 1: Otsu on CLAHE enhanced (best for most plates)
            blur = cv2.GaussianBlur(enhanced1, (3, 3), 0)
            _, otsu1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(self._clean_morph(otsu1))
            
            # Method 2: Otsu on sharpened
            blur = cv2.GaussianBlur(sharpened1, (3, 3), 0)
            _, otsu2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(self._clean_morph(otsu2))
            
            # Method 3: Adaptive threshold (handles uneven lighting)
            adaptive1 = cv2.adaptiveThreshold(enhanced2, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 15, 6)
            variants.append(self._clean_morph(adaptive1))
            
            # Method 4: Adaptive with different parameters
            adaptive2 = cv2.adaptiveThreshold(enhanced3, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 4)
            variants.append(self._clean_morph(adaptive2))
            
            # Method 5: Sauvola thresholding (better for variable lighting)
            try:
                from skimage.filters import threshold_sauvola
                window_size = max(15, min(w, h) // 4)
                thresh_sauvola = threshold_sauvola(enhanced1, window_size=window_size)
                binary_sauvola = (enhanced1 > thresh_sauvola).astype(np.uint8) * 255
                variants.append(self._clean_morph(binary_sauvola))
            except:
                pass
            
            # Method 6: Otsu on gamma corrected (for dark plates)
            blur = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)
            _, otsu3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(self._clean_morph(otsu3))

            # Ensure all variants are inverted correctly (white text on black bg)
            final_variants = []
            for v in variants:
                if np.mean(v) > 127:  # More white than black
                    v = cv2.bitwise_not(v)
                final_variants.append(v)

            return final_variants[:6]  # Limit to 6 best variants

        except Exception as e:
            print(f"[Preprocessor] Error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback
            try:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=3, fy=3,
                                  interpolation=cv2.INTER_LANCZOS4)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if np.mean(thresh) > 127:
                    thresh = cv2.bitwise_not(thresh)
                return [thresh]
            except:
                return [plate_img]

    # ── Private helpers ───────────────────────────────────────────────

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """Enhanced deskew with better angle detection."""
        try:
            # Use edges for better angle detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=5)
            
            # Dilate edges to connect text components
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            edges = cv2.dilate(edges, kernel, iterations=2)
            
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                    threshold=80, minLineLength=30, maxLineGap=5)
            if lines is None:
                return gray

            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if -10 < angle < 10:  # Only near-horizontal lines
                        angles.append(angle)

            if not angles:
                return gray

            median_angle = np.median(angles)

            if abs(median_angle) < 0.3:  # Skip very small angles
                return gray

            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            deskewed = cv2.warpAffine(gray, M, (w, h),
                                      flags=cv2.INTER_LANCZOS4,
                                      borderMode=cv2.BORDER_REPLICATE)
            return deskewed

        except Exception:
            return gray

    def _gamma_correction(self, img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Apply gamma correction for dark images."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    def _clean_morph(self, img: np.ndarray) -> np.ndarray:
        """Clean binary image with morphological operations."""
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        # Remove small white spots
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(img.shape, np.uint8)
        for cnt in contours:
            if cv2.contourArea(cnt) > 10:  # Keep only significant components
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        return cv2.bitwise_and(img, mask)