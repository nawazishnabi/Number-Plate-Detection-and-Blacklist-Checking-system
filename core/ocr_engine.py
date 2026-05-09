import pytesseract
import cv2
import numpy as np

class OCREngine:
    def extract_text(self, image):
        best_text = ''
        variants = self._preprocess(image)

        configs = [
            '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        ]

        for variant in variants:
            for config in configs:
                try:
                    text = pytesseract.image_to_string(
                        variant, config=config
                    ).strip()
                    text = ''.join(c for c in text if c.isalnum())

                    if len(text) > len(best_text):
                        best_text = text

                    # Typical plate is 5-12 characters
                    if 5 <= len(best_text) <= 12:
                        return best_text

                except Exception:
                    continue

        return best_text if best_text else 'UNREADABLE'

    def _preprocess(self, image):
        variants = []

        # Resize small images
        h, w = image.shape[:2]
        if w < 200:
            scale = 200 / w
            image = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LANCZOS4)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Scale up 3x for better OCR accuracy
        gray = cv2.resize(gray, None, fx=3, fy=3,
                          interpolation=cv2.INTER_LANCZOS4)

        # Variant 1: Otsu threshold
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, otsu = cv2.threshold(blur, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
        variants.append(cv2.bitwise_not(otsu))  # inverted version

        # Variant 2: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        variants.append(adaptive)
        variants.append(cv2.bitwise_not(adaptive))

        # Variant 3: CLAHE enhanced
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, clahe_thresh = cv2.threshold(enhanced, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(clahe_thresh)

        # Variant 4: Raw grayscale
        variants.append(gray)

        return variants