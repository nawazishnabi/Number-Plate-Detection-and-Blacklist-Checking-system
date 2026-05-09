import cv2
import numpy as np

class PlateDetector:
    def detect_plates(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return []

        original = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = []

        # ── Method 1: Haar Cascade ─────────────────────────────────────
        try:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            )
            detected = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 20)
            )
            for (x, y, w, h) in detected:
                plates.append(original[y:y+h, x:x+w])
        except:
            pass

        # ── Method 2: Improved contour-based ──────────────────────────
        if not plates:
            blur = cv2.bilateralFilter(gray, 11, 17, 17)
            edges = cv2.Canny(blur, 30, 200)

            contours, _ = cv2.findContours(
                edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)

                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    ratio = w / float(h)
                    area = w * h
                    img_area = img.shape[0] * img.shape[1]

                    if (2.0 < ratio < 6.0 and
                            area > 1000 and
                            area < img_area * 0.3):
                        plates.append(original[y:y+h, x:x+w])

        # ── Method 3: Fallback — full image ────────────────────────────
        if not plates:
            plates = [original]

        return plates