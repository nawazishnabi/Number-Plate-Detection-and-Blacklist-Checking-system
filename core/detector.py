import cv2
import numpy as np

class PlateDetector:
    def detect_plates(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple contour-based detection
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        plates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / float(h)
            if 2.0 < ratio < 5.5 and w > 100:
                plate = img[y:y+h, x:x+w]
                plates.append(plate)
        
        return plates if plates else [img]  # fallback: return full image