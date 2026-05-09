from ultralytics import YOLO
import cv2

class PlateDetector:
    def __init__(self):
        self.model = None  # Don't load at startup!
    
    def _load_model(self):
        if self.model is None:
            self.model = YOLO('50w_best.pt')  # Load only on first request
    
    def detect_plates(self, image_path):
        self._load_model()  # Load here, not in __init__
        results = self.model(image_path)
        plates = []
        for r in results:
            for box in r.boxes:
                img = cv2.imread(image_path)
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                plates.append(img[y1:y2, x1:x2])
        return plates