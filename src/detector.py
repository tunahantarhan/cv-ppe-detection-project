# Detector Fonksiyon:
# Tek sorumluluğu bir frame alıp, o frame'deki ihlalleri ve 
# annotated (kutularla işaretlenmiş) görüntüyü döndürümektir.

import cv2
import numpy as np
from ultralytics import YOLO
from config import MODEL_PATH, CONFIDENCE_THRESHOLD, TARGET_VIOLATIONS

class ViolationDetector:
    def __init__(self):
        print(f"Model yükleniyor: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        print("Model başarıyla yüklendi.")

    # Frame üzerinde ihlal tespiti yapar ve box çizilir.
    def detect(self, frame: np.ndarray) -> tuple[list[str], np.ndarray]:
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, imgsz=640, verbose=False)

        violations = []
        for r in results:
            for c in r.boxes.cls:
                class_name = self.model.names[int(c)]
                if class_name in TARGET_VIOLATIONS:
                    violations.append(class_name)

        annotated_frame = results[0].plot()
        return violations, annotated_frame