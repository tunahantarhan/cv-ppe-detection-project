# Detector Fonksiyon:
# Tek sorumluluğu bir frame alıp, o frame'deki ihlalleri ve 
# annotated (kutularla işaretlenmiş) görüntüyü döndürümektir.

import numpy as np
from ultralytics import YOLO
from config import MODEL_PATH, CONFIDENCE_THRESHOLD, CLASS_CONFIDENCE_THRESHOLDS

class ViolationDetector:
    def __init__(self):
        print(f"-> Model yükleniyor: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        print("-> Model başarıyla yüklendi.")

    def detect(self, frame: np.ndarray) -> tuple[list[str], np.ndarray]:
        min_threshold = min(CLASS_CONFIDENCE_THRESHOLDS.values(), default=CONFIDENCE_THRESHOLD)
        # "verbose=False" terminalde oluşabilecek log birikimi önlenir
        results = self.model.track(frame, conf=min_threshold, imgsz=480, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        detected_classes = []
        for r in results:
            for conf_tensor, cls_tensor in zip(r.boxes.conf, r.boxes.cls):
                class_name = self.model.names[int(cls_tensor)]
                confidence = float(conf_tensor)
                threshold = CLASS_CONFIDENCE_THRESHOLDS.get(class_name, CONFIDENCE_THRESHOLD)
                
                if confidence >= threshold:
                    detected_classes.append(class_name)

        annotated_frame = results[0].plot()
        return detected_classes, annotated_frame