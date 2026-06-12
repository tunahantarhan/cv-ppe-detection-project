import cv2
import numpy as np
from ultralytics import YOLO
from config import MODEL_PATH, CONFIDENCE_THRESHOLD, CLASS_CONFIDENCE_THRESHOLDS, CLASS_NAMES, CLASS_COLORS

class ViolationDetector:
    def __init__(self):
        print(f"-> Model yükleniyor: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        print("-> Model başarıyla yüklendi.")

    def detect(self, frame: np.ndarray) -> tuple[list[str], np.ndarray]:
        min_threshold = min(CLASS_CONFIDENCE_THRESHOLDS.values(), default=CONFIDENCE_THRESHOLD)
        
        # "verbose=False" ile terminalde oluşabilecek log birikimi önlenir
        # tracker devreye alındığında, model.track() fonksiyonu kullanılır
        # nesnelerin takibi sağlanır ve her karede tespit edilen nesneler güncellenir, titreme azaltılır.
        results = self.model.track(frame, conf=min_threshold, imgsz=480, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        detected_classes = []
        annotated_frame = frame.copy()
        
        for r in results:
            if r.boxes is not None:
                for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                    class_id = int(cls)
                    class_name = self.model.names[class_id]
                    confidence = float(conf)
                    threshold = CLASS_CONFIDENCE_THRESHOLDS.get(class_name, CONFIDENCE_THRESHOLD)
                    
                    if confidence >= threshold:
                        detected_classes.append(class_name)
                        
                        # türkçe isimler ve config'deki renklerle çizim yapılır
                        class_name_tr = CLASS_NAMES.get(class_id, class_name)
                        color = CLASS_COLORS.get(class_id, (0, 255, 0))
                        
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{class_name_tr} {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return detected_classes, annotated_frame