# Debug & Test Aracı:
# Canlı kamera veya test videosu üzerinde tespit sonuçlarını, confidence değerlerini ve FPS'i gösterir.

import sys
import os
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    CLASS_CONFIDENCE_THRESHOLDS,
    TARGET_VIOLATIONS,
    CAMERA_INDEX,
    SCREENSHOTS_DIR,
    CLASS_NAMES,
    CLASS_COLORS,
)
from ultralytics import YOLO
from rules import ViolationEvaluator

WINDOW_TITLE = "ISG Debug Ekrani  |  q: cikis  |  s: screenshot"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Renkler ---
COLOR_VIOLATION = (0, 0, 220)      # Kırmızı — İhlal
COLOR_SAFE = (0, 200, 0)           # Yeşil — Güvenli
COLOR_OVERLAY_BG = (20, 20, 20)    # Koyu panel arka planı
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 220, 220)

# --- Video / Kamera Seçimi ---
# Kamera için boş bırakılmalıdır
# Video için videonun path'i verilmelidir
VIDEO_SOURCE: str | None = ""

def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 58), FONT, 0.7, COLOR_YELLOW, 2, cv2.LINE_AA)

def draw_detection_panel(frame: np.ndarray, detections: list[tuple[str, float]]) -> None:
    if not detections:
        return

    panel_x = frame.shape[1] - 280
    panel_y = 10
    row_height = 28
    panel_h = row_height * len(detections) + 16
    panel_w = 270
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 8, panel_y), (panel_x + panel_w, panel_y + panel_h), COLOR_OVERLAY_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (class_name, conf) in enumerate(detections):
        y = panel_y + 18 + i * row_height
        
        # Eğer string içinde "IHLAL" kelimesi geçiyorsa kırmızı yapılır
        is_violation = "IHLAL" in class_name
        color = COLOR_VIOLATION if is_violation else COLOR_SAFE

        cv2.putText(frame, class_name, (panel_x, y), FONT, 0.52, color, 1, cv2.LINE_AA)

        # Confidence bar 
        bar_x = panel_x + 158
        bar_w = 80
        bar_h = 10
        bar_y = y - 9
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)

        filled_w = int(bar_w * conf)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), color, -1)

        cv2.putText(frame, f"{conf:.2f}", (bar_x + bar_w + 5, y), FONT, 0.48, COLOR_WHITE, 1, cv2.LINE_AA)

def save_screenshot(frame: np.ndarray) -> str:
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{SCREENSHOTS_DIR}/debug_{timestamp}.jpg"
    cv2.imwrite(path, frame)
    return path
 
def main() -> None:
    print("=" * 50)
    print("-> ISG Debug Aracı Başlatılıyor...")
    print(f"Model: {MODEL_PATH}")
    print(f"Kaynak: {VIDEO_SOURCE if VIDEO_SOURCE else f'Kamera (index {CAMERA_INDEX})'}")
    print("=" * 50)
 
    model = YOLO(MODEL_PATH)
    evaluator = ViolationEvaluator()
    print("-> Model yüklendi.\n")
    
    source = VIDEO_SOURCE if VIDEO_SOURCE else CAMERA_INDEX
    cap = cv2.VideoCapture(source)
 
    if not cap.isOpened():
        print(f"-> HATA: Kaynak açılamadı: {source}")
        sys.exit(1)
 
    min_threshold = min(CLASS_CONFIDENCE_THRESHOLDS.values(), default=CONFIDENCE_THRESHOLD)
    prev_time = time.time()
    
    print("-> Sistem aktif. q: çıkış | s: ekran görüntüsü")
     
    while True:
        ret, frame = cap.read()
        if not ret:
            print("-> Video bitti veya okunamadı.")
            break
 
        results = model.track(frame, conf=min_threshold, imgsz=480, persist=True, tracker="bytetrack.yaml", verbose=False) 
        
        raw_classes: list[str] = []
        panel_detections: list[tuple[str, float]] = []
 
        for r in results:
            for conf_tensor, cls_tensor in zip(r.boxes.conf, r.boxes.cls):
                class_id = int(cls_tensor)
                class_name_en = model.names[class_id] 
                confidence = float(conf_tensor)
                threshold = CLASS_CONFIDENCE_THRESHOLDS.get(class_name_en, CONFIDENCE_THRESHOLD)

                if confidence < threshold:
                    continue

                raw_classes.append(class_name_en)

                class_name_tr = CLASS_NAMES.get(class_id, class_name_en)
                
                # Güvenli olanlar (Hardhat, Safety Vest) panele eklenir
                if class_name_en not in TARGET_VIOLATIONS:
                    panel_detections.append((class_name_tr, confidence))
 
        # Mantıksal ihlaller panele kırmızı yazıyla eklenir
        logical_violations = evaluator.evaluate(raw_classes)
        for violation in logical_violations:
            violation_tr = f"IHLAL | {violation.replace('_', ' ')}" 
            panel_detections.append((violation_tr, 0.99))
 
        # Conf değerine göre sıralanır
        panel_detections.sort(key=lambda x: x[1], reverse=True)
        
        annotated = frame.copy()
        for r in results:
            for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                class_id = int(cls)
                class_name_tr = CLASS_NAMES.get(class_id, model.names[class_id])
                confidence = float(conf)
                color = CLASS_COLORS.get(class_id, (0, 255, 0))
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{class_name_tr} {confidence:.2f}", (x1, y1 - 10), FONT, 0.5, color, 2)
 
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 1e-6)
        prev_time = current_time
 
        draw_fps(annotated, fps)
        draw_detection_panel(annotated, panel_detections)
 
        if logical_violations:
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 5), COLOR_VIOLATION, -1)
 
        cv2.imshow(WINDOW_TITLE, annotated)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = save_screenshot(annotated)
            print(f"Kanıt kaydedildi: {path}")
 
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()