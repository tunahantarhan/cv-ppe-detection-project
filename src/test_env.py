# Debug & Test Aracı:
# Bu araç, sistem konfigürasyonu ve modeli kullanarak canlı kamera görüntüsü üzerinde
# tespit sonuçlarını, confidence değerlerini ve FPS'i gösterir.

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
)
from ultralytics import YOLO
from rules import ViolationEvaluator  # Kural motoru

WINDOW_TITLE = "ISG Debug Ekrani  |  q: cikis  |  s: screenshot"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Renkler ---
COLOR_VIOLATION = (0, 0, 220)      # kırmızı — ihlal
COLOR_SAFE = (0, 200, 0)           # yeşil — güvenli
COLOR_OVERLAY_BG = (20, 20, 20)    # koyu panel arka planı
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 220, 220)

# Fps değerleri yazılır
def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                FONT, 0.7, COLOR_YELLOW, 2, cv2.LINE_AA)

# Tespit sonuçlarını sağ üst köşede gösteren panel eklenir
def draw_detection_panel(
    frame: np.ndarray,
    detections: list[tuple[str, float]],
) -> None:
    if not detections:
        return

    panel_x = frame.shape[1] - 280
    panel_y = 10
    row_height = 28
    panel_h = row_height * len(detections) + 16
    panel_w = 270

    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x - 8, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  COLOR_OVERLAY_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.2, 0, frame)

    for i, (class_name, conf) in enumerate(detections):
        y = panel_y + 18 + i * row_height
        is_violation = class_name in TARGET_VIOLATIONS
        color = COLOR_VIOLATION if is_violation else COLOR_SAFE

        label = f"{class_name}"
        cv2.putText(frame, label, (panel_x, y),
                    FONT, 0.52, color, 1, cv2.LINE_AA)

        # Confidence bar (arka plan)
        bar_x = panel_x + 158
        bar_w = 80
        bar_h = 10
        bar_y = y - 9
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)

        # Confidence bar (dolu kısım)
        filled_w = int(bar_w * conf)
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + filled_w, bar_y + bar_h),
                      color, -1)

        # Sayısal değer
        cv2.putText(frame, f"{conf:.2f}", (bar_x + bar_w + 5, y),
                    FONT, 0.48, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_threshold_info(frame: np.ndarray, class_name: str, conf: float) -> None:
    threshold = CLASS_CONFIDENCE_THRESHOLDS.get(class_name, CONFIDENCE_THRESHOLD)
    margin = conf - threshold
    return margin


def save_screenshot(frame: np.ndarray) -> str:
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{SCREENSHOTS_DIR}/debug_{timestamp}.jpg"
    cv2.imwrite(path, frame)
    return path


def main() -> None:
    print("=" * 50)
    print("ISG Debug Aracı Başlatılıyor")
    print(f"Model:  {MODEL_PATH}")
    print(f"Kamera: index {CAMERA_INDEX}")
    print("=" * 50)

    print("Model yükleniyor...")
    model = YOLO(MODEL_PATH)
    evaluator = ViolationEvaluator()  # Kural motoru
    print("Model yüklendi.\n")

    video_path = "test_videos/test_video_03.mov"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"HATA: Video dosyası açılamadı: {video_path}")
        sys.exit(1)

    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    prev_time = time.time()
    fps = 0.0

    print("Sistem aktif. q: çıkış | s: ekran görüntüsü")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame okunamadı.")
            break

        results = model(frame, conf=0.15, imgsz=640, verbose=False)

        raw_detections = []
        detected_classes_only = []
        person_confidence = 0.0

        for r in results:
            for conf_tensor, cls_tensor in zip(r.boxes.conf, r.boxes.cls):
                class_name = model.names[int(cls_tensor)]
                confidence = float(conf_tensor)
                
                # Modelin bulduğu her şey listeye atılır 
                detected_classes_only.append(class_name)
                
                if class_name == "Person":
                    person_confidence = max(person_confidence, confidence) # En yüksek güven alınır
                    
                # Flickering indirgeme adına sadece güvenilir tespitler ham listeye atılır
                if not class_name.startswith("NO-") and confidence >= 0.35:
                    raw_detections.append((class_name, confidence))

        # Terminale saniyede 10 kere modelin ne gördüğü yazılarak debug yapılır
        print(f"GÖZÜKENLER: {detected_classes_only}")

        # Kural motoru çalıştırılır
        logical_violations = evaluator.evaluate(detected_classes_only)

        # Bulunan mantıksal ihlaller (NO-Mask, NO-Hardhat) ana panele eklenir
        # Güven skoru olarak person_confidence kullanılır, eğer person yoksa 0.99 gözükür
        final_panel_detections = list(raw_detections)
        for lv in logical_violations:
            final_panel_detections.append((lv, person_confidence if person_confidence > 0 else 0.99))

        # Confidence'a göre sıralanır
        final_panel_detections.sort(key=lambda x: x[1], reverse=True)

        annotated = results[0].plot()

        # FPS ve ana panel çizilir
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 1e-6)
        prev_time = current_time

        draw_fps(annotated, fps)
        draw_detection_panel(annotated, final_panel_detections)

        # İhlal varsa üstte kırmızı bant gösterilir
        active_violations = [d for d in final_panel_detections if d[0] in TARGET_VIOLATIONS]
        if active_violations:
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 5), COLOR_VIOLATION, -1)

        cv2.imshow(WINDOW_TITLE, annotated)

        # --- Klavye kısayolları ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Çıkılıyor...")
            break
        elif key == ord("s"):
            path = save_screenshot(annotated)
            print(f"Ekran görüntüsü kaydedildi: {path}")
                        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()