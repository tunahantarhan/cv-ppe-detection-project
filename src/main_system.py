import cv2
import pandas as pd
from datetime import datetime
import os
from ultralytics import YOLO
import time

# çıktı klasör yolları
os.makedirs('outputs/screenshots', exist_ok=True)
log_file = 'outputs/violations_log.csv'

# eğitilmiş modeli yükleme
model = YOLO('weights/best.pt') 

cooldown_time = 10

# her ihlal türü için son log zamanını tutacak
last_logs = {
    "NO-Hardhat": 0,
    "NO-Safety Vest": 0
}

print("ISG Denetim Sistemi Başlatılıyor...")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # model tarafından kameradaki görüntünün %25 güvenle işlenmesi
    results = model(frame, conf=0.25)
    current_time = time.time()
    
    # çoklu ihlal tespitleri için geçici liste
    current_violations = []

    # ihlal tespiti
    for r in results:
        for c in r.boxes.cls:
            class_name = model.names[int(c)]
            
            if class_name in last_logs:
                if current_time - last_logs[class_name] > cooldown_time:
                    if class_name not in current_violations:
                        current_violations.append(class_name)
                        # sadece yakalanan ihlal türü için cooldown ayarı
                        last_logs[class_name] = current_time 

    if current_violations:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        violation_string = "_ve_".join(current_violations)
        
        # ekran görüntüsü log
        img_name = f"outputs/screenshots/ihlal_{timestamp}.jpg"
        cv2.imwrite(img_name, results[0].plot())
        
        # excel/csv log
        new_log = pd.DataFrame({
            "Tarih_Saat": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Ihlal_Turu": [violation_string],
            "Kanit_Dosyasi": [img_name]
        })
        new_log.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
        
        print(f"!! İHLAL TESPİT EDİLDİ !!: {violation_string} | Kanıt Kaydedildi")

    cv2.imshow("ISG Denetim Kamerasi", results[0].plot())
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()