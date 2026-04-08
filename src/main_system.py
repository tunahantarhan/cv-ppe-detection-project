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


cooldown_time = 5 # ihlal tespit cooldown'ı  
last_log_time = 0

print("ISG Denetim Sistemi Başlatılıyor...")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # model kamera görüntüsünü işler, %25 güven eşiğiyle sonuç verir
    results = model(frame, conf=0.25)
    current_time = time.time()
    
    violation_detected = False
    violation_type = ""

    # ihlal tespiti
    for r in results:
        for c in r.boxes.cls:
            class_name = model.names[int(c)]
            if class_name in ["NO-Hardhat", "NO-Safety Vest"]: 
                violation_detected = True
                violation_type = class_name
                break

    # ihlal tespit edildiğinde ve cooldown süresi geçmişse log tut
    if violation_detected and (current_time - last_log_time > cooldown_time):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ekran görüntüsü log
        img_name = f"outputs/screenshots/ihlal_{timestamp}.jpg"
        cv2.imwrite(img_name, results[0].plot())
        
        # excel/csv log
        new_log = pd.DataFrame({
            "Tarih_Saat": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Ihlal_Turu": [violation_type],
            "Kanit_Dosyasi": [img_name]
        })
        new_log.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
        
        print(f"!! İHLAL TESPİT EDİLDİ !!: {violation_type} | Kanıt Kaydedildi")
        last_log_time = current_time

    cv2.imshow("ISG Denetim Kamerasi", results[0].plot())
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()