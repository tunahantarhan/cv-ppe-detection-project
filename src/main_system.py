import cv2
import pandas as pd
from datetime import datetime
import os
from ultralytics import YOLO
import time

os.makedirs('outputs/screenshots', exist_ok=True)
log_file = 'outputs/violations_log.csv'

model = YOLO('weights/best.pt') 

collection_window = 3 # ihlal tespiti sonrası varsa ekstra ihlallerin algılanma aralığı
cooldown_time = 10 # ihlal grubu tespiti sonrası bekleme süresi 

is_collecting = False
collection_start = 0
gathered_violations = set() # set (küme) ile ihlal türleri her grupta unique olarak tutulur
last_log_time = 0
best_frame_for_log = None # fotoğraf hafızada tutulur

print("ISG Denetim Sistemi Başlatılıyor...")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # model tarafından kameradaki görüntünün %25 güvenle işlenmesi
    results = model(frame, conf=0.25)   
    current_time = time.time()
    
    # == ihlal tespit, veri toplama ve loglama mantığı ==
    
    # ihlal oluşturan durumlar
    TARGET_VIOLATIONS = [
        "NO-Hardhat", 
        "NO-Safety Vest", 
        "NO-Mask", 
        "NO-Gloves", 
        "NO-Goggles", 
        "Fall-Detected"
    ]
    
    current_frame_violations = []
    for r in results:
        for c in r.boxes.cls:
            class_name = model.names[int(c)]
            if class_name in TARGET_VIOLATIONS: 
                current_frame_violations.append(class_name)

    # cooldown süresi geçmiş ve frame'de ihlal tespit edilmişse veri toplama başlatılır
    if (current_time - last_log_time > cooldown_time) and current_frame_violations:
        if not is_collecting:
            is_collecting = True
            collection_start = current_time
            gathered_violations.clear()
            print("İhlal(ler) tespit edildi, veri toplanıyor...")
        
        # birden çok ihlal tespit edilmişse toplanır ve en iyi kanıt görüntüsü alınır
        for v in current_frame_violations:
            gathered_violations.add(v)        
        best_frame_for_log = results[0].plot()

    # loglama
    if is_collecting and (current_time - collection_start > collection_window):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        violation_string = "_ve_".join(sorted(list(gathered_violations)))
        
        # kanıt görüntü kaydedilir
        img_name = f"outputs/screenshots/ihlal_{timestamp}.jpg"
        if best_frame_for_log is not None:
            cv2.imwrite(img_name, best_frame_for_log)
        
        # log dosyasına yeni kayıt eklenir
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