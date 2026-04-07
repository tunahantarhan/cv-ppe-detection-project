from ultralytics import YOLO
import cv2

print("YOLO yükleniyor...")
# YOLO modelleri yüklenir
model = YOLO('yolov8n.pt')

print("Kamera açılıyor...")
# "0" -> cihazın varsayılan kamerası
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # kamera görüntüsü modele aktarılır
    results = model(frame)
    
    # frame'leme yapılır
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Test Ekrani", annotated_frame)

    # kapatmak için klavyeden "q" tuşuna basılır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()