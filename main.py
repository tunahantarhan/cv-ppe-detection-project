import cv2
import sys
import os

# modüllerin bulunması için src dizinini sys.path'e ekler.
sys.path.insert(0, os.path.dirname(__file__))

from camera import Camera
from detector import ViolationDetector
from violation_tracker import ViolationTracker
from logger import ViolationLogger
from config import WINDOW_TITLE
from rules import ViolationEvaluator

def main() -> None:
    # Sistem başlatılır, her sorumluluk ilgili modüllere devredilir.
    print("ISG Denetim Sistemi Başlatılıyor...")

    detector = ViolationDetector()
    tracker = ViolationTracker()
    logger = ViolationLogger()
    evaluator = ViolationEvaluator() # kural motoru

    try:
        with Camera() as camera:
            print("Sistem aktif. Çıkmak için 'q' tuşuna basın.")
            while True:
                success, frame = camera.read_frame()
                if not success:
                    print("Frame okunamadı, döngü sonlandırılıyor.")
                    break

                # 1) Görüntü üzerinden ham nesneler bulunur
                raw_classes, annotated_frame = detector.detect(frame)
                
                # 2) Kural motoru, ham nesneleri alır ve ihlalleri değerlendirir
                violations = evaluator.evaluate(raw_classes)
                
                # 3) İhlaller takip edilir (Slidin Window / Debounce mantığı)
                should_log, gathered, best_frame = tracker.update(violations, annotated_frame)

                # 4) İhlal varsa loglanır, veritabanına ve csv'ye yazılır, kanıt görseli kaydedilir 
                if should_log:
                    logger.log(gathered, best_frame)

                cv2.imshow(WINDOW_TITLE, annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Kullanıcı tarafından durduruldu.")
                    break

    except RuntimeError as e:
        print(f"HATA: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()