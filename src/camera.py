# Kamera Bağlantı Fonksiyonu:
# OpenCV ile kamera bağlantısını yönetir.

import cv2
import numpy as np
from config import CAMERA_INDEX


class Camera:
    def __init__(self):
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "Camera":
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Kamera açılamadı (index: {CAMERA_INDEX}). "
                "Kameranın bağlı ve başka bir uygulama tarafından kullanılmıyor olduğundan emin olun."
            )
        print(f"Kamera başarıyla açıldı (index: {CAMERA_INDEX}).")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._cap is not None:
            self._cap.release()
        cv2.destroyAllWindows()
        print("Kamera kapatıldı.")
        return False

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
       # Kameradan frame okunur.
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        return ret, frame if ret else None