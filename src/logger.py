import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from config import SCREENSHOTS_DIR, LOG_FILE


class ViolationLogger:
    def __init__(self):
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    def log(self, violations: set[str], frame: np.ndarray | None) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        violation_string = "_ve_".join(sorted(violations))
        img_path = self._save_screenshot(frame, timestamp)
        self._write_csv(violation_string, img_path, timestamp)
        print(f"!! İHLAL TESPİT EDİLDİ !!: {violation_string} | Kanıt: {img_path}")

    def _save_screenshot(self, frame: np.ndarray | None, timestamp: str) -> str:
        img_path = f"{SCREENSHOTS_DIR}/ihlal_{timestamp}.jpg"
        if frame is not None:
            cv2.imwrite(img_path, frame)
        return img_path

    def _write_csv(self, violation_string: str, img_path: str, timestamp: str) -> None:
        # runtime kontrolü: cold start (dosya ve içerik var mı?)
        needs_header = not os.path.isfile(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
        
        new_row = pd.DataFrame({
            "Tarih_Saat": [datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")],
            "Ihlal_Turu": [violation_string],
            "Kanit_Dosyasi": [img_path],
        })
        
        new_row.to_csv(
            LOG_FILE,
            mode="a",
            header=needs_header,
            index=False,
        )