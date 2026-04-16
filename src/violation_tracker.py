# İhlal Tespit Fonksiyonu:
# İhlal tespiti sonrası debouncing ve state yönetimini üstlenir.
# "Sliding window" mantığıyla birden fazla ihlali tek bir olay olarak gruplar.

import time
import numpy as np
from config import COLLECTION_WINDOW, COOLDOWN_TIME

class ViolationTracker:
    def __init__(self):
        self._is_collecting: bool = False
        self._collection_start: float = 0.0
        self._last_log_time: float = 0.0
        self._gathered_violations: set[str] = set()
        self._best_frame: np.ndarray | None = None

    def update(
        self, violations: list[str], annotated_frame: np.ndarray
    ) -> tuple[bool, set[str], np.ndarray | None]:
        current_time = time.time()
        cooldown_passed = (current_time - self._last_log_time) > COOLDOWN_TIME

        # Veri toplama başlatma veya devam ettirme
        if cooldown_passed and violations:
            if not self._is_collecting:
                self._is_collecting = True
                self._collection_start = current_time
                self._gathered_violations.clear()
                print("İhlal(ler) tespit edildi, veri toplanıyor...")

            for v in violations:
                self._gathered_violations.add(v)
            self._best_frame = annotated_frame

        # Toplama penceresi dolduğunda log atılır.
        window_elapsed = (current_time - self._collection_start) > COLLECTION_WINDOW
        if self._is_collecting and window_elapsed:
            snapshot_violations = set(self._gathered_violations)
            snapshot_frame = self._best_frame

            self._is_collecting = False
            self._last_log_time = current_time
            self._gathered_violations.clear()
            self._best_frame = None

            return True, snapshot_violations, snapshot_frame

        return False, set(), None