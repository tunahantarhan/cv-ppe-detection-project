import os

MODEL_PATH = "weights/best.pt"
CONFIDENCE_THRESHOLD = 0.60

CLASS_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "Hardhat":         0.60,
    "Safety Vest":     0.60,
    "NO-Hardhat":      0.65,
    "NO-Safety Vest":  0.65,
}

# --- EKRANDA GÖZÜKECEK TÜRKÇE SINIF İSİMLERİ ---
CLASS_NAMES = {
    0: 'Baret',
    1: 'IHLAL | Baret Yok',
    2: 'IHLAL | Is Yelegi Yok',
    3: 'Is Yelegi',
}

# --- İHLAL SINIFLARI ---
TARGET_VIOLATIONS = {
    "NO-Hardhat",
    "NO-Safety Vest"
}

# --- CLASS RENKLERİ ---
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 200, 0),      # Hardhat - Yeşil (Güvenli)
    1: (0, 0, 220),      # NO-Hardhat - Kırmızı (İhlal)
    2: (0, 200, 200),    # NO-Safety Vest - Sarı (İhlal)
    3: (220, 0, 220),    # Safety Vest -  Mor (Güvenli)
}

# --- COOLDOWN VE ÇIKTI ---
COLLECTION_WINDOW = 3   
COOLDOWN_TIME = 10

OUTPUT_DIR = "outputs"
SCREENSHOTS_DIR = "outputs/screenshots"
LOG_FILE = "outputs/violations_log.csv"

CAMERA_INDEX = 0
WINDOW_TITLE = "ISG Denetim Kamerasi"