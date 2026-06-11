import os

MODEL_PATH = "weights/best.pt"
CONFIDENCE_THRESHOLD = 0.50

CLASS_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "Hardhat":         0.50,
    "Safety Vest":     0.50,
    "NO-Hardhat":      0.50,
    "NO-Safety Vest":  0.50,
}

# --- EKRANDA GÖZÜKECEK TÜRKÇE SINIF İSİMLERİ ---
CLASS_NAMES = {
    0: 'Baret',
    1: 'IHLAL | Baret Yok',
    2: 'Is-Yelegi',
    3: 'IHLAL | Is-Yelegi Yok'
}

# --- İHLAL SINIFLARI ---
TARGET_VIOLATIONS = {
    "NO-Hardhat",
    "NO-Safety Vest"
}

# --- CLASS RENKLERİ ---
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 200, 0),      # Hardhat -> Yeşil
    1: (0, 0, 220),      # NO-Hardhat -> Kırmızı
    2: (0, 200, 200),    # Safety Vest -> Cyan
    3: (220, 0, 220),    # NO-Safety Vest -> Mor
}

# --- COOLDOWN VE ÇIKTI ---
COLLECTION_WINDOW = 3   
COOLDOWN_TIME = 10

OUTPUT_DIR = "outputs"
SCREENSHOTS_DIR = "outputs/screenshots"
LOG_FILE = "outputs/violations_log.csv"

CAMERA_INDEX = 0
WINDOW_TITLE = "ISG Denetim Kamerasi"