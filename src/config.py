# Sistem Konfigürasyonu:
# Bir değer değiştirmek için kullanılacak tek yer/dosya.

# --- Model ---
MODEL_PATH = "weights/best.pt"
CONFIDENCE_THRESHOLD = 0.25  # Person ve genel şeyler için taban

CLASS_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "NO-Safety Vest":  0.22,
    "NO-Mask":         0.28,
    "NO-Hardhat":      0.28,
    "NO-Goggles":      0.18,  
    "NO-Gloves":       0.18,
}

# --- İhlal Sınıfları ---
TARGET_VIOLATIONS = {
    "NO-Hardhat",
    "NO-Safety Vest",
    "NO-Mask",
    "NO-Gloves",
    "NO-Goggles",
}

# --- Cooldown ---
COLLECTION_WINDOW = 3   # ihlal sonrası ek ihlallerin toplanma süresi
COOLDOWN_TIME = 10      # bir log grubundan sonra yeni tespite kadar bekleme

# --- Çıktı Yolları ---
OUTPUT_DIR = "outputs"
SCREENSHOTS_DIR = "outputs/screenshots"
LOG_FILE = "outputs/violations_log.csv"

# --- Kamera ---
CAMERA_INDEX = 0
WINDOW_TITLE = "ISG Denetim Kamerasi"