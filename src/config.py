# Sistem Konfigürasyonu:
# Bir değer değiştirmek için kullanılacak tek yer/dosya.

# --- Model ---
MODEL_PATH = "weights/best.pt"
CONFIDENCE_THRESHOLD = 0.25

# --- İhlal Sınıfları ---
TARGET_VIOLATIONS = {
    "NO-Hardhat",
    "NO-Safety Vest",
    "NO-Mask",
    "NO-Gloves",
    "NO-Goggles",
    "Fall-Detected",
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