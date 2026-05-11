# Sistem Konfigürasyonu:
# Bir değer değiştirmek için kullanılacak tek yer/dosya.

# --- Model ---
MODEL_PATH = "weights/best.pt"
CONFIDENCE_THRESHOLD = 0.20  # Person ve genel şeyler için taban

CLASS_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "Gloves":          0.25,
    "Goggles":         0.30,
    "Mask":            0.28,
    "NO-Safety Vest":  0.18,
    "NO-Mask":         0.22,
    "NO-Hardhat":      0.22,
    "NO-Goggles":      0.20,  
    "NO-Gloves":       0.20,
}

# --- Türkçe Class İsimleri ---
CLASS_NAMES = {
    0: 'Dusme-Algilandi',
    1: 'Eldiven',
    2: 'Gozluk',
    3: 'Kask',
    4: 'Merdiven',
    5: 'Maske',
    6: 'IHLAL | Eldiven',
    7: 'IHLAL | Gozluk',
    8: 'IHLAL | Kask',
    9: 'IHLAL | Maske',
    10: 'IHLAL | Is-Yelegi',
    11: 'Insan',
    12: 'Guvenlik-Konisi',
    13: 'Is-Yelegi'
}

# --- İhlal Sınıfları (Türkçe isimleri de kabul eder) ---
TARGET_VIOLATIONS = {
    "NO-Hardhat",
    "IHLAL | Kask",
    "NO-Safety Vest",
    "IHLAL | Is-Yelegi",
    "NO-Mask",
    "IHLAL | Maske",
    "NO-Gloves",
    "IHLAL | Eldiven",
    "NO-Goggles",
    "IHLAL | Gozluk",
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