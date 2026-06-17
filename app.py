import sys
import os
import cv2
import pandas as pd
from werkzeug.utils import secure_filename
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

# src klasörü system path"ine eklenir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from detector import ViolationDetector
from config import CAMERA_INDEX, LOG_FILE
from rules import ViolationEvaluator
from violation_tracker import ViolationTracker
from logger import ViolationLogger

app = Flask(__name__)

# Yükleme Ayarları
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB limit
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "outputs", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Yapay zeka tespit ve loglama/kural motorlarını başlatma
detector = ViolationDetector()
evaluator = ViolationEvaluator()
tracker = ViolationTracker()
system_logger = ViolationLogger()

SCREENSHOT_FOLDER = os.path.join(os.path.dirname(__file__), "outputs", "screenshots")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/camera")
def camera_page():
    return render_template("camera.html")

@app.route("/video")
def video_page():
    uploaded_videos = []
    # eğer "/uploads" klasörü mevcutsa içindeki videolar listelenir (daha önce yüklenmiş videolar)
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.lower().endswith((".mp4", ".avi", ".mov")):
                uploaded_videos.append(filename)
    return render_template("video.html", uploaded_videos=uploaded_videos)

@app.route('/reports')
def reports_page():
    logs = []
    # log dosyası ve içerik varlığı kontrol edilir
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        try:
            df = pd.read_csv(LOG_FILE)
            
            # csv dosyasında istenen sütunlar var ise işlemlere geçilir
            if "Tarih_Saat" in df.columns and "Ihlal_Turu" in df.columns:
                df = df.sort_values(by="Tarih_Saat", ascending=False).fillna("")
                if 'Kanit_Dosyasi' in df.columns:
                    df['Dosya_Adi'] = df['Kanit_Dosyasi'].apply(lambda x: os.path.basename(str(x)))
                logs = df.to_dict('records')
            else:
                # istenen sütun başlığı yoksa veya içerik yanlışsa uyarı mesajı gösterilir
                print("!! UYARI !!: CSV dosyası/şeması hatalı veya eksik.")
                
        except Exception as e:
            print(f"!! LOG OKUMA HATASI !!: {e}")
            
    return render_template('reports.html', logs=logs)

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/evidence/<filename>")
def get_evidence(filename):
    return send_from_directory(SCREENSHOT_FOLDER, filename)

def gen_frames():
    # kamera için canlı akış motoru
    cap = cv2.VideoCapture(CAMERA_INDEX)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        raw_classes, annotated_frame = detector.detect(frame)
        
        logical_violations = evaluator.evaluate(raw_classes)
        should_log, gathered, best_frame = tracker.update(logical_violations, annotated_frame)        
        if should_log:
            system_logger.log(gathered, best_frame)
        
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    cap.release()

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def gen_video_frames(filepath):
    # örnek ve yüklenen videolar için akış motoru
    cap = cv2.VideoCapture(filepath)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break 
        
        raw_classes, annotated_frame = detector.detect(frame)
        
        logical_violations = evaluator.evaluate(raw_classes)
        should_log, gathered, best_frame = tracker.update(logical_violations, annotated_frame)
        if should_log:
            system_logger.log(gathered, best_frame)
            
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    cap.release()

@app.route("/play_video/<video_type>/<filename>")
def play_video(video_type, filename):
    # gelen type parametresine göre videonun yolu bulunur ve motor çalıştırılır
    if video_type == "test":
        filepath = os.path.join(os.path.dirname(__file__), "test_videos", filename)
    elif video_type == "upload":
        filepath = os.path.join(UPLOAD_FOLDER, filename)
    else:
        return "Geçersiz video tipi.", 400
    
    return Response(gen_video_frames(filepath), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/upload", methods=["POST"])
def upload_file():
    # kullanıcının yüklediği video sunucuya kaydedilir
    if "file" not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return jsonify({"filename": filename})

if __name__ == "__main__":
    app.run(debug=True, port=5001)