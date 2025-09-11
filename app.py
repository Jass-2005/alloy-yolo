from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import os, uuid

# ⬅️ Updatet this to your trained model path
MODEL_PATH = r"D:\GGF\runs\classify\train9\weights\best.pt"

app = Flask(__name__, static_folder="frontend", template_folder="frontend")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model once at startup
model = YOLO(MODEL_PATH)# Replace the path below with your actual Python 3.11 install path




@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    file.save(fpath)

    try:
        results = model(fpath)
        r0 = results[0]

        # If it's a classification model (your case)
        if getattr(r0, "probs", None) is not None:
            probs = r0.probs
            topk_idx = probs.top5
            topk_conf = probs.top5conf.tolist()
            topk = [
                {"class": model.names[i], "confidence": float(c)}
                for i, c in zip(topk_idx, topk_conf)
            ]
            return jsonify({"type": "classification", "top": topk})

        # If it happens to be a detection model instead (fallback)
        boxes = r0.boxes
        detections = []
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_idx = int(b.cls)
                detections.append({
                    "class": model.names[cls_idx],
                    "confidence": float(b.conf),
                    "box_xyxy": [float(x) for x in b.xyxy[0].tolist()]
                })
        return jsonify({"type": "detection", "detections": detections})

    finally:
        # Clean up the temp file
        try:
            os.remove(fpath)
        except Exception:
            pass

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # Access at http://127.0.0.1:5000/
    app.run(host="127.0.0.1", port=5000, debug=True)



