# app.py
import os
import io
import time
import csv
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, send_file

# Windows beep
import winsound

# ----------------- Config -----------------
MODEL_PATH_CD = "models/best (CD).pt"  # Crowd Detection
MODEL_PATH_VD = "models/best (VD).pt"  # Violence Detection
MODEL_PATH_UBD = "models/best (UBD).pt" # Abandoned Baggage Detection
MODEL_PATH_SAD = "models/best (SAD).pt" # Suspicious Activity Detection
CSV_LOG = "detection_log.csv"
PEOPLE_LIMIT = 10                # how many people constitute crowd
DISTANCE_THRESHOLD = 80          # pixels; tune per camera view
CONF_THRES = 0.35                # detection confidence threshold
FRAME_SKIP = 0                    # set >0 to skip frames for speed
STREAM_FPS_LIMIT = 20            # max yield fps
# ------------------------------------------

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Load models once
print("Loading YOLO models...")
try:
    model_cd = YOLO(MODEL_PATH_CD)
    model_vd = YOLO(MODEL_PATH_VD)
    model_ubd = YOLO(MODEL_PATH_UBD)
    model_sad = YOLO(MODEL_PATH_SAD)
    print("Models loaded.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Shared state
state_lock = threading.Lock()
current_status = {"crowd_status": "UNKNOWN", "count": 0, "violence_detected": False, "abandoned_count": 0, "suspicious_detected": False}
prev_status_value = "UNKNOWN"

# Video source control
video_source = {"type": "webcam", "value": 0}
capture = None
capture_lock = threading.Lock()
running = True

# Frame store
latest_frame = None
latest_frame_lock = threading.Lock()

# Ensure CSV exists with header
if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "status", "count", "violence_detected", "abandoned_count", "suspicious_detected"])


def log_status(status, count, violence_detected, abandoned_count, suspicious_detected):
    ts = datetime.now().isoformat(sep=' ', timespec='seconds')
    with open(CSV_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, status, count, violence_detected, abandoned_count, suspicious_detected])


def server_beep_once():
    # Single beep on Windows
    try:
        frequency = 1000  # Hz
        duration = 400    # ms
        winsound.Beep(frequency, duration)
    except Exception as e:
        print("Beep failed:", e)


def compute_crowd_from_centroids(centroids, people_limit=PEOPLE_LIMIT, distance_threshold=DISTANCE_THRESHOLD):
    """Return (crowded_bool, count) given list of (x,y) centroids"""
    N = len(centroids)
    if N == 0:
        return False, 0
    pts = np.array(centroids)
    # pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]   # (N,N,2)
    dists = np.sqrt((diff ** 2).sum(axis=2))  # (N,N)
    # For each point count how many other points are within threshold (including self)
    close_counts = (dists < distance_threshold).sum(axis=1)
    # If any point has >= people_limit (i.e. cluster of that many)
    crowded = np.any(close_counts >= people_limit)
    return crowded, N


def process_frame(frame):
    """Run models on frame, draw boxes and return annotated frame and statuses"""
    # Run inferences on all models in parallel using ThreadPoolExecutor
    def run_model(model, frame):
        return model(frame, imgsz=640, conf=CONF_THRES, verbose=False)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_model, model_cd, frame): 'cd',
            executor.submit(run_model, model_vd, frame): 'vd',
            executor.submit(run_model, model_ubd, frame): 'ubd',
            executor.submit(run_model, model_sad, frame): 'sad'
        }
        results = {}
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()

    results_cd = results['cd']
    results_vd = results['vd']
    results_ubd = results['ubd']
    results_sad = results['sad']

    # Process Crowd Detection (CD)
    boxes_cd = []
    centroids = []
    res_cd = results_cd[0]
    if hasattr(res_cd, "boxes") and res_cd.boxes is not None:
        xyxy_cd = res_cd.boxes.xyxy.cpu().numpy()
        confs_cd = res_cd.boxes.conf.cpu().numpy() if hasattr(res_cd.boxes, "conf") else np.ones(len(xyxy_cd))
        for i, (box, conf) in enumerate(zip(xyxy_cd, confs_cd)):
            if conf < CONF_THRES:
                continue
            x1, y1, x2, y2 = map(int, box)
            boxes_cd.append((x1, y1, x2, y2, float(conf)))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centroids.append((cx, cy))

    crowded, count = compute_crowd_from_centroids(centroids)

    # Process Violence Detection (VD)
    violence_detected = False
    boxes_vd = []
    res_vd = results_vd[0]
    if hasattr(res_vd, "boxes") and res_vd.boxes is not None:
        xyxy_vd = res_vd.boxes.xyxy.cpu().numpy()
        confs_vd = res_vd.boxes.conf.cpu().numpy() if hasattr(res_vd.boxes, "conf") else np.ones(len(xyxy_vd))
        cls_vd = None
        try:
            cls_vd = res_vd.boxes.cls.cpu().numpy()
        except Exception:
            cls_vd = None
        for i, (box, conf) in enumerate(zip(xyxy_vd, confs_vd)):
            if conf < CONF_THRES:
                continue
            # Assume class 0 is violence or check if 'violence' in names
            if cls_vd is not None and len(cls_vd) > i:
                class_id = int(cls_vd[i])
                # Assuming model has 'violence' as a class, but since we don't know names, assume any detection is violence
                violence_detected = True
            else:
                violence_detected = True  # If no class info, assume detection means violence
            x1, y1, x2, y2 = map(int, box)
            boxes_vd.append((x1, y1, x2, y2, float(conf)))

    # Process Abandoned Baggage Detection (UBD)
    abandoned_count = 0
    boxes_ubd = []
    res_ubd = results_ubd[0]
    if hasattr(res_ubd, "boxes") and res_ubd.boxes is not None:
        xyxy_ubd = res_ubd.boxes.xyxy.cpu().numpy()
        confs_ubd = res_ubd.boxes.conf.cpu().numpy() if hasattr(res_ubd.boxes, "conf") else np.ones(len(xyxy_ubd))
        for i, (box, conf) in enumerate(zip(xyxy_ubd, confs_ubd)):
            if conf < CONF_THRES:
                continue
            abandoned_count += 1
            x1, y1, x2, y2 = map(int, box)
            boxes_ubd.append((x1, y1, x2, y2, float(conf)))

    # Process Suspicious Activity Detection (SAD)
    suspicious_detected = False
    boxes_sad = []
    res_sad = results_sad[0]
    if hasattr(res_sad, "boxes") and res_sad.boxes is not None:
        xyxy_sad = res_sad.boxes.xyxy.cpu().numpy()
        confs_sad = res_sad.boxes.conf.cpu().numpy() if hasattr(res_sad.boxes, "conf") else np.ones(len(xyxy_sad))
        for i, (box, conf) in enumerate(zip(xyxy_sad, confs_sad)):
            if conf < CONF_THRES:
                continue
            suspicious_detected = True
            x1, y1, x2, y2 = map(int, box)
            boxes_sad.append((x1, y1, x2, y2, float(conf)))

    # Annotate frame
    out = frame.copy()
    # Draw CD boxes in white
    for (x1, y1, x2, y2, conf) in boxes_cd:
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(out, f"Person {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Draw VD boxes in red
    for (x1, y1, x2, y2, conf) in boxes_vd:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, f"Violence {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # Draw UBD boxes in yellow
    for (x1, y1, x2, y2, conf) in boxes_ubd:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, f"Abandoned {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    # Draw SAD boxes in blue
    for (x1, y1, x2, y2, conf) in boxes_sad:
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(out, f"Suspicious {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Overlay statuses
    y_offset = 20
    # Crowd status
    crowd_text = f"Crowd: {'CROWDED' if crowded else 'NON-CROWDED'} ({count})"
    crowd_color = (0, 0, 255) if crowded else (0, 255, 0)
    cv2.putText(out, crowd_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, crowd_color, 2)
    y_offset += 30
    # Violence status
    violence_text = f"Violence: {'DETECTED' if violence_detected else 'NONE'}"
    violence_color = (0, 0, 255) if violence_detected else (0, 255, 0)
    cv2.putText(out, violence_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, violence_color, 2)
    y_offset += 30
    # Abandoned status
    abandoned_text = f"Abandoned: {abandoned_count}"
    abandoned_color = (0, 255, 255) if abandoned_count > 0 else (0, 255, 0)
    cv2.putText(out, abandoned_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, abandoned_color, 2)
    y_offset += 30
    # Suspicious status
    suspicious_text = f"Suspicious: {'DETECTED' if suspicious_detected else 'NONE'}"
    suspicious_color = (255, 0, 0) if suspicious_detected else (0, 255, 0)
    cv2.putText(out, suspicious_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, suspicious_color, 2)

    return out, crowded, count, violence_detected, abandoned_count, suspicious_detected


def video_capture_thread():
    global capture, video_source, latest_frame, current_status, prev_status_value, running
    last_yield = 0
    frame_idx = 0
    while running:
        with capture_lock:
            if capture is None:
                # create capture based on video_source
                try:
                    if video_source["type"] == "webcam":
                        cap_src = int(video_source["value"])
                    else:
                        cap_src = video_source["value"]
                    capture = cv2.VideoCapture(cap_src)
                    # small warm-up
                    time.sleep(0.5)
                except Exception as e:
                    print("Failed to open capture:", e)
                    capture = None

        if capture is None or not capture.isOpened():
            time.sleep(0.5)
            continue

        ret, frame = capture.read()
        if not ret or frame is None:
            # For files, loop by seeking to beginning; for others, restart capture
            if video_source["type"] == "file":
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = capture.read()
                if not ret or frame is None:
                    # If still can't read, release
                    with capture_lock:
                        try:
                            capture.release()
                        except Exception:
                            pass
                        capture = None
                    time.sleep(0.5)
                    continue
            else:
                # restart capture (some rtsp streams drop)
                with capture_lock:
                    try:
                        capture.release()
                    except Exception:
                        pass
                    capture = None
                time.sleep(0.5)
                continue

        frame_idx += 1
        if FRAME_SKIP and (frame_idx % (FRAME_SKIP + 1)) != 0:
            continue

        annotated, crowded, count, violence_detected, abandoned_count, suspicious_detected = process_frame(frame)

        # update shared status (with beep on transition)
        with state_lock:
            new_status_str = "CROWDED" if crowded else "NON-CROWDED"
            # single beep when SAFE -> CROWDED
            global prev_status_value
            prev = current_status.get("crowd_status", "UNKNOWN")
            # update
            current_status["crowd_status"] = new_status_str
            current_status["count"] = int(count)
            current_status["violence_detected"] = violence_detected
            current_status["abandoned_count"] = abandoned_count
            current_status["suspicious_detected"] = suspicious_detected
            # log each time state is evaluated (optional: log only on change)
            log_status(new_status_str, count, violence_detected, abandoned_count, suspicious_detected)

            if prev != "CROWDED" and new_status_str == "CROWDED":
                # server beep once
                try:
                    threading.Thread(target=server_beep_once, daemon=True).start()
                except Exception as e:
                    print("Beep thread error:", e)

        # store latest_frame bytes for MJPEG streaming
        ret2, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret2:
            continue
        frame_bytes = buf.tobytes()
        with latest_frame_lock:
            latest_frame = frame_bytes

        # limit server-side processing fps
        now = time.time()
        if STREAM_FPS_LIMIT > 0:
            elapsed = now - last_yield
            min_interval = 1.0 / STREAM_FPS_LIMIT
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_yield = time.time()

    # cleanup on exit
    with capture_lock:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        capture = None


# Start capture thread
t = threading.Thread(target=video_capture_thread, daemon=True)
t.start()


# ------------- Flask routes --------------
@app.route("/")
def index():
    return render_template("index.html")


def generate_mjpeg():
    """Generator yielding multipart JPEG frames from latest_frame"""
    while True:
        with latest_frame_lock:
            frame = latest_frame
        if frame is None:
            # return a blank frame occasionally
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buf = cv2.imencode(".jpg", blank)
            frame = buf.tobytes()
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        yield boundary
        # small sleep to avoid tight loop
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/status")
def status():
    with state_lock:
        return jsonify(current_status)


@app.route("/set_source", methods=["POST"])
def set_source():
    """Change video source: JSON {type:'webcam'|'rtsp'|'file', value:0 or 'rtsp://..'}"""
    data = request.form or request.json or {}
    src_type = data.get("type")
    value = data.get("value")
    if src_type is None or value is None:
        return jsonify({"ok": False, "error": "Provide 'type' and 'value'"}), 400

    with capture_lock:
        global capture
        # close existing capture
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
            capture = None

        if src_type == "webcam":
            try:
                value_int = int(value)
                video_source["type"] = "webcam"
                video_source["value"] = value_int
            except:
                return jsonify({"ok": False, "error": "webcam value must be int index"}), 400
        elif src_type == "rtsp":
            video_source["type"] = "rtsp"
            video_source["value"] = value
        elif src_type == "file":
            video_source["type"] = "file"
            video_source["value"] = value
        else:
            return jsonify({"ok": False, "error": "invalid type"}), 400

    return jsonify({"ok": True, "type": video_source["type"], "value": video_source["value"]})


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Upload image or video for quick testing.
    For images: we run detection on the image and return annotated result.
    For videos: set as video source for live analysis and redirect to index.
    """
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = file.filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # try read as image first
    img = cv2.imread(save_path)
    if img is not None:
        annotated, crowded, count, violence_detected, abandoned_count, suspicious_detected = process_frame(img)
        # convert to JPEG and send back
        ret, buf = cv2.imencode(".jpg", annotated)
        return send_file(io.BytesIO(buf.tobytes()), mimetype='image/jpeg', as_attachment=False, download_name="annotated.jpg")

    # else assume video: set as source for live analysis
    # Check if it's a valid video by trying to open
    cap_test = cv2.VideoCapture(save_path)
    if not cap_test.isOpened():
        cap_test.release()
        return "Cannot process uploaded file: not a valid video or image", 400
    cap_test.release()

    # Set video source to the uploaded file
    with capture_lock:
        global capture, video_source
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
            capture = None
        video_source["type"] = "file"
        video_source["value"] = save_path

    return redirect(url_for('index'))


@app.route("/download_log")
def download_log():
    return send_file(CSV_LOG, as_attachment=True)


@app.route("/shutdown", methods=["POST"])
def shutdown():
    # for convenience during development
    global running
    running = False
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    return "shutting down..."


if __name__ == "__main__":
    # Run Flask on port 5000, accessible locally
    app.run(host="0.0.0.0", port=5000, threaded=True)
