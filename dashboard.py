import cv2
import numpy as np
import threading
import time
import json
import os
from flask import Flask, render_template, Response, request, jsonify
from detection import ObjectDetector as Detection
from segmentation_tracking import DeepSORTTracker as SegmentationTracking
from safety_engine import SafetyEngine
from visualization import Visualizer
from trajectory_storage import TrajectoryStorage
import logging

app = Flask(__name__)

# Global variables for dashboard state
current_frame = None
processing_active = False
pause_processing = False
reset_tracking = False
metrics = {
    'precision': 0.0,
    'recall': 0.0,
    'fn_rate': 0.0,
    'tracked_persons': 0
}
safety_alerts = {
    'status': 'SAFE',
    'time_to_collision': None
}
failure_insights = {
    'latest_failure': 'None'
}

# Initialize modules
detector = Detection()
tracker = SegmentationTracking()
safety_engine = SafetyEngine()
visualizer = Visualizer()
trajectory_storage = TrajectoryStorage()

# Video capture
cap = None
video_source = 0  # 0 for webcam, or path to video file, or None for mock
mock_mode = False

def initialize_video():
    global cap, mock_mode
    if video_source is None:
        mock_mode = True
        return True
    if isinstance(video_source, str):
        cap = cv2.VideoCapture(video_source)
    else:
        cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logging.warning("Could not open video source. Using mock mode.")
        mock_mode = True
        return True
    return True

def process_frame(frame, frame_index):
    global metrics, safety_alerts, failure_insights

    # Detection
    detections = detector.detect(frame)

    # Tracking
    tracks_dict = tracker.update(detections, frame)

    # Update trajectories and compute risks
    risk_results = []
    tracks = []
    for obj_id, (bbox, mask, is_pred, is_occ, occ_dur) in tracks_dict.items():
        trajectory_storage.update(obj_id, bbox)
        trajectory = trajectory_storage.get_trajectory(obj_id)
        risk = safety_engine.compute_risk(obj_id, bbox, trajectory, frame_index)
        risk['id'] = obj_id  # Add id to risk dict
        risk_results.append(risk)
        tracks.append({'id': obj_id, 'bbox': bbox})

    # Annotate frame
    annotated_frame = visualizer.draw_overlays(frame, tracks, trajectory_storage, risk_results)

    # Update metrics (simplified)
    if detections:
        tp = len([d for d in detections if d[1] > 0.5])  # conf > 0.5
        fp = len(detections) - tp
        fn = 0  # Would need ground truth
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        metrics.update({
            'precision': precision,
            'recall': recall,
            'fn_rate': fn_rate,
            'tracked_persons': len(tracks)
        })

    # Update safety alerts
    if risk_results:
        max_risk = max(risk_results, key=lambda x: x['risk_score'])
        safety_alerts.update({
            'status': max_risk['risk_level'],
            'time_to_collision': max_risk['time_to_collision']
        })

    # Failure analysis (simplified - would need actual pipeline results)
    # For demo, simulate occasional failures
    if np.random.random() < 0.1:  # 10% chance
        failure_types = ['occlusion', 'blur', 'small object', 'lighting issue']
        failure_insights['latest_failure'] = np.random.choice(failure_types)

    return annotated_frame

def video_processing_loop():
    global current_frame, processing_active, pause_processing, reset_tracking

    frame_index = 0
    while processing_active:
        if pause_processing:
            time.sleep(0.1)
            continue

        if reset_tracking:
            trajectory_storage.reset()
            reset_tracking = False

        if mock_mode:
            # Generate mock frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Mock Frame {frame_index}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # Add mock person
            cv2.rectangle(frame, (200, 150), (400, 350), (0, 255, 0), 2)
        else:
            ret, frame = cap.read()
            if not ret:
                break

        annotated_frame = process_frame(frame, frame_index)
        current_frame = annotated_frame

        frame_index += 1
        time.sleep(0.1)  # 10 FPS for mock

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if current_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', current_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def get_metrics():
    return jsonify(metrics)

@app.route('/alerts')
def get_alerts():
    return jsonify(safety_alerts)

@app.route('/failures')
def get_failures():
    return jsonify(failure_insights)

@app.route('/control', methods=['POST'])
def control():
    global processing_active, pause_processing, reset_tracking
    action = request.json.get('action')
    if action == 'start':
        if not processing_active:
            processing_active = True
            threading.Thread(target=video_processing_loop, daemon=True).start()
    elif action == 'stop':
        processing_active = False
    elif action == 'pause':
        pause_processing = not pause_processing
    elif action == 'reset':
        reset_tracking = True
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    if initialize_video():
        app.run(debug=True, threaded=True)
    else:
        print("Failed to initialize video")