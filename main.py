import cv2
import time
import json
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Open camera
cap = cv2.VideoCapture(0)

# Timing + stability
last_action_time = 0
cooldown = 3  # seconds
required_frames = 3

detection_history = {}
last_label = None

# Output file (QNX will read this)
output_file = "detection.json"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.7, imgsz=320)

    current_labels = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Ignore weak detections
            if conf < 0.7:
                continue

            # Get bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)

            # Ignore small (background) detections
            if area < 5000:
                continue

            current_labels.append(label)

            # Stability check (multi-frame detection)
            detection_history[label] = detection_history.get(label, 0) + 1

            if detection_history[label] < required_frames:
                continue

            current_time = time.time()

            # Cooldown + avoid duplicate writes
            if current_time - last_action_time > cooldown and label != last_label:

                # Define action logic
                if label == "slowdown":
                    action = "reduce_speed"
                    speed = 30

                elif label == "speedlimit":
                    action = "set_speed"
                    speed = 40

                elif label == "crossing":
                    action = "reduce_speed"
                    speed = 25

                elif label == "workinprogress":
                    action = "reduce_speed"
                    speed = 20

                else:
                    continue

                # Final data (QNX readable)
                data = {
                    "label": label,
                    "confidence": round(conf, 2),
                    "action": action,
                    "target_speed": speed,
                    "timestamp": int(current_time)
                }

                # 🔥 REAL-TIME WRITE (LATEST ONLY)
                with open(output_file, "w") as f:
                    json.dump(data, f)
                    f.flush()

                print("✅ SENT:", data)

                last_action_time = current_time
                last_label = label

    # Reset missing detections
    for key in list(detection_history.keys()):
        if key not in current_labels:
            detection_history[key] = 0

    # Show detection window
    annotated = results[0].plot()
    cv2.imshow("Traffic Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()