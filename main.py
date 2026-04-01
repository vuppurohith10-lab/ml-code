import cv2
import time
import socket
from ultralytics import YOLO
from collections import deque

# ---------------- SOCKET SETUP ----------------
PI_IP = "192.168.1.7"
PORT = 5000

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    client.connect((PI_IP, PORT))
    print("✅ Connected to Raspberry Pi")
except:
    print("❌ Could not connect to Pi")
    exit()

# ---------------- LOAD MODEL ----------------
model = YOLO("runs/detect/train/weights/best.pt")

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- SETTINGS ----------------
required_frames = 3
cooldown = 1

detection_history = {}
last_action_time = 0
last_label = None

# Buffer for smoothing
label_buffer = deque(maxlen=5)

# Output files
log_file = "detection_log.txt"
latest_file = "latest.txt"

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.7, imgsz=320)

    current_labels = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if conf < 0.7:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)

            if area < 5000:
                continue

            current_labels.append(label)

            # Frame stability check
            detection_history[label] = detection_history.get(label, 0) + 1
            if detection_history[label] < required_frames:
                continue

            # Add to buffer
            label_buffer.append(label)

            # Majority voting
            final_label = max(set(label_buffer), key=label_buffer.count)

            current_time = time.time()

            if current_time - last_action_time > cooldown:

                # -------- ACTION LOGIC --------
                if final_label == "slowdown":
                    action = "notify_only"
                    speed = None

                elif final_label == "speedlimit":
                    action = "set_speed"
                    speed = 30

                elif final_label == "crossing":
                    action = "notify_only"
                    speed = None

                elif final_label == "workinprogress":
                    action = "reduce_speed"
                    speed = 20

                else:
                    continue

                # -------- LOG FORMAT --------
                log_line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {final_label} | Conf:{round(conf,2)} | Action:{action} | Speed:{speed}\n"

                # Write latest
                with open(latest_file, "w") as f:
                    f.write(log_line)
                    f.flush()

                # Write history
                with open(log_file, "a") as f:
                    f.write(log_line)
                    f.flush()

                # -------- SEND TO PI --------
                try:
                    client.send((final_label + "\n").encode())
                    print("📡 SENT TO PI:", final_label)
                except:
                    print("❌ Connection lost")
                    break

                print("✅ LOG:", log_line.strip())

                last_action_time = current_time
                last_label = final_label

    # Reset missing detections
    for key in list(detection_history.keys()):
        if key not in current_labels:
            detection_history[key] = 0

    # Display
    annotated = results[0].plot()
    cv2.imshow("Traffic Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
client.close()