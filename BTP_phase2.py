import cv2
import math
import numpy as np
from ultralytics import YOLO
from collections import deque
import pygame
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# -----------------------------
# FLICKER METRICS VARIABLES
# -----------------------------
prev_fall = None
state_changes = 0
fall_segment_lengths = []
current_segment = 0
total_frames = 0

# -----------------------------
# AUDIO
# -----------------------------
pygame.mixer.init()

def generate_beep():
    arr = np.zeros((22050, 2), dtype=np.int16)
    for i in range(22050):
        val = int(4096 * math.sin(2 * math.pi * 800 * i / 22050))
        arr[i] = [val, val]
    return pygame.sndarray.make_sound(arr)

beep_sound = generate_beep()

# -----------------------------
# EMAIL CONFIGURATION
# -----------------------------
def send_email(frame):
    try:
        sender = "s.mallam@iitg.ac.in"
        password = "*********"
        receiver = "s.mallam@iitg.ac.in"

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = receiver
        msg["Subject"] = "⚠ FALL DETECTED ALERT"

        msg.attach(MIMEText("Fall detected. Please check immediately.", "plain"))

        image_path = "fall_snapshot.jpg"
        cv2.imwrite(image_path, frame)

        with open(image_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment; filename=fall.jpg")
            msg.attach(part)

        server = smtplib.SMTP("smtp.office365.com", 587, timeout=15)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()

        print("📧 Email Sent")

    except Exception as e:
        print("Email failed, continuing:", e)


def trigger_email(frame):
    threading.Thread(target=send_email, args=(frame,), daemon=True).start()

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO("yolov8s.pt")

classnames = []
with open("classes.txt", "r") as f:
    classnames = f.read().splitlines()

# -----------------------------
# VIDEO
# -----------------------------
cap = cv2.VideoCapture("input_1.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if math.isnan(fps) or fps <= 0:
    fps = 25.0

out = cv2.VideoWriter("final_output.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (frame_width, frame_height))

# -----------------------------
# STATE VARIABLES
# -----------------------------
prev_cx, prev_cy = None, None
prev_speed = None

last_box = None
missed_frames = 0
MAX_MISSES = 3

CONF_THRESH = 0.3

# -----------------------------
# TEMPORAL WINDOW
# -----------------------------
WINDOW_SIZE = 6
feature_window = deque(maxlen=WINDOW_SIZE)

fall_counter = 0
FALL_CONFIRM = 2

fallen_state = False
fallen_timer = 0
FALL_HOLD_FRAMES = 20

recovery_counter = 0
RECOVERY_CONFIRM = 3

# -----------------------------
# ALARM + EMAIL CONTROL
# -----------------------------
fall_duration_counter = 0
ALARM_TRIGGER_FRAMES = 30
EMAIL_TRIGGER_FRAMES = 120

alarm_active = False
email_sent = False

# -----------------------------
# LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detected = False
    current_box = None

    for info in results:
        for box in info.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if conf < CONF_THRESH:
                continue

            if classnames[cls_id] != "person":
                continue

            current_box = (x1, y1, x2, y2)
            detected = True
            break

        if detected:
            break

    # CONTINUITY FIX
    if detected:
        last_box = current_box
        missed_frames = 0
    else:
        missed_frames += 1
        if last_box is not None and missed_frames <= MAX_MISSES:
            current_box = last_box
        else:
            current_box = None

    # FEATURE EXTRACTION
    if current_box is not None:
        x1, y1, x2, y2 = current_box

        w = x2 - x1
        h = y2 - y1
        if w == 0:
            continue

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if prev_cx is None:
            dx, dy = 0, 0
        else:
            dx = cx - prev_cx
            dy = cy - prev_cy

        speed = math.sqrt(dx**2 + dy**2)

        if prev_speed is None:
            acc = 0
        else:
            acc = speed - prev_speed

        prev_cx, prev_cy = cx, cy
        prev_speed = speed

        aspect_ratio = h / float(w)

        feature_window.append({"acc": acc, "ar": aspect_ratio})

        if len(feature_window) == WINDOW_SIZE:

            accs = [f["acc"] for f in feature_window]
            ars = [f["ar"] for f in feature_window]

            max_acc = max(accs)
            low_ar_count = sum(1 for a in ars if a < 1.2)

            if max_acc > 1.5 and low_ar_count >= 2:
                fall_counter += 1
            else:
                fall_counter = 0

            fall_event = fall_counter >= FALL_CONFIRM

            if fall_event:
                fallen_state = True
                fallen_timer = FALL_HOLD_FRAMES
                recovery_counter = 0

            elif fallen_state:
                fallen_timer -= 1

                if low_ar_count >= 2:
                    fallen_timer = FALL_HOLD_FRAMES
                    recovery_counter = 0
                else:
                    recovery_counter += 1

                if recovery_counter >= RECOVERY_CONFIRM:
                    fallen_state = False
                    recovery_counter = 0

        fall = fallen_state

        color = (0, 0, 255) if fall else (0, 255, 0)
        label = "FALL DETECTED" if fall else "PERSON"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    else:
        fall = False

    # -----------------------------
    # ALARM + EMAIL 
    # -----------------------------
    if fall:
        fall_duration_counter += 1

        if fall_duration_counter >= ALARM_TRIGGER_FRAMES:
            if not alarm_active:
                print("🔴 FALL CONFIRMED - ALARM")
                alarm_active = True

            if fall_duration_counter % 5 == 0:
                beep_sound.play()

        if fall_duration_counter >= EMAIL_TRIGGER_FRAMES and not email_sent:
            trigger_email(frame.copy())
            email_sent = True

    else:
        fall_duration_counter = 0
        alarm_active = False
        email_sent = False

    # -----------------------------
    # FLICKER METRICS
    # -----------------------------
    total_frames += 1

    if prev_fall is not None and fall != prev_fall:
        state_changes += 1

    if fall:
        current_segment += 1
    else:
        if current_segment > 0:
            fall_segment_lengths.append(current_segment)
            current_segment = 0

    prev_fall = fall

    out.write(frame)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27:
        break

# -----------------------------
# FINAL METRICS
# -----------------------------
if current_segment > 0:
    fall_segment_lengths.append(current_segment)

avg_segment = sum(fall_segment_lengths)/len(fall_segment_lengths) if fall_segment_lengths else 0
flicker_rate = state_changes/total_frames if total_frames else 0

print("\n📊 PHASE 2 METRICS")
print("Total frames:", total_frames)
print("State changes:", state_changes)
print("Avg segment length:", avg_segment)
print("Flicker rate:", flicker_rate)

cap.release()
out.release()
cv2.destroyAllWindows()