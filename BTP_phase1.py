import cv2
import math
import time
import threading
from ultralytics import YOLO
import pygame
import numpy as np
import joblib
import smtplib
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

# ----------------------------------------------------
# EMAIL CONFIGURATION
# ----------------------------------------------------
SEND_EMAIL = True
OUTLOOK_ADDRESS = "s.mallam@iitg.ac.in"
TO_EMAIL = "s.mallam@iitg.ac.in"
OUTLOOK_APP_PASSWORD = "*********"
email_sent_for_this_fall = False

# ----------------------------------------------------
# EMAIL FUNCTIONS
# ----------------------------------------------------
def send_email_outlook(image_path):
    msg = MIMEMultipart()
    msg["From"] = OUTLOOK_ADDRESS
    msg["To"] = TO_EMAIL
    msg["Subject"] = "⚠ FALL DETECTED ALERT"

    body = "A fall has been detected and confirmed for more than 10 seconds.\nImage is attached."
    msg.attach(MIMEText(body, "plain"))

    try:
        with open(image_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition",
                            f"attachment; filename=fall_snapshot.jpg")
            msg.attach(part)
    except Exception as e:
        print("⚠ Could not attach image:", e)

    try:
        server = smtplib.SMTP("smtp.office365.com", 587)
        server.starttls()
        server.login(OUTLOOK_ADDRESS, OUTLOOK_APP_PASSWORD)
        server.sendmail(OUTLOOK_ADDRESS, TO_EMAIL, msg.as_string())
        server.quit()
        print("📧 Outlook alert sent (with image)!")
    except Exception as e:
        print("❌ Outlook error:", e)

def send_fall_alert_email(frame):
    image_path = "fall_snapshot.jpg"
    cv2.imwrite(image_path, frame)
    send_email_outlook(image_path)

# ----------------------------------------------------
# LOAD ML
# ----------------------------------------------------
try:
    svm_model = joblib.load("fall_svm.pkl")
    scaler = joblib.load("svm_scaler.pkl")
    ML_ENABLED = True
    print("✔ ML Model Loaded")
except:
    ML_ENABLED = False
    print("⚠ ML Model NOT FOUND")

# ----------------------------------------------------
# AUDIO
# ----------------------------------------------------
pygame.mixer.pre_init(22050, -16, 2, 512)
pygame.mixer.init()

def generate_beep_sound(frequency=1000, duration=0.1, sample_rate=22050):
    frames = int(duration * sample_rate)
    arr = np.zeros((frames, 2), dtype=np.int16)
    for i in range(frames):
        wave = int(4096 * math.sin(2 * math.pi * frequency * i / sample_rate))
        arr[i] = [wave, wave]
    return arr

beep_sound = pygame.sndarray.make_sound(generate_beep_sound())

alarm_thread = None
stop_event = threading.Event()

def play_alarm():
    while not stop_event.is_set():
        beep_sound.play()
        if stop_event.wait(0.3):
            break

def start_alarm():
    global alarm_thread
    if alarm_thread is None or not alarm_thread.is_alive():
        stop_event.clear()
        alarm_thread = threading.Thread(target=play_alarm, daemon=True)
        alarm_thread.start()

def stop_alarm():
    global alarm_thread
    if alarm_thread and alarm_thread.is_alive():
        stop_event.set()
        pygame.mixer.stop()
        alarm_thread.join(timeout=1)
        stop_event.clear()

# ----------------------------------------------------
# YOLO + VIDEO
# ----------------------------------------------------
model = YOLO("yolov8s.pt")

classnames = []
with open("classes.txt", "r") as f:
    classnames = f.read().splitlines()

cap = cv2.VideoCapture("input_1.mp4")

w = int(cap.get(3))
h = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
if math.isnan(fps) or fps <= 0:
    fps = 25

out = cv2.VideoWriter("phase1_output.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (w, h))

# ----------------------------------------------------
# STATE
# ----------------------------------------------------
prev_cy = None
prev_h = None

fall_start_time = None
fall_threshold_sec = 5
EMAIL_DELAY = 10

RECOVERY_AR_THRESHOLD = 1.1
RECOVERY_STABLE_FRAMES = 12
recovery_ar_counter = 0
last_frame_max_ar = None

# ----------------------------------------------------
# LOOP
# ----------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    fall_detected_any = False
    person_ars = []

    for info in results:
        for box in info.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf < 0.8:
                continue
            if classnames[cls] != "person":
                continue

            w = x2-x1
            h = y2-y1
            if w == 0:
                continue

            ar = h/float(w)
            orient = w/float(h) if h>0 else 0

            cy = (y1+y2)//2

            dy = None if prev_cy is None else cy-prev_cy
            dh = None if prev_h is None else prev_h-h

            prev_cy = cy
            prev_h = h

            person_ars.append(ar)

            if ML_ENABLED and dy is not None and dh is not None:
                area = w*h
                features = scaler.transform([[ar,orient,dy,dh,area]])
                ml = svm_model.predict(features)[0]
            else:
                ml = 0

            rule = (ar<=0.75 and orient>=1.2) or (dy and dy>5) or (dh and dh>7)

            if ar > 1.25:
                rule = False
                ml = 0

            fall = rule or (ml==1)

            if fall:
                fall_detected_any = True
                color=(0,0,255)
                label="FALL DETECTED"
            else:
                color=(0,255,0)
                label="PERSON"

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

    if person_ars:
        last_frame_max_ar = max(person_ars)
    else:
        last_frame_max_ar = None

    if fall_detected_any:
        recovery_ar_counter = 0

        if fall_start_time is None:
            fall_start_time = time.time()
        else:
            elapsed = time.time() - fall_start_time

            if elapsed >= fall_threshold_sec:
                start_alarm()

            if SEND_EMAIL and elapsed >= EMAIL_DELAY and not email_sent_for_this_fall:
                threading.Thread(target=send_fall_alert_email,
                                 args=(frame.copy(),),daemon=True).start()
                email_sent_for_this_fall = True
    else:
        if fall_start_time is not None:
            if last_frame_max_ar and last_frame_max_ar > RECOVERY_AR_THRESHOLD:
                recovery_ar_counter += 1
            else:
                recovery_ar_counter = 0

            if recovery_ar_counter >= RECOVERY_STABLE_FRAMES:
                stop_alarm()
                fall_start_time = None
                email_sent_for_this_fall = False
                recovery_ar_counter = 0

    # -----------------------------
    # FLICKER METRICS 
    # -----------------------------
    total_frames += 1

    if prev_fall is not None and fall_detected_any != prev_fall:
        state_changes += 1

    if fall_detected_any:
        current_segment += 1
    else:
        if current_segment > 0:
            fall_segment_lengths.append(current_segment)
            current_segment = 0

    prev_fall = fall_detected_any

    out.write(frame)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1)==27:
        break

# -----------------------------
# FINAL METRICS
# -----------------------------

if current_segment > 0:
    fall_segment_lengths.append(current_segment)

avg_segment = sum(fall_segment_lengths)/len(fall_segment_lengths) if fall_segment_lengths else 0
flicker_rate = state_changes/total_frames if total_frames else 0

print("\n📊 PHASE 1 METRICS")
print("Total frames:", total_frames)
print("State changes:", state_changes)
print("Avg segment length:", avg_segment)
print("Flicker rate:", flicker_rate)

cap.release()
out.release()
cv2.destroyAllWindows()