# 🧠 Fall Detection System (BTP)

This project implements a **real-time fall detection system** using:
- YOLOv8 for human detection  
- Rule-based + ML-based fall detection (Phase 1)  
- Event-based stable detection (Phase 2)  
- Alarm and Email alert system  

---

## ⚙️ Requirements

### 🐍 Python Version
Use **Python 3.10.11 (recommended)**

Download from:  
👉 https://www.python.org/downloads/release/python-31011/

---

## 📥 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/mallamsathwika/BTP_Sathwika.git
cd BTP_Sathwika
```

---

### 2️⃣ Download Required Files (IMPORTANT)

Download the following files from Google Drive (https://drive.google.com/drive/folders/1n0UkmiUDoWGVqDDmmMgcJpGU86D6c_YL?usp=sharing):

- fall_svm.pkl
- svm_scaler.pkl
- input_1.mp4
- yolov8s.pt

Place them inside the project folder:

BTP_Sathwika/
  BTP_phase1.py
  BTP_phase2.py
  classes.txt
  fall_svm.pkl
  svm_scaler.pkl
  yolov8s.pt
  input_1.mp4

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the System

#### ▶ Phase 1 (ML + Rule-Based)
```bash
python BTP_phase1.py
```

#### ▶ Phase 2 (Event-Based Detection)
```bash
python BTP_phase2.py
```

---

## ⚠️ Notes


- Ensure internet connection for initial setup.
- Use correct file names as expected in the code.
- Email alerts require valid SMTP credentials.

---

## ✅ Output

- Real-time video with detection
- Alarm triggering after confirmed fall
- Email alert (optional)
- Flicker metrics printed in terminal

---

## 👨‍💻 Author

Mallam Sathwika  
B.Tech CSE, IIT Guwahati
