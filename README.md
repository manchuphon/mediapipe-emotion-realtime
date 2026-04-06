# 😊 MediaPipe Emotion & Body Language — Real-time Detection

> ตรวจจับ **อารมณ์และภาษากาย** (Body Language) แบบ Real-time ผ่านกล้องเว็บแคม  
> โดยใช้ **MediaPipe Holistic + Face Mesh** สกัด landmarks → **Machine Learning** จำแนกอารมณ์

---


## 📌 Overview

โปรเจกต์นี้สร้างระบบที่ตรวจจับท่าทางร่างกายและใบหน้าจากกล้องเว็บแคมแบบ real-time แล้วจำแนกเป็นอารมณ์ 4 ประเภท โดยใช้ MediaPipe Holistic ร่วมกับ Face Mesh เพื่อเก็บ landmarks ทั้งร่างกายและใบหน้า จากนั้นฝึก ML classifier และแสดงผลทำนายบนหน้าจอแบบ real-time

**อารมณ์ที่รองรับ:**

| คลาส | ความหมาย | จำนวนตัวอย่าง (newdata.csv) |
|------|----------|--------------------------|
| 😊 Happy | มีความสุข | 206 |
| 😢 Sad | เศร้า | 363 |
| 🏆 Victorious | ฉลองชัยชนะ | 335 |
| 😭 Crying | ร้องไห้ | 221 |

---
<img width="751" height="562" alt="Screenshot 2026-04-07 040747" src="https://github.com/user-attachments/assets/01ed8d79-f7f1-4cb4-80f0-3743cea3feb1" />
## 🗂️ โครงสร้างโปรเจกต์

```
mediapipe-emotion-realtime/
│
├── Body_Language.ipynb     # Notebook หลัก — Pipeline ตั้งแต่ detect → train → predict
├── body_language.pkl       # โมเดล Random Forest ที่ train แล้ว (พร้อมใช้งาน)
├── coords.csv              # Dataset v1 — 3 คลาส (Happy, Sad, Victorious) | 838 rows
└── newdata.csv             # Dataset v2 — 4 คลาส (เพิ่ม Crying) | 1,125 rows
```

---

## ⚙️ Pipeline การทำงาน (Body_Language.ipynb)

```
[กล้อง] → [MediaPipe Holistic + Face Mesh] → [Export landmarks → CSV]
         → [Train ML Models] → [Load Model] → [Real-time Prediction + Display]
```

### ขั้นตอนที่ 1 — Make Detections (Section 1)

ใช้ MediaPipe พร้อมกัน 3 โมดูลเพื่อวาด landmarks บนหน้าจอ:

| ส่วน | โมดูล | สี |
|------|-------|---|
| ใบหน้า (468 จุด) | `mp.solutions.face_mesh` + FACEMESH_TESSELATION | เขียว |
| มือขวา (21 จุด) | `mp.solutions.holistic` | น้ำเงิน-ม่วง |
| มือซ้าย (21 จุด) | `mp.solutions.holistic` | ชมพู-ม่วง |
| ท่าทางร่างกาย (33 จุด) | `mp.solutions.holistic` | ส้ม |

### ขั้นตอนที่ 2 — Capture Landmarks & Export to CSV (Section 2)

สกัด landmarks จาก 2 ส่วนหลักแล้วรวมเป็น feature vector เดียว:

```
Pose landmarks:   33 จุด × 4 ค่า (x, y, z, visibility) =   132 features
Face landmarks:  468 จุด × 4 ค่า (x, y, z, visibility) = 1,872 features
                                               รวมทั้งหมด = 2,004 features + 1 label
```

**วิธีเก็บข้อมูล:** กด `s` เพื่อบันทึก frame ปัจจุบันพร้อม label → append ต่อท้าย CSV

**Dataset ที่ได้:**

| ไฟล์ | คลาส | จำนวน rows | จำนวน features |
|------|------|-----------|---------------|
| `coords.csv` | Happy, Sad, Victorious | 838 | 2,004 |
| `newdata.csv` | Happy, Sad, Victorious, Crying | 1,125 | 2,004 |

### ขั้นตอนที่ 3 — Train ML Models (Section 3)

ฝึก 4 โมเดลพร้อม `StandardScaler` ผ่าน sklearn Pipeline:

| ชื่อย่อ | โมเดล | Pipeline |
|--------|-------|---------|
| `lr` | Logistic Regression | StandardScaler → LogisticRegression |
| `rc` | Ridge Classifier | StandardScaler → RidgeClassifier |
| `rf` | Random Forest | StandardScaler → RandomForestClassifier |
| `gb` | Gradient Boosting | StandardScaler → GradientBoostingClassifier |

**แบ่งข้อมูล:** Train 70% / Test 30% (`random_state=1234`)  
**บันทึกโมเดล:** เลือก `rf` (Random Forest) → บันทึกเป็น `body_language.pkl`

### ขั้นตอนที่ 4 — Real-time Prediction (Section 4)

โหลด `body_language.pkl` → ตรวจจับ landmarks ทุก frame → ทำนายอารมณ์  
แสดงผลบนหน้าจอที่ตำแหน่ง **LEFT_EAR** ของ pose landmark พร้อม confidence จาก `predict_proba`

```python
# ตำแหน่งแสดงผล — คำนวณจาก LEFT_EAR landmark
coords = tuple(np.multiply(
    [pose.LEFT_EAR.x, pose.LEFT_EAR.y],
    [640, 480]
).astype(int))

# วาด label box + ชื่ออารมณ์
cv2.rectangle(image, (coords[0], coords[1]+5),
              (coords[0] + len(class_name)*20, coords[1]-30), (245,117,16), -1)
cv2.putText(image, class_name, coords, ...)
```

---

## 🛠️ การติดตั้ง

### ความต้องการของระบบ

- Python 3.8+
- เว็บแคม
- macOS / Windows / Linux

### ติดตั้ง Dependencies

```bash
pip install mediapipe opencv-python pandas scikit-learn numpy tensorflow
```

> หมายเหตุ: `tensorflow` ใช้ในการตรวจสอบ version เท่านั้น ไม่ได้ใช้ใน training

---

## 🚀 วิธีใช้งาน

### วิธีที่ 1: ใช้โมเดลที่มีอยู่แล้ว (Quickstart)

โมเดล `body_language.pkl` ที่ train เสร็จแล้วอยู่ในโปรเจกต์  
รัน Section 4 ของ Notebook ได้เลยโดยไม่ต้อง train ใหม่:

```bash
jupyter notebook Body_Language.ipynb
# → รัน Section 4: Make Detections with Model
# → กด 'q' เพื่อออก
```

### วิธีที่ 2: Train โมเดลใหม่จากศูนย์

**ขั้นตอนที่ 1 — เก็บข้อมูล (Section 2)**

```python
# ตั้งชื่อคลาสที่ต้องการบันทึก
class_name = "Happy"   # หรือ "Sad", "Victorious", "Crying"
```

รัน cell เก็บข้อมูล → แสดงใบหน้าและท่าทางต่อกล้อง → กด `s` เพื่อบันทึก → กด `q` เมื่อเสร็จ

**ขั้นตอนที่ 2 — Train โมเดล (Section 3)**

รัน cell ทั้งหมดใน Section 3 โมเดลจะถูกบันทึกอัตโนมัติเป็น `body_language.pkl`

**ขั้นตอนที่ 3 — ทดสอบ Real-time (Section 4)**

รัน Section 4 → กด `q` เพื่อออกจากหน้าต่างกล้อง

---

## 📊 โครงสร้าง Feature Vector

MediaPipe ตรวจจับ landmarks ทั้งหมด **501 จุด** ซึ่งแต่ละจุดมี 4 ค่า:

```
ค่าแต่ละ landmark:
  x = พิกัดแนวนอน (normalized 0-1)
  y = พิกัดแนวตั้ง (normalized 0-1)
  z = ความลึก (relative depth)
  v = visibility score (0-1)
```

```
Pose  (Holistic):  33 landmarks × 4 =   132 features
Face  (Face Mesh): 468 landmarks × 4 = 1,872 features
──────────────────────────────────────────────────────
Total:                                 2,004 features + 1 label column
```

---

## 🧰 Tech Stack

| ส่วน | เทคโนโลยี |
|------|-----------|
| Body Tracking | [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic) |
| Face Tracking | [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh) |
| Computer Vision | OpenCV |
| Machine Learning | scikit-learn (LR, Ridge, RF, GradientBoosting) |
| Data Processing | NumPy, pandas |
| Model Serialization | pickle |
| Notebook | Jupyter Notebook |

---

## 💡 หมายเหตุ

- `coords.csv` คือ dataset รุ่นแรก (3 คลาส) — `newdata.csv` คือรุ่นใหม่ที่เพิ่มคลาส Crying
- โมเดล `body_language.pkl` train จาก Random Forest ซึ่งให้ผลดีที่สุดจากการทดสอบ
- ประสิทธิภาพการตรวจจับขึ้นอยู่กับแสงสว่างและระยะห่างจากกล้อง
- ยิ่งเก็บข้อมูลหลาย frame ต่อคลาส โมเดลยิ่งแม่นยำมากขึ้น

---

## 👤 ผู้พัฒนา

**Manchuphon (Aoy)**  


---

## 📄 License

This project is for educational purposes.
