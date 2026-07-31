# Real-Time Object Tracking System 🎯

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen)](https://ultralytics.com)
[![ByteTrack](https://img.shields.io/badge/Tracker-ByteTrack-orange)](https://github.com/ifzhang/ByteTrack)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/CV-OpenCV-green)](https://opencv.org)

A production-quality, fully modular real-time object tracking system with a sleek Streamlit UI.  
Supports **webcam streams**, **video file upload**, and **static image processing** — all in one app.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Detection Engine** | YOLOv8n / YOLOv8s / YOLOv8m (auto-download) |
| **Tracking Algorithm** | ByteTrack via Ultralytics — persistent IDs across frames |
| **Input Sources** | Webcam · MP4/AVI/MOV upload · JPG/PNG upload |
| **Class Filtering** | All 80 COCO classes, multi-select whitelist |
| **Confidence Slider** | 0.10 – 1.00, adjustable at runtime |
| **Live Metrics** | FPS · Inference ms · Active objects · Total unique IDs |
| **Analytics** | Class-wise bar chart + scrollable detection history log |
| **Export** | Annotated video → `./outputs/` + one-click Streamlit download |
| **Color Coding** | Deterministic HSV colors per track ID — no flickering |

---

## 🗂️ Project Structure

```
real-time-object-tracking-opencv/
│
├── app.py              ← Streamlit UI entry point
├── detector.py         ← YOLOv8Detector (detection-only helper)
├── tracker.py          ← ObjectTracker (ByteTrack integration)
├── utils.py            ← Drawing, FPS, VideoWriter, helpers
├── config.py           ← Constants, paths, CSS, color palette
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
│
├── models/             ← YOLOv8 weight files (auto-downloaded)
├── outputs/            ← Processed video / image exports
├── uploads/            ← Temporary uploaded media
├── assets/             ← Custom CSS / branding
└── screenshots/        ← UI screenshots for docs
```

---

## 🚀 Quick Start

### 1. Clone & enter the repo

```bash
git clone https://github.com/Vinyaaggarwal/real-time-object-tracking-opencv.git
cd real-time-object-tracking-opencv
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

The browser will open at **http://localhost:8501** automatically.  
On first run, YOLOv8 weights are downloaded to `./models/` (~6 MB for nano).

---

## 🎛️ Usage Guide

### Sidebar Controls

| Control | Description |
|---|---|
| **Input Source** | Choose Webcam / Upload Video / Upload Image |
| **Model Weights** | YOLOv8n (fastest) · YOLOv8s (balanced) · YOLOv8m (accurate) |
| **Confidence** | Minimum detection score (0.10 – 1.00) |
| **Target Classes** | Whitelist specific COCO classes (blank = all 80) |
| **Display Toggles** | Show/hide boxes, confidence, IDs, FPS overlay, stats panel |

### Webcam Mode
1. Select **Webcam** in the sidebar.
2. Click **▶ Start Detection**.
3. Watch live bounding boxes + track IDs update in real time.
4. Click **⏹ Stop Detection** to end the session.

### Video Mode
1. Select **Upload Video** and drop an MP4/AVI/MOV file.
2. Click **▶ Process Video**.
3. Watch the live preview + progress bar.
4. Download the processed MP4 via the **⬇️ Download** button.

### Image Mode
1. Select **Upload Image** and drop a JPG/PNG.
2. Click **🔍 Run Detection**.
3. Compare original vs. annotated side-by-side.
4. Download the annotated PNG.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                   app.py (UI)                   │
│  Sidebar ─ Header ─ Mode Router ─ Metrics       │
└────────────────────┬────────────────────────────┘
                     │ calls
        ┌────────────▼────────────┐
        │    tracker.py           │
        │  ObjectTracker          │
        │  ├─ YOLO.track()        │
        │  ├─ ByteTrack state     │
        │  └─ Session stats       │
        └────────────┬────────────┘
                     │ uses
        ┌────────────▼────────────┐
        │    utils.py             │
        │  draw_tracked_boxes()   │
        │  FPSCounter             │
        │  VideoWriterCtx         │
        │  save_annotated_image() │
        └─────────────────────────┘
        ┌─────────────────────────┐
        │    config.py            │
        │  Paths · Models · CSS   │
        │  COCO classes · Colors  │
        └─────────────────────────┘
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.28 | Web UI framework |
| `ultralytics` | ≥ 8.0 | YOLOv8 + ByteTrack |
| `opencv-python-headless` | ≥ 4.8 | Frame I/O, drawing |
| `numpy` | ≥ 1.24 | Array operations |
| `Pillow` | ≥ 9.5 | Image codec |
| `pandas` | ≥ 2.0 | Analytics tables |

---

## ⚙️ Configuration

Edit `config.py` to customise:
- `DEFAULT_CONFIDENCE` — default confidence threshold
- `DEFAULT_IOU` — NMS IOU threshold
- `DEFAULT_IMG_SIZE` — inference resolution
- `MODEL_CONFIGS` — add custom YOLOv8 variants
- `STREAMLIT_CUSTOM_CSS` — tweak the UI theme

---

## 📝 License

MIT — see `LICENSE` for details.

---

## 🤝 Contributing

Pull requests are welcome!  
Please open an issue first to discuss major changes.
