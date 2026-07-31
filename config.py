"""
config.py
=========
Central configuration module for the Real-Time Object Tracking System.

Defines:
  - Model weight configurations
  - Default filesystem paths
  - COCO 80-class name list
  - Deterministic color palette generation
  - Streamlit custom CSS
  - Default hyper-parameters
"""

from __future__ import annotations

import colorsys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filesystem Paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).parent.resolve()

DEFAULT_PATHS: Dict[str, Path] = {
    "models": BASE_DIR / "models",
    "outputs": BASE_DIR / "outputs",
    "uploads": BASE_DIR / "uploads",
    "assets": BASE_DIR / "assets",
    "screenshots": BASE_DIR / "screenshots",
}

# Ensure every required directory exists at import time
for _dir in DEFAULT_PATHS.values():
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model Configurations
# ---------------------------------------------------------------------------
MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "YOLOv8n (Fastest)": {
        "weights": "yolov8n.pt",
        "description": "Nano — best for real-time on CPU / edge devices",
    },
    "YOLOv8s (Balanced)": {
        "weights": "yolov8s.pt",
        "description": "Small — good accuracy / speed tradeoff",
    },
    "YOLOv8m (Accurate)": {
        "weights": "yolov8m.pt",
        "description": "Medium — higher accuracy, needs a decent GPU",
    },
}

DEFAULT_MODEL: str = "YOLOv8n (Fastest)"

# ---------------------------------------------------------------------------
# Detection Hyper-parameters
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE: float = 0.25
DEFAULT_IOU: float = 0.45
DEFAULT_IMG_SIZE: int = 640
MAX_DETECTIONS: int = 300

# ByteTrack configuration passed to Ultralytics .track()
# Ultralytics >= 8.0 requires the full '.yaml' suffix for tracker names
BYTETRACK_CONFIG: Dict[str, object] = {
    "tracker": "bytetrack.yaml",
    "persist": True,
}

# ---------------------------------------------------------------------------
# COCO 80 Class Names
# ---------------------------------------------------------------------------
COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# COCO class index → name mapping
COCO_CLASS_MAP: Dict[int, str] = {i: name for i, name in enumerate(COCO_CLASSES)}

# ---------------------------------------------------------------------------
# Color Palette (deterministic, HSV-based)
# ---------------------------------------------------------------------------
_NUM_COLORS: int = 128  # enough unique hues for most tracking sessions


def _build_palette(n: int) -> List[Tuple[int, int, int]]:
    """Generate *n* visually distinct BGR colors using the golden-ratio hue step."""
    palette: List[Tuple[int, int, int]] = []
    golden_ratio_conjugate: float = 0.618033988749895
    hue: float = 0.0
    for _ in range(n):
        hue = (hue + golden_ratio_conjugate) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        palette.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return palette


COLOR_PALETTE: List[Tuple[int, int, int]] = _build_palette(_NUM_COLORS)


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Return a deterministic BGR color for a given tracking ID."""
    return COLOR_PALETTE[int(track_id) % len(COLOR_PALETTE)]


def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """Return a deterministic BGR color for a COCO class index."""
    return COLOR_PALETTE[(class_id * 7 + 11) % len(COLOR_PALETTE)]


# ---------------------------------------------------------------------------
# UI / Streamlit configuration
# ---------------------------------------------------------------------------
APP_TITLE: str = "Real-Time Object Tracking System"
APP_SUBTITLE: str = "YOLOv8 + ByteTrack | Multi-Source | Production-Ready"
APP_ICON: str = "🎯"

STREAMLIT_CUSTOM_CSS: str = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e0e0e0;
}

/* ── Header gradient ── */
.hero-header {
    background: linear-gradient(90deg, #6c63ff 0%, #3ecf8e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    color: #8892a4;
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 1.4rem;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(6px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(108,99,255,0.25);
}
div[data-testid="metric-container"] label {
    color: #8892a4 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #6c63ff !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15,15,30,0.92);
    border-right: 1px solid rgba(108,99,255,0.2);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, #6c63ff, #3ecf8e);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
    transition: opacity 0.2s, transform 0.2s;
}
.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-2px);
}

/* ── Stop button variant ── */
.stop-btn button {
    background: linear-gradient(90deg, #ff4d6d, #c9184a) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(90deg, #3ecf8e, #0bbf73);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

/* ── Section dividers ── */
hr {
    border: none;
    border-top: 1px solid rgba(108,99,255,0.2);
    margin: 1.2rem 0;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    background: rgba(108,99,255,0.18);
    border: 1px solid rgba(108,99,255,0.45);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 500;
    color: #a5a0ff;
    margin-right: 0.4rem;
}

/* ── Image / video containers ── */
.stImage img, .stVideo video {
    border-radius: 12px;
    border: 1px solid rgba(108,99,255,0.25);
}

/* ── DataFrame tables ── */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Selectbox & slider labels ── */
.stSelectbox label, .stSlider label, .stMultiSelect label, .stRadio label {
    color: #c0c8d8 !important;
    font-weight: 500 !important;
}

/* ── Info / warning boxes ── */
.stAlert {
    border-radius: 10px;
}
</style>
"""

logger.info("config.py loaded — base dir: %s", BASE_DIR)
