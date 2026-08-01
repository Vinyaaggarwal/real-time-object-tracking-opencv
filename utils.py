"""
utils.py
========
Visualization, metrics, and file-I/O helpers for the tracking system.

Exports:
  - draw_tracked_boxes()   — render bounding boxes + labels on a frame
  - FPSCounter             — rolling FPS calculator
  - VideoWriterCtx         — context manager for annotated video export
  - save_annotated_image() — write a single annotated frame to disk
  - format_stats_dataframe() — convert tracking dict → pandas DataFrame
  - resize_frame()         — constrained aspect-ratio resize
  - bgr_to_rgb()           — OpenCV → PIL / Streamlit colour conversion
  - timestamp_filename()   — generate timestamped output filenames
"""

from __future__ import annotations

import base64
import collections
import io
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from config import DEFAULT_PATHS, get_color_for_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame Drawing
# ---------------------------------------------------------------------------


def draw_tracked_boxes(
    frame: np.ndarray,
    tracked_objects: list,                  # List[TrackedObject] — avoid circular import
    show_bbox: bool = True,
    show_confidence: bool = True,
    show_track_id: bool = True,
    show_class: bool = True,
    box_thickness: int = 2,
    font_scale: float = 0.55,
    font_thickness: int = 1,
) -> np.ndarray:
    """
    Draw bounding boxes and annotation labels on a copy of ``frame``.

    Each box is coloured deterministically by track_id so the same
    object always appears in the same colour across frames.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (H, W, 3).
    tracked_objects : list of TrackedObject
        Tracking results for the current frame.
    show_bbox : bool
        Render bounding rectangles.
    show_confidence : bool
        Include confidence score in label.
    show_track_id : bool
        Include track ID in label.
    show_class : bool
        Include class name in label.
    box_thickness : int
        Rectangle line thickness (px).
    font_scale : float
        OpenCV font scale factor.
    font_thickness : int
        OpenCV font stroke thickness.

    Returns
    -------
    np.ndarray
        Annotated copy of the input frame (BGR).
    """
    output = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.bbox
        color = get_color_for_id(obj.track_id if obj.track_id >= 0 else 0)

        if show_bbox:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness)

        # Build label string
        label_parts: List[str] = []
        if show_class:
            label_parts.append(obj.class_name)
        if show_track_id and obj.track_id >= 0:
            label_parts.append(f"ID:{obj.track_id}")
        if show_confidence:
            label_parts.append(f"{obj.confidence:.0%}")

        if label_parts:
            label = "  ".join(label_parts)
            (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            # Background pill
            pad = 4
            cv2.rectangle(
                output,
                (x1, y1 - lh - baseline - pad * 2),
                (x1 + lw + pad * 2, y1),
                color,
                cv2.FILLED,
            )
            # Label text (white for readability)
            cv2.putText(
                output,
                label,
                (x1 + pad, y1 - baseline - pad),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

    return output


def draw_fps_overlay(frame: np.ndarray, fps: float, inference_ms: float) -> np.ndarray:
    """
    Overlay FPS and inference-time text in the top-left corner.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (H, W, 3).
    fps : float
        Current frames-per-second.
    inference_ms : float
        Inference latency in milliseconds.

    Returns
    -------
    np.ndarray
        Frame with overlay (in-place modification, reference returned).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.60, 2
    lines = [
        f"FPS: {fps:.1f}",
        f"Inf: {inference_ms:.1f} ms",
    ]
    y = 28
    for line in lines:
        # Drop shadow
        cv2.putText(frame, line, (12, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), font, scale, (50, 255, 120), thickness, cv2.LINE_AA)
        y += 26
    return frame


# ---------------------------------------------------------------------------
# FPS Calculator
# ---------------------------------------------------------------------------


class FPSCounter:
    """
    Rolling-window FPS calculator.

    Parameters
    ----------
    window : int
        Number of frame timestamps to keep for the rolling average.
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: Deque[float] = collections.deque(maxlen=window)
        self._last_tick: Optional[float] = None

    def tick(self) -> None:
        """Record a new frame timestamp."""
        now = time.perf_counter()
        self._timestamps.append(now)
        self._last_tick = now

    @property
    def fps(self) -> float:
        """Return the rolling average FPS (0.0 if fewer than 2 ticks)."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        """Clear all stored timestamps."""
        self._timestamps.clear()
        self._last_tick = None


# ---------------------------------------------------------------------------
# Video Writer Context Manager
# ---------------------------------------------------------------------------


class VideoWriterCtx:
    """
    Context manager that writes BGR frames to an MP4 file in ``./outputs/``.

    Usage::

        with VideoWriterCtx(frame_width=1280, frame_height=720, fps=30) as vw:
            for frame in frames:
                vw.write(annotated_frame)
        print(vw.output_path)

    Parameters
    ----------
    frame_width : int
    frame_height : int
    fps : float
        Target output frame rate.
    stem : str
        Filename stem (without extension).  A timestamp is appended.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        fps: float = 25.0,
        stem: str = "tracked",
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.stem = stem
        self.output_path: Optional[Path] = None
        self._writer: Optional[cv2.VideoWriter] = None

    def __enter__(self) -> "VideoWriterCtx":
        filename = timestamp_filename(self.stem, ".mp4")
        self.output_path = DEFAULT_PATHS["outputs"] / filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter at {self.output_path}")
        logger.info("VideoWriter opened: %s", self.output_path)
        return self

    def write(self, frame: np.ndarray) -> None:
        """Write one BGR frame to the output file."""
        if self._writer and self._writer.isOpened():
            self._writer.write(frame)

    def __exit__(self, *args: object) -> None:
        if self._writer:
            self._writer.release()
        logger.info("VideoWriter closed: %s", self.output_path)


# ---------------------------------------------------------------------------
# Image Save Helper
# ---------------------------------------------------------------------------


def save_annotated_image(frame: np.ndarray, stem: str = "detection") -> Path:
    """
    Save a single annotated BGR frame as a PNG in ``./outputs/``.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to save.
    stem : str
        Output filename stem (timestamp is appended).

    Returns
    -------
    Path
        Absolute path of the saved PNG file.
    """
    filename = timestamp_filename(stem, ".png")
    out_path = DEFAULT_PATHS["outputs"] / filename
    success = cv2.imwrite(str(out_path), frame)
    if not success:
        raise OSError(f"cv2.imwrite failed for path: {out_path}")
    logger.info("Saved annotated image: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# DataFrame Formatter
# ---------------------------------------------------------------------------


def format_stats_dataframe(class_counts: Dict[str, int]) -> pd.DataFrame:
    """
    Convert a class-count dictionary into a sorted pandas DataFrame.

    Parameters
    ----------
    class_counts : dict
        Mapping ``{class_name: count}``.

    Returns
    -------
    pd.DataFrame
        Columns: ``["Class", "Count"]``, sorted descending by Count.
    """
    if not class_counts:
        return pd.DataFrame(columns=["Class", "Count"])
    rows = [{"Class": cls, "Count": cnt} for cls, cnt in class_counts.items()]
    df = pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)
    return df


def format_detection_history(history: List[Dict]) -> pd.DataFrame:
    """
    Convert a list of per-frame detection snapshots into a DataFrame.

    Each entry in ``history`` should be a dict with keys:
    ``frame``, ``timestamp``, ``class``, ``track_id``, ``confidence``.

    Parameters
    ----------
    history : list of dict

    Returns
    -------
    pd.DataFrame
    """
    if not history:
        return pd.DataFrame(
            columns=["Frame", "Timestamp", "Class", "Track ID", "Confidence"]
        )
    df = pd.DataFrame(history)
    df.columns = ["Frame", "Timestamp", "Class", "Track ID", "Confidence"]
    return df.tail(200)   # Keep last 200 rows to avoid memory bloat


# ---------------------------------------------------------------------------
# Misc Helpers
# ---------------------------------------------------------------------------


def resize_frame(
    frame: np.ndarray,
    max_width: int = 1280,
    max_height: int = 720,
) -> np.ndarray:
    """
    Resize ``frame`` to fit within ``(max_width, max_height)`` while
    preserving aspect ratio.  Returns the original frame if it already fits.
    """
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return frame
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR OpenCV frame to RGB for Streamlit / PIL display."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def frame_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert a BGR OpenCV frame to a PIL Image (RGB)."""
    return Image.fromarray(bgr_to_rgb(frame))


def frame_to_base64_html(
    frame: np.ndarray,
    quality: int = 80,
    border_radius: str = "12px",
) -> str:
    """
    Encode a BGR frame as a JPEG data-URL and return an HTML ``<img>`` tag.

    Using a base64 data-URL completely bypasses Streamlit's media-file
    storage system, eliminating the ``MediaFileStorageError`` that occurs
    when ``st.rerun()`` evicts cached images before the browser fetches them.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (H, W, 3).
    quality : int
        JPEG compression quality (1-95).  Lower = smaller payload, faster.
    border_radius : str
        CSS border-radius for the ``<img>`` element.

    Returns
    -------
    str
        A self-contained HTML string ready for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    pil = frame_to_pil(frame)          # BGR → RGB PIL Image
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return (
        f'<img src="data:image/jpeg;base64,{b64}" '
        f'style="width:100%;border-radius:{border_radius};'
        f'border:1px solid rgba(108,99,255,0.25);display:block;">'
    )


def timestamp_filename(stem: str, ext: str) -> str:
    """
    Generate a filename with a UTC timestamp appended.

    Example: ``"tracked_20240715_143022.mp4"``
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{ts}{ext}"


def open_video_capture(source: object) -> cv2.VideoCapture:
    """
    Open a ``cv2.VideoCapture`` from a file path or device index.

    For integer (webcam) sources on Windows, tries the DirectShow (DSHOW)
    backend first, which fixes blank/black frames caused by the default
    MSMF backend's slow warm-up. Falls back to the default backend.
    Also performs warm-up reads so the first real ``cap.read()`` call
    returns a valid frame.

    Parameters
    ----------
    source : str | Path | int
        File path string, ``Path`` object, or integer device index.

    Returns
    -------
    cv2.VideoCapture

    Raises
    ------
    RuntimeError
        If the capture could not be opened or returns no frames.
    """
    import platform

    if isinstance(source, Path):
        source = str(source)

    # ── File path — open normally ────────────────────────────────────────────
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {source!r}")
        return cap

    # ── Webcam (integer index) ────────────────────────────────────────────────
    index: int = int(source)
    cap: Optional[cv2.VideoCapture] = None

    on_windows = platform.system() == "Windows"

    # On Windows try DSHOW first — avoids MSMF blank-frame bug
    backends = (
        [cv2.CAP_DSHOW, cv2.CAP_ANY] if on_windows else [cv2.CAP_ANY]
    )

    for backend in backends:
        try:
            c = cv2.VideoCapture(index, backend)
            if not c.isOpened():
                c.release()
                continue
            # Set a reasonable resolution so frames come through quickly
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            c.set(cv2.CAP_PROP_FPS, 30)
            # Warm-up: discard a few frames — Windows cameras need this
            for _ in range(5):
                c.read()
            # Verify we actually get a real frame
            ret, test = c.read()
            if ret and test is not None and test.size > 0:
                logger.info(
                    "Webcam index=%d opened with backend=%d (W=%d H=%d)",
                    index,
                    backend,
                    int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
                cap = c
                break
            c.release()
        except Exception as exc:
            logger.warning("Backend %d failed for index %d: %s", backend, index, exc)

    if cap is None or not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {index}. "
            "Make sure no other application (Zoom, Teams, OBS) is using the camera "
            "and that camera permissions are granted."
        )
    return cap


def find_working_webcam(max_index: int = 3) -> int:
    """
    Probe camera indices 0 … ``max_index`` and return the first one that
    delivers a real frame.

    Parameters
    ----------
    max_index : int
        Highest camera index to try (inclusive).

    Returns
    -------
    int
        The working camera index.

    Raises
    ------
    RuntimeError
        If no working camera is found.
    """
    import platform

    on_windows = platform.system() == "Windows"
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if on_windows else [cv2.CAP_ANY]

    for idx in range(max_index + 1):
        for backend in backends:
            try:
                c = cv2.VideoCapture(idx, backend)
                if not c.isOpened():
                    c.release()
                    continue
                for _ in range(3):
                    c.read()
                ret, frame = c.read()
                c.release()
                if ret and frame is not None and frame.size > 0:
                    logger.info("Found working webcam at index %d (backend %d)", idx, backend)
                    return idx
            except Exception:
                pass

    raise RuntimeError(
        "No working webcam found (tried indices 0–"
        f"{max_index}). Check camera connection and permissions."
    )


def get_video_properties(cap: cv2.VideoCapture) -> Tuple[int, int, float, int]:
    """
    Read basic properties from an open VideoCapture.

    Returns
    -------
    tuple
        ``(width, height, fps, total_frames)``
    """
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return w, h, fps, total


def save_upload_to_disk(uploaded_file: object, dest_dir: Optional[Path] = None) -> Path:
    """
    Write a Streamlit ``UploadedFile`` to ``dest_dir`` (defaults to ``./uploads/``).

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
    dest_dir : Path, optional

    Returns
    -------
    Path
        Absolute path of the saved file.
    """
    if dest_dir is None:
        dest_dir = DEFAULT_PATHS["uploads"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Use a timestamped name to avoid collisions
    name = Path(uploaded_file.name)
    out_name = timestamp_filename(name.stem, name.suffix)
    out_path = dest_dir / out_name

    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    logger.info("Saved upload to %s (%d bytes)", out_path, out_path.stat().st_size)
    return out_path
