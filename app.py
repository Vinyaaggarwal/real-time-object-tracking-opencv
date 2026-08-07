"""
app.py
======
Main Streamlit entry point for the Real-Time Object Tracking System.

Launch with:
    streamlit run app.py

Architecture:
  - All heavy inference lives in ``tracker.py`` (ObjectTracker)
  - Visualization helpers come from ``utils.py``
  - Constants / CSS from ``config.py``
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Local modules
from config import (
    APP_ICON,
    APP_SUBTITLE,
    APP_TITLE,
    COCO_CLASSES,
    DEFAULT_CONFIDENCE,
    MODEL_CONFIGS,
    STREAMLIT_CUSTOM_CSS,
)
from tracker import ObjectTracker, TrackingResult, TrackedObject
from utils import (
    FPSCounter,
    VideoWriterCtx,
    bgr_to_rgb,
    draw_fps_overlay,
    draw_tracked_boxes,
    find_working_webcam,
    format_detection_history,
    format_stats_dataframe,
    frame_to_base64_html,
    frame_to_pil,
    get_video_properties,
    open_video_capture,
    resize_frame,
    save_annotated_image,
    save_upload_to_disk,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
st.markdown(STREAMLIT_CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    """Ensure all required session-state keys exist with sensible defaults."""
    defaults: Dict[str, object] = {
        "running": False,
        "tracker": None,
        # Webcam-specific persistent state
        "webcam_cap": None,           # cv2.VideoCapture kept alive across reruns
        "webcam_fps_counter": None,   # FPSCounter kept alive across reruns
        "webcam_frame_idx": 0,
        "webcam_last_frame_html": None,    # last annotated frame as base64 HTML for display after stop
        "webcam_class_counts": {},
        # Shared metrics
        "detection_history": [],
        "frame_count": 0,
        "last_fps": 0.0,
        "last_inference_ms": 0.0,
        "last_active_count": 0,
        "last_unique_ids": 0,
        "last_class_counts": {},
        "output_path": None,
        "annotated_image_bytes": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


# ---------------------------------------------------------------------------
# Helper: get or (re)create tracker
# ---------------------------------------------------------------------------
def _get_tracker(
    model_name: str,
    confidence: float,
    target_classes: Optional[List[str]],
) -> ObjectTracker:
    """
    Return the cached ``ObjectTracker``, rebuilding it if model settings changed.
    """
    tracker: Optional[ObjectTracker] = st.session_state.tracker

    if tracker is None:
        tracker = ObjectTracker(
            model_name=model_name,
            confidence=confidence,
            target_classes=target_classes if target_classes else None,
        )
        st.session_state.tracker = tracker
        return tracker

    # Update dynamic settings without reloading model
    tracker.set_confidence(confidence)
    tracker.set_classes(target_classes if target_classes else None)

    # Reload model only if variant changed
    if tracker.model_name != model_name:
        tracker.set_model(model_name)

    return tracker


# ---------------------------------------------------------------------------
# Helper: append detection history rows
# ---------------------------------------------------------------------------
def _append_history(result: TrackingResult, frame_idx: int) -> None:
    ts = time.strftime("%H:%M:%S")
    for obj in result.tracked_objects:
        st.session_state.detection_history.append(
            [frame_idx, ts, obj.class_name, obj.track_id, f"{obj.confidence:.2f}"]
        )
    # Cap history size
    if len(st.session_state.detection_history) > 5000:
        st.session_state.detection_history = st.session_state.detection_history[-5000:]


# ---------------------------------------------------------------------------
# Helper: update metrics placeholders
# ---------------------------------------------------------------------------
def _update_metrics(
    col_fps,
    col_inf,
    col_active,
    col_unique,
    fps: float,
    inf_ms: float,
    active: int,
    unique: int,
) -> None:
    col_fps.metric("⚡ FPS", f"{fps:.1f}")
    col_inf.metric("🕐 Inference", f"{inf_ms:.1f} ms")
    col_active.metric("📦 Active Objects", active)
    col_unique.metric("🔖 Unique IDs", unique)


# ---------------------------------------------------------------------------
# Sidebar — Controls
# ---------------------------------------------------------------------------
def render_sidebar() -> Dict[str, object]:
    """
    Render sidebar UI and return a dict of all user-selected settings.
    """
    with st.sidebar:
        st.markdown(
            "## 🎯 Control Panel",
        )
        st.markdown("---")

        # ── Input Source ──────────────────────────────────────────────
        source_mode = st.radio(
            "📹 Input Source",
            options=["Webcam", "Upload Video", "Upload Image"],
            index=0,
            key="source_mode",
        )

        st.markdown("---")

        # ── Model Selection ───────────────────────────────────────────
        model_display_name = st.selectbox(
            "🤖 Model Weights",
            options=list(MODEL_CONFIGS.keys()),
            index=0,
            key="model_select",
        )
        st.caption(MODEL_CONFIGS[model_display_name]["description"])

        st.markdown("---")

        # ── Confidence Threshold ──────────────────────────────────────
        confidence = st.slider(
            "🎚️ Confidence Threshold",
            min_value=0.10,
            max_value=1.00,
            value=DEFAULT_CONFIDENCE,
            step=0.05,
            key="confidence_slider",
        )

        st.markdown("---")

        # ── Target Class Filter ───────────────────────────────────────
        target_classes_raw = st.multiselect(
            "🏷️ Target Classes (blank = all)",
            options=COCO_CLASSES,
            default=[],
            key="class_filter",
        )
        target_classes = target_classes_raw if target_classes_raw else []

        st.markdown("---")

        # ── Display Toggles ───────────────────────────────────────────
        st.markdown("**🖥️ Display Toggles**")
        show_bbox = st.checkbox("Bounding Boxes", value=True, key="show_bbox")
        show_conf = st.checkbox("Confidence Scores", value=True, key="show_conf")
        show_tid = st.checkbox("Track IDs", value=True, key="show_tid")
        show_fps = st.checkbox("FPS Overlay", value=True, key="show_fps")
        show_stats = st.checkbox("Statistics Panel", value=True, key="show_stats")

        st.markdown("---")
        st.markdown(
            "<small style='color:#555'>Built with YOLOv8 + ByteTrack + Streamlit</small>",
            unsafe_allow_html=True,
        )

    return {
        "source_mode": source_mode,
        "model_name": model_display_name,
        "confidence": confidence,
        "target_classes": target_classes,
        "show_bbox": show_bbox,
        "show_conf": show_conf,
        "show_tid": show_tid,
        "show_fps": show_fps,
        "show_stats": show_stats,
    }


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def render_header() -> None:
    """Render the gradient hero header and badge row."""
    st.markdown(
        f'<div class="hero-header">{APP_ICON} {APP_TITLE}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="hero-subtitle">{APP_SUBTITLE}</div>',
        unsafe_allow_html=True,
    )
    badges = [
        "YOLOv8", "ByteTrack", "OpenCV", "Streamlit", "Real-Time", "COCO-80",
    ]
    badge_html = "".join(f'<span class="badge">{b}</span>' for b in badges)
    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown("---")


# ---------------------------------------------------------------------------
# Metrics Dashboard
# ---------------------------------------------------------------------------
def render_metrics_dashboard(settings: Dict) -> tuple:
    """
    Render the four live metric cards.

    Returns
    -------
    tuple of four st.empty() placeholders: (fps, inf, active, unique)
    """
    if not settings["show_stats"]:
        return None, None, None, None

    st.markdown("### 📊 Live Metrics")
    c1, c2, c3, c4 = st.columns(4)
    ph_fps = c1.empty()
    ph_inf = c2.empty()
    ph_active = c3.empty()
    ph_unique = c4.empty()

    # Initial values
    ph_fps.metric("⚡ FPS", "—")
    ph_inf.metric("🕐 Inference", "—")
    ph_active.metric("📦 Active Objects", "—")
    ph_unique.metric("🔖 Unique IDs", "—")

    return ph_fps, ph_inf, ph_active, ph_unique


# ---------------------------------------------------------------------------
# Class Breakdown + History
# ---------------------------------------------------------------------------
def render_analytics(settings: Dict) -> tuple:
    """
    Render class-breakdown chart placeholder and detection-history table.

    Returns
    -------
    (chart_placeholder, history_placeholder)
    """
    if not settings["show_stats"]:
        return None, None

    st.markdown("### 📈 Analytics")
    tab_chart, tab_hist = st.tabs(["Class Breakdown", "Detection History"])

    with tab_chart:
        chart_ph = st.empty()

    with tab_hist:
        hist_ph = st.empty()

    return chart_ph, hist_ph


def update_analytics(
    chart_ph,
    hist_ph,
    class_counts: Dict[str, int],
    show_stats: bool,
) -> None:
    """Push fresh data to the analytics placeholders."""
    if not show_stats or chart_ph is None:
        return

    df_counts = format_stats_dataframe(class_counts)
    if not df_counts.empty:
        chart_ph.bar_chart(df_counts.set_index("Class"))
    else:
        chart_ph.info("No detections yet.")

    df_hist = format_detection_history(st.session_state.detection_history)
    if not df_hist.empty:
        hist_ph.dataframe(df_hist, hide_index=True)
    else:
        hist_ph.info("Detection history will appear here.")


# ---------------------------------------------------------------------------
# MODE: Webcam  (one-frame-per-rerun pattern — Stop button always works)
# ---------------------------------------------------------------------------
def _webcam_release() -> None:
    """Safely release the persisted VideoCapture and clear webcam state."""
    cap = st.session_state.get("webcam_cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
        st.session_state.webcam_cap = None
    st.session_state.webcam_fps_counter = None
    logger.info("Webcam released.")


def run_webcam_mode(settings: Dict) -> None:
    """
    Webcam live-stream detection.

    Uses the one-frame-per-rerun pattern:
      1. Render buttons → user clicks Start/Stop → session state updates.
      2. If running=True, open (or reuse) VideoCapture, read ONE frame,
         annotate it, display it, save it to session state, then call
         st.rerun() to fetch the next frame.
      3. If running=False, release the cap and show the last saved frame.
    """
    st.markdown("### 📷 Webcam Live Stream")

    col_vid, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        start_btn = st.button("▶ Start Detection", key="start_webcam")
        stop_btn  = st.button("⏹ Stop Detection",  key="stop_webcam")

        if start_btn and not st.session_state.running:
            st.session_state.running = True
            st.session_state.detection_history = []
            st.session_state.webcam_frame_idx = 0
            st.session_state.webcam_class_counts = {}
            # Release any stale cap before starting fresh
            _webcam_release()

        if stop_btn and st.session_state.running:
            st.session_state.running = False
            _webcam_release()

        st.markdown("---")
        st.markdown("**Session Info**")
        info_ph = st.empty()
        info_ph.markdown(
            f"**Frames:** {st.session_state.webcam_frame_idx}  \n"
            f"**Status:** {'🟢 Running' if st.session_state.running else '🔴 Stopped'}"
        )

    with col_vid:
        frame_ph = st.empty()

    ph_fps, ph_inf, ph_active, ph_unique = render_metrics_dashboard(settings)
    chart_ph, hist_ph = render_analytics(settings)

    # ── Stopped state — show last annotated frame if available ──────────────
    if not st.session_state.running:
        last_html = st.session_state.get("webcam_last_frame_html")
        if last_html is not None:
            frame_ph.markdown(last_html, unsafe_allow_html=True)
            st.caption("Last captured frame (detection stopped)")
            # Restore metrics from saved state
            _update_metrics(
                ph_fps, ph_inf, ph_active, ph_unique,
                st.session_state.last_fps,
                st.session_state.last_inference_ms,
                st.session_state.last_active_count,
                st.session_state.last_unique_ids,
            )
            update_analytics(
                chart_ph, hist_ph,
                st.session_state.webcam_class_counts,
                settings["show_stats"],
            )
        else:
            frame_ph.info("Click **▶ Start Detection** to begin streaming from your webcam.")
        return

    # ── Running state — open (or reuse) VideoCapture ─────────────────────────
    if st.session_state.webcam_cap is None:
        with st.spinner("🔍 Detecting camera… (first launch may take a few seconds)"):
            try:
                cam_index = find_working_webcam(max_index=3)
                st.session_state.webcam_cap = open_video_capture(cam_index)
                st.session_state.webcam_fps_counter = FPSCounter(window=30)
                logger.info("Webcam opened at index %d.", cam_index)
            except RuntimeError as exc:
                st.error(
                    f"❌ Could not open webcam: {exc}\n\n"
                    "**Tips:**\n"
                    "- Close Zoom / Teams / OBS that may be using the camera\n"
                    "- Check Windows camera privacy settings → Allow apps to access camera\n"
                    "- Try a different USB port if using an external camera"
                )
                st.session_state.running = False
                return

    cap: cv2.VideoCapture = st.session_state.webcam_cap
    fps_counter: FPSCounter = st.session_state.webcam_fps_counter

    # ── Get / configure tracker ───────────────────────────────────────────────
    tracker = _get_tracker(
        settings["model_name"],
        settings["confidence"],
        settings["target_classes"],
    )

    # ── Read ONE frame ────────────────────────────────────────────────────────
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Webcam feed lost or end of stream.")
        _webcam_release()
        st.session_state.running = False
        return

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        result = tracker.update(frame)
    except Exception as exc:
        logger.error("Inference error: %s", exc, exc_info=True)
        st.error(f"❌ Inference error: {exc}")
        _webcam_release()
        st.session_state.running = False
        return

    fps_counter.tick()
    st.session_state.webcam_frame_idx += 1
    frame_idx = st.session_state.webcam_frame_idx

    # ── Annotate ──────────────────────────────────────────────────────────────
    annotated = draw_tracked_boxes(
        frame,
        result.tracked_objects,
        show_bbox=settings["show_bbox"],
        show_confidence=settings["show_conf"],
        show_track_id=settings["show_tid"],
    )
    if settings["show_fps"]:
        annotated = draw_fps_overlay(annotated, fps_counter.fps, result.inference_ms)

    # ── Encode as base64 HTML — bypasses MediaFileStorageError completely ──────
    # st.image() stores frames in Streamlit's media cache.  When st.rerun()
    # fires before the browser GETs the image, the cache evicts it and the
    # browser gets a 404.  Embedding as a data-URL avoids this entirely.
    html_frame = frame_to_base64_html(annotated, quality=75)

    # ── Persist encoded frame for display after stop ─────────────────────────────
    st.session_state.webcam_last_frame_html = html_frame

    # ── Display ───────────────────────────────────────────────────────────────
    frame_ph.markdown(html_frame, unsafe_allow_html=True)

    # ── Update cumulative class counts ────────────────────────────────────────
    cc = st.session_state.webcam_class_counts
    for cls, cnt in result.class_counts.items():
        cc[cls] = cc.get(cls, 0) + cnt

    # ── Persist metrics into session state ────────────────────────────────────
    stats = tracker.get_stats()
    st.session_state.last_fps = fps_counter.fps
    st.session_state.last_inference_ms = result.inference_ms
    st.session_state.last_active_count = result.active_count
    st.session_state.last_unique_ids = stats["unique_id_count"]

    # ── Update UI metrics + analytics every 3 frames ─────────────────────────
    if frame_idx % 3 == 0:
        _update_metrics(
            ph_fps, ph_inf, ph_active, ph_unique,
            fps_counter.fps, result.inference_ms,
            result.active_count, stats["unique_id_count"],
        )
        update_analytics(chart_ph, hist_ph, cc, settings["show_stats"])
        info_ph.markdown(
            f"**Frames:** {frame_idx}  \n"
            f"**Unique IDs:** {stats['unique_id_count']}  \n"
            f"**Status:** 🟢 Running"
        )

    _append_history(result, frame_idx)

    # ── Throttle: give the browser time to fetch the frame ────────────────────
    # Without this sleep, st.rerun() fires faster than Streamlit's media
    # file server can serve the image, causing MediaFileStorageError.
    time.sleep(0.03)  # ~33 fps cap — adjust down if your hardware is faster

    # ── Trigger next frame ────────────────────────────────────────────────────
    st.rerun()


# ---------------------------------------------------------------------------
# MODE: Video Upload
# ---------------------------------------------------------------------------
def run_video_mode(settings: Dict) -> None:
    """Video file upload → frame-by-frame processing → preview + export."""
    st.markdown("### 🎬 Video File Processing")

    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    if uploaded is None:
        st.info("⬆️ Upload an MP4 / AVI / MOV file to begin.")
        return

    col_vid, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        process_btn = st.button("▶ Process Video", key="process_video")
        stop_btn = st.button("⏹ Stop", key="stop_video")

        if stop_btn:
            st.session_state.running = False

        st.markdown("---")
        progress_ph = st.empty()
        info_ph = st.empty()

    with col_vid:
        frame_ph = st.empty()

    ph_fps, ph_inf, ph_active, ph_unique = render_metrics_dashboard(settings)
    chart_ph, hist_ph = render_analytics(settings)

    if not process_btn:
        st.session_state.running = False
        return

    # Save upload to disk
    try:
        video_path = save_upload_to_disk(uploaded)
    except Exception as exc:
        st.error(f"❌ Could not save uploaded file: {exc}")
        return

    # Open capture
    try:
        cap = open_video_capture(video_path)
    except RuntimeError as exc:
        st.error(f"❌ Could not open video: {exc}")
        return

    w, h, src_fps, total_frames = get_video_properties(cap)
    out_fps = max(src_fps, 1.0)

    tracker = _get_tracker(
        settings["model_name"],
        settings["confidence"],
        settings["target_classes"],
    )
    tracker.reset()

    fps_counter = FPSCounter(window=30)
    frame_idx = 0
    last_inference_ms = 0.0
    cumulative_class_counts: Dict[str, int] = {}
    output_path: Optional[Path] = None

    st.session_state.running = True
    st.session_state.detection_history = []

    try:
        with VideoWriterCtx(frame_width=w, frame_height=h, fps=out_fps, stem="tracked_video") as vw:
            output_path = vw.output_path

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break

                result = tracker.update(frame)
                last_inference_ms = result.inference_ms
                fps_counter.tick()
                frame_idx += 1

                annotated = draw_tracked_boxes(
                    frame,
                    result.tracked_objects,
                    show_bbox=settings["show_bbox"],
                    show_confidence=settings["show_conf"],
                    show_track_id=settings["show_tid"],
                )
                if settings["show_fps"]:
                    annotated = draw_fps_overlay(annotated, fps_counter.fps, last_inference_ms)

                vw.write(annotated)

                # Preview every 3 frames via base64 to avoid cache errors
                if frame_idx % 3 == 0:
                    display_frame = resize_frame(annotated, max_width=960)
                    frame_ph.markdown(
                        frame_to_base64_html(display_frame, quality=75),
                        unsafe_allow_html=True,
                    )

                # Metrics every 10 frames
                if frame_idx % 10 == 0:
                    stats = tracker.get_stats()
                    for cls, cnt in result.class_counts.items():
                        cumulative_class_counts[cls] = cumulative_class_counts.get(cls, 0) + cnt

                    pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    progress_ph.progress(min(pct / 100, 1.0), text=f"{pct:.0f}%")
                    info_ph.markdown(
                        f"**Frame:** {frame_idx} / {total_frames}  \n"
                        f"**Unique IDs:** {stats['unique_id_count']}"
                    )
                    _update_metrics(
                        ph_fps, ph_inf, ph_active, ph_unique,
                        fps_counter.fps, last_inference_ms,
                        result.active_count, stats["unique_id_count"],
                    )
                    update_analytics(chart_ph, hist_ph, cumulative_class_counts, settings["show_stats"])

                _append_history(result, frame_idx)

    except Exception as exc:
        logger.error("Video processing error: %s", exc, exc_info=True)
        st.error(f"❌ Error during processing: {exc}")
    finally:
        cap.release()
        st.session_state.running = False

    # ── Download button ──
    if output_path and output_path.exists():
        st.success(f"✅ Video processed! Saved to `{output_path.name}`")
        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Processed Video",
                data=f,
                file_name=output_path.name,
                mime="video/mp4",
            )


# ---------------------------------------------------------------------------
# MODE: Image Upload
# ---------------------------------------------------------------------------
def run_image_mode(settings: Dict) -> None:
    """Single-image detection and download."""
    st.markdown("### 🖼️ Image Detection")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload",
    )

    if uploaded is None:
        st.info("⬆️ Upload a JPG or PNG image to run detection.")
        return

    # Decode uploaded bytes
    try:
        pil_img = Image.open(uploaded).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        st.error(f"❌ Could not decode image: {exc}")
        return

    col_orig, col_ann = st.columns(2)

    with col_orig:
        st.markdown("**Original**")
        st.image(pil_img, width='stretch')

    process_btn = st.button("🔍 Run Detection", key="run_image_detect")

    if not process_btn:
        return

    tracker = _get_tracker(
        settings["model_name"],
        settings["confidence"],
        settings["target_classes"],
    )

    with st.spinner("Running YOLOv8 + ByteTrack …"):
        result = tracker.update(frame)

    annotated = draw_tracked_boxes(
        frame,
        result.tracked_objects,
        show_bbox=settings["show_bbox"],
        show_confidence=settings["show_conf"],
        show_track_id=settings["show_tid"],
    )

    with col_ann:
        st.markdown("**Annotated**")
        st.image(frame_to_pil(annotated), width='stretch')

    # ── Metrics ──
    if settings["show_stats"]:
        st.markdown("### 📊 Detection Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📦 Objects Detected", result.active_count)
        m2.metric("🕐 Inference Time", f"{result.inference_ms:.1f} ms")
        m3.metric("🏷️ Classes Found", len(result.class_counts))
        m4.metric("🔖 Unique IDs", tracker.unique_id_count)

        if result.class_counts:
            st.markdown("#### Class Breakdown")
            df = format_stats_dataframe(result.class_counts)
            st.bar_chart(df.set_index("Class"))
            st.dataframe(df, hide_index=True)

    # ── Save & download ──
    try:
        out_path = save_annotated_image(annotated, stem="detection")
        st.success(f"✅ Saved to `{out_path.name}`")
        with open(out_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Annotated Image",
                data=f,
                file_name=out_path.name,
                mime="image/png",
            )
    except Exception as exc:
        st.warning(f"⚠️ Could not save image: {exc}")


# ---------------------------------------------------------------------------
# Main App Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    render_header()
    settings = render_sidebar()

    mode: str = settings["source_mode"]

    if mode == "Webcam":
        run_webcam_mode(settings)
    elif mode == "Upload Video":
        run_video_mode(settings)
    elif mode == "Upload Image":
        run_image_mode(settings)
    else:
        st.error(f"Unknown source mode: {mode!r}")


if __name__ == "__main__":
    main()
