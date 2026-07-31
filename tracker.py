"""
tracker.py
==========
ByteTrack multi-object tracking module.

Provides the ``ObjectTracker`` class, which wraps the Ultralytics
``model.track()`` API (ByteTrack backend) and maintains session-wide
statistics such as unique track IDs and per-class counts.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import (
    BYTETRACK_CONFIG,
    COCO_CLASS_MAP,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_IMG_SIZE,
    DEFAULT_PATHS,
    MODEL_CONFIGS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class TrackedObject:
    """A single tracked object in one frame."""

    track_id: int
    bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float


@dataclass
class TrackingResult:
    """Aggregated tracking output for one frame."""

    tracked_objects: List[TrackedObject] = field(default_factory=list)
    inference_ms: float = 0.0
    active_count: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.tracked_objects)


# ---------------------------------------------------------------------------
# ObjectTracker
# ---------------------------------------------------------------------------


class ObjectTracker:
    """
    Multi-object tracker using YOLOv8 + ByteTrack.

    ByteTrack is engaged via Ultralytics' built-in ``.track()`` method with
    ``tracker="bytetrack"`` and ``persist=True`` so internal Kalman-filter
    state is maintained across consecutive ``update()`` calls.

    Parameters
    ----------
    model_name : str
        Display name matching a key in ``config.MODEL_CONFIGS``.
    confidence : float
        Detection confidence threshold in [0, 1].
    iou_threshold : float
        NMS IOU threshold.
    device : str
        Inference device (``"cpu"``, ``"cuda"``, ``"mps"``, or ``""`` for auto).
    target_classes : Optional[List[str]]
        COCO class name whitelist.  ``None`` keeps all 80 classes.
    img_size : int
        Inference image size (longest edge).
    """

    def __init__(
        self,
        model_name: str = "YOLOv8n (Fastest)",
        confidence: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU,
        device: str = "",
        target_classes: Optional[List[str]] = None,
        img_size: int = DEFAULT_IMG_SIZE,
    ) -> None:
        self.model_name: str = model_name
        self.confidence: float = confidence
        self.iou_threshold: float = iou_threshold
        self.device: str = device
        self.target_classes: Optional[List[str]] = target_classes
        self.img_size: int = img_size

        self._model = None

        # Session-wide tracking statistics
        self._unique_ids: Set[int] = set()
        self._class_counts: Dict[str, int] = {}
        self._frame_count: int = 0
        self._class_filter: Optional[List[int]] = self._build_class_filter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the YOLOv8 model weights (downloads if absent).

        Raises
        ------
        RuntimeError
            If Ultralytics is not installed or download fails.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics not installed. Run: pip install ultralytics"
            ) from exc

        if self.model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: '{self.model_name}'")

        weights_filename: str = MODEL_CONFIGS[self.model_name]["weights"]
        local_path: Path = DEFAULT_PATHS["models"] / weights_filename

        if not local_path.exists():
            logger.info("Downloading '%s' weights …", weights_filename)
            try:
                import shutil
                tmp = YOLO(weights_filename)
                downloaded = Path(weights_filename)
                if downloaded.exists():
                    shutil.move(str(downloaded), str(local_path))
                    logger.info("Saved weights to %s", local_path)
                else:
                    logger.info("Using Ultralytics cache for '%s'", weights_filename)
                    self._model = tmp
                    return
            except Exception as exc:
                raise RuntimeError(f"Weight download failed: {exc}") from exc
        else:
            logger.info("Loading weights from %s", local_path)

        from ultralytics import YOLO
        self._model = YOLO(str(local_path))
        logger.info("Tracker model '%s' ready on device='%s'.", self.model_name, self.device or "auto")

    def update(self, frame: np.ndarray) -> TrackingResult:
        """
        Run detection + ByteTrack on one BGR frame.

        Persists ByteTrack state across calls (``persist=True`` in Ultralytics).
        Updates session-wide unique-ID and class-count accumulators.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).

        Returns
        -------
        TrackingResult
            Per-frame tracking snapshot.
        """
        if self._model is None:
            self.load_model()

        result = TrackingResult()
        self._frame_count += 1

        t0 = time.perf_counter()
        try:
            raw = self._model.track(
                source=frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device if self.device else None,
                classes=self._class_filter,
                tracker=BYTETRACK_CONFIG["tracker"],
                persist=BYTETRACK_CONFIG["persist"],
                verbose=False,
            )
        except Exception as exc:
            logger.error("Tracking inference failed: %s", exc)
            return result

        result.inference_ms = (time.perf_counter() - t0) * 1000.0

        if not raw or raw[0].boxes is None:
            return result

        boxes = raw[0].boxes

        for box in boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = COCO_CLASS_MAP.get(cls_id, f"cls_{cls_id}")

                # ByteTrack may not assign an ID on the very first frame
                tid_tensor = box.id
                if tid_tensor is not None:
                    track_id = int(tid_tensor[0].item())
                else:
                    track_id = -1

                tracked_obj = TrackedObject(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                )
                result.tracked_objects.append(tracked_obj)

                # Accumulate session stats
                if track_id >= 0:
                    self._unique_ids.add(track_id)
                self._class_counts[cls_name] = self._class_counts.get(cls_name, 0) + 1

            except (IndexError, ValueError) as exc:
                logger.warning("Skipping malformed tracking box: %s", exc)

        result.active_count = len(result.tracked_objects)
        result.class_counts = self._get_frame_class_counts(result.tracked_objects)
        return result

    def reset(self) -> None:
        """
        Reset all session-wide tracking state.

        Call this when starting a new session (new video / webcam restart).
        Also re-initialises the Ultralytics ByteTrack internal state by
        reloading the model (clearing persisted kalman state).
        """
        self._unique_ids.clear()
        self._class_counts.clear()
        self._frame_count = 0
        # Force ByteTrack state reset by reloading model
        if self._model is not None:
            self.load_model()
        logger.info("Tracker session reset.")

    def get_stats(self) -> Dict[str, object]:
        """
        Return a snapshot of session-wide statistics.

        Returns
        -------
        dict with keys:
            ``unique_id_count``, ``class_totals``, ``frame_count``
        """
        return {
            "unique_id_count": len(self._unique_ids),
            "class_totals": dict(self._class_counts),
            "frame_count": self._frame_count,
        }

    # ------------------------------------------------------------------
    # Configuration Updates
    # ------------------------------------------------------------------

    def set_confidence(self, confidence: float) -> None:
        """Dynamically update the confidence threshold."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be in [0.0, 1.0]")
        self.confidence = confidence

    def set_classes(self, target_classes: Optional[List[str]]) -> None:
        """Update the class whitelist filter."""
        self.target_classes = target_classes
        self._class_filter = self._build_class_filter()

    def set_model(self, model_name: str) -> None:
        """Switch to a different YOLOv8 model variant."""
        if model_name == self.model_name and self._model is not None:
            return
        logger.info("Switching tracker model to '%s'", model_name)
        self.model_name = model_name
        self._model = None
        self.load_model()

    @property
    def unique_id_count(self) -> int:
        """Total unique track IDs seen in the current session."""
        return len(self._unique_ids)

    @property
    def is_loaded(self) -> bool:
        """True if the underlying YOLO model is loaded."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _build_class_filter(self) -> Optional[List[int]]:
        """Build integer class filter list from COCO name whitelist."""
        if not self.target_classes:
            return None
        name_to_id = {v: k for k, v in COCO_CLASS_MAP.items()}
        indices = [name_to_id[n] for n in self.target_classes if n in name_to_id]
        return indices if indices else None

    @staticmethod
    def _get_frame_class_counts(objects: List[TrackedObject]) -> Dict[str, int]:
        """Count active objects per class in the current frame."""
        counts: Dict[str, int] = {}
        for obj in objects:
            counts[obj.class_name] = counts.get(obj.class_name, 0) + 1
        return counts
