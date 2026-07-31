"""
detector.py
===========
YOLOv8 object detection module.

Provides the ``YOLOv8Detector`` class, which handles:
  - Automatic weight download / caching inside ``./models/``
  - Model instantiation with configurable device placement
  - Stateless ``detect()`` for single-frame inference
  - Dynamic confidence threshold and class-filter updates
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
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
class Detection:
    """Represents a single object detection result."""

    bbox: Tuple[int, int, int, int]      # (x1, y1, x2, y2) in pixel coords
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None       # populated after tracking


@dataclass
class DetectionResult:
    """Aggregated output of one inference pass."""

    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    frame_shape: Tuple[int, int] = (0, 0)   # (height, width)

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# YOLOv8Detector
# ---------------------------------------------------------------------------


class YOLOv8Detector:
    """
    YOLOv8-based object detector backed by the Ultralytics library.

    Parameters
    ----------
    model_name : str
        Display name matching a key in ``config.MODEL_CONFIGS``
        (e.g. ``"YOLOv8n (Fastest)"``).
    confidence : float
        Minimum confidence threshold in [0, 1].
    iou_threshold : float
        Non-maximum suppression IOU threshold.
    device : str
        Inference device string — ``"cpu"``, ``"cuda"``, ``"mps"``, etc.
        Pass ``""`` to let Ultralytics auto-select.
    target_classes : Optional[List[str]]
        Whitelist of COCO class names to keep.  ``None`` keeps all classes.
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

        self._model = None          # lazy-loaded on first call to load_model()
        self._weights_path: Optional[Path] = None
        self._class_filter: Optional[List[int]] = self._build_class_filter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load YOLOv8 weights.

        Downloads weights to ``./models/`` if the file is absent.
        Raises ``RuntimeError`` if the download fails.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics package not found.  "
                "Install it with: pip install ultralytics"
            ) from exc

        if self.model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model '{self.model_name}'.  "
                f"Choose from: {list(MODEL_CONFIGS.keys())}"
            )

        weights_filename: str = MODEL_CONFIGS[self.model_name]["weights"]
        models_dir: Path = DEFAULT_PATHS["models"]
        local_path: Path = models_dir / weights_filename

        if not local_path.exists():
            logger.info(
                "Weights '%s' not found locally — initiating download …",
                weights_filename,
            )
            try:
                # Ultralytics downloads weights to CWD; we'll move them.
                tmp_model = YOLO(weights_filename)
                downloaded: Path = Path(weights_filename)
                if downloaded.exists():
                    shutil.move(str(downloaded), str(local_path))
                    logger.info("Weights saved to %s", local_path)
                else:
                    # Ultralytics may cache in ~/.ultralytics — just keep ref
                    logger.info(
                        "Using Ultralytics cached weights for '%s'",
                        weights_filename,
                    )
                    self._weights_path = Path(weights_filename)
                    self._model = tmp_model
                    logger.info("Model '%s' loaded successfully.", self.model_name)
                    return
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download '{weights_filename}': {exc}"
                ) from exc
        else:
            logger.info("Loading weights from local cache: %s", local_path)

        self._weights_path = local_path
        self._model = YOLO(str(local_path))
        logger.info("Model '%s' loaded on device='%s'.", self.model_name, self.device or "auto")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run YOLOv8 inference on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image array (H, W, 3).

        Returns
        -------
        DetectionResult
            Structured detection output including timing.
        """
        if self._model is None:
            self.load_model()

        h, w = frame.shape[:2]
        result = DetectionResult(frame_shape=(h, w))

        t0 = time.perf_counter()
        try:
            raw_results = self._model.predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device if self.device else None,
                classes=self._class_filter,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            return result

        result.inference_ms = (time.perf_counter() - t0) * 1000.0

        if not raw_results or raw_results[0].boxes is None:
            return result

        boxes = raw_results[0].boxes
        for box in boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = COCO_CLASS_MAP.get(cls_id, f"cls_{cls_id}")
                result.detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                    )
                )
            except (IndexError, ValueError) as exc:
                logger.warning("Skipping malformed box: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Configuration Updates
    # ------------------------------------------------------------------

    def set_confidence(self, confidence: float) -> None:
        """Update detection confidence threshold."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be in [0.0, 1.0]")
        self.confidence = confidence
        logger.debug("Confidence threshold updated to %.2f", confidence)

    def set_classes(self, target_classes: Optional[List[str]]) -> None:
        """Update the class whitelist filter."""
        self.target_classes = target_classes
        self._class_filter = self._build_class_filter()
        logger.debug("Class filter updated: %s", self._class_filter)

    def set_model(self, model_name: str) -> None:
        """Switch to a different YOLOv8 variant and reload weights."""
        if model_name == self.model_name and self._model is not None:
            return
        logger.info("Switching model from '%s' to '%s'", self.model_name, model_name)
        self.model_name = model_name
        self._model = None
        self.load_model()

    @property
    def is_loaded(self) -> bool:
        """True if model weights have been loaded."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _build_class_filter(self) -> Optional[List[int]]:
        """Translate class name whitelist into COCO integer indices."""
        if not self.target_classes:
            return None
        name_to_id = {v: k for k, v in COCO_CLASS_MAP.items()}
        indices = [name_to_id[name] for name in self.target_classes if name in name_to_id]
        return indices if indices else None
