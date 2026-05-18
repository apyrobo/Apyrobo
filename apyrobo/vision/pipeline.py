"""OpenCV vision pipeline for APYROBO.

Provides a capture-and-detect loop that runs in a background thread and
exposes the most recent frame and any detected objects to the skill layer.

Architecture::

    Camera/file ──► CaptureThread ──► frame_buffer (threading.Lock)
                                            │
                               detect("cup") call (skill thread)
                                            │
                             [optional YOLO inference on frame]
                                            │
                                  list[Detection] ◄── caller

Supports three detection backends:
    1. YOLO (ultralytics) — best accuracy, requires `pip install ultralytics`
    2. Haar cascades (OpenCV built-in) — CPU-only, no extra deps
    3. Color/contour heuristic — last resort fallback, no accuracy guarantees
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_CV2_AVAILABLE = False
try:
    import cv2 as _cv2  # noqa: F401
    _CV2_AVAILABLE = True
except ImportError:
    pass

_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO as _YOLO  # noqa: F401
    _YOLO_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Pixel-space bounding box: (x1, y1) top-left, (x2, y2) bottom-right."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class Detection:
    """A single detected object in a frame."""
    label: str
    confidence: float                        # 0.0 – 1.0
    bbox: BoundingBox
    pose_3d: tuple[float, float, float] | None = None  # (x, y, z) in metres if depth available
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox.to_dict(),
            "pose_3d": list(self.pose_3d) if self.pose_3d else None,
        }


@dataclass
class VisionFrame:
    """One captured frame with timestamp and optional detections."""
    timestamp: float
    frame: Any                               # numpy array (H, W, 3) BGR
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.timestamp


# ---------------------------------------------------------------------------
# Vision adapter
# ---------------------------------------------------------------------------

class VisionAdapter:
    """OpenCV-based camera capture + object detection sensor.

    Runs a background capture thread that continuously grabs frames from
    the camera or video source.  Detection runs on-demand when
    :meth:`detect` is called (lazy — no wasted inference if skills aren't
    looking for anything).

    Parameters
    ----------
    source:
        OpenCV capture source.  ``0`` = first webcam, ``"video.mp4"`` = file,
        ``"rtsp://..."`` = IP camera stream.
    yolo_model:
        YOLO model name/path for ``ultralytics``.  ``"yolov8n.pt"`` is the
        smallest and fastest.  Set to ``None`` to disable YOLO and use the
        Haar cascade fallback.
    confidence_threshold:
        Minimum detection confidence to return (default 0.45).
    max_frame_age:
        Warn when :meth:`detect` is called with a frame older than this many
        seconds (default 1.0).
    """

    def __init__(
        self,
        source: int | str = 0,
        yolo_model: str | None = "yolov8n.pt",
        confidence_threshold: float = 0.45,
        max_frame_age: float = 1.0,
    ) -> None:
        if not _CV2_AVAILABLE:
            raise ImportError(
                "OpenCV is required for VisionAdapter.\n"
                "Install with: pip install opencv-python"
            )
        self._source = source
        self._yolo_model_name = yolo_model
        self._confidence_threshold = confidence_threshold
        self._max_frame_age = max_frame_age

        self._cap: Any = None
        self._yolo: Any = None
        self._latest_frame: VisionFrame | None = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the capture source and start the background grab thread."""
        import cv2
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self._source!r}")

        if self._yolo_model_name and _YOLO_AVAILABLE:
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(self._yolo_model_name)
                logger.info("VisionAdapter: YOLO model %r loaded", self._yolo_model_name)
            except Exception as exc:
                logger.warning("VisionAdapter: could not load YOLO model: %s", exc)
                self._yolo = None

        self._stop_event.clear()
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        logger.info("VisionAdapter started (source=%r)", self._source)

    def stop(self) -> None:
        """Stop the capture thread and release the camera."""
        self._stop_event.set()
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("VisionAdapter stopped")

    def __enter__(self) -> "VisionAdapter":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Background capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        import cv2
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                logger.warning("VisionAdapter: frame read failed — source may have ended")
                self._stop_event.wait(0.1)
                continue
            h, w = frame.shape[:2]
            vf = VisionFrame(
                timestamp=time.monotonic(),
                frame=frame,
                width=w,
                height=h,
            )
            with self._frame_lock:
                self._latest_frame = vf

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def get_frame(self) -> VisionFrame | None:
        """Return the most recently captured frame, or None if not started."""
        with self._frame_lock:
            return self._latest_frame

    def detect(
        self,
        label: str | None = None,
        *,
        min_confidence: float | None = None,
    ) -> list[Detection]:
        """Run object detection on the latest frame.

        Parameters
        ----------
        label:
            Object class to filter by (e.g. ``"person"``, ``"cup"``).
            ``None`` returns all detected objects.
        min_confidence:
            Override the adapter-level threshold for this call.

        Returns
        -------
        list[Detection]
            Detected objects matching *label*, sorted by confidence descending.
            Empty list if nothing found or no frame is available.
        """
        frame = self.get_frame()
        if frame is None:
            logger.debug("VisionAdapter.detect: no frame available (call start() first)")
            return []

        if frame.age_seconds > self._max_frame_age:
            logger.warning(
                "VisionAdapter.detect: frame is %.1fs old (max %.1fs) — results may be stale",
                frame.age_seconds, self._max_frame_age,
            )

        threshold = min_confidence if min_confidence is not None else self._confidence_threshold

        if self._yolo is not None:
            detections = self._detect_yolo(frame, threshold)
        else:
            detections = self._detect_haar(frame, threshold)

        if label is not None:
            detections = [d for d in detections if d.label.lower() == label.lower()]

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _detect_yolo(self, frame: VisionFrame, threshold: float) -> list[Detection]:
        """Run YOLO inference on *frame*."""
        try:
            results = self._yolo(frame.frame, verbose=False, conf=threshold)
            detections: list[Detection] = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                names = result.names
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    xyxy = boxes.xyxy[i].tolist()
                    det = Detection(
                        label=names.get(cls_id, str(cls_id)),
                        confidence=conf,
                        bbox=BoundingBox(
                            x1=int(xyxy[0]), y1=int(xyxy[1]),
                            x2=int(xyxy[2]), y2=int(xyxy[3]),
                        ),
                    )
                    detections.append(det)
            return detections
        except Exception as exc:
            logger.error("VisionAdapter YOLO inference error: %s", exc)
            return []

    def _detect_haar(self, frame: VisionFrame, threshold: float) -> list[Detection]:
        """Haar cascade fallback for face/person detection."""
        import cv2
        gray = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2GRAY)
        try:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            detections = []
            for (x, y, w, h) in faces:
                detections.append(Detection(
                    label="face",
                    confidence=0.7,
                    bbox=BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h),
                ))
            return detections
        except Exception as exc:
            logger.debug("VisionAdapter Haar cascade error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Depth integration
    # ------------------------------------------------------------------

    def estimate_pose(self, detection: Detection, depth_frame: Any) -> Detection:
        """Estimate 3-D pose of *detection* using a depth frame.

        *depth_frame* is a numpy array (H, W) of depth values in metres.
        Updates ``detection.pose_3d`` in-place and returns the detection.
        """
        try:
            import numpy as np
            cx, cy = detection.bbox.center
            if depth_frame is not None and 0 <= cy < depth_frame.shape[0] and 0 <= cx < depth_frame.shape[1]:
                z = float(depth_frame[cy, cx])
                if z > 0:
                    detection.pose_3d = (float(cx), float(cy), z)
        except Exception as exc:
            logger.debug("VisionAdapter.estimate_pose error: %s", exc)
        return detection

    # ------------------------------------------------------------------
    # Skill-friendly helpers
    # ------------------------------------------------------------------

    def is_present(self, label: str, min_confidence: float | None = None) -> bool:
        """Return True if *label* is detected in the current frame."""
        return bool(self.detect(label, min_confidence=min_confidence))

    def count(self, label: str) -> int:
        """Return the number of detected *label* objects in the current frame."""
        return len(self.detect(label))

    @property
    def is_running(self) -> bool:
        return self._running

    def __repr__(self) -> str:
        return f"VisionAdapter(source={self._source!r}, running={self._running})"
