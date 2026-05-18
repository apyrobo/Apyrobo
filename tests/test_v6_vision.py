"""Tests for apyrobo.vision — OpenCV vision pipeline (v6.0.0)."""
from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from apyrobo.vision.pipeline import (
    BoundingBox,
    Detection,
    VisionAdapter,
    VisionFrame,
)


@contextmanager
def _mock_cv2(**extras):
    """Inject a fake cv2 module into sys.modules for the duration of the block."""
    mock_cv2 = MagicMock(name="cv2")
    # cv2.data.haarcascades needs to be a real string so path concatenation works
    mock_cv2.data.haarcascades = "/fake/haarcascades/"
    mock_cv2.COLOR_BGR2GRAY = 6
    for k, v in extras.items():
        setattr(mock_cv2, k, v)
    old = sys.modules.get("cv2")
    sys.modules["cv2"] = mock_cv2
    try:
        yield mock_cv2
    finally:
        if old is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = old


# ---------------------------------------------------------------------------
# BoundingBox tests
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_width(self):
        bb = BoundingBox(x1=10, y1=20, x2=110, y2=80)
        assert bb.width == 100

    def test_height(self):
        bb = BoundingBox(x1=10, y1=20, x2=110, y2=80)
        assert bb.height == 60

    def test_center(self):
        bb = BoundingBox(x1=0, y1=0, x2=100, y2=200)
        assert bb.center == (50, 100)

    def test_center_odd(self):
        bb = BoundingBox(x1=1, y1=1, x2=100, y2=100)
        cx, cy = bb.center
        assert cx == 50
        assert cy == 50

    def test_to_dict(self):
        bb = BoundingBox(x1=5, y1=10, x2=55, y2=60)
        d = bb.to_dict()
        assert d == {"x1": 5, "y1": 10, "x2": 55, "y2": 60}


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestDetection:
    def _make(self, label="cup", confidence=0.9, pose_3d=None):
        return Detection(
            label=label,
            confidence=confidence,
            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=120),
            pose_3d=pose_3d,
        )

    def test_to_dict_no_pose(self):
        det = self._make()
        d = det.to_dict()
        assert d["label"] == "cup"
        assert d["confidence"] == 0.9
        assert d["pose_3d"] is None
        assert "x1" in d["bbox"]

    def test_to_dict_with_pose(self):
        det = self._make(pose_3d=(1.0, 2.0, 3.5))
        d = det.to_dict()
        assert d["pose_3d"] == [1.0, 2.0, 3.5]

    def test_confidence_rounded(self):
        det = self._make(confidence=0.123456)
        assert det.to_dict()["confidence"] == 0.123


# ---------------------------------------------------------------------------
# VisionFrame tests
# ---------------------------------------------------------------------------

class TestVisionFrame:
    def test_age_seconds(self):
        before = time.monotonic()
        vf = VisionFrame(
            timestamp=before - 0.5,
            frame=None,
            width=640,
            height=480,
        )
        age = vf.age_seconds
        assert 0.4 < age < 1.0

    def test_detections_empty_by_default(self):
        vf = VisionFrame(timestamp=time.monotonic(), frame=None, width=1, height=1)
        assert vf.detections == []


# ---------------------------------------------------------------------------
# VisionAdapter — no-CV2 guard
# ---------------------------------------------------------------------------

class TestVisionAdapterNoCv2:
    def test_raises_importerror_without_opencv(self):
        import apyrobo.vision.pipeline as pipeline_mod
        orig = pipeline_mod._CV2_AVAILABLE
        try:
            pipeline_mod._CV2_AVAILABLE = False
            with pytest.raises(ImportError, match="opencv"):
                VisionAdapter(source=0)
        finally:
            pipeline_mod._CV2_AVAILABLE = orig


# ---------------------------------------------------------------------------
# Helpers to build a mock cv2 + VideoCapture
# ---------------------------------------------------------------------------

def _make_fake_frame(w=640, h=480):
    """Return a minimal numpy-like array stub."""
    try:
        import numpy as np
        return np.zeros((h, w, 3), dtype="uint8")
    except ImportError:
        # Minimal stub if numpy isn't available
        class _Arr:
            shape = (h, w, 3)
        return _Arr()


def _make_mock_cap(fake_frame=None, isOpened=True):
    cap = MagicMock()
    cap.isOpened.return_value = isOpened
    frame = fake_frame if fake_frame is not None else _make_fake_frame()
    cap.read.return_value = (True, frame)
    return cap


# ---------------------------------------------------------------------------
# VisionAdapter lifecycle (with mocked cv2)
# ---------------------------------------------------------------------------

class TestVisionAdapterLifecycle:
    def _make_adapter(self, **kwargs):
        """Create a VisionAdapter with cv2 mocked out."""
        import apyrobo.vision.pipeline as pipeline_mod
        pipeline_mod._CV2_AVAILABLE = True
        return VisionAdapter(source=0, yolo_model=None, **kwargs)

    def test_is_running_false_before_start(self):
        va = self._make_adapter()
        assert va.is_running is False

    def test_start_stop(self):
        va = self._make_adapter()
        mock_cap = _make_mock_cap()
        with _mock_cv2(VideoCapture=MagicMock(return_value=mock_cap)):
            va.start()
            assert va.is_running is True
            va.stop()
            assert va.is_running is False

    def test_context_manager(self):
        va = self._make_adapter()
        mock_cap = _make_mock_cap()
        with _mock_cv2(VideoCapture=MagicMock(return_value=mock_cap)):
            with va:
                assert va.is_running is True
            assert va.is_running is False

    def test_start_raises_when_cap_not_opened(self):
        va = self._make_adapter()
        mock_cap = _make_mock_cap(isOpened=False)
        with _mock_cv2(VideoCapture=MagicMock(return_value=mock_cap)):
            with pytest.raises(RuntimeError, match="Could not open"):
                va.start()

    def test_repr(self):
        va = self._make_adapter()
        r = repr(va)
        assert "VisionAdapter" in r
        assert "running=False" in r


# ---------------------------------------------------------------------------
# VisionAdapter — get_frame
# ---------------------------------------------------------------------------

class TestVisionAdapterGetFrame:
    def test_get_frame_returns_none_before_start(self):
        import apyrobo.vision.pipeline as pipeline_mod
        pipeline_mod._CV2_AVAILABLE = True
        va = VisionAdapter(source=0, yolo_model=None)
        assert va.get_frame() is None

    def test_get_frame_returns_visionframe_after_capture(self):
        import apyrobo.vision.pipeline as pipeline_mod
        pipeline_mod._CV2_AVAILABLE = True
        va = VisionAdapter(source=0, yolo_model=None)
        fake_frame = _make_fake_frame()
        mock_cap = _make_mock_cap(fake_frame=fake_frame)
        with _mock_cv2(VideoCapture=MagicMock(return_value=mock_cap)):
            va.start()
            time.sleep(0.1)  # let capture thread grab a frame
            vf = va.get_frame()
            va.stop()
        assert vf is not None
        assert isinstance(vf, VisionFrame)
        assert vf.width == 640
        assert vf.height == 480


# ---------------------------------------------------------------------------
# VisionAdapter — detect (Haar cascade path)
# ---------------------------------------------------------------------------

class TestVisionAdapterDetectHaar:
    def _started_adapter(self):
        import apyrobo.vision.pipeline as pipeline_mod
        pipeline_mod._CV2_AVAILABLE = True
        va = VisionAdapter(source=0, yolo_model=None, confidence_threshold=0.5)
        fake_frame = _make_fake_frame()
        mock_cap = _make_mock_cap(fake_frame=fake_frame)
        return va, mock_cap

    def test_detect_returns_empty_when_no_frame(self):
        import apyrobo.vision.pipeline as pipeline_mod
        pipeline_mod._CV2_AVAILABLE = True
        va = VisionAdapter(source=0, yolo_model=None)
        result = va.detect("face")
        assert result == []

    def _haar_cv2(self, cascade_results):
        """Return a mock cv2 module configured with the given Haar results."""
        fake_cascade = MagicMock()
        fake_cascade.detectMultiScale.return_value = cascade_results
        mock_cv2 = MagicMock(name="cv2")
        mock_cv2.data.haarcascades = "/fake/"
        mock_cv2.COLOR_BGR2GRAY = 6
        mock_cv2.cvtColor.return_value = _make_fake_frame()
        mock_cv2.CascadeClassifier.return_value = fake_cascade
        return mock_cv2

    def test_detect_haar_faces(self):
        va, mock_cap = self._started_adapter()
        fake_cv2 = self._haar_cv2([(10, 20, 50, 60)])
        fake_cv2.VideoCapture = MagicMock(return_value=mock_cap)

        with _mock_cv2(**{k: getattr(fake_cv2, k) for k in
                          ("VideoCapture", "cvtColor", "CascadeClassifier", "COLOR_BGR2GRAY")}):
            import cv2
            cv2.data = fake_cv2.data
            va.start()
            time.sleep(0.1)
            dets = va.detect("face")
            va.stop()

        assert len(dets) == 1
        assert dets[0].label == "face"
        assert dets[0].confidence == 0.7

    def test_detect_haar_label_filter(self):
        va, mock_cap = self._started_adapter()
        # Inject via direct _detect_haar override
        va._latest_frame = VisionFrame(
            timestamp=time.monotonic(), frame=_make_fake_frame(), width=640, height=480
        )
        with patch.object(va, "_detect_haar", return_value=[
            Detection(label="face", confidence=0.7, bbox=BoundingBox(10, 20, 60, 80))
        ]):
            dets = va.detect("person")  # "person" ≠ "face"
        assert dets == []

    def test_detect_label_case_insensitive(self):
        va, mock_cap = self._started_adapter()
        va._latest_frame = VisionFrame(
            timestamp=time.monotonic(), frame=_make_fake_frame(), width=640, height=480
        )
        with patch.object(va, "_detect_haar", return_value=[
            Detection(label="face", confidence=0.7, bbox=BoundingBox(10, 20, 60, 80))
        ]):
            dets = va.detect("FACE")
        assert len(dets) == 1

    def test_detect_no_label_returns_all(self):
        va, mock_cap = self._started_adapter()
        va._latest_frame = VisionFrame(
            timestamp=time.monotonic(), frame=_make_fake_frame(), width=640, height=480
        )
        all_dets = [
            Detection(label="face", confidence=0.7, bbox=BoundingBox(0, 0, 30, 30)),
            Detection(label="face", confidence=0.7, bbox=BoundingBox(100, 100, 30, 30)),
        ]
        with patch.object(va, "_detect_haar", return_value=all_dets):
            dets = va.detect()
        assert len(dets) == 2

    def test_detect_sorted_by_confidence_descending(self):
        """All Haar detections get 0.7; sorting is stable but order is consistent."""
        va, mock_cap = self._started_adapter()
        va._latest_frame = VisionFrame(
            timestamp=time.monotonic(), frame=_make_fake_frame(), width=640, height=480
        )
        all_dets = [
            Detection(label="face", confidence=0.7, bbox=BoundingBox(0, 0, 30, 30)),
            Detection(label="face", confidence=0.7, bbox=BoundingBox(50, 50, 40, 40)),
        ]
        with patch.object(va, "_detect_haar", return_value=all_dets):
            dets = va.detect()
        assert all(d.confidence == 0.7 for d in dets)

    def test_detect_bbox_values(self):
        va, mock_cap = self._started_adapter()
        va._latest_frame = VisionFrame(
            timestamp=time.monotonic(), frame=_make_fake_frame(), width=640, height=480
        )
        with patch.object(va, "_detect_haar", return_value=[
            Detection(label="face", confidence=0.7, bbox=BoundingBox(x1=10, y1=20, x2=60, y2=80))
        ]):
            dets = va.detect("face")

        bb = dets[0].bbox
        assert bb.x1 == 10
        assert bb.y1 == 20
        assert bb.x2 == 60
        assert bb.y2 == 80


# ---------------------------------------------------------------------------
# VisionAdapter — detect (YOLO path)
# ---------------------------------------------------------------------------

class TestVisionAdapterDetectYolo:
    def _make_yolo_result(self, detections):
        """Build a minimal mock matching ultralytics Results structure."""
        import types
        result = MagicMock()
        boxes = MagicMock()
        boxes.__len__ = lambda self: len(detections)

        cls_vals = [d["cls"] for d in detections]
        conf_vals = [d["conf"] for d in detections]
        xyxy_vals = [d["xyxy"] for d in detections]

        boxes.cls = [MagicMock(item=lambda cls=c: cls) for c in cls_vals]
        boxes.conf = [MagicMock(item=lambda cf=cf: cf) for cf in conf_vals]
        boxes.xyxy = [MagicMock(tolist=lambda xy=xy: xy) for xy in xyxy_vals]

        result.boxes = boxes
        result.names = {0: "person", 1: "cup", 2: "dog"}
        return [result]

    def test_detect_yolo_returns_detections(self):
        import apyrobo.vision.pipeline as pipeline_mod
        pipeline_mod._CV2_AVAILABLE = True
        va = VisionAdapter(source=0, yolo_model=None, confidence_threshold=0.4)
        # Inject a pre-seeded frame and mock _detect_yolo
        va._latest_frame = VisionFrame(
            timestamp=time.monotonic(), frame=_make_fake_frame(), width=640, height=480
        )
        yolo_dets = [
            Detection(label="person", confidence=0.85,
                      bbox=BoundingBox(10, 20, 100, 120)),
            Detection(label="cup", confidence=0.65,
                      bbox=BoundingBox(200, 50, 300, 150)),
        ]
        va._yolo = MagicMock()
        with patch.object(va, "_detect_yolo", return_value=yolo_dets):
            dets = va.detect()

        # sorted descending
        assert dets[0].confidence >= dets[-1].confidence

    def test_detect_yolo_error_returns_empty(self):
        import apyrobo.vision.pipeline as pipeline_mod
        pipeline_mod._CV2_AVAILABLE = True
        va = VisionAdapter(source=0, yolo_model=None)
        va._latest_frame = VisionFrame(
            timestamp=time.monotonic(), frame=_make_fake_frame(), width=640, height=480
        )
        va._yolo = MagicMock()
        with patch.object(va, "_detect_yolo", return_value=[]):
            result = va.detect()
        assert result == []


# ---------------------------------------------------------------------------
# VisionAdapter — stale frame warning
# ---------------------------------------------------------------------------

class TestVisionAdapterStaleFrame:
    def test_stale_frame_logs_warning(self, caplog):
        import apyrobo.vision.pipeline as pipeline_mod
        pipeline_mod._CV2_AVAILABLE = True
        va = VisionAdapter(source=0, yolo_model=None, max_frame_age=0.01)

        old_ts = time.monotonic() - 10.0
        vf = VisionFrame(timestamp=old_ts, frame=_make_fake_frame(), width=640, height=480)

        with va._frame_lock:
            va._latest_frame = vf

        import logging
        with caplog.at_level(logging.WARNING, logger="apyrobo.vision.pipeline"), \
             patch.object(va, "_detect_haar", return_value=[]):
            va.detect("face")

        assert any("stale" in r.message.lower() or "old" in r.message.lower()
                   for r in caplog.records)


# ---------------------------------------------------------------------------
# VisionAdapter — estimate_pose
# ---------------------------------------------------------------------------

class TestVisionAdapterPose:
    def test_estimate_pose_sets_pose_3d(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        import apyrobo.vision.pipeline as pipeline_mod
        pipeline_mod._CV2_AVAILABLE = True
        va = VisionAdapter(source=0, yolo_model=None)

        depth = np.ones((480, 640), dtype="float32") * 2.5
        det = Detection(
            label="cup",
            confidence=0.9,
            bbox=BoundingBox(x1=100, y1=200, x2=200, y2=300),
        )
        result = va.estimate_pose(det, depth)
        assert result.pose_3d is not None
        cx, cy, z = result.pose_3d
        assert z == pytest.approx(2.5, abs=0.01)

    def test_estimate_pose_zero_depth_ignored(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        import apyrobo.vision.pipeline as pipeline_mod
        pipeline_mod._CV2_AVAILABLE = True
        va = VisionAdapter(source=0, yolo_model=None)

        depth = np.zeros((480, 640), dtype="float32")
        det = Detection(
            label="cup",
            confidence=0.9,
            bbox=BoundingBox(x1=100, y1=200, x2=200, y2=300),
        )
        result = va.estimate_pose(det, depth)
        assert result.pose_3d is None

    def test_estimate_pose_none_depth(self):
        import apyrobo.vision.pipeline as pipeline_mod
        pipeline_mod._CV2_AVAILABLE = True
        va = VisionAdapter(source=0, yolo_model=None)

        det = Detection(
            label="cup",
            confidence=0.9,
            bbox=BoundingBox(x1=50, y1=50, x2=100, y2=100),
        )
        result = va.estimate_pose(det, None)
        assert result.pose_3d is None


# ---------------------------------------------------------------------------
# VisionAdapter — skill helpers
# ---------------------------------------------------------------------------

class TestVisionAdapterHelpers:
    def _adapter_with_frame(self, detections):
        """Return a VisionAdapter with a pre-seeded frame and mocked detect."""
        import apyrobo.vision.pipeline as pipeline_mod
        pipeline_mod._CV2_AVAILABLE = True
        va = VisionAdapter(source=0, yolo_model=None)
        va._latest_frame = VisionFrame(
            timestamp=time.monotonic(),
            frame=_make_fake_frame(),
            width=640,
            height=480,
            detections=detections,
        )
        va.detect = MagicMock(return_value=detections)
        return va

    def test_is_present_true(self):
        dets = [Detection(label="cup", confidence=0.9, bbox=BoundingBox(0, 0, 10, 10))]
        va = self._adapter_with_frame(dets)
        assert va.is_present("cup") is True

    def test_is_present_false(self):
        va = self._adapter_with_frame([])
        assert va.is_present("person") is False

    def test_count_zero(self):
        va = self._adapter_with_frame([])
        assert va.count("cup") == 0

    def test_count_multiple(self):
        dets = [
            Detection(label="cup", confidence=0.9, bbox=BoundingBox(0, 0, 10, 10)),
            Detection(label="cup", confidence=0.8, bbox=BoundingBox(50, 50, 60, 60)),
        ]
        va = self._adapter_with_frame(dets)
        assert va.count("cup") == 2


# ---------------------------------------------------------------------------
# VisionAdapter — min_confidence override
# ---------------------------------------------------------------------------

class TestVisionAdapterMinConfidence:
    def test_min_confidence_passed_to_haar(self):
        import apyrobo.vision.pipeline as pipeline_mod
        pipeline_mod._CV2_AVAILABLE = True
        va = VisionAdapter(source=0, yolo_model=None, confidence_threshold=0.5)

        vf = VisionFrame(timestamp=time.monotonic(), frame=_make_fake_frame(), width=640, height=480)
        with va._frame_lock:
            va._latest_frame = vf

        captured_threshold = []

        def fake_detect_haar(frame, threshold):
            captured_threshold.append(threshold)
            return []

        va._detect_haar = fake_detect_haar

        va.detect("face", min_confidence=0.9)
        assert captured_threshold == [0.9]

    def test_default_threshold_used_when_no_override(self):
        import apyrobo.vision.pipeline as pipeline_mod
        pipeline_mod._CV2_AVAILABLE = True
        va = VisionAdapter(source=0, yolo_model=None, confidence_threshold=0.6)

        vf = VisionFrame(timestamp=time.monotonic(), frame=_make_fake_frame(), width=640, height=480)
        with va._frame_lock:
            va._latest_frame = vf

        captured_threshold = []

        def fake_detect_haar(frame, threshold):
            captured_threshold.append(threshold)
            return []

        va._detect_haar = fake_detect_haar
        va.detect("face")
        assert captured_threshold == [0.6]


# ---------------------------------------------------------------------------
# VisionAdapter — module imports export the right names
# ---------------------------------------------------------------------------

class TestVisionModuleExports:
    def test_vision_init_exports(self):
        import apyrobo.vision as vision
        assert hasattr(vision, "VisionAdapter")
        assert hasattr(vision, "Detection")
        assert hasattr(vision, "VisionFrame")

    def test_bounding_box_importable_from_pipeline(self):
        from apyrobo.vision.pipeline import BoundingBox
        bb = BoundingBox(1, 2, 3, 4)
        assert bb.width == 2
