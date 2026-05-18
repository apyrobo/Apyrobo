"""APYROBO Vision — OpenCV-based sensor pipeline.

Wraps OpenCV capture and optional YOLO/SAM object detection as an
APYROBO sensor source. Skills call ``vision.detect("cup")`` and receive
a list of :class:`Detection` objects with class name, bounding box, and
an optional 3-D pose estimate (when depth data is available).

Works on CPU — no GPU required for detection, though GPU accelerates
YOLO inference considerably.

Quick start::

    from apyrobo.vision import VisionAdapter
    vision = VisionAdapter(source=0)  # webcam
    vision.start()

    detections = vision.detect("person")
    for det in detections:
        print(det.label, det.confidence, det.bbox)

    vision.stop()

Skills can use VisionAdapter as a robot sensor::

    @skill(name="find_object", description="Locate an object using the robot camera")
    def find_object(robot, label: str = "cup") -> bool:
        vision = robot.get_sensor("vision")
        detections = vision.detect(label)
        return bool(detections)

Requires: ``pip install opencv-python`` for capture.
Optional: ``pip install ultralytics`` for YOLO detection.
"""

from apyrobo.vision.pipeline import VisionAdapter, Detection, VisionFrame

__all__ = ["VisionAdapter", "Detection", "VisionFrame"]
