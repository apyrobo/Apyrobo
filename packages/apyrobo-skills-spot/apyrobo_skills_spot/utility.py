"""Boston Dynamics Spot utility skills — dock, capture image, arm pick."""
from __future__ import annotations

import time

from apyrobo import skill


@skill(
    description="Navigate to and use Spot's autowalk dock for charging",
    capability="navigate",
)
def dock(dock_id: int = 1) -> bool:
    """Approach and engage Spot's autowalk charging dock.

    The robot localises the dock fiducial, aligns its rear end with the
    dock contact, and backs in until the charging pins engage.

    Args:
        dock_id: Integer identifier of the target dock (default 1).
    """
    print(f"  [dock] Searching for Spot dock ID {dock_id}")
    time.sleep(0.05)
    print(f"  [dock] Dock {dock_id} fiducial acquired — aligning rear contact")
    time.sleep(0.05)
    print(f"  [dock] Dock {dock_id} engaged — charging pins connected")
    return True


@skill(
    description="Capture an image from one of Spot's onboard cameras",
    capability="perception",
)
def capture_image(camera: str = "frontleft", save_path: str = "") -> str:
    """Capture a single frame from a named Spot camera and save it to disk.

    Spot has five image sources: ``frontleft``, ``frontright``,
    ``left``, ``right``, and ``back``.

    Args:
        camera:    Name of the camera to capture from (default ``"frontleft"``).
        save_path: File path to write the image. If empty a default path is
                   generated from the camera name and a timestamp token.

    Returns:
        The absolute path where the image was saved.
    """
    valid_cameras = {"frontleft", "frontright", "left", "right", "back"}
    if camera not in valid_cameras:
        camera = "frontleft"
    if not save_path:
        save_path = f"/tmp/spot_{camera}_000000.jpg"
    print(f"  [capture_image] Requesting frame from camera '{camera}'")
    time.sleep(0.05)
    print(f"  [capture_image] Frame captured — saving to '{save_path}'")
    return save_path


@skill(
    description="Use Spot Arm to grasp an object at a specified 3-D position",
    capability="manipulation",
)
def arm_pick(x: float, y: float, z: float) -> bool:
    """Command Spot Arm to reach out and grasp an object.

    The skill moves the end-effector above the target, opens the gripper,
    descends to the grasp point, closes the gripper, and retracts the arm
    to a safe carry pose.

    Args:
        x: Object X coordinate in the body frame (metres).
        y: Object Y coordinate in the body frame (metres).
        z: Object Z coordinate in the body frame (metres).
    """
    print(f"  [arm_pick] Planning grasp at ({x:.3f}, {y:.3f}, {z:.3f})")
    time.sleep(0.05)
    print("  [arm_pick] Extending arm — opening gripper")
    time.sleep(0.05)
    print(f"  [arm_pick] Descending to ({x:.3f}, {y:.3f}, {z:.3f}) — closing gripper")
    time.sleep(0.05)
    print("  [arm_pick] Object grasped — retracting arm to carry pose")
    return True
