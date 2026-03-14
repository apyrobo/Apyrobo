from pathlib import Path

from apyrobo.core import ros2_bridge


def test_apply_namespace_prefixes_topics_not_actions():
    cfg = {
        "cmd_vel": "/cmd_vel",
        "odom": "odom",
        "nav2_action": "navigate_to_pose",
    }
    out = ros2_bridge._apply_namespace(cfg, "robot_0")
    assert out["cmd_vel"] == "/robot_0/cmd_vel"
    assert out["odom"] == "/robot_0/odom"
    assert out["nav2_action"] == "navigate_to_pose"


def test_load_yaml_file_returns_empty_for_missing(tmp_path: Path):
    data = ros2_bridge._load_yaml_file(str(tmp_path / "missing.yaml"))
    assert data == {}


def test_load_yaml_file_parses_mapping(tmp_path: Path):
    p = tmp_path / "bridge.yaml"
    p.write_text("ros2_bridge:\n  namespace: robot_0\n  topics:\n    odom: /odom\n")
    data = ros2_bridge._load_yaml_file(str(p))
    assert data["ros2_bridge"]["namespace"] == "robot_0"


def test_ros_compat_layer_reports_status():
    compat = ros2_bridge._ros_compat_layer()
    assert "distro" in compat
    assert compat["status"] in {"supported", "unknown"}
