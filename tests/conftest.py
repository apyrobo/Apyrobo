# When real ROS 2 is available (the integration Docker image), pre-import the
# packages that tests/test_ros2_*_mocked.py would otherwise replace with
# MagicMocks via sys.modules.setdefault(). Loading them here first means those
# setdefault() calls become no-ops, so the mocked tests can still run in pytest
# sessions that also collect integration tests without poisoning the real
# Odometry / NavigateToPose / etc. classes used by apyrobo.core.ros2_bridge.
#
# Outside ROS 2 environments (normal CI), every import raises ImportError, the
# real modules stay absent, and the mocked tests install their MagicMocks
# exactly as before.

try:
    import rclpy  # noqa: F401
    import rclpy.node  # noqa: F401
    import rclpy.action  # noqa: F401
    import rclpy.qos  # noqa: F401
    import rclpy.callback_groups  # noqa: F401
    import rclpy.executors  # noqa: F401
    import geometry_msgs  # noqa: F401
    import geometry_msgs.msg  # noqa: F401
    import nav_msgs  # noqa: F401
    import nav_msgs.msg  # noqa: F401
    import sensor_msgs  # noqa: F401
    import sensor_msgs.msg  # noqa: F401
    import nav2_msgs  # noqa: F401
    import nav2_msgs.action  # noqa: F401
    import std_msgs  # noqa: F401
    import std_msgs.msg  # noqa: F401
    import action_msgs  # noqa: F401
    import action_msgs.msg  # noqa: F401
    import builtin_interfaces  # noqa: F401
    import builtin_interfaces.msg  # noqa: F401
except ImportError:
    pass
