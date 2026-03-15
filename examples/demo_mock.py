"""
APYROBO — Quick demo (mock mode, no ROS 2 required).

Run this from the repo root:
    python examples/demo_mock.py
"""

from apyrobo.core.robot import Robot


def main() -> None:
    # Discover a mock robot — no ROS 2 or hardware needed
    print("Discovering robot...")
    robot = Robot.discover("mock://turtlebot4")
    print(f"  Found: {robot}")

    # Query capabilities
    print("\nCapabilities:")
    caps = robot.capabilities()
    print(f"  Name:       {caps.name}")
    print(f"  Max speed:  {caps.max_speed} m/s")
    print(f"  Skills:     {[c.name for c in caps.capabilities]}")
    print(f"  Sensors:    {[f'{s.sensor_id} ({s.sensor_type.value})' for s in caps.sensors]}")

    # Send commands
    print("\nCommands:")
    robot.move(x=2.0, y=3.0, speed=0.5)
    print(f"  Moved to (2.0, 3.0)")

    robot.move(x=5.0, y=1.0)
    print(f"  Moved to (5.0, 1.0)")

    robot.stop()
    print(f"  Stopped")

    print("\nDone! The APYROBO capability API works.")
    print("Next step: connect to a real ROS 2 robot via the gazebo:// or ros2:// adapter.")


if __name__ == "__main__":
    main()
