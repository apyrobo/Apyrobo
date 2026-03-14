"""SIM-02 smoke check: verify GazeboNative adapter spawn + topics."""

from apyrobo.core.robot import Robot


def main() -> int:
    robot = Robot.discover("gazebo_native://smoke_bot")
    adapter = robot._adapter
    report = adapter.smoke_test()
    ok = report.get("spawned") and report.get("has_odom_topic")
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
