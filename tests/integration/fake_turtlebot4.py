#!/usr/bin/env python3
"""
Fake TurtleBot4 node for integration testing.

Simulates the ROS 2 interface that ROS2Adapter (ros2_bridge.py) expects:
  - Publishes /odom (nav_msgs/Odometry) at 10 Hz with BEST_EFFORT QoS
  - Subscribes to /cmd_vel (geometry_msgs/Twist) and integrates into pose
  - Serves NavigateToPose action at 'navigate_to_pose'
  - Publishes /battery_state (sensor_msgs/BatteryState) at 1 Hz

Run standalone:
    python tests/integration/fake_turtlebot4.py
"""

from __future__ import annotations

import math
import threading
import time

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_group import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState


class FakeTurtleBot4(Node):
    """Minimal ROS 2 node that acts like a TurtleBot4."""

    ODOM_HZ = 10.0
    BATTERY_HZ = 1.0
    NAV_STEP_HZ = 10.0
    GOAL_TOLERANCE_M = 0.05

    def __init__(self) -> None:
        super().__init__("fake_turtlebot4")

        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._lock = threading.Lock()

        cb_group = ReentrantCallbackGroup()

        # Publish /odom with BEST_EFFORT QoS to match adapter's subscription
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5,
        )
        self._odom_pub = self.create_publisher(Odometry, "/odom", odom_qos)

        # Publish /battery_state with default reliable QoS
        self._battery_pub = self.create_publisher(BatteryState, "/battery_state", 10)

        # Subscribe to /cmd_vel for pose integration
        self._cmd_vel_sub = self.create_subscription(
            Twist,
            "/cmd_vel",
            self._cmd_vel_callback,
            10,
            callback_group=cb_group,
        )

        # Timers
        self.create_timer(1.0 / self.ODOM_HZ, self._publish_odom, callback_group=cb_group)
        self.create_timer(1.0 / self.BATTERY_HZ, self._publish_battery, callback_group=cb_group)

        # NavigateToPose action server — action name matches adapter DEFAULT_CONFIG['nav2_action']
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            "navigate_to_pose",
            execute_callback=self._navigate_execute,
            callback_group=cb_group,
        )

        self.get_logger().info("FakeTurtleBot4 ready — publishing /odom, serving navigate_to_pose")

    # ------------------------------------------------------------------
    # cmd_vel — integrate velocity into pose (very rough, no physics)
    # ------------------------------------------------------------------

    def _cmd_vel_callback(self, msg: Twist) -> None:
        dt = 1.0 / self.ODOM_HZ
        with self._lock:
            self._x += msg.linear.x * math.cos(self._yaw) * dt
            self._y += msg.linear.x * math.sin(self._yaw) * dt
            self._yaw += msg.angular.z * dt

    # ------------------------------------------------------------------
    # Odometry publisher
    # ------------------------------------------------------------------

    def _publish_odom(self) -> None:
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        with self._lock:
            msg.pose.pose.position.x = self._x
            msg.pose.pose.position.y = self._y
            msg.pose.pose.position.z = 0.0
            msg.pose.pose.orientation = self._yaw_to_quat(self._yaw)
        self._odom_pub.publish(msg)

    # ------------------------------------------------------------------
    # Battery publisher
    # ------------------------------------------------------------------

    def _publish_battery(self) -> None:
        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.percentage = 0.85
        msg.voltage = 25.2
        msg.present = True
        self._battery_pub.publish(msg)

    # ------------------------------------------------------------------
    # NavigateToPose action server
    # ------------------------------------------------------------------

    def _navigate_execute(self, goal_handle: object) -> NavigateToPose.Result:
        """
        Handle a NavigateToPose goal.

        Linearly interpolates from current position to target, publishing
        feedback at NAV_STEP_HZ. No real physics — just moves the pose.
        """
        goal = goal_handle.request
        target_x = float(goal.pose.pose.position.x)
        target_y = float(goal.pose.pose.position.y)

        self.get_logger().info(
            "NavigateToPose goal received: target=(%.2f, %.2f)", target_x, target_y
        )

        with self._lock:
            start_x = self._x
            start_y = self._y

        dist = math.sqrt((target_x - start_x) ** 2 + (target_y - start_y) ** 2)
        # Simulate at ~0.5 m/s — at least 5 steps for any goal
        steps = max(5, int(dist * self.NAV_STEP_HZ / 0.5))
        dt = 1.0 / self.NAV_STEP_HZ

        for i in range(steps + 1):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info("NavigateToPose: goal cancelled")
                return NavigateToPose.Result()

            t = i / steps
            cur_x = start_x + t * (target_x - start_x)
            cur_y = start_y + t * (target_y - start_y)

            with self._lock:
                self._x = cur_x
                self._y = cur_y
                # Point toward target
                dx = target_x - cur_x
                dy = target_y - cur_y
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    self._yaw = math.atan2(dy, dx)

            remaining = dist * (1.0 - t)

            feedback = NavigateToPose.Feedback()
            feedback.current_pose = PoseStamped()
            feedback.current_pose.header.stamp = self.get_clock().now().to_msg()
            feedback.current_pose.header.frame_id = "map"
            feedback.current_pose.pose.position.x = cur_x
            feedback.current_pose.pose.position.y = cur_y
            feedback.current_pose.pose.position.z = 0.0
            feedback.current_pose.pose.orientation = self._yaw_to_quat(self._yaw)
            feedback.distance_remaining = float(remaining)

            goal_handle.publish_feedback(feedback)
            time.sleep(dt)

        # Snap to exact target
        with self._lock:
            self._x = target_x
            self._y = target_y

        goal_handle.succeed()
        self.get_logger().info("NavigateToPose: goal succeeded")
        return NavigateToPose.Result()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _yaw_to_quat(yaw: float) -> Quaternion:
        q = Quaternion()
        q.w = math.cos(yaw / 2.0)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        return q


def main() -> None:
    rclpy.init()
    node = FakeTurtleBot4()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
