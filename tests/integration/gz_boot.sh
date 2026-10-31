#!/usr/bin/env bash
# Boot a headless Gazebo Harmonic (ROS 2 Jazzy) with a physics-simulated
# TurtleBot3 burger: the Jazzy counterpart of gazebo_boot.sh.
#
# gz sim runs the world, ros_gz_sim spawns the burger, and ros_gz_bridge
# carries /clock, /odom, /tf, /joint_states and /imu from Gazebo and /cmd_vel
# to it. The world is gz_empty_world.sdf (no Fuel download, no renderer, so
# no /scan). The TurtleBot3 package's own bridge config maps /cmd_vel as
# geometry_msgs/TwistStamped; the ros2:// adapter publishes Twist, so this
# script uses its own config with Twist on /cmd_vel. No GUI, no Nav2.
set -e

source /opt/ros/jazzy/setup.bash
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"

TB3=/opt/ros/jazzy/share/turtlebot3_gazebo
WORLD="$(cd "$(dirname "$0")" && pwd)/gz_empty_world.sdf"
SDF="$TB3/models/turtlebot3_${TURTLEBOT3_MODEL}/model.sdf"
export GZ_SIM_RESOURCE_PATH="$TB3/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"

echo "[gz_boot] model=$TURTLEBOT3_MODEL world=$WORLD"
[ -f "$SDF" ] || { echo "[gz_boot] FATAL: $SDF not found" >&2; exit 1; }

BRIDGE=$(mktemp --suffix=.yaml)
cat > "$BRIDGE" <<'YAML'
- {ros_topic_name: clock, gz_topic_name: clock, ros_type_name: rosgraph_msgs/msg/Clock, gz_type_name: gz.msgs.Clock, direction: GZ_TO_ROS}
- {ros_topic_name: joint_states, gz_topic_name: joint_states, ros_type_name: sensor_msgs/msg/JointState, gz_type_name: gz.msgs.Model, direction: GZ_TO_ROS}
- {ros_topic_name: odom, gz_topic_name: odom, ros_type_name: nav_msgs/msg/Odometry, gz_type_name: gz.msgs.Odometry, direction: GZ_TO_ROS}
- {ros_topic_name: tf, gz_topic_name: tf, ros_type_name: tf2_msgs/msg/TFMessage, gz_type_name: gz.msgs.Pose_V, direction: GZ_TO_ROS}
- {ros_topic_name: imu, gz_topic_name: imu, ros_type_name: sensor_msgs/msg/Imu, gz_type_name: gz.msgs.IMU, direction: GZ_TO_ROS}
- {ros_topic_name: cmd_vel, gz_topic_name: cmd_vel, ros_type_name: geometry_msgs/msg/Twist, gz_type_name: gz.msgs.Twist, direction: ROS_TO_GZ}
YAML

# Headless server (-s), running from the start (-r).
echo "[gz_boot] starting gz sim (headless)…"
gz sim -s -r -v 2 "$WORLD" &
GZ_PID=$!

echo "[gz_boot] spawning $TURTLEBOT3_MODEL at origin…"
spawned=0
for _ in $(seq 1 30); do
    if ! kill -0 "$GZ_PID" 2>/dev/null; then
        echo "[gz_boot] FATAL: gz sim exited during startup" >&2
        exit 1
    fi
    if ros2 run ros_gz_sim create -name "$TURTLEBOT3_MODEL" -file "$SDF" \
            -x 0 -y 0 -z 0.01; then
        spawned=1
        break
    fi
    sleep 2
done
[ "$spawned" = 1 ] || { echo "[gz_boot] FATAL: could not spawn the robot" >&2; exit 1; }

echo "[gz_boot] starting ros_gz_bridge…"
ros2 run ros_gz_bridge parameter_bridge --ros-args -p "config_file:=$BRIDGE" &

echo "[gz_boot] up — /cmd_vel (sub) and /odom (pub) live. Waiting on gz sim…"
wait "$GZ_PID"
