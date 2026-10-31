#!/usr/bin/env bash
# Boot the full navigation stack in headless Gazebo Harmonic (ROS 2 Jazzy):
# the Jazzy counterpart of gazebo_nav_boot.sh.
#
#   gz sim (gz_nav_world.sdf: turtlebot3_world + lidar) + TurtleBot3 burger
#     + ros_gz_bridge            (/clock /odom /tf /joint_states /imu /scan
#                                 from Gazebo, /cmd_vel to it)
#     + robot_state_publisher    (base_footprint -> base_link -> base_scan)
#     + Nav2 in SLAM mode        (slam_toolbox builds the map live,
#                                 TurtleBot3's Jazzy burger parameters)
#
# This is the sim side of test_gazebo_nav2_e2e.py on Jazzy. The lidar is a
# gpu_lidar, so gz sim renders headless (--headless-rendering, Mesa software
# EGL; no GPU or display). /cmd_vel is bridged as Twist, which the ros2://
# adapter publishes and Nav2 is set to publish; see gz_boot.sh.
set -e

source /opt/ros/jazzy/setup.bash
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"

TB3=/opt/ros/jazzy/share/turtlebot3_gazebo
WORLD="$(cd "$(dirname "$0")" && pwd)/gz_nav_world.sdf"
SDF="$TB3/models/turtlebot3_${TURTLEBOT3_MODEL}/model.sdf"
URDF="/opt/ros/jazzy/share/turtlebot3_description/urdf/turtlebot3_${TURTLEBOT3_MODEL}.urdf"
# TurtleBot3's own Jazzy Nav2 parameters (DWB controller at 10 Hz, as on
# Humble; Nav2's default MPPI misses its rate next to software rendering).
# They publish TwistStamped; the ros2:// adapter and our bridge use Twist.
PARAMS_SRC=/opt/ros/jazzy/share/turtlebot3_navigation2/param/${TURTLEBOT3_MODEL}.yaml
# Unused in SLAM mode, but bringup_launch.py requires a map argument.
MAP=/opt/ros/jazzy/share/nav2_bringup/maps/tb3_sandbox.yaml
export GZ_SIM_RESOURCE_PATH="$TB3/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"

echo "[gz_nav_boot] model=$TURTLEBOT3_MODEL world=$WORLD"
for f in "$SDF" "$URDF" "$MAP" "$PARAMS_SRC" "$TB3/models/turtlebot3_world/model.sdf"; do
    [ -f "$f" ] || { echo "[gz_nav_boot] FATAL: $f not found" >&2; exit 1; }
done

BRIDGE=$(mktemp --suffix=.yaml)
cat > "$BRIDGE" <<'YAML'
- {ros_topic_name: clock, gz_topic_name: clock, ros_type_name: rosgraph_msgs/msg/Clock, gz_type_name: gz.msgs.Clock, direction: GZ_TO_ROS}
- {ros_topic_name: joint_states, gz_topic_name: joint_states, ros_type_name: sensor_msgs/msg/JointState, gz_type_name: gz.msgs.Model, direction: GZ_TO_ROS}
- {ros_topic_name: odom, gz_topic_name: odom, ros_type_name: nav_msgs/msg/Odometry, gz_type_name: gz.msgs.Odometry, direction: GZ_TO_ROS}
- {ros_topic_name: tf, gz_topic_name: tf, ros_type_name: tf2_msgs/msg/TFMessage, gz_type_name: gz.msgs.Pose_V, direction: GZ_TO_ROS}
- {ros_topic_name: imu, gz_topic_name: imu, ros_type_name: sensor_msgs/msg/Imu, gz_type_name: gz.msgs.IMU, direction: GZ_TO_ROS}
- {ros_topic_name: scan, gz_topic_name: scan, ros_type_name: sensor_msgs/msg/LaserScan, gz_type_name: gz.msgs.LaserScan, direction: GZ_TO_ROS}
- {ros_topic_name: cmd_vel, gz_topic_name: cmd_vel, ros_type_name: geometry_msgs/msg/Twist, gz_type_name: gz.msgs.Twist, direction: ROS_TO_GZ}
YAML

# Odometry in world coordinates. Gazebo Classic's TurtleBot3 diff-drive
# published odom from the world pose, so odom started at the spawn pose and
# test_gazebo_nav2_e2e.py's START/GOAL are world coordinates. The gz
# DiffDrive plugin integrates wheel odometry from (0, 0) instead. Spawn a
# copy of the model whose DiffDrive still drives the wheels but publishes its
# odometry and TF on unbridged topics, and whose OdometryPublisher (the model
# pose, as Classic did) owns /odom and odom -> base_footprint.
MODEL=$(mktemp --suffix=.sdf)
sed -e 's#<odom_topic>odom</odom_topic>#<odom_topic>wheel_odom</odom_topic>#' \
    -e 's#<tf_topic>/tf</tf_topic>#<tf_topic>/wheel_tf</tf_topic>#' \
    -e 's#</model>#<plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher"><odom_frame>odom</odom_frame><robot_base_frame>base_footprint</robot_base_frame><odom_publish_frequency>30</odom_publish_frequency><odom_topic>odom</odom_topic><tf_topic>/tf</tf_topic><dimensions>2</dimensions></plugin></model>#' \
    "$SDF" > "$MODEL"
grep -q wheel_odom "$MODEL" && grep -q OdometryPublisher "$MODEL" \
    || { echo "[gz_nav_boot] FATAL: could not patch $SDF" >&2; exit 1; }

echo "[gz_nav_boot] starting gz sim (headless, rendering for the lidar)…"
gz sim -s -r -v 2 --headless-rendering "$WORLD" &
GZ_PID=$!

# Fixed frames slam_toolbox needs to place /scan. The Jazzy URDF is
# namespace-templated like Humble's; strip the placeholder.
echo "[gz_nav_boot] starting robot_state_publisher…"
ros2 run robot_state_publisher robot_state_publisher --ros-args \
    -p use_sim_time:=true \
    -p robot_description:="$(sed 's/\${namespace}//g' "$URDF")" &

# Standard turtlebot3_world start pose, free corridor ahead along +x.
echo "[gz_nav_boot] spawning $TURTLEBOT3_MODEL at (-2.0, -0.5)…"
spawned=0
for _ in $(seq 1 30); do
    if ! kill -0 "$GZ_PID" 2>/dev/null; then
        echo "[gz_nav_boot] FATAL: gz sim exited during startup" >&2
        exit 1
    fi
    if ros2 run ros_gz_sim create -name "$TURTLEBOT3_MODEL" -file "$MODEL" \
            -x -2.0 -y -0.5 -z 0.01; then
        spawned=1
        break
    fi
    sleep 2
done
[ "$spawned" = 1 ] || { echo "[gz_nav_boot] FATAL: could not spawn the robot" >&2; exit 1; }

echo "[gz_nav_boot] starting ros_gz_bridge…"
ros2 run ros_gz_bridge parameter_bridge --ros-args -p "config_file:=$BRIDGE" &

echo "[gz_nav_boot] waiting for /odom and /scan before Nav2 bringup…"
for topic in /odom /scan; do
    for _ in $(seq 1 30); do
        if timeout 3 ros2 topic echo "$topic" --once > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
done

PARAMS=$(mktemp --suffix=.yaml)
sed 's/enable_stamped_cmd_vel: true/enable_stamped_cmd_vel: false/' "$PARAMS_SRC" > "$PARAMS"

echo "[gz_nav_boot] launching Nav2 (SLAM mode)…"
ros2 launch nav2_bringup bringup_launch.py \
    slam:=True \
    use_sim_time:=True \
    autostart:=True \
    params_file:="$PARAMS" \
    map:="$MAP" &

echo "[gz_nav_boot] up — waiting on gz sim (navigate_to_pose comes up with Nav2)…"
wait "$GZ_PID"
