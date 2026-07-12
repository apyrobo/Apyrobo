#!/usr/bin/env bash
# Boot the full navigation stack in headless Gazebo Classic:
#
#   gzserver (turtlebot3_world) + TurtleBot3 burger
#     + robot_state_publisher      (TF: base_footprint → base_link → base_scan)
#     + Nav2 in SLAM mode          (slam_toolbox builds the map live — no
#                                   pre-built map, no AMCL initial pose)
#
# This is the sim side of the NL → plan → Nav2 → robot-moves end-to-end test
# (test_gazebo_nav2_e2e.py). Unlike gazebo_boot.sh (cmd_vel only), the
# NavigateToPose action server is live here, so ROS2Adapter.move() takes its
# preferred Nav2 path instead of the proportional-controller fallback.
set -e

source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"

WORLD=/opt/ros/humble/share/turtlebot3_gazebo/worlds/turtlebot3_world.world
SDF="/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_${TURTLEBOT3_MODEL}/model.sdf"
URDF="/opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_${TURTLEBOT3_MODEL}.urdf"

echo "[gazebo_nav_boot] model=$TURTLEBOT3_MODEL world=$WORLD"
[ -f "$WORLD" ] || { echo "[gazebo_nav_boot] FATAL: turtlebot3_world not found" >&2; exit 1; }
[ -f "$SDF" ]   || { echo "[gazebo_nav_boot] FATAL: burger SDF not found" >&2; exit 1; }
[ -f "$URDF" ]  || { echo "[gazebo_nav_boot] FATAL: burger URDF not found" >&2; exit 1; }

echo "[gazebo_nav_boot] starting gzserver (headless)…"
gzserver --verbose \
    -s libgazebo_ros_init.so \
    -s libgazebo_ros_factory.so \
    "$WORLD" &
GZ_PID=$!

# TF tree for the lidar: the raw SDF spawn gives odom→base_footprint (from
# the diff-drive plugin) but not the fixed frames slam_toolbox needs to place
# /scan. robot_state_publisher supplies base_footprint→base_link→base_scan.
#
# Humble's turtlebot3_description URDF is namespace-templated — fed raw, RSP
# publishes frames literally named "${namespace}base_link" and Nav2/SLAM see
# no base_link at all. Strip the placeholder (the launch files normally do).
echo "[gazebo_nav_boot] starting robot_state_publisher…"
ros2 run robot_state_publisher robot_state_publisher --ros-args \
    -p use_sim_time:=true \
    -p robot_description:="$(sed 's/\${namespace}//g' "$URDF")" &

echo "[gazebo_nav_boot] waiting for /spawn_entity service…"
for _ in $(seq 1 40); do
    if ros2 service list 2>/dev/null | grep -q "/spawn_entity"; then
        break
    fi
    if ! kill -0 "$GZ_PID" 2>/dev/null; then
        echo "[gazebo_nav_boot] FATAL: gzserver exited during startup" >&2
        exit 1
    fi
    sleep 1
done

# Spawn at the standard turtlebot3_world start pose (free corridor ahead
# along +x — the test's goal is a short hop through open space).
echo "[gazebo_nav_boot] spawning $TURTLEBOT3_MODEL at (-2.0, -0.5)…"
ros2 run gazebo_ros spawn_entity.py \
    -entity "$TURTLEBOT3_MODEL" \
    -file "$SDF" \
    -x -2.0 -y -0.5 -z 0.01

echo "[gazebo_nav_boot] waiting for /odom before Nav2 bringup…"
for _ in $(seq 1 30); do
    if timeout 3 ros2 topic echo /odom --once > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# SLAM mode: slam_toolbox publishes map→odom immediately (no localization
# bootstrap), and the turtlebot3_world walls give it features to match on.
# The map argument is unused with slam:=True but bringup_launch.py requires
# it; nav2_bringup ships a matching one.
echo "[gazebo_nav_boot] launching Nav2 (SLAM mode)…"
ros2 launch nav2_bringup bringup_launch.py \
    slam:=True \
    use_sim_time:=True \
    autostart:=True \
    map:=/opt/ros/humble/share/nav2_bringup/maps/turtlebot3_world.yaml &

echo "[gazebo_nav_boot] up — waiting on gzserver (healthcheck watches for navigate_to_pose)…"
wait "$GZ_PID"
