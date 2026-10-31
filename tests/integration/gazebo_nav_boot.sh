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

# slam_toolbox only publishes the map frame once it has placed a scan, and
# Nav2's global costmap waits on base_link→map until it does. Make sure the
# lidar is publishing before Nav2 comes up, so a missing map frame later
# points at SLAM rather than at a silent sensor.
echo "[gazebo_nav_boot] waiting for /scan before Nav2 bringup…"
SCAN_OK=0
for _ in $(seq 1 30); do
    if timeout 3 ros2 topic echo /scan --once > /dev/null 2>&1; then
        SCAN_OK=1
        break
    fi
    sleep 1
done
[ "$SCAN_OK" = 1 ] || echo "[gazebo_nav_boot] WARNING: no /scan message after 30 s" >&2

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

# If slam_toolbox has not published map→odom within MAP_WAIT_S, the
# healthcheck will eventually fail with only "Timed out waiting for transform
# from base_link to map" in the log. Print what the map frame depends on so
# the failure can be diagnosed from the CI log.
MAP_WAIT_S="${MAP_WAIT_S:-120}"
(
    for _ in $(seq 1 "$MAP_WAIT_S"); do
        if timeout 4 ros2 run tf2_ros tf2_echo map odom 2>/dev/null | grep -q "Translation"; then
            echo "[gazebo_nav_boot] map frame is up"
            exit 0
        fi
        sleep 1
    done
    echo "[gazebo_nav_boot] DIAG: no map→odom transform after ${MAP_WAIT_S} s; dumping state" >&2
    ros2 node list 2>&1 | sed 's/^/[gazebo_nav_boot] DIAG node: /' >&2
    ros2 topic info /scan --verbose 2>&1 | sed 's/^/[gazebo_nav_boot] DIAG \/scan: /' >&2
    timeout 5 ros2 topic echo /scan --once --field header 2>&1 | sed 's/^/[gazebo_nav_boot] DIAG scan header: /' >&2
    timeout 5 ros2 topic echo /clock --once 2>&1 | sed 's/^/[gazebo_nav_boot] DIAG clock: /' >&2
    timeout 6 ros2 run tf2_ros tf2_echo odom base_scan 2>&1 | head -n 8 | sed 's/^/[gazebo_nav_boot] DIAG tf odom→base_scan: /' >&2
    timeout 5 ros2 topic echo /map --once --field info 2>&1 | head -n 12 | sed 's/^/[gazebo_nav_boot] DIAG map info: /' >&2
) &

echo "[gazebo_nav_boot] up — waiting on gzserver (healthcheck watches for navigate_to_pose)…"
wait "$GZ_PID"
