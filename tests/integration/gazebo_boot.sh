#!/usr/bin/env bash
# Boot a headless Gazebo Classic with a physics-simulated TurtleBot3 burger.
#
# The burger model's SDF carries the gazebo_ros diff-drive plugin, which
# subscribes /cmd_vel and publishes /odom — exactly the interface the
# ros2:// adapter speaks. No GUI (gzserver only), no Nav2: this is the
# minimal real-physics robot the adapter can drive.
set -e

source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"

WORLD=/opt/ros/humble/share/turtlebot3_gazebo/worlds/empty_world.world
[ -f "$WORLD" ] || WORLD=/usr/share/gazebo/worlds/empty.world
[ -f "$WORLD" ] || WORLD=/usr/share/gazebo-11/worlds/empty.world
SDF="/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_${TURTLEBOT3_MODEL}/model.sdf"

echo "[gazebo_boot] model=$TURTLEBOT3_MODEL world=$WORLD"
echo "[gazebo_boot] sdf=$SDF"
[ -f "$SDF" ] || { echo "[gazebo_boot] FATAL: burger SDF not found" >&2; exit 1; }

# Headless server with the ROS init + factory system plugins.
echo "[gazebo_boot] starting gzserver (headless)…"
gzserver --verbose \
    -s libgazebo_ros_init.so \
    -s libgazebo_ros_factory.so \
    "$WORLD" &
GZ_PID=$!

# Wait for the factory service so spawn_entity can succeed.
echo "[gazebo_boot] waiting for /spawn_entity service…"
for _ in $(seq 1 40); do
    if ros2 service list 2>/dev/null | grep -q "/spawn_entity"; then
        break
    fi
    if ! kill -0 "$GZ_PID" 2>/dev/null; then
        echo "[gazebo_boot] FATAL: gzserver exited during startup" >&2
        exit 1
    fi
    sleep 1
done

echo "[gazebo_boot] spawning $TURTLEBOT3_MODEL at origin…"
ros2 run gazebo_ros spawn_entity.py \
    -entity "$TURTLEBOT3_MODEL" \
    -file "$SDF" \
    -x 0 -y 0 -z 0.01

echo "[gazebo_boot] up — /cmd_vel (sub) and /odom (pub) live. Waiting on gzserver…"
wait "$GZ_PID"
