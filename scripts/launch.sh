#!/bin/bash
# =============================================================================
# APYROBO Launch Script
# Boots Gazebo + TurtleBot4 + Nav2 + APYROBO in a single tmux session
#
# Usage:
#   bash scripts/launch.sh              # full launch
#   bash scripts/launch.sh --no-gui     # headless (no Gazebo GUI)
#   bash scripts/launch.sh --world warehouse  # custom world
#
# Tmux panes:
#   0: Gazebo + TurtleBot4 simulator
#   1: Nav2 navigation stack
#   2: APYROBO interactive shell (ready for commands)
#   3: Sensor monitor (topic echo)
#
# Controls:
#   Ctrl+B then arrow keys — switch panes
#   Ctrl+B then d — detach (everything keeps running)
#   tmux attach -t apyrobo — reattach
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SESSION="apyrobo"
WORLD="${APYROBO_WORLD:-warehouse}"
USE_GUI=true
NAV2_PARAMS="/opt/ros/humble/share/turtlebot4_navigation/config/nav2.yaml"

# Parse args
for arg in "$@"; do
    case $arg in
        --no-gui)    USE_GUI=false ;;
        --world)     shift; WORLD="${1:-warehouse}" ;;
        --help|-h)
            echo "Usage: bash scripts/launch.sh [--no-gui] [--world NAME]"
            echo ""
            echo "Options:"
            echo "  --no-gui    Run Gazebo headless (no GUI window)"
            echo "  --world     Gazebo world name (default: warehouse)"
            exit 0 ;;
    esac
done

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

echo "══════════════════════════════════════════════"
echo "  APYROBO Launch"
echo "══════════════════════════════════════════════"
echo ""

# Source ROS 2
source /opt/ros/humble/setup.bash
export PYTHONPATH=/workspace:$PYTHONPATH

# Check ROS 2 is working
if ! ros2 --help > /dev/null 2>&1; then
    echo "ERROR: ROS 2 not found. Are you inside the Docker container?"
    echo "  Run: docker compose -f docker/docker-compose.yml exec apyrobo bash"
    exit 1
fi
echo "✓ ROS 2 Humble"

# Check Gazebo
if ! which gzserver > /dev/null 2>&1; then
    echo "ERROR: Gazebo not found."
    exit 1
fi
echo "✓ Gazebo"

# Check DISPLAY for GUI
if [ "$USE_GUI" = true ]; then
    if [ -z "${DISPLAY:-}" ]; then
        echo "⚠ DISPLAY not set — switching to headless mode"
        USE_GUI=false
    else
        echo "✓ Display: $DISPLAY"
    fi
fi

# Check tmux
if ! which tmux > /dev/null 2>&1; then
    echo "ERROR: tmux not found. Install with: apt install tmux"
    exit 1
fi
echo "✓ tmux"

echo ""
echo "Configuration:"
echo "  World:    $WORLD"
echo "  GUI:      $USE_GUI"
echo "  Session:  $SESSION"
echo ""

# ---------------------------------------------------------------------------
# Kill any existing session
# ---------------------------------------------------------------------------
tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 1

# ---------------------------------------------------------------------------
# Build Gazebo launch command
# ---------------------------------------------------------------------------

if [ "$USE_GUI" = true ]; then
    GAZEBO_CMD="ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py world:=${WORLD}"
else
    GAZEBO_CMD="ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py world:=${WORLD} headless:=true"
fi

# Fallback if turtlebot4_ignition isn't available (older setup)
if ! ros2 pkg list 2>/dev/null | grep -q turtlebot4_ignition_bringup; then
    echo "⚠ turtlebot4_ignition_bringup not found, trying turtlebot4_gazebo..."
    if [ "$USE_GUI" = true ]; then
        GAZEBO_CMD="ros2 launch turtlebot4_gazebo turtlebot4_spawn.launch.py"
    else
        GAZEBO_CMD="ros2 launch turtlebot4_gazebo turtlebot4_spawn.launch.py headless:=true"
    fi
fi

# ---------------------------------------------------------------------------
# Create tmux session with 4 panes
# ---------------------------------------------------------------------------

echo "Starting tmux session: $SESSION"
echo ""

# Create session with first pane (Gazebo)
tmux new-session -d -s "$SESSION" -n "main"

# Pane 0: Gazebo + TurtleBot4
tmux send-keys -t "$SESSION:main.0" "source /opt/ros/humble/setup.bash" Enter
tmux send-keys -t "$SESSION:main.0" "echo '── Pane 0: Gazebo + TurtleBot4 ──'" Enter
tmux send-keys -t "$SESSION:main.0" "echo 'Starting simulator (this takes 30-60s)...'" Enter
tmux send-keys -t "$SESSION:main.0" "$GAZEBO_CMD" Enter

# Split horizontally for Nav2
tmux split-window -h -t "$SESSION:main"

# Pane 1: Nav2
tmux send-keys -t "$SESSION:main.1" "source /opt/ros/humble/setup.bash" Enter
tmux send-keys -t "$SESSION:main.1" "echo '── Pane 1: Nav2 Navigation ──'" Enter
tmux send-keys -t "$SESSION:main.1" "echo 'Waiting 20s for Gazebo to start...'" Enter
tmux send-keys -t "$SESSION:main.1" "sleep 20 && ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true params_file:=${NAV2_PARAMS} 2>&1 || echo 'Nav2 launch failed — may need a map file. Try: ros2 launch turtlebot4_navigation nav2.launch.py'" Enter

# Split pane 1 vertically for APYROBO shell
tmux split-window -v -t "$SESSION:main.1"

# Pane 2: APYROBO interactive shell
tmux send-keys -t "$SESSION:main.2" "source /opt/ros/humble/setup.bash" Enter
tmux send-keys -t "$SESSION:main.2" "export PYTHONPATH=/workspace:\$PYTHONPATH" Enter
tmux send-keys -t "$SESSION:main.2" "cd /workspace" Enter
tmux send-keys -t "$SESSION:main.2" "echo '── Pane 2: APYROBO Shell ──'" Enter
tmux send-keys -t "$SESSION:main.2" "echo ''" Enter
tmux send-keys -t "$SESSION:main.2" "echo 'Waiting 40s for Gazebo + Nav2 to initialise...'" Enter
tmux send-keys -t "$SESSION:main.2" "sleep 40" Enter
tmux send-keys -t "$SESSION:main.2" "echo ''" Enter
tmux send-keys -t "$SESSION:main.2" "echo 'Ready! Try:'" Enter
tmux send-keys -t "$SESSION:main.2" "echo '  python3 scripts/integration_test.py'" Enter
tmux send-keys -t "$SESSION:main.2" "echo '  python3 scripts/demo_gazebo.py'" Enter
tmux send-keys -t "$SESSION:main.2" "echo '  python3 scripts/record_demo.py'" Enter
tmux send-keys -t "$SESSION:main.2" "echo ''" Enter

# Split pane 0 vertically for sensor monitor
tmux split-window -v -t "$SESSION:main.0"

# Pane 3: Sensor monitor
tmux send-keys -t "$SESSION:main.3" "source /opt/ros/humble/setup.bash" Enter
tmux send-keys -t "$SESSION:main.3" "echo '── Pane 3: Sensor Monitor ──'" Enter
tmux send-keys -t "$SESSION:main.3" "echo 'Waiting 30s then listing topics...'" Enter
tmux send-keys -t "$SESSION:main.3" "sleep 30 && echo '── Active ROS 2 Topics ──' && ros2 topic list" Enter

# Select the APYROBO pane
tmux select-pane -t "$SESSION:main.2"

echo "══════════════════════════════════════════════"
echo "  Launched! Attaching to tmux session..."
echo ""
echo "  Ctrl+B, arrows  = switch panes"
echo "  Ctrl+B, d        = detach"
echo "  tmux attach -t $SESSION = reattach"
echo "══════════════════════════════════════════════"
echo ""

# Attach
tmux attach-session -t "$SESSION"
