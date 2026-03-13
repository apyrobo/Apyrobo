# =============================================================================
# APYROBO Development Container
# ROS 2 Humble + Gazebo + Nav2 + TurtleBot4 + Python AI stack
# =============================================================================
#
# Build:    docker compose -f docker/docker-compose.yml build
# Run:      docker compose -f docker/docker-compose.yml up -d
# Shell:    docker compose -f docker/docker-compose.yml exec apyrobo bash
# Demo:     docker compose -f docker/docker-compose.yml exec apyrobo bash scripts/demo.sh
#
# =============================================================================

FROM ros:humble-desktop

LABEL maintainer="APYROBO Contributors"
LABEL description="APYROBO — full MVP development environment"

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------------------------
# 1. System packages + ROS 2 packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    git \
    curl \
    wget \
    software-properties-common \
    # Python
    python3-pip \
    python3-venv \
    python3-dev \
    python3-colcon-common-extensions \
    # ─── ROS 2 core ───
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    # ─── TurtleBot4 ───
    ros-humble-turtlebot4-simulator \
    ros-humble-turtlebot4-description \
    ros-humble-turtlebot4-msgs \
    ros-humble-turtlebot4-navigation \
    ros-humble-turtlebot4-bringup \
    # ─── Nav2 (full stack) ───
    ros-humble-nav2-bringup \
    ros-humble-nav2-bt-navigator \
    ros-humble-nav2-controller \
    ros-humble-nav2-planner \
    ros-humble-nav2-recoveries \
    ros-humble-nav2-waypoint-follower \
    ros-humble-nav2-lifecycle-manager \
    ros-humble-nav2-map-server \
    ros-humble-nav2-amcl \
    ros-humble-nav2-msgs \
    # ─── Perception + Transforms ───
    ros-humble-tf2-ros \
    ros-humble-tf2-tools \
    ros-humble-tf2-geometry-msgs \
    ros-humble-robot-localization \
    ros-humble-slam-toolbox \
    # ─── Visualization ───
    ros-humble-rviz2 \
    ros-humble-rqt \
    ros-humble-rqt-graph \
    ros-humble-rqt-topic \
    # ─── Useful tools ───
    vim \
    less \
    tree \
    tmux \
    htop \
    # ─── Screen recording ───
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Python dependencies
# ---------------------------------------------------------------------------
WORKDIR /workspace

# Install Python deps first (cache-friendly layer)
RUN pip3 install --no-cache-dir --break-system-packages \
    # APYROBO core
    pydantic>=2.0 \
    networkx>=3.0 \
    pyyaml>=6.0 \
    # AI agent providers
    litellm>=1.0 \
    # ROS 2 Python helpers
    transforms3d \
    # Dev tools
    pytest>=8.0 \
    pytest-cov>=5.0 \
    pytest-asyncio>=0.23 \
    ruff>=0.4 \
    mypy>=1.10 \
    # Utilities
    rich \
    click

# ---------------------------------------------------------------------------
# 3. APYROBO source
# ---------------------------------------------------------------------------
COPY . .
RUN pip3 install --no-cache-dir --break-system-packages -e ".[dev]" || \
    pip3 install --no-cache-dir --break-system-packages -e .

# ---------------------------------------------------------------------------
# 4. Shell configuration
# ---------------------------------------------------------------------------
RUN echo '# ROS 2 environment' >> /root/.bashrc && \
    echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo 'export PYTHONPATH=/workspace:$PYTHONPATH' >> /root/.bashrc && \
    echo 'export ROS_DOMAIN_ID=0' >> /root/.bashrc && \
    echo 'export RCUTILS_COLORIZED_OUTPUT=1' >> /root/.bashrc && \
    echo '' >> /root/.bashrc && \
    echo '# APYROBO aliases' >> /root/.bashrc && \
    echo 'alias apytest="cd /workspace && python3 run_tests.py"' >> /root/.bashrc && \
    echo 'alias apydemo="cd /workspace && python3 examples/demo_full.py"' >> /root/.bashrc && \
    echo 'alias apylaunch="cd /workspace && bash scripts/launch.sh"' >> /root/.bashrc && \
    echo '' >> /root/.bashrc && \
    echo 'echo "══════════════════════════════════════════════"' >> /root/.bashrc && \
    echo 'echo "  APYROBO Dev Container"' >> /root/.bashrc && \
    echo 'echo "  ROS 2 Humble + Gazebo + Nav2 + TurtleBot4"' >> /root/.bashrc && \
    echo 'echo ""' >> /root/.bashrc && \
    echo 'echo "  apytest   — run test suite"' >> /root/.bashrc && \
    echo 'echo "  apydemo   — run mock demo"' >> /root/.bashrc && \
    echo 'echo "  apylaunch — launch Gazebo + full pipeline"' >> /root/.bashrc && \
    echo 'echo "══════════════════════════════════════════════"' >> /root/.bashrc

# ---------------------------------------------------------------------------
# 5. Create scripts directory
# ---------------------------------------------------------------------------
RUN mkdir -p /workspace/scripts

# ---------------------------------------------------------------------------
# 6. Healthcheck
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "from apyrobo import Robot; print('OK')" || exit 1

CMD ["bash"]
