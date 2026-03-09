# =============================================================================
# APYROBO Development Container
# ROS 2 Humble + Gazebo + Python dev tools
# =============================================================================
# Build:   docker compose build
# Run:     docker compose up
# Shell:   docker compose exec apyrobo bash
# =============================================================================

FROM ros:humble-desktop

LABEL maintainer="APYROBO Contributors"
LABEL description="APYROBO development environment — ROS 2 Humble + Gazebo"

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    git \
    curl \
    wget \
    # Python
    python3-pip \
    python3-venv \
    python3-dev \
    # ROS 2 extras
    ros-humble-gazebo-ros-pkgs \
    ros-humble-turtlebot4-simulator \
    ros-humble-turtlebot4-description \
    ros-humble-turtlebot4-msgs \
    ros-humble-turtlebot4-navigation \
    ros-humble-nav2-bringup \
    ros-humble-rviz2 \
    # Useful tools
    vim \
    less \
    tree \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
WORKDIR /workspace

# Copy only dependency files first (cache-friendly)
COPY pyproject.toml .
RUN pip3 install --no-cache-dir -e ".[dev]" 2>/dev/null || true

# Copy full source
COPY . .

# Install the package in editable mode
RUN pip3 install --no-cache-dir -e ".[dev]"

# ---------------------------------------------------------------------------
# ROS 2 workspace setup
# ---------------------------------------------------------------------------
# Source ROS 2 in every new shell
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "export PYTHONPATH=/workspace:\$PYTHONPATH" >> /root/.bashrc

# ---------------------------------------------------------------------------
# Default command
# ---------------------------------------------------------------------------
CMD ["bash"]
