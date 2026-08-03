#!/bin/bash
set -e

# Source ROS2 Humble
source /opt/ros/humble/setup.bash

# Build workspace if not built yet
if [ ! -d "/ws/install" ]; then
    echo "[entrypoint] Building workspace for the first time..."
    cd /ws && colcon build --symlink-install
fi

# Source workspace overlay
if [ -f "/ws/install/setup.bash" ]; then
    source /ws/install/setup.bash
fi

exec "$@"
