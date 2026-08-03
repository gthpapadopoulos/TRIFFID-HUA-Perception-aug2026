FROM ros:humble-perception-jammy

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DOMAIN_ID=42

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-image-transport \
    && rm -rf /var/lib/apt/lists/*

# Python deps
# Pin numpy<2 because ROS2 Humble cv_bridge was compiled against NumPy 1.x.
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    "packaging>=22" \
    opencv-python-headless \
    ultralytics \
    scipy

# Workspace setup 
WORKDIR /ws
# src/ is bind-mounted at runtime via docker-compose

# Ensure ROS2 + workspace overlay are sourced in every shell
# (docker compose exec runs non-interactive bash, which skips .bashrc)
# BASH_ENV is sourced by every non-interactive bash invocation.
ENV BASH_ENV=/etc/triffid_ros_env.sh
RUN echo '#!/bin/bash'                                          >  /etc/triffid_ros_env.sh && \
    echo 'source /opt/ros/humble/setup.bash'                    >> /etc/triffid_ros_env.sh && \
    echo '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' >> /etc/triffid_ros_env.sh && \
    chmod +x /etc/triffid_ros_env.sh

# Entrypoint 
COPY docker_entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh
ENTRYPOINT ["/docker_entrypoint.sh"]
CMD ["bash"]
