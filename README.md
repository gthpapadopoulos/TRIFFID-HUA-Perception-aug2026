# TRIFFID UGV/UAV Perception Module (T4.1) - AUGUST 2026 VERSION

**Harokopio University of Athens, TRIFFID Research Project**

**Contact: amichailidou@hua.gr**

## General Information

Two independent modules, each in its own Docker container, both running on
live data:

| Module | Input | Output |
|---|---|---|
| **UGV** (`triffid_ugv_perception`) | Live RGB-D camera topics + GPS/odometry | `Detection3DArray` + GeoJSON on ROS topics, periodic GeoJSON snapshot to disk and/or upsert to the TELESTO backend |
| **UAV** (`triffid_uav_perception`) | Drone video (MP4 + DJI SRT telemetry sidecar), polled live from API | One pixel-space GeoJSON per video, saved to disk, optional upload to the TELESTO backend |

For instructions on how to run each module see below.

## 1. UGV Module

Real-time 3D detection: YOLOv11-seg on RGB, depth sampled at detection
pixels (pixel-aligned RGB-D), back-projection to 3D, TF to `b2/base_link`,
ByteTrack IDs, GeoJSON geo-referencing via GPS + heading (will be changed to be via tf2 tree once available).

### Build and start the container

```bash
docker compose up -d
docker exec -it triffid_perception bash
# inside the container:
cd /ws && source /opt/ros/humble/setup.bash && colcon build --symlink-install
source /ws/install/setup.bash
```

Requires the NVIDIA container toolkit (GPU). The container uses
`network_mode: host`, so it sees your ROS 2 traffic directly
(`ROS_DOMAIN_ID=42`, CycloneDDS , see `cyclonedds.xml`).

### Run the perception node

Inside the container (one terminal per node, or append `&`):

```bash
ros2 run triffid_ugv_perception ugv_node --ros-args \
    -p model_path:=/ws/best.pt \
    -p rgb_image_topic:=/camera_front_435i/realsense_front_435i/color/image_raw \
    -p depth_image_topic:=/camera_front_435i/realsense_front_435i/depth/image_rect_raw \
    -p camera_info_topic:=/camera_front_435i/realsense_front_435i/color/camera_info
```

Set the three topic parameters to wherever the camera publishes. The
camera must be pixel-aligned RGB-D: `bgr8` (or YUYV) colour, `16UC1` depth
in millimetres, one shared `CameraInfo`. A TF from the camera optical
frame (read from `CameraInfo.header.frame_id`) to `b2/base_link` must be
available.

Other `ugv_node` parameters (defaults): `confidence_threshold` (0.35),
`yolo_imgsz` (1280), `target_frame` (`b2/base_link`), `nms_merge_dist_m`
(0.5), and tracker tuning , `tracker_iou_threshold` (0.30),
`tracker_iou_threshold_low` (0.15), `tracker_conf_high` (0.40),
`tracker_max_age` (30), `tracker_n_init` (3), `tracker_pos_gate` (2.0).

### Run the GeoJSON bridge

```bash
ros2 run triffid_ugv_perception geojson_bridge --ros-args \
    -p save_to_disk:=true \
    -p publish_to_api:=false \
    -p publish_period_s:=5.0 \
    -p output_dir:=/ws/samples
```

| Parameter | Default | Meaning |
|---|---|---|
| `save_to_disk` | `true` | Every period, write the accumulated deduplicated feature set atomically to `output_dir/ugv_detections.geojson` (host: `./samples/`) |
| `publish_to_api` | `false` | Every period, **upsert** the accumulated set to `api_url` (GET remote → PUT new / PATCH improved / skip unchanged |
| `publish_period_s` | `5.0` | Seconds between periodic flushes |
| `output_dir` | `/ws/samples` | Snapshot directory inside the container |
| `api_url` | `https://crispres.com/wp-json/map-manager/v1/features` | TELESTO Map Manager API endpoint |
| `dedup_radius_m` | `3.0` | Same-class features within this distance merge (highest confidence kept) |
| `gps_origin_lat/lon/alt` | `0.0` | Optional GPS seed used until the first `/fix` arrives |

Both flags are independent, enable either or both. The bridge also
publishes the per-frame GeoJSON on the ROS topic
`/ugv/detections/front/geojson` (`std_msgs/String`) regardless.

The bridge subscribes to `/fix` (`NavSatFix`) and `/dog_odom`
(`Odometry`) for geo-referencing. Until a GPS fix arrives, features are
emitted in body-frame coordinates with `"local_frame": true`.

### UGV topics

| Direction | Topic | Type |
|---|---|---|
| in | *( RGB / depth / CameraInfo topics)* | `sensor_msgs/Image`, `CameraInfo` |
| in | `/fix`, `/dog_odom`, `/tf`, `/tf_static` | `NavSatFix`, `Odometry`, TF |
| out | `/ugv/detections/front/detections_3d` | `vision_msgs/Detection3DArray` |
| out | `/ugv/detections/front/segmentation` | `sensor_msgs/Image` (mono8 label map) |
| out | `/ugv/detections/front/debug_image` | `sensor_msgs/Image` (lazy) |
| out | `/ugv/detections/front/geojson` | `std_msgs/String` (GeoJSON) |

**Alternatively, both nodes start together via the launch file:**

```bash
ros2 launch triffid_ugv_perception ugv_perception.launch.py \
    rgb_image_topic:=... depth_image_topic:=... camera_info_topic:=... \
    save_to_disk:=true publish_to_api:=false
```

---

## 2. UAV Module

Drone-video detection producing pixel-space GeoJSON, with
persistent ByteTrack IDs, each detection reduced to a few independent
pixel sample points (`Point`/`MultiPoint`), DJI SRT
telemetry joined per frame so every feature also carries the camera pose
(`drone_latitude/longitude/altitude_m`, `gimbal_yaw/pitch/roll`). A
downstream consumer raycasts the pixel points against a 3DGS
reconstruction to obtain real-world coordinates.

### Build and start the container

```bash
docker compose -f docker-compose.uav.yml up -d
```

### Live mode - poll the API

Polls for newly uploaded videos (`.mp4`/`.mov`) and their `.srt`
telemetry files, processes each video, and writes one
GeoJSON per video:

```bash
docker exec -e FUTURISED_MEDIA_API_KEY='your-key' triffid_uav_perception bash -c \
  "PYTHONPATH=/app/src/triffid_uav_perception:/app/src \
   python -m triffid_uav_perception.uav_node --poll-api \
       --api-camera Wide --api-poll-interval 30 \
       --api-download-dir /app/videos --output /app/samples"
```

Outputs land in `./uav_samples/<video-stem>_detections.geojson` on the
host. Add `--post-telesto` to also upload each result to the TELESTO
backend (off by default).

### Local-file test mode

To verify the module place the mp4 file under
`./uav_videos/`, with its `.srt`, auto-detected by name):

```bash
docker exec triffid_uav_perception bash -c \
  "PYTHONPATH=/app/src/triffid_uav_perception:/app/src \
   python -m triffid_uav_perception.uav_node /app/videos/flight.mp4 \
       --output /app/samples"
```

Key flags (defaults): `--model /app/best.pt`* , `--confidence 0.5`,
`--stride 5` (every Nth frame), `--sample-seconds 1.0` (each track id
emitted at most once per second), `--classes id,label,...` (allowlist),
`--srt path` (override sidecar auto-detect).

### UAV output schema (summary)

One RFC-7946 FeatureCollection per video with a top-level `metadata`
member (video/model/fps/stride/`srt_file`/counts). Every feature:

```json
{
  "type": "Feature",
  "id": "7",
  "geometry": {"type": "MultiPoint",
               "coordinates": [[1825.0, 1310.0], [1980.0, 1330.0],
                               [1972.0, 1700.0], [1817.0, 1680.0]]},
  "properties": {
    "class": "building", "class_id": 14, "id": "7", "confidence": 0.81,
    "frame_index": 30, "timestamp_s": 1.001,
    "coordinate_space": "pixel", "classes_txt_geometry": "Polygon",
    "category": "infrastructure", "detection_type": "seg", "source": "uav",
    "drone_latitude": 49.726349, "drone_longitude": 13.350951,
    "drone_altitude_m": 431.53, "drone_rel_altitude_m": 36.28,
    "gimbal_yaw": 84.1, "gimbal_pitch": -24.5, "gimbal_roll": 0.0,
    "marker-color": "#708090", "marker-size": "medium",
    "marker-symbol": "building"
  }
}
```

- `geometry` coordinates are **pixels** (origin top-left, x = column) ,
  independent sample points for raycasting, not an outline. `id` is the
  persistent track id.
- `drone_*` / `gimbal_*` describe the camera at that frame (from the
  SRT), they are omitted when no SRT is available.


## Repository layout

```
├── docker-compose.yml / Dockerfile           # UGV container (ROS 2 Humble + CUDA)
├── docker-compose.uav.yml / Dockerfile.uav   # UAV container (Python + CUDA torch)
├── best.pt                                   # shared YOLOv11-seg model (Git LFS)
└── src/
    ├── triffid_ugv_perception/               # ROS 2 package (ugv_node, geojson_bridge, tracker)
    ├── triffid_uav_perception/               # UAV pipeline (uav_node, srt_metadata, api_client)
    └── triffid_telesto/                      # TELESTO REST client (used by --post-telesto)
```
