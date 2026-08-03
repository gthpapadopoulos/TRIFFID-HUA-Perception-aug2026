# TRIFFID UAV Perception

Standalone (no ROS2) drone-video perception pipeline: video in, pixel-space
GeoJSON out, a downstream consumer raycasts those pixel points against a **3D Gaussian Splatting
(3DGS)** reconstruction to recover real-world coordinates.

## Quick Start

```bash
# Build + start the container (CUDA torch, falls back to CPU without a GPU)
docker compose -f docker-compose.uav.yml up -d

# Process a local video (place it under ./uav_videos/, its .srt next to it)
docker exec triffid_uav_perception bash -c \
  "PYTHONPATH=/app/src/triffid_uav_perception:/app/src \
   python -m triffid_uav_perception.uav_node /app/videos/clip.mp4 --output /app/samples"
# → ./uav_samples/clip_detections.geojson

# Live mode: poll FUTURISED for new video uploads (MP4 + SRT)
docker exec -e FUTURISED_MEDIA_API_KEY='your-key' triffid_uav_perception bash -c \
  "PYTHONPATH=/app/src/triffid_uav_perception:/app/src \
   python -m triffid_uav_perception.uav_node --poll-api \
       --api-download-dir /app/videos --output /app/samples"

# Stop
docker rm -f triffid_uav_perception
```

## How It Works

- Runs the detection model
- Processes every `--stride`-th frame (default `5`), tracking always runs on
  every processed frame so IDs stay stable.
- **Per-track dedup:** a given track id is emitted at most once every
  `--sample-seconds` (default `1.0`).
- **Sample point count by ontology geometry type** this only controls
  how many independent pixel points get sampled per detection, it does
  not describe a shape.

  | Geometry type | Points | How |
  |---|---|---|
  | `Point` | 1 | mask centroid (bbox centre fallback) |
  | `Line` | 4 | `cv2.minAreaRect` corners — spread across the structure's height, not just its ground line |
  | `Polygon` (compact) | 4 | `cv2.minAreaRect` corners |
  | `Polygon` (large-area) | 8–12 | simplified contour (`cv2.approxPolyDP`) for water, debris, dirt road, road, green/dry/burnt grass, pavement, mud — more samples across a bigger/irregular area |

- `--classes id,label,...` restricts output to an allowlist (default: all).

## Output GeoJSON

Written to `<output>/<video-stem>_detections.geojson` (default: next to the
input video, the runner always uses `./uav_samples/`). One RFC-7946
FeatureCollection per video with a top-level `metadata` foreign member
(permitted by RFC 7946 §6.1) carrying the run parameters.

**Coordinates are pixels** in the original frame resolution, origin top-left
(`x` = column, `y` = row) — flagged per feature via
`properties.coordinate_space: "pixel"`. Geometry is always **`Point`** (1
position) or **`MultiPoint`** (several independent positions, no closure,
no implied shape) — never `LineString`/`Polygon` — since these are raw
raycasting samples, not a detection outline.

```json
{
  "type": "FeatureCollection",
  "metadata": {
    "video": "clip.mp4",
    "model": "/app/best.pt",
    "fps": 29.97,
    "frame_width": 3840,
    "frame_height": 2160,
    "total_frames": 21071,
    "stride": 5,
    "tracker": "bytetrack.yaml",
    "confidence": 0.5,
    "sample_seconds": 1.0,
    "classes": "all",
    "geometry_source": "classes.txt",
    "processed_frames": 4215,
    "total_detections": 5821,
    "coordinate_space": "pixels in original frame resolution, origin top-left (x = column, y = row)",
    "generated_at": "2026-07-17T12:00:00+00:00"
  },
  "features": [
    {
      "type": "Feature",
      "id": "7",
      "geometry": {
        "type": "MultiPoint",
        "coordinates": [[1825.0, 1310.0], [1980.0, 1330.0],
                        [1972.0, 1700.0], [1817.0, 1680.0]]
      },
      "properties": {
        "class": "building",
        "class_id": 14,
        "id": "7",
        "confidence": 0.8123,
        "frame_index": 30,
        "timestamp_s": 1.001,
        "category": "infrastructure",
        "detection_type": "seg",
        "source": "uav",
        "coordinate_space": "pixel",
        "classes_txt_geometry": "Polygon",
        "marker-color": "#708090",
        "marker-size": "medium",
        "marker-symbol": "building"
      }
    }
  ]
}
```

| Property | Meaning |
|---|---|
| `id` (feature + property) | Persistent ByteTrack track ID (stable across frames, the same object emitted in a later window shares the id) |
| `class` / `class_id` | class name / index |
| `frame_index` / `timestamp_s` | Where in the video this emission happened |
| `coordinate_space` | Always `"pixel"` until the 3DGS raycast replaces coordinates |
| `classes_txt_geometry` | The ontology's geometry classification (`Point`/`Line`/`Polygon`), kept for traceability of why this detection got N points |
| `marker-*` | SimpleStyle marker styling, identical scheme to the UGV `geojson_bridge` output |

`geometry.type` is `Point` for single-sample classes, `MultiPoint` for
everything else.

## CLI Reference

```bash
python -m triffid_uav_perception.uav_node VIDEO [options]
python -m triffid_uav_perception.uav_node --poll-api [options]   # see below
```

| Option | Default | Meaning |
|---|---|---|
| `--model` | `best.pt` | YOLO segmentation model |
| `--confidence` | `0.5` | Confidence threshold |
| `--imgsz` | `1280` | YOLO input size |
| `--stride` | `5` | Process every Nth frame |
| `--tracker` | `bytetrack.yaml` | Ultralytics tracker config |
| `--sample-seconds` | `1.0` | Per-track emission throttle (`0` = off) |
| `--classes` | *(all)* | Comma-separated class ids and/or labels |
| `--output` | *(video dir)* | Output directory for the GeoJSON |
| `--srt` | *(auto-detect)* | DJI SRT telemetry sidecar (default: `<video-stem>.srt` next to the video) |
| `--post-telesto` | off | Upload the collection to TELESTO after each video |
| `--telesto-base-url` | env `TELESTO_BASE_URL` / built-in | Backend override |
| `--poll-api` | off | Poll FUTURISED for new video uploads instead of a local file (PoC — see below) |
| `--api-media-key` | env `FUTURISED_MEDIA_API_KEY` | Media Files API key (required with `--poll-api`) |
| `--api-camera` | `Wide` | Camera filter: `Wide`/`Zoom`/`Thermal`/empty |
| `--api-poll-interval` | `30` | Seconds between polls |
| `--api-download-dir` | `./uav_videos` | Where downloaded videos land |
| `-v` | | Debug logging |

Exactly one of a positional `VIDEO` or `--poll-api` is required.

## SRT Telemetry Sidecar

FUTURISED delivers each drone video as an **MP4 + an `.SRT` file**
carrying per-frame telemetry (DJI drones write one subtitle block per
frame). When an SRT is available, `--srt path.srt`, or auto-detected as
`<video-stem>.srt`/`.SRT` next to the video, the pipeline joins it to
frames by frame number (nearest-timestamp fallback for lower-rate SRTs)
and attaches the drone/camera telemetry to every emitted feature:

| Property | Source (modern DJI SRT key) |
|---|---|
| `drone_latitude` / `drone_longitude` | `latitude` / `longitude` (or legacy `GPS(lon,lat,alt)`) |
| `drone_altitude_m` | `abs_alt` |
| `drone_rel_altitude_m` | `rel_alt` (or legacy `H <n>m`) |
| `gimbal_yaw` / `gimbal_pitch` / `gimbal_roll` | `gb_yaw` / `gb_pitch` / `gb_roll` |

In `--poll-api` mode, `.SRT` files are polled and downloaded alongside
`.MP4`/`.MOV` into the same directory, where sidecar auto-detection pairs
them by basename. Limitation: an SRT that appears *after* its video
was already processed does not trigger reprocessing.

## TELESTO Upload (optional)

`--post-telesto` (or `POST_TELESTO=1` with the runner) PUTs every feature to
the TELESTO Map Manager REST API via `triffid_telesto.telesto_client` after
each video finishes.

## Docker Setup

Separate image from the UGV (no ROS2): Python 3.10 slim + CUDA PyTorch
(cu121 wheels — they fall back to CPU on hosts without a GPU). GPU access is
wired via `deploy.resources` in `docker-compose.uav.yml` and the
`nvidia-container-toolkit`, remove the `deploy` block to run CPU-only.

| Host Path | Container Path | Purpose |
|---|---|---|
| `./src/triffid_uav_perception/` | `/app/src/triffid_uav_perception/` | Source code (editable) |
| `./src/triffid_telesto/` | `/app/src/triffid_telesto/` | TELESTO REST client (for `--post-telesto`) |
| `./uav_videos/` | `/app/videos/` | Input videos |
| `./uav_data/` | `/app/uav_data/` | Additional input media |
| `./uav_samples/` | `/app/samples/` | Output GeoJSON |
| `./best.pt` | `/app/best.pt` | YOLO model (read-only) |
