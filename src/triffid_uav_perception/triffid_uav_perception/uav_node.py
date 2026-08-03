"""
TRIFFID UAV Perception Node

Usage:
  python -m triffid_uav_perception.uav_node video.mp4
  python -m triffid_uav_perception.uav_node video.mp4 \\
      --model best.pt --stride 5 --sample-seconds 1.0 --output ./samples

  poll API for new video uploads instead of a local file
  python -m triffid_uav_perception.uav_node --poll-api \\
      --api-media-key "$FUTURISED_MEDIA_API_KEY" --output ./uav_samples
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from triffid_uav_perception.api_client import FuturisedClient
from triffid_uav_perception.srt_metadata import SrtIndex, find_sidecar_srt

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

log = logging.getLogger('triffid_uav')


TARGET_CLASSES = {
    0: 'water', 1: 'fence', 2: 'green tree', 3: 'helmet',
    4: 'flame', 5: 'smoke', 6: 'first responder', 7: 'destroyed vehicle',
    8: 'fire hose', 9: 'scba', 10: 'boot', 11: 'green plant',
    12: 'mask', 13: 'window', 14: 'building', 15: 'destroyed building',
    16: 'debris', 17: 'ladder', 18: 'dirt road', 19: 'dry tree',
    20: 'wall', 21: 'civilian vehicle', 22: 'road', 23: 'citizen',
    24: 'green grass', 25: 'pole', 26: 'boat', 27: 'pavement',
    28: 'dry grass', 29: 'animal', 30: 'excavator', 31: 'door',
    32: 'mud', 33: 'barrier', 34: 'hole in the ground', 35: 'bag',
    36: 'burnt tree', 37: 'ambulance', 38: 'fire truck', 39: 'cone',
    40: 'bicycle', 41: 'tower', 42: 'silo', 43: 'military personnel',
    44: 'burnt grass', 45: 'ax', 46: 'glove', 47: 'crane',
    48: 'stairs', 49: 'dry plant', 50: 'furniture', 51: 'tank',
    52: 'protective glasses', 53: 'barrel', 54: 'shovel',
    55: 'fire hydrant', 56: 'police vehicle', 57: 'burnt plant',
    58: 'army vehicle', 59: 'chainsaw', 60: 'aerial vehicle',
    61: 'lifesaver', 62: 'extinguisher',
}

# Same styling helpers as UGV geojson_bridge — kept here to avoid
# importing from the UGV package (they're independent deployments).

_CLASS_COLORS = {
    'flame': '#ff0000', 'smoke': '#ff4500', 'burnt tree': '#8b0000',
    'burnt grass': '#a52a2a', 'burnt plant': '#b22222',
    'fire hose': '#dc143c', 'fire hydrant': '#ff6347',
    'fire truck': '#ff0000', 'extinguisher': '#ff1493',
    'first responder': '#1e90ff', 'citizen': '#4169e1',
    'military personnel': '#000080',
    'civilian vehicle': '#0000ff', 'destroyed vehicle': '#00008b',
    'ambulance': '#4682b4', 'police vehicle': '#191970',
    'army vehicle': '#2f4f4f', 'boat': '#5f9ea0', 'bicycle': '#00ff00',
    'aerial vehicle': '#87ceeb',
    'green tree': '#228b22', 'green plant': '#32cd32',
    'green grass': '#7cfc00', 'dry tree': '#daa520',
    'dry grass': '#bdb76b', 'dry plant': '#f0e68c', 'animal': '#ff8c00',
    'building': '#708090', 'destroyed building': '#696969',
    'wall': '#808080', 'road': '#a9a9a9', 'pavement': '#c0c0c0',
    'dirt road': '#d2b48c', 'window': '#b0c4de', 'door': '#8b4513',
    'stairs': '#a0522d', 'pole': '#778899', 'tower': '#556b2f',
    'silo': '#6b8e23',
    'debris': '#ff8c00', 'fence': '#daa520', 'barrier': '#ffd700',
    'cone': '#ff7f50', 'hole in the ground': '#8b4513',
    'mud': '#a0522d', 'water': '#00bfff',
    'helmet': '#9370db', 'scba': '#8a2be2', 'boot': '#4b0082',
    'mask': '#9400d3', 'glove': '#da70d6', 'protective glasses': '#ba55d3',
    'ladder': '#ff8c00', 'ax': '#cd853f', 'shovel': '#d2691e',
    'chainsaw': '#b8860b', 'bag': '#bc8f8f', 'barrel': '#8b8682',
    'furniture': '#deb887', 'tank': '#2e8b57', 'crane': '#b8860b',
    'excavator': '#daa520', 'lifesaver': '#ff4500',
}

_CLASS_CATEGORIES = {
    'flame': 'hazard', 'smoke': 'hazard', 'burnt tree': 'hazard',
    'burnt grass': 'hazard', 'burnt plant': 'hazard',
    'first responder': 'person', 'citizen': 'person',
    'military personnel': 'person',
    'civilian vehicle': 'vehicle', 'destroyed vehicle': 'vehicle',
    'ambulance': 'vehicle', 'police vehicle': 'vehicle',
    'fire truck': 'vehicle', 'army vehicle': 'vehicle',
    'boat': 'vehicle', 'bicycle': 'vehicle', 'aerial vehicle': 'vehicle',
    'green tree': 'nature', 'green plant': 'nature',
    'green grass': 'nature', 'dry tree': 'nature',
    'dry grass': 'nature', 'dry plant': 'nature', 'animal': 'nature',
    'building': 'infrastructure', 'destroyed building': 'infrastructure',
    'wall': 'infrastructure', 'road': 'infrastructure',
    'pavement': 'infrastructure', 'dirt road': 'infrastructure',
    'window': 'infrastructure', 'door': 'infrastructure',
    'stairs': 'infrastructure', 'pole': 'infrastructure',
    'tower': 'infrastructure', 'silo': 'infrastructure',
    'debris': 'obstacle', 'fence': 'obstacle', 'barrier': 'obstacle',
    'cone': 'obstacle', 'hole in the ground': 'obstacle',
    'mud': 'obstacle', 'water': 'obstacle',
    'fire hose': 'equipment', 'fire hydrant': 'equipment',
    'extinguisher': 'equipment', 'helmet': 'equipment',
    'scba': 'equipment', 'boot': 'equipment', 'mask': 'equipment',
    'glove': 'equipment', 'protective glasses': 'equipment',
    'ladder': 'equipment', 'ax': 'equipment', 'shovel': 'equipment',
    'chainsaw': 'equipment', 'bag': 'equipment', 'barrel': 'equipment',
    'furniture': 'equipment', 'tank': 'equipment', 'crane': 'equipment',
    'excavator': 'equipment', 'lifesaver': 'equipment',
}

_CLASS_SYMBOLS = {
    'first responder': 'pitch', 'citizen': 'pitch',
    'military personnel': 'pitch',
    'civilian vehicle': 'car', 'destroyed vehicle': 'car',
    'ambulance': 'hospital', 'police vehicle': 'police',
    'fire truck': 'fire-station', 'army vehicle': 'car',
    'boat': 'harbor', 'bicycle': 'bicycle', 'aerial vehicle': 'airfield',
    'flame': 'fire-station', 'smoke': 'fire-station',
    'building': 'building', 'destroyed building': 'building',
    'water': 'water', 'animal': 'dog-park',
}


# ── Geometry mapping (single source of truth: classes.txt) ───────────
#
# Each class id maps to the "GeoJSON Type" agreed in classes.txt, which
# decides how many boundary pixel points we emit per detection for
# downstream 3DGS raycasting:
#   Point   → 1 point  (centroid)
#   Line    → 4 points (min-area-rect corners — keeps the structure's height,
#                       not just its ground line)
#   Polygon → 4 points (min-area-rect corners), or a simplified 8–12 pt
#             contour for the large/irregular area classes below.
# Transcribed from classes.txt; TestClassesTxtConsistency in the unit suite
# asserts this table matches the file, so drift fails CI-style.
_GEOMETRY_TYPE_BY_ID = {
    0: 'Polygon',   # Water
    1: 'Line',      # Fence
    2: 'Polygon',   # Green tree
    3: 'Point',     # Helmet
    4: 'Polygon',   # Flame
    5: 'Polygon',   # Smoke
    6: 'Point',     # First responder
    7: 'Point',     # Destroyed vehicle
    8: 'Point',     # Fire hose
    9: 'Point',     # SCBA
    10: 'Point',    # Boot
    11: 'Polygon',  # Green plant
    12: 'Point',    # Mask
    13: 'Point',    # Window
    14: 'Polygon',  # Building
    15: 'Polygon',  # Destroyed building
    16: 'Polygon',  # Debris
    17: 'Polygon',  # Ladder
    18: 'Polygon',  # Dirt road
    19: 'Polygon',  # Dry tree
    20: 'Line',     # Wall
    21: 'Point',    # Civilian vehicle
    22: 'Polygon',  # Road
    23: 'Point',    # Citizen
    24: 'Polygon',  # Green grass
    25: 'Point',    # Pole
    26: 'Polygon',  # Boat
    27: 'Polygon',  # Pavement
    28: 'Polygon',  # Dry grass
    29: 'Point',    # Animal
    30: 'Polygon',  # Excavator
    31: 'Point',    # Door
    32: 'Polygon',  # Mud
    33: 'Polygon',  # Barrier
    34: 'Point',    # Hole in the ground
    35: 'Point',    # Bag
    36: 'Polygon',  # Burnt tree
    37: 'Point',    # Ambulance
    38: 'Point',    # Fire truck
    39: 'Point',    # Cone
    40: 'Polygon',  # Bicycle
    41: 'Polygon',  # Tower
    42: 'Polygon',  # Silo
    43: 'Point',    # Military personnel
    44: 'Polygon',  # Burnt grass
    45: 'Point',    # Ax
    46: 'Point',    # Glove
    47: 'Polygon',  # Crane
    48: 'Point',    # Stairs
    49: 'Polygon',  # Dry plant
    50: 'Polygon',  # Furniture
    51: 'Polygon',  # Tank
    52: 'Point',    # Protective glasses
    53: 'Polygon',  # Barrel
    54: 'Point',    # Shovel
    55: 'Point',    # Fire hydrant
    56: 'Point',    # Police vehicle
    57: 'Polygon',  # Burnt plant
    58: 'Point',    # Army vehicle
    59: 'Point',    # Chainsaw
    60: 'Point',    # aerial vehicle
    61: 'Point',    # Lifesaver
    62: 'Point',    # Extinguisher
}

# Large / irregular Polygon classes whose shape isn't captured by 4 corners —
# these get a simplified contour (~8–12 points) instead. Editable.
_LARGE_AREA_CLASS_IDS = {
    0,   # Water
    16,  # Debris
    18,  # Dirt road
    22,  # Road
    24,  # Green grass
    27,  # Pavement
    28,  # Dry grass
    32,  # Mud
    44,  # Burnt grass
}


def _simplify_contour(pts: np.ndarray, lo: int = 8, hi: int = 12,
                      max_iter: int = 16):
    """Reduce a contour to between ``lo`` and ``hi`` points via approxPolyDP.

    Bisection-searches the Douglas–Peucker epsilon (as a fraction of the
    contour perimeter): a larger epsilon yields fewer points. Returns an
    ``(N, 2)`` array with ``lo <= N <= hi`` when possible (capped at ``hi``),
    or ``None`` if the contour is degenerate.
    """
    cnt = pts.reshape(-1, 1, 2).astype(np.float32)
    peri = cv2.arcLength(cnt, True)
    if peri <= 0:
        return None

    lo_eps, hi_eps = 0.0005, 0.2
    best = None
    for _ in range(max_iter):
        eps = (lo_eps + hi_eps) / 2.0
        approx = cv2.approxPolyDP(cnt, eps * peri, True).reshape(-1, 2)
        n = len(approx)
        if lo <= n <= hi:
            return approx
        if n > hi:
            lo_eps = eps          # too many points → need a larger epsilon
        else:                     # n < lo → need a smaller epsilon
            hi_eps = eps
            best = approx         # densest under-shoot seen so far
    # No exact landing: cap an over-shoot at hi, else return densest under-shoot.
    approx = cv2.approxPolyDP(cnt, lo_eps * peri, True).reshape(-1, 2)
    if len(approx) > hi:
        return approx[:hi]
    if len(approx) >= lo:
        return approx
    return best


def _boundary_points(contour_xy, geometry_type: str, class_id: int,
                     bbox=None) -> list:
    """Reduce a detection to a few boundary pixel points for raycasting.

    Parameters
    ----------
    contour_xy : array-like or None
        ``(N, 2)`` mask contour points in original-frame pixels (e.g.
        ``masks.xy[i]``), or None when the model produced no mask.
    geometry_type : str
        ``'Point' | 'Line' | 'Polygon'`` from ``_GEOMETRY_TYPE_BY_ID``.
    class_id : int
        Selects the large-area contour branch for Polygon classes.
    bbox : sequence or None
        ``[x1, y1, x2, y2]`` fallback when no usable contour is available.

    Returns a list of ``[x, y]`` points rounded to 1 decimal:
    Point → 1, Line → 4 (min-area-rect corners), Polygon → 4 corners or
    8–12 simplified-contour points for large-area classes.
    """
    def _round(arr):
        return [[round(float(x), 1), round(float(y), 1)] for x, y in arr]

    pts = None
    if contour_xy is not None:
        arr = np.asarray(contour_xy, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 3:
            pts = arr

    if pts is None:
        if bbox is None:
            return []
        x1, y1, x2, y2 = bbox
        pts = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32,
        )

    if geometry_type == 'Point':
        return _round([[float(pts[:, 0].mean()), float(pts[:, 1].mean())]])

    if geometry_type == 'Polygon' and class_id in _LARGE_AREA_CLASS_IDS:
        simplified = _simplify_contour(pts)
        if simplified is not None and len(simplified) >= 3:
            return _round(simplified)
        # fall through to the 4-corner min-area-rect on failure

    # Line and compact Polygon → 4 corners of the minimum-area rectangle.
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    return _round(box)


def _detection_to_pixel_feature(det: dict, frame_index: int,
                                timestamp_s: float,
                                frame_meta=None) -> dict:
    """Convert one tracked detection into a pixel-space GeoJSON Feature.

    The geometry is a **bag of independent sample points** for raycasting,
    not an outline: single-point classes (``classes.txt`` "Point") become a
    GeoJSON ``Point``; everything else (``Line``/``Polygon`` classes, which
    just get more sample points — 4, or 8-12 for large-area classes) becomes
    a GeoJSON ``MultiPoint``. There is no implied ordering, connectivity, or
    closed boundary between the points — each one is meant to be raycast
    independently against the 3DGS reconstruction. ``classes_txt_geometry``
    keeps the original classes.txt classification for traceability only; it
    does not describe the shape of ``geometry`` here.

    Coordinates are ``[x, y]`` pixels in the original frame resolution.

    When ``frame_meta`` (an ``SrtFrameMeta`` from the sidecar SRT) is
    given, the *drone/camera* telemetry for this frame is attached as
    ``drone_*`` / ``gimbal_*`` properties — these describe where the
    camera was, not where the detection is, and are meant as input for
    the downstream raycast.

    Properties otherwise mirror the UGV geojson_bridge feature schema so
    both platforms render consistently on the TELESTO map.
    """
    label = det['label']
    points = det['points']
    geom_type = det['geometry_type']

    if geom_type == 'Point':
        geometry = {"type": "Point", "coordinates": points[0]}
    else:
        geometry = {"type": "MultiPoint", "coordinates": points}

    track_id = str(det['id'])
    feature = {
        "type": "Feature",
        "id": track_id,
        "geometry": geometry,
        "properties": {
            "class": label,
            "class_id": det['class_id'],
            "id": track_id,
            "confidence": det['confidence'],
            "frame_index": frame_index,
            "timestamp_s": timestamp_s,
            "category": _CLASS_CATEGORIES.get(label, 'unknown'),
            "detection_type": "seg",
            "source": "uav",
            "coordinate_space": "pixel",
            "classes_txt_geometry": geom_type,
            "marker-color": _CLASS_COLORS.get(label, '#808080'),
            "marker-size": "medium",
            "marker-symbol": _CLASS_SYMBOLS.get(label, 'marker'),
        },
    }

    if frame_meta is not None:
        props = feature["properties"]
        telemetry = {
            "drone_latitude": frame_meta.lat,
            "drone_longitude": frame_meta.lon,
            "drone_altitude_m": frame_meta.abs_alt,
            "drone_rel_altitude_m": frame_meta.rel_alt,
            "gimbal_yaw": frame_meta.gimbal_yaw,
            "gimbal_pitch": frame_meta.gimbal_pitch,
            "gimbal_roll": frame_meta.gimbal_roll,
        }
        props.update({k: v for k, v in telemetry.items() if v is not None})

    return feature


def _build_feature_collection(features: list, metadata: dict) -> dict:
    """Assemble the output FeatureCollection.

    ``metadata`` is attached as a top-level foreign member (permitted by
    RFC 7946 §6.1) so per-video context travels with the features without
    colliding with per-feature ``properties``.
    """
    return {
        "type": "FeatureCollection",
        "metadata": metadata,
        "features": features,
    }


def _write_geojson(collection: dict, video_path: str,
                   output_dir: Optional[str]) -> Path:
    """Write the collection as ``<video-stem>_detections.geojson``.

    ``output_dir=None`` writes next to the input video.
    """
    out = Path(output_dir) if output_dir else Path(video_path).parent
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{Path(video_path).stem}_detections.geojson'
    path.write_text(json.dumps(collection, indent=2))
    return path


def _parse_class_filter(spec: Optional[str]) -> Optional[set]:
    """Parse a comma-separated allowlist of class ids and/or labels."""
    if not spec:
        return None
    class_filter = set()
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        class_filter.add(int(tok) if tok.isdigit() else tok.lower())
    return class_filter or None


def _open_video(video_path: str):
    """Open a video and return ``(cap, fps, width, height, total_frames)``.

    Returns ``None`` when the file cannot be opened. Falls back to 30 fps
    when the container reports no frame rate (timestamps stay usable).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f'Failed to open video: {video_path}')
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        log.warning('Video FPS unavailable; assuming 30.0 for timestamps.')
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, width, height, total


class UAVPipeline:
    """UAV perception pipeline: video in, pixel-space GeoJSON out."""

    def __init__(
        self,
        model_path: str = 'best.pt',
        confidence: float = 0.5,
        yolo_imgsz: int = 1280,
    ):
        self.conf_thresh = confidence
        self.yolo_imgsz = yolo_imgsz

        if not _HAS_YOLO:
            raise RuntimeError(
                'ultralytics is required for the UAV pipeline — '
                'pip install ultralytics'
            )
        log.info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        log.info('Model loaded.')

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        stride: int = 5,
        tracker: str = 'bytetrack.yaml',
        sample_seconds: float = 1.0,
        classes: Optional[set] = None,
        srt_path: Optional[str] = None,
    ) -> Optional[dict]:
        """Run detection + tracking over a video, emit pixel-space GeoJSON.

        For every ``stride``-th frame, runs the YOLO-seg model with
        persistent track IDs. Each detection is reduced to a few independent
        pixel sample points (count decided by its ``classes.txt`` geometry
        type — not an outline), in the *original* frame resolution
        (origin = top-left), and becomes one GeoJSON Feature.

        To keep the output small, emission is **deduplicated per track**: a
        given track id is written at most once every ``sample_seconds``.
        Tracking still runs on every ``stride`` frame so ids stay stable;
        only the writing is throttled.

        Parameters
        ----------
        video_path : str
            Path to the input video file.
        output_path : str, optional
            Directory to write ``<video-stem>_detections.geojson`` into
            (None = don't write, just return the collection).
        stride : int
            Process every Nth frame (1 = every frame).
        tracker : str
            Ultralytics tracker config (``bytetrack.yaml`` or
            ``botsort.yaml``).
        sample_seconds : float
            Minimum seconds between successive emissions of the same track
            id (0 = emit on every processed frame).
        classes : set of int/str, optional
            Allowlist of class ids and/or labels to keep (None = all).
        srt_path : str, optional
            DJI SRT telemetry sidecar. When None, ``<video-stem>.srt`` /
            ``.SRT`` next to the video is auto-detected. Per-frame drone
            position/gimbal telemetry is attached to each feature.

        Returns the FeatureCollection dict, or None on failure.
        """
        video_path = str(video_path)
        if stride < 1:
            stride = 1
        if sample_seconds < 0:
            sample_seconds = 0.0

        srt_file = Path(srt_path) if srt_path else find_sidecar_srt(video_path)
        srt_index = None
        if srt_file is not None:
            try:
                srt_index = SrtIndex.from_file(srt_file)
                log.info(
                    f'SRT telemetry: {srt_file} ({len(srt_index)} frames)'
                )
                if len(srt_index) == 0:
                    log.warning('SRT contained no parsable telemetry.')
                    srt_index = None
            except OSError as e:
                log.warning(f'Could not read SRT {srt_file}: {e}')
                srt_index = None
        else:
            log.info('No SRT telemetry sidecar found — features will carry '
                     'pixel coordinates only.')

        opened = _open_video(video_path)
        if opened is None:
            return None
        cap, fps, width, height, total = opened

        log.info(
            f'Video: {video_path} — {width}x{height} @ {fps:.3f} fps, '
            f'{total} frames. Processing every {stride} frame(s), '
            f'tracker={tracker}.'
        )

        # Fresh tracker state for this video.
        try:
            self.model.predictor = None
        except Exception:
            pass

        features = []
        frame_idx = 0
        processed = 0
        last_emit = {}          # track_id -> timestamp_s of last written record
        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                detections = self._track_frame(frame, tracker)
                processed += 1

                timestamp_s = round(frame_idx / fps, 3)
                frame_meta = (
                    srt_index.at(frame_idx, timestamp_s)
                    if srt_index is not None else None
                )
                for det in detections:
                    if classes is not None and not (
                        det['class_id'] in classes or det['label'] in classes
                    ):
                        continue
                    tid = det['id']
                    last = last_emit.get(tid)
                    # New track, untracked (-1), or enough time elapsed → emit.
                    if (sample_seconds <= 0 or tid < 0 or last is None
                            or timestamp_s - last >= sample_seconds):
                        features.append(_detection_to_pixel_feature(
                            det, frame_idx, timestamp_s,
                            frame_meta=frame_meta,
                        ))
                        if tid >= 0:
                            last_emit[tid] = timestamp_s

                if processed % 100 == 0:
                    elapsed = time.time() - t_start
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    log.info(
                        f'  frame {frame_idx}/{total} '
                        f'({processed} processed, {len(features)} detections, '
                        f'{rate:.1f} proc-fps)'
                    )

            frame_idx += 1

        cap.release()

        metadata = {
            'video': os.path.basename(video_path),
            'model': getattr(self.model, 'ckpt_path', None) or 'unknown',
            'fps': round(fps, 3),
            'frame_width': width,
            'frame_height': height,
            'total_frames': total,
            'stride': stride,
            'tracker': tracker,
            'confidence': self.conf_thresh,
            'sample_seconds': sample_seconds,
            'classes': (
                sorted(str(c) for c in classes) if classes is not None
                else 'all'
            ),
            'geometry_source': 'classes.txt',
            'srt_file': srt_file.name if srt_index is not None else None,
            'srt_frames': len(srt_index) if srt_index is not None else 0,
            'processed_frames': processed,
            'total_detections': len(features),
            'coordinate_space': (
                'pixels in original frame resolution, origin top-left '
                '(x = column, y = row)'
            ),
            'generated_at': datetime.now(timezone.utc).isoformat(
                timespec='seconds',
            ),
        }
        collection = _build_feature_collection(features, metadata)

        elapsed = time.time() - t_start
        log.info(
            f'Done: {processed} frames processed in {elapsed:.1f}s, '
            f'{len(features)} detections emitted.'
        )

        if output_path:
            path = _write_geojson(collection, video_path, output_path)
            log.info(f'Saved GeoJSON: {path}')

        return collection

    def _track_frame(self, frame: np.ndarray, tracker: str) -> list:
        """Run tracked YOLO-seg on one frame, return pixel-space detections.

        Every detection carries the persistent track ``id``, class
        ``label``/``class_id``, ``confidence``, ``bbox`` ([x1,y1,x2,y2]),
        a ``geometry_type`` (from ``classes.txt`` via
        ``_GEOMETRY_TYPE_BY_ID``) and a short ``points`` list — Point→1,
        Line→4, Polygon→4 (or 8–12 for large-area classes) — ready for
        downstream 3DGS raycasting.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_thresh,
            imgsz=self.yolo_imgsz,
            tracker=tracker,
            verbose=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            masks = r.masks
            ids = boxes.id

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                if cls_id not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                bbox = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                track_id = int(ids[i]) if ids is not None else -1

                # Mask contour in original-frame pixel coordinates (if any).
                contour = None
                if masks is not None and masks.xy is not None and i < len(masks.xy):
                    pts = masks.xy[i]
                    if pts is not None and len(pts) >= 3:
                        contour = pts

                geom = _GEOMETRY_TYPE_BY_ID.get(cls_id, 'Polygon')
                detections.append({
                    'id': track_id,
                    'label': TARGET_CLASSES[cls_id],
                    'class_id': cls_id,
                    'confidence': round(float(boxes.conf[i]), 4),
                    'bbox': bbox,
                    'geometry_type': geom,
                    'points': _boundary_points(contour, geom, cls_id, bbox=bbox),
                })

        return detections


def _upload_to_telesto(collection: dict, telesto_base_url: str = '') -> int:
    """Upload a FeatureCollection to TELESTO via telesto_client. Returns
    the number of features uploaded. Raises SystemExit if the package
    (repo-local, not pip-installed) isn't on PYTHONPATH."""
    try:
        from triffid_telesto.telesto_client import TelestoClient
    except ImportError as e:
        raise SystemExit(
            f'--post-telesto needs the triffid_telesto package on '
            f'PYTHONPATH (src/ of this repo): {e}'
        )
    client = (
        TelestoClient(base_url=telesto_base_url)
        if telesto_base_url else TelestoClient()
    )
    results = client.upload_collection(collection)
    log.info(f'Uploaded {len(results)} features to TELESTO.')
    return len(results)


def _poll_api_video_once(pipeline: UAVPipeline, client: FuturisedClient,
                         camera: str, output_path: Optional[str],
                         post_telesto: bool = False,
                         telesto_base_url: str = '',
                         **video_kwargs) -> int:
    """One poll+download+process pass. Returns the number of videos processed.

    ``.SRT`` sidecars are polled alongside the videos: they download into
    the same directory, where ``process_video``'s ``find_sidecar_srt``
    pairs them with their MP4 by matching basename. PoC limitation: if an
    SRT upload appears only *after* its MP4 was already processed, the
    video is not reprocessed.

    Split out from the poll loop so it's unit-testable without mocking
    ``time.sleep``/``KeyboardInterrupt``.
    """
    new_files = client.poll_new_images(
        camera_filter=camera, extensions={'.MP4', '.MOV', '.SRT'},
    )
    new_videos = [p for p in new_files if p.suffix.upper() != '.SRT']
    processed = 0
    for video_path in new_videos:
        log.info(f'Processing downloaded video: {video_path}')
        collection = pipeline.process_video(
            str(video_path), output_path=output_path, **video_kwargs,
        )
        processed += 1
        if collection is not None and post_telesto:
            _upload_to_telesto(collection, telesto_base_url)
    return processed


def _poll_api_video(pipeline: UAVPipeline, client: FuturisedClient,
                    poll_interval: float, camera: str,
                    output_path: Optional[str], **video_kwargs):
    """Poll FUTURISED for new video uploads and process each one (PoC).

    FUTURISED's Media Files API serves *uploaded* files, not a continuous
    stream — this polls for newly-appeared ``.MP4``/``.MOV`` files the same
    way the old still-image poll mode watched for JPEGs, and runs the full
    video pipeline on each as it's found.
    """
    log.info(
        f'Polling FUTURISED for new video uploads every {poll_interval}s '
        f'(camera={camera or "all"}) — PoC.'
    )
    while True:
        try:
            _poll_api_video_once(pipeline, client, camera, output_path,
                                 **video_kwargs)
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            break
    log.info('API polling stopped.')


def main():
    parser = argparse.ArgumentParser(
        description='TRIFFID UAV Perception Pipeline — video in, '
                    'pixel-space GeoJSON out (for 3DGS raycasting)',
    )
    parser.add_argument('video', type=str, nargs='?', default=None,
                        help='Input video file (omit when using --poll-api)')

    parser.add_argument('--model', type=str, default='best.pt',
                        help='YOLO model path (default: best.pt)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='YOLO input size (default: 1280)')
    parser.add_argument('--stride', type=int, default=5,
                        help='Process every Nth video frame (default: 5)')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml',
                        help='Ultralytics tracker config: bytetrack.yaml '
                             'or botsort.yaml (default: bytetrack.yaml)')
    parser.add_argument('--sample-seconds', type=float, default=1.0,
                        help='Min seconds between emissions of the same '
                             'track id; 0 = every processed frame '
                             '(default: 1.0)')
    parser.add_argument('--classes', type=str, default=None,
                        help='Comma-separated class ids and/or labels to '
                             'keep (default: all classes)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for the GeoJSON '
                             '(default: next to the input video)')
    parser.add_argument('--srt', type=str, default=None,
                        help='DJI SRT telemetry sidecar for the input video '
                             '(default: auto-detect <video-stem>.srt next '
                             'to it). Attaches per-frame drone position/'
                             'gimbal properties to each feature.')
    parser.add_argument('--post-telesto', action='store_true',
                        help='Upload the resulting FeatureCollection to the '
                             'TELESTO backend (default: off). Only meaningful '
                             'once coordinates are geographic — the pixel-'
                             'space output must first go through 3DGS '
                             'raycasting.')
    parser.add_argument('--telesto-base-url', type=str,
                        default=os.environ.get('TELESTO_BASE_URL', ''),
                        help='TELESTO API base URL (or env TELESTO_BASE_URL; '
                             'default: the client\'s built-in production URL)')
    parser.add_argument('-v', '--verbose', action='store_true')

    api_group = parser.add_argument_group(
        'FUTURISED API poll mode (PoC) — poll for new video uploads '
        'instead of processing a local file',
    )
    api_group.add_argument(
        '--poll-api', action='store_true',
        help='Poll the FUTURISED Media Files API for new video uploads '
             '(.mp4/.mov) and process each one as it appears. Proof of '
             'concept: FUTURISED serves uploaded files, not a continuous '
             'live stream, and video support there is unconfirmed.',
    )
    api_group.add_argument(
        '--api-media-key', type=str,
        default=os.environ.get('FUTURISED_MEDIA_API_KEY', ''),
        help='Media Files API key (or env FUTURISED_MEDIA_API_KEY)',
    )
    api_group.add_argument(
        '--api-org-id', type=str,
        default=os.environ.get(
            'FUTURISED_ORG_ID', '66f9f3ae-cd33-4313-b474-ae24e923a185',
        ),
        help='Organisation UUID for the media API (or env FUTURISED_ORG_ID)',
    )
    api_group.add_argument(
        '--api-camera', type=str, default='Wide',
        help='Camera filter: Wide, Zoom, Thermal, or empty for all '
             '(default: Wide)',
    )
    api_group.add_argument(
        '--api-poll-interval', type=float, default=30.0,
        help='Seconds between polls (default: 30)',
    )
    api_group.add_argument(
        '--api-download-dir', type=str, default='./uav_videos',
        help='Local directory for downloaded videos (default: ./uav_videos)',
    )

    args = parser.parse_args()

    if args.poll_api and args.video:
        parser.error('pass either a video file or --poll-api, not both')
    if not args.poll_api and not args.video:
        parser.error('a video file is required unless --poll-api is given')
    if args.poll_api and not args.api_media_key:
        parser.error(
            '--poll-api requires --api-media-key or '
            'env FUTURISED_MEDIA_API_KEY'
        )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    pipeline = UAVPipeline(
        model_path=args.model,
        confidence=args.confidence,
        yolo_imgsz=args.imgsz,
    )
    video_kwargs = dict(
        stride=args.stride,
        tracker=args.tracker,
        sample_seconds=args.sample_seconds,
        classes=_parse_class_filter(args.classes),
    )

    if args.poll_api:
        client = FuturisedClient(
            media_api_key=args.api_media_key,
            org_id=args.api_org_id,
            download_dir=args.api_download_dir,
        )
        try:
            _poll_api_video(
                pipeline, client,
                poll_interval=args.api_poll_interval,
                camera=args.api_camera,
                output_path=args.output,
                post_telesto=args.post_telesto,
                telesto_base_url=args.telesto_base_url,
                **video_kwargs,
            )
        except KeyboardInterrupt:
            pass
        return

    try:
        collection = pipeline.process_video(
            args.video,
            output_path=args.output or str(Path(args.video).parent),
            srt_path=args.srt,
            **video_kwargs,
        )
    except KeyboardInterrupt:
        return

    if collection is None:
        raise SystemExit(1)

    print(json.dumps(collection['metadata'], indent=2))

    if args.post_telesto:
        _upload_to_telesto(collection, args.telesto_base_url)


if __name__ == '__main__':
    main()
