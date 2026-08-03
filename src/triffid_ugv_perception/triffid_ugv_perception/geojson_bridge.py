"""
TRIFFID GeoJSON Bridge

Subscribes to UGV detection topic (Detection3DArray in b2/base_link)
and converts them to RFC-7946 GeoJSON, then:
  1. Publishes as a ROS2 String topic (for debugging / other nodes)
  2. Periodically (every ``publish_period_s``) flushes the accumulated,
     deduplicated feature set — independently controllable:
     ``save_to_disk`` (default true) writes an atomic GeoJSON snapshot to
     ``output_dir``/ugv_detections.geojson; ``publish_to_api`` (default
     false) PUTs one feature per request to the TRIFFID mapping API.
     Both may be enabled at once.

Coordinate handling:
  - Detections arrive in ``b2/base_link`` (X=forward, Y=left, Z=up).
  - The robot's current GPS position is tracked from ``/fix``
    (median-filtered over a sliding window to reduce noise).
  - The robot's heading (yaw) is obtained from ``/dog_odom``
    orientation quaternion (Go2 state estimator, magnetometer-fused).
  - Body-frame detection offsets are rotated by the robot's yaw into
    East-North-Up (ENU) and added to the current GPS position.
  - When no GPS is available, raw local (x, y, z) are emitted and a
    ``"local_frame": true`` property is added.
  - 2D coordinates are emitted: [lon, lat] (RFC 7946 §3.1.1).
   - WGS-84 ellipsoidal altitude is stored in ``altitude_m``.

API endpoint (when enabled): https://crispres.com/wp-json/map-manager/v1/features
"""

import collections
import json
import math
import os
import queue
import threading
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from vision_msgs.msg import Detection3DArray
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from std_msgs.msg import String

# Earth radius (WGS-84 semi-major axis) in metres
_R_EARTH = 6378137.0

# GPS sliding-window size for median filter
_GPS_WINDOW = 7

# Minimum polygon extent (metres) — used when one bbox dimension is 0 so that a visible polygon is still emitted instead of a point 
_MIN_EXTENT = 0.3

# Per-class geometry types from the TRIFFID 63-class disaster-response
# ontology.  See geojson_geometries.txt for the authoritative mapping.
#
#   Point      – small / mobile objects (persons, equipment, vehicles)
#   LineString – linear structures (fences, walls)
#   Polygon    – everything else (areas, buildings, vegetation, …)

_POINT_CLASSES = frozenset([
    'helmet', 'first responder', 'destroyed vehicle', 'fire hose',
    'scba', 'boot', 'mask', 'window', 'citizen', 'pole', 'animal',
    'door', 'civilian vehicle', 'hole in the ground', 'bag',
    'ambulance', 'fire truck', 'cone', 'military personnel', 'ax',
    'glove', 'stairs', 'protective glasses', 'shovel', 'fire hydrant',
    'police vehicle', 'army vehicle', 'chainsaw', 'aerial vehicle',
    'lifesaver', 'extinguisher',
])

_LINE_CLASSES = frozenset([
    'fence', 'wall',
])


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return the Haversine distance in metres between two GPS points."""
    import math as _math
    r = 6378137.0
    d_lat = _math.radians(lat2 - lat1)
    d_lon = _math.radians(lon2 - lon1)
    a = (_math.sin(d_lat / 2) ** 2
         + _math.cos(_math.radians(lat1)) * _math.cos(_math.radians(lat2))
         * _math.sin(d_lon / 2) ** 2)
    return r * 2.0 * _math.asin(_math.sqrt(a))


def _feature_centroid(feature: dict):
    """Return (lon, lat) centroid of a GeoJSON feature, or None."""
    geom = feature.get('geometry') or {}
    geom_type = geom.get('type')
    coords = geom.get('coordinates')
    if not coords:
        return None
    if geom_type == 'Point':
        return (coords[0], coords[1])
    if geom_type == 'LineString':
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return (sum(lons) / len(lons), sum(lats) / len(lats))
    if geom_type == 'Polygon' and coords:
        ring = coords[0]
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        return (sum(lons) / len(lons), sum(lats) / len(lats))
    return None


def _deduplicate_features(features: list, radius_m: float) -> list:
    """Remove duplicate features: same class + overlapping location.

    Within each class group, if two features have centroids within
    *radius_m* metres of each other, only the higher-confidence one is
    kept.  Features with no GPS coordinates (local_frame) are passed
    through unchanged.
    """
    if len(features) <= 1:
        return features

    # Sort descending by confidence so greedy keep is always the best
    def _conf(f):
        return f.get('properties', {}).get('confidence', 0.0)

    sorted_feats = sorted(features, key=_conf, reverse=True)
    kept = []
    for candidate in sorted_feats:
        props = candidate.get('properties', {})
        if props.get('local_frame', False):
            kept.append(candidate)
            continue
        centroid = _feature_centroid(candidate)
        if centroid is None:
            kept.append(candidate)
            continue
        cls = props.get('class', '')
        suppressed = False
        for existing in kept:
            if existing.get('properties', {}).get('class', '') != cls:
                continue
            existing_centroid = _feature_centroid(existing)
            if existing_centroid is None:
                continue
            dist = _haversine_m(
                centroid[0], centroid[1],
                existing_centroid[0], existing_centroid[1],
            )
            if dist < radius_m:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
    return kept


def _merge_features(accumulated: list, new: list, radius_m: float) -> list:
    """Merge new features into the accumulated set with spatial dedup.

    Growth is bounded: GPS features collapse to the single highest-
    confidence one within *radius_m* (per class), and local-frame features
    — which bypass the spatial dedup because they have no GPS coordinates —
    collapse to the newest one per (class, track id).
    """
    latest_local = {}
    gps_feats = []
    for feat in accumulated + new:
        props = feat.get('properties', {})
        if props.get('local_frame', False):
            key = (props.get('class', ''), props.get('id', ''))
            latest_local[key] = feat        # later (newer) entries win
        else:
            gps_feats.append(feat)
    return _deduplicate_features(gps_feats, radius_m) + list(latest_local.values())


def _atomic_write_geojson(path, collection: dict) -> None:
    """Write a GeoJSON snapshot atomically (tmp file + os.replace).

    Readers never see a half-written file, even if they poll mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(collection, indent=2))
    os.replace(tmp, path)


class GeoJSONBridge(Node):
    """Convert ROS2 detections to GeoJSON and push to TRIFFID API."""

    def __init__(self):
        super().__init__('geojson_bridge')

        # Parameters 
        self.declare_parameter('api_url',
                               'https://crispres.com/wp-json/map-manager/v1/features')
        self.declare_parameter('publish_to_api', False)
        self.declare_parameter('save_to_disk', True)
        self.declare_parameter('gps_origin_lat', 0.0)
        self.declare_parameter('gps_origin_lon', 0.0)
        self.declare_parameter('gps_origin_alt', 0.0)
        self.declare_parameter('dedup_radius_m', 3.0)
        self.declare_parameter('publish_period_s', 5.0)
        self.declare_parameter('output_dir', '/ws/samples')

        self.api_url = self.get_parameter('api_url').value
        self.publish_to_api = self.get_parameter('publish_to_api').value
        self.save_to_disk = self.get_parameter('save_to_disk').value
        self.publish_period_s = float(self.get_parameter('publish_period_s').value)
        self.output_dir = self.get_parameter('output_dir').value

        # GPS state
        # Sliding window for median filtering of noisy GPS fixes
        self._gps_lat_buf = collections.deque(maxlen=_GPS_WINDOW)
        self._gps_lon_buf = collections.deque(maxlen=_GPS_WINDOW)
        self._gps_alt_buf = collections.deque(maxlen=_GPS_WINDOW)

        # Current filtered robot GPS position (updated every /fix)
        self.robot_lat = 0.0
        self.robot_lon = 0.0
        self.robot_alt = 0.0
        self.gps_valid = False

        # Seed from parameters if provided
        param_lat = self.get_parameter('gps_origin_lat').value
        param_lon = self.get_parameter('gps_origin_lon').value
        param_alt = self.get_parameter('gps_origin_alt').value
        if param_lat != 0.0 and param_lon != 0.0:
            self.robot_lat = param_lat
            self.robot_lon = param_lon
            self.robot_alt = param_alt
            self.gps_valid = True

        # Heading state 
        # Yaw from /dog_odom quaternion (radians, ENU convention:
        # 0 = East, π/2 = North, counter-clockwise positive)
        self.robot_yaw = 0.0
        self.heading_valid = False

        # Subscribers 
        self.sub_ugv = self.create_subscription(
            Detection3DArray,
            '/ugv/detections/front/detections_3d',
            self.ugv_callback,
            10,
        )
        self.sub_gps = self.create_subscription(
            NavSatFix,
            '/fix',
            self.gps_callback,
            10,
        )
        self.sub_odom = self.create_subscription(
            Odometry,
            '/dog_odom',
            self.odom_callback,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )

        # Publishers
        self.pub_geojson = self.create_publisher(
            String,
            '/ugv/detections/front/geojson',
            10,
        )

        outputs = []
        if self.save_to_disk:
            outputs.append(f'snapshot to {self.output_dir}/ugv_detections.geojson')
        if self.publish_to_api:
            outputs.append(f'PUT to {self.api_url}')
        self.get_logger().info('GeoJSON Bridge started.')
        self.get_logger().info(
            f'  Every {self.publish_period_s}s: '
            + (' AND '.join(outputs) if outputs
               else 'nothing (both save_to_disk and publish_to_api are off)')
        )
        if self.gps_valid:
            self.get_logger().info(
                f'  GPS seeded from params: ({self.robot_lat}, {self.robot_lon})'
            )
        else:
            self.get_logger().warn(
                'GPS not yet available — emitting local-frame coordinates. '
                'Waiting for /fix topic.'
            )

        # Spatial deduplication radius (metres)
        self._dedup_radius_m = float(self.get_parameter('dedup_radius_m').value)

        # ── Periodic flush (disk snapshot / API upload) ─────────────
        # Features accumulate across frames (spatially deduplicated) and
        # are flushed every publish_period_s. API uploads run on a single
        # long-lived worker so a slow backend can never pile up threads;
        # the depth-1 queue drops the older snapshot when the worker lags
        # (each snapshot supersedes the previous one anyway).
        self._accumulated = []
        self._uploaded_local_ids = set()   # local-frame features PUT once
        self._api_queue = queue.Queue(maxsize=1)
        self._api_worker = None
        if self.publish_to_api:
            self._api_worker = threading.Thread(
                target=self._api_worker_loop, daemon=True,
            )
            self._api_worker.start()
        self.create_timer(self.publish_period_s, self._periodic_flush)

    #  GPS (position tracking with median filter)
    def gps_callback(self, msg: NavSatFix):
        """Update current robot position from every valid GPS fix.

        Uses a sliding-window median filter to suppress GPS noise
        (typical consumer GPS jitter is ±2-5 m).
        """
        if msg.latitude == 0.0 and msg.longitude == 0.0:
            return

        self._gps_lat_buf.append(msg.latitude)
        self._gps_lon_buf.append(msg.longitude)
        self._gps_alt_buf.append(msg.altitude)

        # Median of the sliding window
        self.robot_lat = float(sorted(self._gps_lat_buf)[len(self._gps_lat_buf) // 2])
        self.robot_lon = float(sorted(self._gps_lon_buf)[len(self._gps_lon_buf) // 2])
        self.robot_alt = float(sorted(self._gps_alt_buf)[len(self._gps_alt_buf) // 2])

        if not self.gps_valid:
            self.gps_valid = True
            self.get_logger().info(
                f'GPS acquired: ({self.robot_lat:.7f}, {self.robot_lon:.7f}, '
                f'{self.robot_alt:.1f}m)'
            )


    #  Heading (from /dog_odom orientation quaternion)
    def odom_callback(self, msg: Odometry):
        """Extract yaw from the Go2 state estimator odometry.

        The orientation quaternion from /dog_odom is expected to be in
        an ENU-aligned frame (magnetometer-fused on the Unitree Go2).
        Yaw = angle from East axis, counter-clockwise positive.
        """
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        self.robot_yaw = math.atan2(siny, cosy)

        if not self.heading_valid:
            self.heading_valid = True
            self.get_logger().info(
                f'Heading acquired: {math.degrees(self.robot_yaw):.1f}° '
                f'(ENU yaw)'
            )

    #  Coordinate conversion (body-frame → GPS)
    @staticmethod
    def _body_to_enu(x_fwd, y_left, z_up, yaw):
        """Rotate a body-frame offset into East-North-Up.

        Body frame (b2/base_link): X=forward, Y=left, Z=up.
        ENU world frame: X=East, Y=North, Z=Up.

        The robot's heading *yaw* is the angle from East (CCW positive)
        in the ENU frame.

        Robot forward direction in ENU: (cos(yaw), sin(yaw))
        Robot left direction in ENU:    (-sin(yaw), cos(yaw))

        Returns (east, north, up) in metres.
        """
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        east  = x_fwd * cos_y - y_left * sin_y
        north = x_fwd * sin_y + y_left * cos_y
        up    = z_up
        return east, north, up

    def body_to_gps(self, x_fwd, y_left, z_up=0.0):
        """Convert a detection in b2/base_link to [lon, lat, alt].

        1. Rotate body offset by robot yaw → ENU metres
        2. Convert ENU metres → degree offsets (equirectangular)
        3. Add to current robot GPS position

        Returns (lon, lat, alt) tuple.
        When GPS not available, returns raw body-frame coords.
        """
        if not self.gps_valid:
            return (x_fwd, y_left, z_up)

        yaw = self.robot_yaw if self.heading_valid else 0.0

        # Step 1: body → ENU
        east, north, up = self._body_to_enu(x_fwd, y_left, z_up, yaw)

        # Step 2: ENU metres → degree offsets
        lat_rad = math.radians(self.robot_lat)
        d_lat = north / _R_EARTH * (180.0 / math.pi)
        d_lon = east / (_R_EARTH * math.cos(lat_rad)) * (180.0 / math.pi)

        # Step 3: add to current robot position
        lat = self.robot_lat + d_lat
        lon = self.robot_lon + d_lon
        alt = self.robot_alt + up

        return (lon, lat, alt)  # GeoJSON order: [lon, lat, alt]

    # ═══════════════════════════════════════════════════════════════
    #  Detection → GeoJSON conversion
    # ═══════════════════════════════════════════════════════════════

    # Class-dependent geometry type
    @staticmethod
    def _geometry_type_for_class(class_name: str) -> str:
        """Return the GeoJSON geometry type to use for a given class.

        - Person classes → Point (small, mobile targets)
        - Linear structures (Fence) → LineString
        - Everything else → Polygon (footprint)
        """
        if class_name in _POINT_CLASSES:
            return 'Point'
        if class_name in _LINE_CLASSES:
            return 'LineString'
        return 'Polygon'

    def detections_to_geojson(self, detections, source='ugv',
                               detection_type='seg'):
        """Build a GeoJSON FeatureCollection from detection dicts.

        Geometry coordinates are 2D [lon, lat] only (RFC 7946).
        WGS-84 ellipsoidal altitude is stored in ``altitude_m``.
        Geometry type is class-dependent (Point / LineString / Polygon).
        """
        features = []
        for det in detections:
            lon, lat, alt = det['coordinates']
            sx, sy, sz = det.get('size', (0.0, 0.0, 0.0))
            pos_x, pos_y, pos_z = det.get('position', (0.0, 0.0, 0.0))

            class_name = det.get('class_name', 'unknown')
            geom_type = self._geometry_type_for_class(class_name)

            if geom_type == 'Point':
                # Point: just the centre position in [lon, lat]
                geometry = {
                    "type": "Point",
                    "coordinates": [lon, lat],
                }

            elif geom_type == 'LineString':
                # LineString: centre-line along the longer bbox axis
                half_long = max(sx, sy, _MIN_EXTENT) / 2.0
                if sx >= sy:
                    # longer along X (forward)
                    endpoints = [
                        list(self.body_to_gps(pos_x + half_long, pos_y, pos_z))[:2],
                        list(self.body_to_gps(pos_x - half_long, pos_y, pos_z))[:2],
                    ]
                else:
                    # longer along Y (left-right)
                    endpoints = [
                        list(self.body_to_gps(pos_x, pos_y + half_long, pos_z))[:2],
                        list(self.body_to_gps(pos_x, pos_y - half_long, pos_z))[:2],
                    ]
                geometry = {
                    "type": "LineString",
                    "coordinates": endpoints,
                }

            else:  # Polygon (default)
                # Build polygon from body-frame corners, each
                # individually converted to GPS via body_to_gps.
                # Use minimum extent so a zero dimension still
                # produces a visible polygon (not a degenerate line).
                half_x = max(sx, _MIN_EXTENT) / 2.0
                half_y = max(sy, _MIN_EXTENT) / 2.0
                body_corners = [
                    (pos_x + half_x, pos_y + half_y, pos_z),
                    (pos_x + half_x, pos_y - half_y, pos_z),
                    (pos_x - half_x, pos_y - half_y, pos_z),
                    (pos_x - half_x, pos_y + half_y, pos_z),
                ]
                ring = [list(self.body_to_gps(cx, cy, cz))[:2]
                        for cx, cy, cz in body_corners]
                ring.append(ring[0])  # close the ring (RFC-7946)
                geometry = {
                    "type": "Polygon",
                    "coordinates": [ring],
                }

            feature = {
                "type": "Feature",
                "id": det.get('track_id', ''),
                "geometry": geometry,
                "properties": {
                    "class": class_name,
                    "id": det.get('track_id', ''),
                    "confidence": det.get('confidence', 0.0),
                    "category": self._class_category(class_name),
                    "detection_type": detection_type,
                    "source": source,
                    "local_frame": not self.gps_valid,
                    "altitude_m": round(alt, 2),
                    "height_m": round(float(sz), 2),
                    "marker-color": self._class_color(class_name),
                    "marker-size": "medium",
                    "marker-symbol": self._class_symbol(class_name),
                }
            }
            # SimpleStyle: stroke/fill for Polygon, stroke for LineString
            if geometry['type'] == 'Polygon':
                feature["properties"]["stroke"] = feature["properties"]["marker-color"]
                feature["properties"]["stroke-width"] = 2
                feature["properties"]["stroke-opacity"] = 1.0
                feature["properties"]["fill"] = feature["properties"]["marker-color"]
                feature["properties"]["fill-opacity"] = 0.25
            elif geometry['type'] == 'LineString':
                feature["properties"]["stroke"] = feature["properties"]["marker-color"]
                feature["properties"]["stroke-width"] = 2
                feature["properties"]["stroke-opacity"] = 1.0
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features
        }

    #  Detection callback
    def ugv_callback(self, msg: Detection3DArray):
        """Process UGV 3D detections (b2/base_link) → GeoJSON."""
        detections = []
        for det in msg.detections:
            # 3D position in b2/base_link (X=fwd, Y=left, Z=up)
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y
            z = det.bbox.center.position.z

            # 3D bbox extent (metres)
            sx = det.bbox.size.x
            sy = det.bbox.size.y
            sz = det.bbox.size.z

            # Convert body-frame centre to GPS [lon, lat, alt]
            lon, lat, alt = self.body_to_gps(x, y, z)

            # Extract class and confidence
            class_name = 'unknown'
            confidence = 0.0
            if det.results:
                class_name = det.results[0].hypothesis.class_id
                confidence = det.results[0].hypothesis.score

            detections.append({
                'coordinates': (lon, lat, alt),
                'size': (sx, sy, sz),
                'position': (x, y, z),
                'class_name': class_name,
                'confidence': confidence,
                'track_id': det.id,
            })

        if not detections:
            return

        if not self.gps_valid:
            self.get_logger().warn(
                'No GPS fix — publishing with local_frame=true (body-frame coords).',
                throttle_duration_sec=5.0,
            )

        geojson = self.detections_to_geojson(
            detections, source='ugv', detection_type='seg',
        )
        geojson['features'] = _deduplicate_features(
            geojson['features'], self._dedup_radius_m
        )
        self._accumulated = _merge_features(
            self._accumulated, geojson['features'], self._dedup_radius_m,
        )
        self._publish(geojson)

    #  Publishing
    def _publish(self, geojson: dict):
        """Publish GeoJSON to the ROS2 topic (per frame)."""
        msg = String()
        msg.data = json.dumps(geojson, indent=2)
        self.pub_geojson.publish(msg)

        self.get_logger().info(
            f'Published GeoJSON with {len(geojson["features"])} features',
            throttle_duration_sec=1.0,
        )

    #  Periodic flush (timer)
    def _periodic_flush(self):
        """Flush the accumulated feature set: disk snapshot and/or API
        upload, each independently controlled by its parameter."""
        if not self._accumulated:
            return
        collection = {
            "type": "FeatureCollection",
            "features": list(self._accumulated),
        }
        if self.save_to_disk:
            path = Path(self.output_dir) / 'ugv_detections.geojson'
            try:
                _atomic_write_geojson(path, collection)
                self.get_logger().info(
                    f'Snapshot: {len(collection["features"])} features → {path}',
                    throttle_duration_sec=30.0,
                )
            except OSError as e:
                self.get_logger().warn(
                    f'GeoJSON snapshot write failed: {e}',
                    throttle_duration_sec=30.0,
                )
        if self.publish_to_api:
            # Drop the stale queued snapshot (if any) — the newest one
            # contains everything the older one did.
            try:
                self._api_queue.put_nowait(collection)
            except queue.Full:
                try:
                    self._api_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._api_queue.put_nowait(collection)
                except queue.Full:
                    pass

    def _api_worker_loop(self):
        """Single long-lived uploader; a None item is the shutdown sentinel."""
        while True:
            collection = self._api_queue.get()
            if collection is None:
                return
            self._send_to_api(collection)

    def _api_request(self, url: str, method: str, feature=None):
        """One JSON request to the TELESTO Map Manager. Returns the parsed
        response body (dict) or None on failure."""
        data = None
        headers = {}
        if feature is not None:
            data = json.dumps({
                'geometry': feature.get('geometry'),
                'properties': feature.get('properties'),
            }).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        req = Request(url, data=data, method=method, headers=headers)
        with urlopen(req, timeout=5) as resp:
            if resp.status not in (200, 201):
                self.get_logger().warn(
                    f'API {method} returned {resp.status}'
                )
                return None
            try:
                return json.loads(resp.read().decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}

    def _send_to_api(self, collection: dict):
        """Upsert the accumulated features to the TELESTO Map Manager.

        On this API, ``PUT /features`` *creates* a feature (the server
        assigns a new POI id) — so blindly re-PUTting the accumulated set
        every flush period would pile up server-side duplicates. Instead,
        each flush:

          1. GET the current remote features
          2. PUT features with no same-class remote within the dedup radius
          3. PATCH the nearby remote when ours has higher confidence
          4. skip when the nearby remote is already as good

        Local-frame features (no GPS fix yet) cannot be geo-compared
        against remote features, so they are PUT once per (class, track
        id) and remembered to avoid re-uploading every period.
        """
        features = collection.get('features', [])
        if not features:
            return

        try:
            remote = self._api_request(self.api_url, 'GET') or {}
            remote_features = remote.get('features', [])
        except (URLError, OSError) as e:
            self.get_logger().warn(
                f'API GET failed — skipping this flush (no safe dedup): {e}',
                throttle_duration_sec=10.0,
            )
            return

        # (class, centroid, confidence, poi_id) snapshot of the remote set;
        # successful PUTs are appended so one flush can't self-duplicate.
        remote_index = []
        for rf in remote_features:
            centroid = _feature_centroid(rf)
            if centroid is None:
                continue
            props = rf.get('properties', {})
            remote_index.append((
                str(props.get('class', '')).lower(),
                centroid,
                float(props.get('confidence', 0.0)),
                rf.get('id'),
            ))

        created = updated = skipped = errors = 0
        for feature in features:
            props = feature.get('properties', {})
            centroid = _feature_centroid(feature)
            try:
                if props.get('local_frame', False) or centroid is None:
                    key = (props.get('class', ''), props.get('id', ''))
                    if key in self._uploaded_local_ids:
                        skipped += 1
                        continue
                    self._api_request(self.api_url, 'PUT', feature)
                    self._uploaded_local_ids.add(key)
                    created += 1
                    continue

                cls = str(props.get('class', '')).lower()
                conf = float(props.get('confidence', 0.0))
                best = None
                best_dist = float('inf')
                for i, (r_cls, r_centroid, r_conf, r_id) in enumerate(remote_index):
                    if r_cls != cls:
                        continue
                    d = _haversine_m(centroid[0], centroid[1],
                                     r_centroid[0], r_centroid[1])
                    if d < self._dedup_radius_m and d < best_dist:
                        best_dist = d
                        best = i

                if best is None:
                    resp = self._api_request(self.api_url, 'PUT', feature)
                    poi_id = (resp or {}).get('id')
                    remote_index.append((cls, centroid, conf, poi_id))
                    created += 1
                elif conf > remote_index[best][2]:
                    poi_id = remote_index[best][3]
                    if poi_id:
                        self._api_request(
                            f'{self.api_url}/{poi_id}', 'PATCH', feature,
                        )
                    remote_index[best] = (cls, centroid, conf, poi_id)
                    updated += 1
                else:
                    skipped += 1
            except (URLError, OSError) as e:
                errors += 1
                self.get_logger().warn(
                    f'API upsert failed for a feature: {e}',
                    throttle_duration_sec=10.0,
                )
            except Exception as e:
                errors += 1
                self.get_logger().error(f'API upsert error: {e}')

        self.get_logger().info(
            f'API upsert: created={created} updated={updated} '
            f'skipped={skipped} errors={errors}',
            throttle_duration_sec=10.0,
        )


    #  SimpleStyle helpers
    @staticmethod
    def _class_color(class_name: str) -> str:

        colors = {
            # Fire / hazard (reds)
            'flame': '#ff0000',
            'smoke': '#ff4500',
            'burnt tree': '#8b0000',
            'burnt grass': '#a52a2a',
            'burnt plant': '#b22222',
            'fire hose': '#dc143c',
            'fire hydrant': '#ff6347',
            'fire truck': '#ff0000',
            'extinguisher': '#ff1493',
            # People (blues)
            'first responder': '#1e90ff',
            'citizen': '#4169e1',
            'military personnel': '#000080',
            # Vehicles (dark blues)
            'civilian vehicle': '#0000ff',
            'destroyed vehicle': '#00008b',
            'ambulance': '#4682b4',
            'police vehicle': '#191970',
            'army vehicle': '#2f4f4f',
            'boat': '#5f9ea0',
            'bicycle': '#00ff00',
            'aerial vehicle': '#87ceeb',
            # Nature (greens)
            'green tree': '#228b22',
            'green plant': '#32cd32',
            'green grass': '#7cfc00',
            'dry tree': '#daa520',
            'dry grass': '#bdb76b',
            'dry plant': '#f0e68c',
            'animal': '#ff8c00',
            # Infrastructure (greys)
            'building': '#708090',
            'destroyed building': '#696969',
            'wall': '#808080',
            'road': '#a9a9a9',
            'pavement': '#c0c0c0',
            'dirt road': '#d2b48c',
            'window': '#b0c4de',
            'door': '#8b4513',
            'stairs': '#a0522d',
            'pole': '#778899',
            'tower': '#556b2f',
            'silo': '#6b8e23',
            # Obstacles (oranges / yellows)
            'debris': '#ff8c00',
            'fence': '#daa520',
            'barrier': '#ffd700',
            'cone': '#ff7f50',
            'hole in the ground': '#8b4513',
            'mud': '#a0522d',
            'water': '#00bfff',
            # Equipment (purples)
            'helmet': '#9370db',
            'scba': '#8a2be2',
            'boot': '#4b0082',
            'mask': '#9400d3',
            'glove': '#da70d6',
            'protective glasses': '#ba55d3',
            'ladder': '#ff8c00',
            'ax': '#cd853f',
            'shovel': '#d2691e',
            'chainsaw': '#b8860b',
            'bag': '#bc8f8f',
            'barrel': '#8b8682',
            'furniture': '#deb887',
            'tank': '#2e8b57',
            'crane': '#b8860b',
            'excavator': '#daa520',
            'lifesaver': '#ff4500',
        }
        return colors.get(class_name, '#808080')

    @staticmethod
    def _class_category(class_name: str) -> str:

        categories = {
            # Hazard
            'flame': 'hazard', 'smoke': 'hazard',
            'burnt tree': 'hazard', 'burnt grass': 'hazard', 'burnt plant': 'hazard',
            # People
            'first responder': 'person', 'citizen': 'person',
            'military personnel': 'person',
            # Vehicles
            'civilian vehicle': 'vehicle', 'destroyed vehicle': 'vehicle',
            'ambulance': 'vehicle', 'police vehicle': 'vehicle',
            'fire truck': 'vehicle', 'army vehicle': 'vehicle',
            'boat': 'vehicle', 'bicycle': 'vehicle', 'aerial vehicle': 'vehicle',
            # Nature
            'green tree': 'nature', 'green plant': 'nature',
            'green grass': 'nature', 'dry tree': 'nature',
            'dry grass': 'nature', 'dry plant': 'nature', 'animal': 'nature',
            # Infrastructure
            'building': 'infrastructure', 'destroyed building': 'infrastructure',
            'wall': 'infrastructure', 'road': 'infrastructure',
            'pavement': 'infrastructure', 'dirt road': 'infrastructure',
            'window': 'infrastructure', 'door': 'infrastructure',
            'stairs': 'infrastructure', 'pole': 'infrastructure',
            'tower': 'infrastructure', 'silo': 'infrastructure',
            # Obstacle
            'debris': 'obstacle', 'fence': 'obstacle', 'barrier': 'obstacle',
            'cone': 'obstacle', 'hole in the ground': 'obstacle',
            'mud': 'obstacle', 'water': 'obstacle',
            # Equipment
            'fire hose': 'equipment', 'fire hydrant': 'equipment',
            'extinguisher': 'equipment', 'helmet': 'equipment',
            'scba': 'equipment', 'boot': 'equipment', 'mask': 'equipment',
            'glove': 'equipment', 'protective glasses': 'equipment',
            'ladder': 'equipment', 'ax': 'equipment', 'shovel': 'equipment',
            'chainsaw': 'equipment', 'bag': 'equipment', 'barrel': 'equipment',
            'furniture': 'equipment', 'tank': 'equipment', 'crane': 'equipment',
            'excavator': 'equipment', 'lifesaver': 'equipment',
        }
        return categories.get(class_name, 'unknown')

    @staticmethod
    def _class_symbol(class_name: str) -> str:

        symbols = {
            'first responder': 'pitch',
            'citizen': 'pitch',
            'military personnel': 'pitch',
            'civilian vehicle': 'car',
            'destroyed vehicle': 'car',
            'ambulance': 'hospital',
            'police vehicle': 'police',
            'fire truck': 'fire-station',
            'army vehicle': 'car',
            'boat': 'harbor',
            'bicycle': 'bicycle',
            'aerial vehicle': 'airfield',
            'flame': 'fire-station',
            'smoke': 'fire-station',
            'building': 'building',
            'destroyed building': 'building',
            'water': 'water',
            'animal': 'dog-park',
        }
        return symbols.get(class_name, 'marker')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GeoJSONBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if node._api_worker is not None:
                # Best-effort worker stop: clear any queued snapshot and
                # deliver the shutdown sentinel (worker is a daemon, so a
                # full queue at exit is not fatal).
                try:
                    node._api_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    node._api_queue.put_nowait(None)
                except queue.Full:
                    pass
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
