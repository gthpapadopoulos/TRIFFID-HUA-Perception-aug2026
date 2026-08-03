"""
TRIFFID UGV Perception Node

Pixel-aligned RGB-D perception pipeline:

  RGB and depth come from a single pixel-aligned camera (RealSense).
  The depth image shares the same resolution and intrinsics as the RGB image, so depth can be sampled
  directly at each detection's pixel coordinates.

  1. Subscribe to RGB + aligned Depth + shared CameraInfo + TF
  2. Run YOLO on the RGB image → 2D bounding boxes + instance masks
  3. For each detection, sample depth at mask/bbox pixels
  4. Back-project sampled pixels → 3D points in camera_optical_frame
  5. Median position + extent estimation
  6. Transform camera_optical_frame → b2/base_link via TF2
  7. Assign persistent tracking ID (IoU tracker on RGB bboxes)
  8. Publish vision_msgs/Detection3DArray in b2/base_link
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from geometry_msgs.msg import PoseStamped

from cv_bridge import CvBridge
import cv2
import numpy as np

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa – registers PoseStamped transform

from triffid_ugv_perception.tracker import ByteTracker

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False


TARGET_CLASSES = {
    0: 'water',
    1: 'fence',
    2: 'green tree',
    3: 'helmet',
    4: 'flame',
    5: 'smoke',
    6: 'first responder',
    7: 'destroyed vehicle',
    8: 'fire hose',
    9: 'scba',
    10: 'boot',
    11: 'green plant',
    12: 'mask',
    13: 'window',
    14: 'building',
    15: 'destroyed building',
    16: 'debris',
    17: 'ladder',
    18: 'dirt road',
    19: 'dry tree',
    20: 'wall',
    21: 'civilian vehicle',
    22: 'road',
    23: 'citizen',
    24: 'green grass',
    25: 'pole',
    26: 'boat',
    27: 'pavement',
    28: 'dry grass',
    29: 'animal',
    30: 'excavator',
    31: 'door',
    32: 'mud',
    33: 'barrier',
    34: 'hole in the ground',
    35: 'bag',
    36: 'burnt tree',
    37: 'ambulance',
    38: 'fire truck',
    39: 'cone',
    40: 'bicycle',
    41: 'tower',
    42: 'silo',
    43: 'military personnel',
    44: 'burnt grass',
    45: 'ax',
    46: 'glove',
    47: 'crane',
    48: 'stairs',
    49: 'dry plant',
    50: 'furniture',
    51: 'tank',
    52: 'protective glasses',
    53: 'barrel',
    54: 'shovel',
    55: 'fire hydrant',
    56: 'police vehicle',
    57: 'burnt plant',
    58: 'army vehicle',
    59: 'chainsaw',
    60: 'aerial vehicle',
    61: 'lifesaver',
    62: 'extinguisher',
}

# Frame IDs
BASE_FRAME = 'b2/base_link'

# Default topic names
DEFAULT_RGB_TOPIC = '/camera_front/raw_image'
DEFAULT_DEPTH_TOPIC = '/camera_front/realsense_front/depth/image_rect_raw'
DEFAULT_CAMERA_INFO_TOPIC = '/camera_front/camera_info'

# Maximum number of depth pixels to sample per detection (for efficiency)
_MAX_DEPTH_SAMPLES = 500


class UGVPerceptionNode(Node):
    """Main UGV perception node – pixel-aligned RGB-D pipeline."""

    def __init__(self):
        super().__init__('ugv_perception_node')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('model_path', '/ws/best.pt')
        self.declare_parameter('confidence_threshold', 0.35)
        self.declare_parameter('target_frame', BASE_FRAME)
        self.declare_parameter('use_dummy_detections', False)
        self.declare_parameter('yolo_imgsz', 1280)
        self.declare_parameter('tracker_iou_threshold', 0.30)
        self.declare_parameter('tracker_iou_threshold_low', 0.15)
        self.declare_parameter('tracker_conf_high', 0.40)
        self.declare_parameter('tracker_max_age', 30)
        self.declare_parameter('tracker_n_init', 3)
        self.declare_parameter('tracker_pos_gate', 2.0)
        self.declare_parameter('nms_merge_dist_m', 0.5)
        self.declare_parameter('publish_debug_image', True)
        # Topic names (placeholders — override in launch file when final names known)
        self.declare_parameter('rgb_image_topic', DEFAULT_RGB_TOPIC)
        self.declare_parameter('depth_image_topic', DEFAULT_DEPTH_TOPIC)
        self.declare_parameter('camera_info_topic', DEFAULT_CAMERA_INFO_TOPIC)

        self.model_path = self.get_parameter('model_path').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value
        self.target_frame = self.get_parameter('target_frame').value
        self.use_dummy = self.get_parameter('use_dummy_detections').value
        self.yolo_imgsz = self.get_parameter('yolo_imgsz').value
        self.tracker_iou = self.get_parameter('tracker_iou_threshold').value
        self.tracker_iou_low = self.get_parameter('tracker_iou_threshold_low').value
        self.tracker_conf_high = self.get_parameter('tracker_conf_high').value
        self.tracker_max_age = self.get_parameter('tracker_max_age').value
        self.tracker_n_init = self.get_parameter('tracker_n_init').value
        self.tracker_pos_gate = self.get_parameter('tracker_pos_gate').value
        self.nms_merge_dist_m = self.get_parameter('nms_merge_dist_m').value
        self.publish_debug_image = self.get_parameter('publish_debug_image').value
        self.rgb_topic = self.get_parameter('rgb_image_topic').value
        self.depth_topic = self.get_parameter('depth_image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value

        # ── YOLO model ──────────────────────────────────────────────
        if _HAS_YOLO:
            self.get_logger().info(f'Loading YOLO model: {self.model_path}')
            self.model = YOLO(self.model_path)
            self.get_logger().info('Model loaded.')
        else:
            self.model = None
            self.get_logger().error(
                'ultralytics not installed — no detections will be produced. '
                'Install with: pip install ultralytics'
            )

        # ── State ───────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.camera_info = None          # shared CameraInfo (pixel-aligned)
        self.camera_frame = None         # read from CameraInfo header.frame_id
        self.depth_image = None          # latest depth frame (uint16, mm)
        self.depth_stamp = None
        self.tracker = ByteTracker(
            iou_threshold=float(self.tracker_iou),
            iou_threshold_low=float(self.tracker_iou_low),
            conf_threshold_high=float(self.tracker_conf_high),
            max_age=int(self.tracker_max_age),
            n_init=int(self.tracker_n_init),
            pos_gate=float(self.tracker_pos_gate),
        )

        # ── TF2 ─────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── QoS ─────────────────────────────────────────────────────
        # The rosbag records sensor topics as RELIABLE + VOLATILE.
        # We match that exactly so ros2 bag play data flows correctly.
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Subscribers ─────────────────────────────────────────────
        self.sub_rgb = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            sensor_qos,
        )
        self.sub_depth = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            sensor_qos,
        )
        self.sub_camera_info = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            reliable_qos,
        )

        # ── Publishers ───────────────────────────────────────────────
        self.pub_det3d = self.create_publisher(
            Detection3DArray,
            '/ugv/detections/front/detections_3d',
            10,
        )
        self.pub_seg = self.create_publisher(
            Image,
            '/ugv/detections/front/segmentation',
            10,
        )
        self.pub_debug = self.create_publisher(
            Image,
            '/ugv/detections/front/debug_image',
            10,
        )

        self.get_logger().info('UGV Perception node started (pixel-aligned RGB-D pipeline).')
        self.get_logger().info(f'  RGB topic:       {self.rgb_topic}')
        self.get_logger().info(f'  Depth topic:     {self.depth_topic}')
        self.get_logger().info(f'  CameraInfo topic: {self.camera_info_topic}')
        self.get_logger().info(f'  Output topic:    /ugv/detections/front/detections_3d  (frame: {self.target_frame})')
        self.get_logger().info(f'  Seg topic:       /ugv/detections/front/segmentation  (mono8 label map)')
        self.get_logger().info(f'  Debug topic:     /ugv/detections/front/debug_image  (RGB+ID overlay)')
        self.get_logger().info(
            f'  Tracker: IoU={self.tracker_iou:.2f}, max_age={int(self.tracker_max_age)}'
        )
        if self.use_dummy:
            self.get_logger().warn('*** DUMMY DETECTION MODE — bypassing YOLO ***')

    # ─── Callbacks ──────────────────────────────────────────────────

    def camera_info_callback(self, msg: CameraInfo):
        """Store shared camera intrinsics (pixel-aligned RGB-D)."""
        if self.camera_info is None:
            self.get_logger().info(
                f'CameraInfo received: {msg.width}x{msg.height}, '
                f'frame={msg.header.frame_id}')
        self.camera_info = msg
        self.camera_frame = msg.header.frame_id

    def depth_callback(self, msg: Image):
        """Store latest depth image (16UC1, millimetres)."""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough'
            )
            self.depth_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Depth conversion failed: {e}')

    def rgb_callback(self, msg: Image):
        """Main processing trigger – runs on every RGB frame."""

        # Guard: need CameraInfo and a depth image
        if self.camera_info is None:
            self.get_logger().warn(
                f'Waiting for CameraInfo ({self.camera_info_topic})…',
                throttle_duration_sec=5.0,
            )
            return
        if self.depth_image is None:
            self.get_logger().warn(
                'Waiting for Depth image…', throttle_duration_sec=5.0,
            )
            return

        # Convert RGB image — handle both standard bgr8/rgb8 and YUYV (RealSense 435i)
        try:
            if msg.encoding in ('yuv422', 'yuv422_yuy2', 'YUV422'):
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (msg.height, msg.width, 2)
                )
                cv_image = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUY2)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB conversion failed: {e}')
            return

        # Check pixel alignment: depth and RGB should share resolution
        depth_h, depth_w = self.depth_image.shape[:2]
        rgb_h, rgb_w = cv_image.shape[:2]
        depth_scale_x = 1.0
        depth_scale_y = 1.0
        if (depth_h, depth_w) != (rgb_h, rgb_w):
            depth_scale_x = float(depth_w) / float(rgb_w)
            depth_scale_y = float(depth_h) / float(rgb_h)
            self.get_logger().warn(
                f'Depth ({depth_w}×{depth_h}) and RGB ({rgb_w}×{rgb_h}) '
                f'resolutions differ — using scaled depth sampling fallback.',
                throttle_duration_sec=10.0,
            )

        # ── Step 1: YOLO detection on RGB ────────────────────────────
        if self.use_dummy:
            raw_detections = self._dummy_detection(cv_image)
        else:
            raw_detections = self._detect(cv_image)
        if not raw_detections:
            self._publish_empty(msg.header.stamp)
            self._publish_debug_overlay(cv_image, [], msg.header)
            return

        # Camera intrinsics (shared for pixel-aligned RGB-D)
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]
        if fx == 0 or fy == 0:
            self.get_logger().warn(
                'CameraInfo has zero focal length', throttle_duration_sec=5.0)
            return

        camera_frame = self.camera_frame

        # ── Step 2–4: For each detection, sample depth & back-project ─
        detections_3d = []

        for det in raw_detections:
            x1, y1, x2, y2 = det['bbox']
            mask = det.get('mask')

            # Sample depth at detection pixels (pixel-aligned: same coords)
            matched_pts = self._sample_depth_for_detection(
                det, self.depth_image, fx, fy, cx, cy,
                depth_scale_x=depth_scale_x,
                depth_scale_y=depth_scale_y,
                rgb_shape=(rgb_h, rgb_w),
            )
            if matched_pts is None or len(matched_pts) == 0:
                continue

            n_inside = len(matched_pts)

            # Median position in camera_optical_frame
            median_pt = np.median(matched_pts, axis=0)  # (3,)

            # ── Step 5: Transform camera_optical_frame → b2/base_link ─
            pt_base = self._transform_point(
                tuple(median_pt), camera_frame, self.target_frame,
                msg.header.stamp,
            )
            if pt_base is None:
                # No valid transform — a camera-frame position labeled as
                # base_link would poison downstream geo-referencing.
                continue

            # ── Compute 3D bbox extent in base_link ─────────────────
            pt_min_cam = np.min(matched_pts, axis=0)
            pt_max_cam = np.max(matched_pts, axis=0)
            cloud_extent = pt_max_cam - pt_min_cam

            if np.any(cloud_extent > 1e-3):
                corners_cam = np.array([
                    [pt_min_cam[0], pt_min_cam[1], pt_min_cam[2]],
                    [pt_min_cam[0], pt_min_cam[1], pt_max_cam[2]],
                    [pt_min_cam[0], pt_max_cam[1], pt_min_cam[2]],
                    [pt_min_cam[0], pt_max_cam[1], pt_max_cam[2]],
                    [pt_max_cam[0], pt_min_cam[1], pt_min_cam[2]],
                    [pt_max_cam[0], pt_min_cam[1], pt_max_cam[2]],
                    [pt_max_cam[0], pt_max_cam[1], pt_min_cam[2]],
                    [pt_max_cam[0], pt_max_cam[1], pt_max_cam[2]],
                ])
            else:
                corners_cam = self._bbox_to_3d_corners(
                    x1, y1, x2, y2, median_pt[2], fx, fy, cx, cy,
                )

            corners_base = self._transform_points_batch(
                corners_cam, camera_frame, self.target_frame,
                msg.header.stamp,
            )
            if corners_base is not None:
                extent_base = tuple(
                    float(v) for v in np.ptp(corners_base, axis=0)
                )
            else:
                extent_base = (0.0, 0.0, 0.0)

            detections_3d.append({
                'position': pt_base,
                'extent': extent_base,
                'class_id': det['class_id'],
                'class_name': det['class_name'],
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'n_depth_pts': n_inside,
            })

        # ── Step 5.5: 3-D NMS — deduplicate overlapping detections ───
        detections_3d = self._nms_3d(
            detections_3d, dist_thresh=float(self.nms_merge_dist_m),
        )

        # ── Step 6: Track across frames (IoU on RGB bboxes) ─────────
        tracked = self.tracker.update(detections_3d)

        # ── Step 7: Publish Detection3DArray ─────────────────────────
        det_array = Detection3DArray()
        det_array.header.stamp = msg.header.stamp       # timestamp from RGB
        det_array.header.frame_id = self.target_frame    # b2/base_link

        for t in tracked:
            d3d = Detection3D()
            d3d.header = det_array.header

            if t['position'] is not None:
                d3d.bbox.center.position.x = float(t['position'][0])
                d3d.bbox.center.position.y = float(t['position'][1])
                d3d.bbox.center.position.z = float(t['position'][2])

            ext = t.get('extent', (0.0, 0.0, 0.0))
            d3d.bbox.size.x = float(ext[0])
            d3d.bbox.size.y = float(ext[1])
            d3d.bbox.size.z = float(ext[2])

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(t['class_name'])
            hyp.hypothesis.score = float(t['confidence'])
            d3d.results.append(hyp)

            d3d.id = str(t['track_id'])
            det_array.detections.append(d3d)

        self.pub_det3d.publish(det_array)

        # ── Step 8: Publish segmentation label map (mono8) ───────────
        self._publish_segmentation(cv_image, raw_detections, msg.header)
        self._publish_debug_overlay(cv_image, tracked, msg.header)

        if tracked:
            self.get_logger().info(
                f'Published {len(tracked)} detections '
                f'(IDs: {[t["track_id"] for t in tracked]})',
                throttle_duration_sec=1.0,
            )

    # ─── YOLO detection ────────────────────────────────────────────

    def _detect(self, cv_image):
        """Run YOLO-seg on a BGR image.

        Returns list of detection dicts, each containing:
          bbox, class_id, class_name, confidence, mask (H×W bool ndarray or None)
        """
        if self.model is None:
            return []

        results = self.model(
            cv_image, conf=self.conf_thresh, imgsz=self.yolo_imgsz, verbose=False,
        )
        detections = []
        for r in results:
            masks_data = r.masks  # may be None if model produced no masks
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0])
                if cls_id not in TARGET_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Extract per-instance binary mask (resized to input image)
                mask = None
                if masks_data is not None and i < len(masks_data.data):
                    mask = masks_data.data[i].cpu().numpy().astype(bool)
                    # Resize mask to match the input image dimensions
                    h, w = cv_image.shape[:2]
                    if mask.shape != (h, w):
                        mask = cv2.resize(
                            mask.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)

                detections.append({
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'class_id': cls_id,
                    'class_name': TARGET_CLASSES[cls_id],
                    'confidence': float(box.conf[0]),
                    'mask': mask,
                })
        return detections

    def _dummy_detection(self, cv_image):
        """Return a single fake detection at the image centre.
        Useful for testing the depth/TF/tracking pipeline without YOLO."""
        h, w = cv_image.shape[:2]
        # Full-image bbox so projections from the tilted depth camera
        # (which hits the lower portion of the RGB image) are captured.
        return [{
            'bbox': (0.0, 0.0, float(w), float(h)),
            'class_id': 0,
            'class_name': TARGET_CLASSES[0],
            'confidence': 0.99,
            'mask': None,
        }]

    # ─── Segmentation label-map publisher ──────────────────────────

    def _publish_segmentation(self, cv_image, detections, header):
        """Publish a semantic segmentation label image (mono8).

        Each pixel value is the 1-based YOLO class ID of the detection
        that covers it (0 = background).  With 63 classes this fits
        comfortably in uint8.  When masks overlap, the higher-confidence
        detection wins.
        """
        if self.pub_seg.get_subscription_count() == 0:
            return  # no subscribers, skip

        h, w = cv_image.shape[:2]
        label_img = np.zeros((h, w), dtype=np.uint8)

        # Sort ascending by confidence so higher-conf overwrites lower
        sorted_dets = sorted(detections,
                             key=lambda d: d['confidence'])
        for det in sorted_dets:
            mask = det.get('mask')
            if mask is None:
                continue
            # class_id is 0-based from YOLO; store as 1-based (0 = bg)
            label_img[mask] = det['class_id'] + 1

        seg_msg = self.bridge.cv2_to_imgmsg(label_img, encoding='mono8')
        seg_msg.header = header
        self.pub_seg.publish(seg_msg)

    def _publish_debug_overlay(self, cv_image, tracked, header):
        """Publish RGB debug image with bbox + class + track ID overlay."""
        if not self.publish_debug_image:
            return
        if self.pub_debug.get_subscription_count() == 0:
            return

        dbg = cv_image.copy()
        for t in tracked:
            bbox = t.get('bbox')
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            tid = int(t.get('track_id', -1))
            cls = t.get('class_name', 'unknown')
            conf = float(t.get('confidence', 0.0))
            npts = int(t.get('n_depth_pts', 0))

            # Deterministic per-track color makes ID switches obvious.
            color = (
                int((37 * tid) % 255),
                int((97 * tid) % 255),
                int((173 * tid) % 255),
            )
            cv2.rectangle(dbg, (x1, y1), (x2, y2), color, 2)
            label = f'{cls} #{tid} {conf:.2f} d={npts}'
            ytxt = max(18, y1 - 6)
            cv2.putText(
                dbg, label, (x1, ytxt),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
            )

        msg = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
        msg.header = header
        self.pub_debug.publish(msg)

    # ─── Cross-camera depth → RGB-frame geometry ───────────────────

    def _sample_depth_for_detection(
        self,
        det,
        depth_img,
        fx,
        fy,
        cx,
        cy,
        depth_scale_x=1.0,
        depth_scale_y=1.0,
        rgb_shape=None,
    ):
        """Sample depth at detection pixels and back-project to 3D.

        For a pixel-aligned RGB-D camera the depth image shares the
        same pixel grid and intrinsics as the RGB image, so we sample
        depth directly at the detection's mask (or bbox) pixels.

        Returns:
            np.ndarray (M, 3) in camera_optical_frame
            (X=right, Y=down, Z=forward), or None.
        """
        depth_h, depth_w = depth_img.shape[:2]
        if rgb_shape is None:
            rgb_h, rgb_w = depth_h, depth_w
        else:
            rgb_h, rgb_w = rgb_shape
        mask = det.get('mask')
        x1, y1, x2, y2 = det['bbox']

        if mask is not None:
            ys, xs = np.where(mask)
            if len(xs) == 0:
                return None
            if len(xs) > _MAX_DEPTH_SAMPLES:
                step = max(1, len(xs) // _MAX_DEPTH_SAMPLES)
                xs = xs[::step]
                ys = ys[::step]
        else:
            # Bbox grid fallback (dummy mode / no mask)
            ix1 = max(0, min(int(x1), rgb_w - 1))
            ix2 = max(ix1 + 1, min(int(x2), rgb_w))
            iy1 = max(0, min(int(y1), rgb_h - 1))
            iy2 = max(iy1 + 1, min(int(y2), rgb_h))
            side = max(1, min(ix2 - ix1, iy2 - iy1) // 10)
            xs_g, ys_g = np.meshgrid(
                np.arange(ix1, ix2, max(1, side)),
                np.arange(iy1, iy2, max(1, side)),
            )
            xs = xs_g.ravel()
            ys = ys_g.ravel()
            if len(xs) > _MAX_DEPTH_SAMPLES:
                step = max(1, len(xs) // _MAX_DEPTH_SAMPLES)
                xs = xs[::step]
                ys = ys[::step]

        # Map RGB coordinates onto depth indices for non-aligned fallback.
        xs_depth = np.rint(xs.astype(np.float64) * depth_scale_x).astype(np.int32)
        ys_depth = np.rint(ys.astype(np.float64) * depth_scale_y).astype(np.int32)

        # Clip to depth image bounds
        ok = (
            (xs_depth >= 0) & (xs_depth < depth_w) &
            (ys_depth >= 0) & (ys_depth < depth_h)
        )
        xs = xs[ok]
        ys = ys[ok]
        xs_depth = xs_depth[ok]
        ys_depth = ys_depth[ok]
        if len(xs_depth) == 0:
            return None

        # Read depth (uint16 mm)
        z_mm = depth_img[ys_depth, xs_depth].astype(np.float64)
        valid = z_mm > 0
        if not np.any(valid):
            return None

        xs = xs[valid].astype(np.float64)
        ys = ys[valid].astype(np.float64)
        z_m = z_mm[valid] / 1000.0

        # Pinhole back-projection → camera_optical_frame
        X = (xs - cx) * z_m / fx
        Y = (ys - cy) * z_m / fy
        Z = z_m
        return np.column_stack([X, Y, Z])

    def _transform_points_batch(self, points, source_frame, target_frame, stamp):
        """Transform an (N,3) array of 3D points between frames using TF2.

        Looks up the transform once and applies it as a matrix multiply
        for efficiency (instead of N individual TF calls).

        Returns np.ndarray (N, 3) in the target frame, or None on failure.
        """
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except Exception as e:
            self.get_logger().warn(
                f'TF {source_frame}→{target_frame} lookup failed: {e}',
                throttle_duration_sec=5.0,
            )
            return None

        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation

        # Quaternion → 3×3 rotation matrix
        rot = self._quat_to_matrix(q.x, q.y, q.z, q.w)
        tvec = np.array([t.x, t.y, t.z])

        # Apply: p_target = R @ p_source + t
        pts_out = (rot @ points.T).T + tvec
        return pts_out

    def _transform_point(self, point_cam, source_frame, target_frame, stamp):
        """Transform a single 3D point between frames using TF2.
        Returns (x, y, z) in target frame, or None on failure (consistent
        with _transform_points_batch — callers must skip the detection).
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = source_frame
        pose_msg.pose.position.x = float(point_cam[0])
        pose_msg.pose.position.y = float(point_cam[1])
        pose_msg.pose.position.z = float(point_cam[2])
        pose_msg.pose.orientation.w = 1.0

        try:
            transformed = self.tf_buffer.transform(
                pose_msg, target_frame,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            return (
                transformed.pose.position.x,
                transformed.pose.position.y,
                transformed.pose.position.z,
            )
        except Exception as e:
            self.get_logger().warn(
                f'TF {source_frame}→{target_frame} failed: {e}',
                throttle_duration_sec=5.0,
            )
            return None

    def _bbox_to_3d_corners(self, u1, v1, u2, v2, depth, fx, fy, cx, cy):
        """Back-project 2D bbox corners to 3D at a given depth.

        Uses camera_optical_frame convention (X right, Y down, Z forward).
        ``depth`` is the Z component (forward distance from the camera).

        Returns np.ndarray (8, 3) – axis-aligned bounding box corners.
        """
        x_left  = (u1 - cx) * depth / fx
        x_right = (u2 - cx) * depth / fx
        y_top   = (v1 - cy) * depth / fy
        y_bot   = (v2 - cy) * depth / fy

        x_min, x_max = min(x_left, x_right), max(x_left, x_right)
        y_min, y_max = min(y_top, y_bot), max(y_top, y_bot)

        z = float(depth)
        return np.array([
            [x_min, y_min, z],
            [x_min, y_max, z],
            [x_max, y_min, z],
            [x_max, y_max, z],
            [x_min, y_min, z],
            [x_min, y_max, z],
            [x_max, y_min, z],
            [x_max, y_max, z],
        ])

    # ─── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _nms_3d(detections, dist_thresh=0.5):
        """Suppress duplicate 3-D detections at (nearly) identical positions.

        When several YOLO predictions overlap the same region (e.g.
        "Destroyed building" and "Building"), they match the same sparse
        depth points and produce identical 3-D positions.  This function
        keeps only the highest-confidence detection within *dist_thresh*
        metres (Euclidean in base_link).
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence descending — greedy NMS
        dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        keep = []
        for det in dets:
            pos = np.array(det['position'])
            suppressed = False
            for kept in keep:
                if np.linalg.norm(pos - np.array(kept['position'])) < dist_thresh:
                    suppressed = True
                    break
            if not suppressed:
                keep.append(det)
        return keep

    def _publish_empty(self, stamp):
        """Publish an empty Detection3DArray (keeps downstream aware)."""
        msg = Detection3DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target_frame
        self.pub_det3d.publish(msg)

    @staticmethod
    def _quat_to_matrix(qx, qy, qz, qw):
        """Convert quaternion (x, y, z, w) to a 3×3 rotation matrix."""
        # Normalise
        n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if n == 0:
            return np.eye(3)
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

        return np.array([
            [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ])


def main(args=None):
    rclpy.init(args=args)
    node = UGVPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
