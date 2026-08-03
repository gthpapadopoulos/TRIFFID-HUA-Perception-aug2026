"""
ByteTrack-style Tracker for TRIFFID Perception

Multi-object tracker using:
  - Kalman filter for bbox state prediction (constant-velocity model)
  - Hungarian algorithm (scipy) for optimal assignment
  - Two-pass association (high-confidence first, then low-confidence)
  - Track confirmation gate (tentative → confirmed)
  - 3D position fallback cost for small or occluded objects

Effectively:
  - IDs are persistent and never reused
  - If an object disappears, its ID is retired
  - Counter never resets
"""

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
#  Lightweight Kalman filter for 2D bbox tracking
# ---------------------------------------------------------------------------

class _KalmanBBox:
    """Constant-velocity Kalman filter on (cx, cy, aspect, h).

    State  x = [cx, cy, a, h, vx, vy, va, vh]^T   (8-dim)
    Measurement z = [cx, cy, a, h]^T                (4-dim)

    This is the same model used in SORT / DeepSORT / ByteTrack.
    """

    _STD_WEIGHT_POS = 1.0 / 20.0
    _STD_WEIGHT_VEL = 1.0 / 160.0

    def __init__(self, bbox):
        """Initialise from (x1, y1, x2, y2)."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        a = w / h if h > 0 else 0.0

        self.x = np.array([cx, cy, a, h, 0, 0, 0, 0], dtype=np.float64)
        self.P = np.eye(8, dtype=np.float64)
        self.P[4:, 4:] *= 10.0  # high initial velocity uncertainty

        self.F = np.eye(8, dtype=np.float64)
        self.F[:4, 4:] = np.eye(4)

        self.H = np.eye(4, 8, dtype=np.float64)

    def predict(self):
        """Advance state by one time step.  Returns predicted bbox."""
        std = [
            self._STD_WEIGHT_POS * self.x[3],
            self._STD_WEIGHT_POS * self.x[3],
            1e-2,
            self._STD_WEIGHT_POS * self.x[3],
            self._STD_WEIGHT_VEL * self.x[3],
            self._STD_WEIGHT_VEL * self.x[3],
            1e-5,
            self._STD_WEIGHT_VEL * self.x[3],
        ]
        Q = np.diag(np.square(std))

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

        self.x[2] = max(self.x[2], 1e-4)
        self.x[3] = max(self.x[3], 1.0)

        return self._state_to_bbox()

    def update(self, bbox):
        """Correct state with a measurement (x1, y1, x2, y2)."""
        z = self._bbox_to_measurement(bbox)

        std = [
            self._STD_WEIGHT_POS * self.x[3],
            self._STD_WEIGHT_POS * self.x[3],
            1e-1,
            self._STD_WEIGHT_POS * self.x[3],
        ]
        R = np.diag(np.square(std))

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def _state_to_bbox(self):
        """Convert internal state → (x1, y1, x2, y2)."""
        cx, cy, a, h = self.x[:4]
        w = a * h
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @staticmethod
    def _bbox_to_measurement(bbox):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        a = w / h if h > 0 else 0.0
        return np.array([cx, cy, a, h], dtype=np.float64)


# ---------------------------------------------------------------------------
#  Track object
# ---------------------------------------------------------------------------

class _Track:
    """Single tracked object with state machine."""

    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2

    def __init__(self, track_id, detection, n_init):
        self.id = track_id
        self.kf = _KalmanBBox(detection['bbox'])
        self.bbox = detection['bbox']
        self.position = detection.get('position')
        self.extent = detection.get('extent', (0.0, 0.0, 0.0))
        self.n_depth_pts = detection.get('n_depth_pts', 0)
        self.class_id = detection.get('class_id', -1)
        self.class_name = detection['class_name']
        self.confidence = detection['confidence']
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.n_init = n_init
        self.state = self.TENTATIVE if n_init > 1 else self.CONFIRMED
        # Class vote histogram: {class_name: count}
        self._class_votes = {self.class_name: 1}

    @property
    def is_confirmed(self):
        return self.state == self.CONFIRMED

    @property
    def is_lost(self):
        return self.state == self.LOST

    def predict(self):
        """Kalman predict; returns predicted bbox."""
        self.age += 1
        self.time_since_update += 1
        self.bbox = self.kf.predict()
        return self.bbox

    def update(self, detection):
        """Kalman update with matched detection."""
        self.kf.update(detection['bbox'])
        self.bbox = detection['bbox']
        self.position = detection.get('position')
        self.extent = detection.get('extent', (0.0, 0.0, 0.0))
        self.n_depth_pts = detection.get('n_depth_pts', 0)
        self.confidence = detection['confidence']
        self.hits += 1
        self.time_since_update = 0
        # Majority-vote class: accumulate votes, pick the winner
        det_cls = detection['class_name']
        self._class_votes[det_cls] = self._class_votes.get(det_cls, 0) + 1
        best_cls = max(self._class_votes, key=self._class_votes.get)
        self.class_name = best_cls
        det_cid = detection.get('class_id', -1)
        if best_cls == det_cls:
            self.class_id = det_cid

        if self.state == self.TENTATIVE and self.hits >= self.n_init:
            self.state = self.CONFIRMED
        elif self.state == self.LOST:
            self.state = self.CONFIRMED

    def mark_lost(self):
        self.state = self.LOST


# ---------------------------------------------------------------------------
#  Cost matrix helpers
# ---------------------------------------------------------------------------

def _iou_batch(bboxes_a, bboxes_b):
    """Vectorised IoU between two sets of (x1,y1,x2,y2) boxes.

    Returns (M, N) IoU matrix.
    """
    a = np.asarray(bboxes_a, dtype=np.float64)
    b = np.asarray(bboxes_b, dtype=np.float64)
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[np.newaxis, :]

    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter
    union = np.maximum(union, 1e-12)

    return inter / union


def _pos_distance_batch(positions_a, positions_b):
    """Euclidean distance between two sets of 3D positions.

    Returns (M, N) distance matrix.  Entries where either position is
    None are set to inf.
    """
    M = len(positions_a)
    N = len(positions_b)
    dist = np.full((M, N), np.inf, dtype=np.float64)
    for i, pa in enumerate(positions_a):
        if pa is None:
            continue
        pa = np.asarray(pa, dtype=np.float64)
        for j, pb in enumerate(positions_b):
            if pb is None:
                continue
            pb = np.asarray(pb, dtype=np.float64)
            dist[i, j] = np.linalg.norm(pa - pb)
    return dist


# ---------------------------------------------------------------------------
#  Main tracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """ByteTrack-style multi-object tracker with Kalman prediction.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU for first-pass association (high-confidence).
    iou_threshold_low : float
        Minimum IoU for second-pass association (low-confidence).
    conf_threshold_high : float
        Confidence split between high and low association passes.
    max_age : int
        Maximum frames a lost track is kept before removal.
    n_init : int
        Consecutive detections needed to confirm a tentative track.
        Unconfirmed tracks are not published.
    pos_gate : float
        Maximum 3D Euclidean distance (metres) for a match.
        Used as auxiliary gate when IoU is unreliable (small boxes).
    """

    def __init__(
        self,
        iou_threshold=0.3,
        iou_threshold_low=0.15,
        conf_threshold_high=0.4,
        max_age=30,
        n_init=3,
        pos_gate=2.0,
    ):
        self.iou_threshold = iou_threshold
        self.iou_threshold_low = iou_threshold_low
        self.conf_threshold_high = conf_threshold_high
        self.max_age = max_age
        self.n_init = n_init
        self.pos_gate = pos_gate

        self.next_id = 1
        self.tracks = []   # list of _Track objects

    # ------------------------------------------------------------------
    #  Public API  (same return format as old IoUTracker.update)
    # ------------------------------------------------------------------

    def update(self, detections):
        """Update tracker with new detections.

        Args:
            detections: list of dicts with keys:
                'bbox': (x1, y1, x2, y2)
                'class_id': int
                'class_name': str
                'confidence': float
                'position': (x, y, z) or None

        Returns:
            list of dicts (confirmed tracks only) with added
            'track_id' key.
        """
        for track in self.tracks:
            track.predict()

        if not detections:
            # No detections → all tracks become lost
            for track in self.tracks:
                track.mark_lost()
            self._remove_dead_tracks()
            return self._output()

        # Split detections by confidence
        dets_high = []
        dets_low = []
        for det in detections:
            if det['confidence'] >= self.conf_threshold_high:
                dets_high.append(det)
            else:
                dets_low.append(det)

        # First pass: high-confidence vs confirmed + lost tracks
        confirmed_tracks = [t for t in self.tracks
                           if t.state != _Track.TENTATIVE]
        tentative_tracks = [t for t in self.tracks
                           if t.state == _Track.TENTATIVE]

        matched_t, matched_d, unmatched_tracks, unmatched_dets = \
            self._associate(confirmed_tracks, dets_high,
                            self.iou_threshold)

        for ti, di in zip(matched_t, matched_d):
            confirmed_tracks[ti].update(dets_high[di])
        remaining_tracks = [confirmed_tracks[i] for i in unmatched_tracks]

        # Second pass: low-confidence vs remaining tracks
        if dets_low and remaining_tracks:
            matched_t2, matched_d2, unmatched_tracks2, _ = \
                self._associate(remaining_tracks, dets_low,
                                self.iou_threshold_low)
            for ti, di in zip(matched_t2, matched_d2):
                remaining_tracks[ti].update(dets_low[di])
            for i in unmatched_tracks2:
                remaining_tracks[i].mark_lost()
        else:
            for t in remaining_tracks:
                t.mark_lost()

        # Tentative tracks: match with remaining unmatched high-conf dets
        unmatched_high_dets = [dets_high[i] for i in unmatched_dets]
        if tentative_tracks and unmatched_high_dets:
            mt, md, ut, ud = self._associate(
                tentative_tracks, unmatched_high_dets,
                self.iou_threshold)
            for ti, di in zip(mt, md):
                tentative_tracks[ti].update(unmatched_high_dets[di])
            for i in ut:
                tentative_tracks[i].mark_lost()
            unmatched_high_dets = [unmatched_high_dets[i] for i in ud]
        else:
            for t in tentative_tracks:
                t.mark_lost()

        # Create new tentative tracks for remaining unmatched high-conf dets
        for det in unmatched_high_dets:
            self._create_track(det)

        self._remove_dead_tracks()
        return self._output()

    # ------------------------------------------------------------------
    #  Association
    # ------------------------------------------------------------------

    def _associate(self, tracks, detections, iou_thresh):
        """Match tracks to detections using IoU + 3D position gate.

        Returns (matched_track_idx, matched_det_idx,
                 unmatched_track_idx, unmatched_det_idx).
        """
        if not tracks or not detections:
            return ([], [],
                    list(range(len(tracks))),
                    list(range(len(detections))))

        track_bboxes = [t.bbox for t in tracks]
        det_bboxes = [d['bbox'] for d in detections]
        iou_matrix = _iou_batch(track_bboxes, det_bboxes)

        track_pos = [t.position for t in tracks]
        det_pos = [d.get('position') for d in detections]
        dist_matrix = _pos_distance_batch(track_pos, det_pos)

        # Cost = 1 - IoU (lower is better)
        cost = 1.0 - iou_matrix

        # Class gate: penalise cross-class matches
        track_cls = [t.class_name for t in tracks]
        det_cls = [d['class_name'] for d in detections]
        for i, tc in enumerate(track_cls):
            for j, dc in enumerate(det_cls):
                if tc != dc:
                    cost[i, j] += 1e5

        # Gate: reject pairs that are both low-IoU AND far in 3D
        gate_mask = (iou_matrix < iou_thresh) & (dist_matrix > self.pos_gate)
        cost[gate_mask] = 1e5

        no_info = (iou_matrix == 0) & np.isinf(dist_matrix)
        cost[no_info] = 1e5

        if _HAS_SCIPY:
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            row_idx, col_idx = self._greedy_assignment(cost)

        matched_t = []
        matched_d = []
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] >= 1e5:
                continue
            if iou_matrix[r, c] < iou_thresh:
                # Below IoU threshold — accept only if 3D match is close
                if dist_matrix[r, c] > self.pos_gate:
                    continue
            matched_t.append(r)
            matched_d.append(c)

        unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
        unmatched_d = [i for i in range(len(detections)) if i not in matched_d]

        return matched_t, matched_d, unmatched_t, unmatched_d

    @staticmethod
    def _greedy_assignment(cost):
        """Fallback greedy assignment when scipy is not available."""
        used_r = set()
        used_c = set()
        pairs_r = []
        pairs_c = []

        indices = np.unravel_index(
            np.argsort(cost, axis=None), cost.shape
        )
        for r, c in zip(indices[0], indices[1]):
            if r in used_r or c in used_c:
                continue
            pairs_r.append(r)
            pairs_c.append(c)
            used_r.add(r)
            used_c.add(c)

        return np.array(pairs_r), np.array(pairs_c)

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    def _create_track(self, det):
        track = _Track(self.next_id, det, self.n_init)
        self.next_id += 1
        self.tracks.append(track)
        return track

    def _remove_dead_tracks(self):
        self.tracks = [
            t for t in self.tracks
            if not (t.state == _Track.LOST
                    and t.time_since_update > self.max_age)
            and not (t.state == _Track.TENTATIVE
                     and t.time_since_update > 0)
        ]

    def _output(self):
        """Return list of confirmed-track dicts (compatible with old API)."""
        results = []
        for t in self.tracks:
            if not t.is_confirmed:
                continue
            results.append({
                'bbox': t.bbox,
                'class_name': t.class_name,
                'class_id': t.class_id,
                'confidence': t.confidence,
                'position': t.position,
                'extent': t.extent,
                'n_depth_pts': t.n_depth_pts,
                'track_id': t.id,
            })
        return results


# ---------------------------------------------------------------------------
#  Legacy alias — keeps old imports working
# ---------------------------------------------------------------------------

class IoUTracker(ByteTracker):
    """Drop-in replacement: delegates to ByteTracker.

    Accepts the old constructor signature
    ``IoUTracker(iou_threshold=0.3, max_age=10)``
    and maps it to ByteTracker defaults.  ``n_init=1`` preserves the
    original behaviour of publishing tracks immediately (no confirmation gate).
    """

    def __init__(self, iou_threshold=0.3, max_age=10, **kwargs):
        super().__init__(
            iou_threshold=iou_threshold,
            max_age=max_age,
            n_init=kwargs.pop('n_init', 1),
            pos_gate=kwargs.pop('pos_gate', 2.0),
            **kwargs,
        )

    @staticmethod
    def _compute_iou(bbox_a, bbox_b):
        """Compute IoU between two bounding boxes (x1, y1, x2, y2).

        Kept for backward-compatibility with existing tests.
        """
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
        area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0
