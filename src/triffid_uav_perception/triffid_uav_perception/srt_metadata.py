"""
DJI SRT Telemetry Parser
"""

import bisect
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger('triffid_uav.srt')


# ── Regexes ──────────────────────────────────────────────────────────

_TIMECODE_RE = re.compile(
    r'(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->'
)
_FONT_TAG_RE = re.compile(r'</?font[^>]*>')
_FRAMECNT_RE = re.compile(r'FrameCnt\s*[:=]?\s*(\d+)', re.IGNORECASE)

# Modern bracketed keys: [latitude: 49.72] / [rel_alt: 36.2 abs_alt: 431.5]
# The bracket grouping varies between models, so match key/value pairs
# anywhere rather than whole brackets.
_KV_RE = re.compile(
    r'(latitude|longitude|rel_alt|abs_alt|altitude'
    r'|gb_yaw|gb_pitch|gb_roll)\s*[:=]\s*(-?\d+(?:\.\d+)?)',
    re.IGNORECASE,
)

# Legacy: GPS(lon,lat,alt) — DJI order is (longitude, latitude, altitude)
_GPS_RE = re.compile(
    r'GPS\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)'
    r'\s*(?:,\s*(-?\d+(?:\.\d+)?))?\s*\)',
    re.IGNORECASE,
)
# Legacy relative altitude: "H 36.2m"
_H_RE = re.compile(r'\bH\s+(-?\d+(?:\.\d+)?)\s*m\b', re.IGNORECASE)


@dataclass
class SrtFrameMeta:
    """Telemetry for one video frame, scraped from one SRT block."""
    frame_index: int            # 0-based, aligned with cv2 frame numbers
    timestamp_s: float          # block start time, seconds from video start
    lat: Optional[float] = None
    lon: Optional[float] = None
    abs_alt: Optional[float] = None
    rel_alt: Optional[float] = None
    gimbal_yaw: Optional[float] = None
    gimbal_pitch: Optional[float] = None
    gimbal_roll: Optional[float] = None

    @property
    def has_position(self) -> bool:
        return self.lat is not None and self.lon is not None


def _parse_block(block: str) -> Optional[SrtFrameMeta]:
    """Parse one SRT block (index line + timecode + payload). Returns
    None when the block has no usable timecode or no telemetry at all."""
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    # Locate the timecode line (usually line 2, after the numeric index).
    tc_match = None
    tc_line_i = None
    for i, ln in enumerate(lines[:3]):
        tc_match = _TIMECODE_RE.search(ln)
        if tc_match:
            tc_line_i = i
            break
    if tc_match is None:
        return None

    h, m, s, ms = tc_match.groups()
    timestamp_s = int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, '0')) / 1000.0

    # SRT block index (the line before the timecode, when numeric).
    block_index = None
    if tc_line_i and tc_line_i >= 1 and lines[tc_line_i - 1].isdigit():
        block_index = int(lines[tc_line_i - 1])

    payload = ' '.join(lines[tc_line_i + 1:])
    payload = _FONT_TAG_RE.sub(' ', payload)

    meta = SrtFrameMeta(frame_index=-1, timestamp_s=round(timestamp_s, 3))

    fc = _FRAMECNT_RE.search(payload)
    if fc:
        # DJI FrameCnt is 1-based; cv2 frame indices are 0-based.
        meta.frame_index = int(fc.group(1)) - 1
    elif block_index is not None:
        meta.frame_index = block_index - 1

    found_any = False
    for key, value in _KV_RE.findall(payload):
        key = key.lower()
        value = float(value)
        found_any = True
        if key == 'latitude':
            meta.lat = value
        elif key == 'longitude':
            meta.lon = value
        elif key in ('abs_alt', 'altitude'):
            meta.abs_alt = value
        elif key == 'rel_alt':
            meta.rel_alt = value
        elif key == 'gb_yaw':
            meta.gimbal_yaw = value
        elif key == 'gb_pitch':
            meta.gimbal_pitch = value
        elif key == 'gb_roll':
            meta.gimbal_roll = value

    if meta.lat is None:
        gps = _GPS_RE.search(payload)
        if gps:
            found_any = True
            meta.lon = float(gps.group(1))
            meta.lat = float(gps.group(2))
            if gps.group(3) is not None:
                meta.abs_alt = float(gps.group(3))
    if meta.rel_alt is None:
        h_match = _H_RE.search(payload)
        if h_match:
            found_any = True
            meta.rel_alt = float(h_match.group(1))

    if not found_any:
        return None
    if meta.frame_index < 0:
        # No FrameCnt and no numeric block index — keep the record, the
        # index will fall back to the block's position (set by caller).
        pass
    return meta


def parse_srt(text: str) -> List[SrtFrameMeta]:
    """Parse a whole SRT file's text into per-frame telemetry records.

    Malformed or telemetry-free blocks are skipped. Records without an
    explicit frame number get their sequential block position.
    """
    records: List[SrtFrameMeta] = []
    for pos, block in enumerate(re.split(r'\n\s*\n', text)):
        if not block.strip():
            continue
        meta = _parse_block(block)
        if meta is None:
            continue
        if meta.frame_index < 0:
            meta.frame_index = pos
        records.append(meta)
    return records


class SrtIndex:
    """Frame-indexed lookup over parsed SRT telemetry.

    ``at(frame_index)`` returns the exact frame's record when present,
    else the record nearest in time (SRT blocks are usually one per
    frame, but some models emit them at a lower rate).
    """

    def __init__(self, records: List[SrtFrameMeta]):
        self._by_frame = {r.frame_index: r for r in records}
        self._sorted = sorted(records, key=lambda r: r.timestamp_s)
        self._times = [r.timestamp_s for r in self._sorted]

    @classmethod
    def from_file(cls, path) -> 'SrtIndex':
        text = Path(path).read_text(errors='replace')
        return cls(parse_srt(text))

    def __len__(self) -> int:
        return len(self._sorted)

    def at(self, frame_index: int,
           timestamp_s: Optional[float] = None) -> Optional[SrtFrameMeta]:
        """Exact frame lookup, falling back to nearest-by-time."""
        exact = self._by_frame.get(frame_index)
        if exact is not None:
            return exact
        if timestamp_s is None or not self._sorted:
            return None
        i = bisect.bisect_left(self._times, timestamp_s)
        candidates = []
        if i > 0:
            candidates.append(self._sorted[i - 1])
        if i < len(self._sorted):
            candidates.append(self._sorted[i])
        return min(candidates,
                   key=lambda r: abs(r.timestamp_s - timestamp_s))


def find_sidecar_srt(video_path) -> Optional[Path]:
    """Return ``<video-stem>.srt``/``.SRT`` next to the video, if any."""
    video_path = Path(video_path)
    for suffix in ('.srt', '.SRT'):
        candidate = video_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None
