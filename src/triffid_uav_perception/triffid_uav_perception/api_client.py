"""
API Client
Fetches DJI drone images and telemetry from the FUTURISED platform.

1. **Media Files API** (``dji.getfuturised.com``)
   - Lists uploaded media files (images + video)
   - Provides temporary S3 download URLs
   - Auth: ``x-api-key`` header

2. **Telemetry API** (``api.getfuturised.com/getDJIData``)
   - Returns drone state records (position, heading, batteries, gimbal target)
   - Auth: ``Authorization: Bearer <token>``
"""

import json
import logging
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

log = logging.getLogger('triffid_uav.api')

# Default timeout for HTTP requests (seconds)
_HTTP_TIMEOUT = 30

# DJI file name suffixes → camera type
_CAMERA_SUFFIX = {
    '_W': 'Wide',
    '_Z': 'Zoom',
    '_T': 'Thermal',
    '_S': 'Split',   # screen recording / IR
}


@dataclass
class MediaFile:
    """A media file discovered via the FUTURISED Media Files API."""
    id: str
    name: str
    uploaded_at: int         # epoch millis
    path: str                # DJI album path
    camera: str              # Wide / Zoom / Thermal / Split / Unknown
    extension: str           # .JPG / .MP4 / etc.
    metadata: Optional[dict] = None  # sparse metadata from the API
    download_url: Optional[str] = None


class FuturisedClient:
    """Client for the FUTURISED DJI image and telemetry APIs.

    Parameters
    ----------
    media_api_key : str
        API key for ``dji.getfuturised.com`` (``x-api-key`` header).
    org_id : str
        Organisation UUID for the media files endpoint.
    media_base_url : str
        Base URL for the media API.
    telemetry_token : str, optional
        Bearer token for ``api.getfuturised.com/getDJIData``.
    telemetry_project : str
        Project name in the telemetry path (e.g. ``Triffid_test``).
    telemetry_base_url : str
        Base URL for the telemetry API.
    download_dir : str or Path
        Local directory where downloaded images are saved.
    """

    def __init__(
        self,
        media_api_key: str,
        org_id: str = '66f9f3ae-cd33-4313-b474-ae24e923a185',
        media_base_url: str = 'https://dji.getfuturised.com/api/v0',
        telemetry_token: Optional[str] = None,
        telemetry_project: str = 'Triffid_test',
        telemetry_base_url: str = 'https://api.getfuturised.com',
        download_dir: str = './uav_images',
    ):
        self._media_key = media_api_key
        self._org_id = org_id
        self._media_base = media_base_url.rstrip('/')
        self._telem_token = telemetry_token
        self._telem_project = telemetry_project
        self._telem_base = telemetry_base_url.rstrip('/')
        self._download_dir = Path(download_dir)
        self._download_dir.mkdir(parents=True, exist_ok=True)
        self._ssl_ctx = ssl.create_default_context()
        # Track already-processed file IDs to avoid re-downloading
        self._seen_ids: Set[str] = set()

    # ── Media Files API ─────────────────────────────────────────

    def list_media(self, uploaded_after: int = 0) -> List[MediaFile]:
        """List available media files from the FUTURISED cloud.

        Parameters
        ----------
        uploaded_after : int
            Unix timestamp in milliseconds. Only files uploaded after this
            time are returned. Use 0 to list all files.

        Returns
        -------
        list of MediaFile
        """
        url = (
            f'{self._media_base}/organization/{self._org_id}'
            f'/dji_media_files?uploaded_after={uploaded_after}'
        )
        data = self._get_json(url, headers={'x-api-key': self._media_key})
        if data is None:
            return []

        files = []
        for entry in data:
            name = entry.get('name', '')
            ext = Path(name).suffix.upper()
            camera = 'Unknown'
            for suffix, cam in _CAMERA_SUFFIX.items():
                if suffix in Path(name).stem:
                    camera = cam
                    break

            files.append(MediaFile(
                id=entry['id'],
                name=name,
                uploaded_at=entry.get('uploaded_at', 0),
                path=entry.get('path', ''),
                camera=camera,
                extension=ext,
                metadata=entry.get('metadata'),
            ))

        return files

    def get_file_details(self, file_id: str) -> Optional[MediaFile]:
        """Fetch metadata and a temporary download URL for a media file.

        Parameters
        ----------
        file_id : str
            UUID of the media file (from ``list_media``).

        Returns
        -------
        MediaFile with ``download_url`` populated, or None on error.
        """
        url = (
            f'{self._media_base}/organization/{self._org_id}'
            f'/dji_media_files/{file_id}'
        )
        data = self._get_json(url, headers={'x-api-key': self._media_key})
        if data is None:
            return None

        name = data.get('name', '')
        ext = Path(name).suffix.upper()
        camera = 'Unknown'
        for suffix, cam in _CAMERA_SUFFIX.items():
            if suffix in Path(name).stem:
                camera = cam
                break

        return MediaFile(
            id=data['id'],
            name=name,
            uploaded_at=data.get('uploaded_at', 0),
            path=data.get('path', ''),
            camera=camera,
            extension=ext,
            metadata=data.get('metadata'),
            download_url=data.get('url'),
        )

    def download_image(self, file_id: str) -> Optional[Path]:
        """Download a media file to the local download directory.

        Fetches a fresh temporary S3 URL each time (URLs expire after ~1h).
        Skips download if the file already exists locally with the same name.

        Returns the local file path, or None on failure.
        """
        details = self.get_file_details(file_id)
        if details is None or details.download_url is None:
            log.error(f'Could not get download URL for {file_id}')
            return None

        local_path = self._download_dir / details.name
        if local_path.exists():
            log.debug(f'Already downloaded: {local_path}')
            self._seen_ids.add(file_id)
            return local_path

        log.info(f'Downloading {details.name} ...')
        try:
            req = urllib.request.Request(details.download_url)
            resp = urllib.request.urlopen(
                req, context=self._ssl_ctx, timeout=_HTTP_TIMEOUT,
            )
            img_bytes = resp.read()
            local_path.write_bytes(img_bytes)
            self._seen_ids.add(file_id)
            log.info(f'  Saved {len(img_bytes)} bytes → {local_path}')
            return local_path
        except (urllib.error.URLError, OSError) as exc:
            log.error(f'Download failed for {details.name}: {exc}')
            return None

    def poll_new_images(
        self,
        camera_filter: str = 'Wide',
        extensions: Set[str] = frozenset({'.JPG', '.JPEG', '.TIFF', '.TIF'}),
    ) -> List[Path]:
        """Check for new image files and download them.

        Filters by camera type and file extension. Skips files already
        downloaded in this session.

        Parameters
        ----------
        camera_filter : str
            Only download files from this camera (e.g. ``'Wide'``).
            Use ``''`` or ``None`` to accept all cameras.
        extensions : set of str
            Allowed file extensions (uppercase, with dot).

        Returns
        -------
        list of Path
            Local paths of newly downloaded images.
        """
        all_files = self.list_media()

        candidates = [
            f for f in all_files
            if f.extension in extensions
            and f.id not in self._seen_ids
            and (not camera_filter or f.camera == camera_filter)
        ]

        if not candidates:
            log.debug('No new images found.')
            return []

        log.info(f'Found {len(candidates)} new image(s) to download.')
        downloaded = []
        for mf in candidates:
            path = self.download_image(mf.id)
            if path is not None:
                downloaded.append(path)

        return downloaded

    # ── Telemetry API ───────────────────────────────────────────

    def get_telemetry(self, count: int = 1) -> Optional[List[dict]]:
        """Fetch recent telemetry records from the getDJIData endpoint.

        Parameters
        ----------
        count : int
            Number of recent records to retrieve. Use ``1`` for latest only.

        Returns
        -------
        list of dicts (raw JSON), or None on error.

        Notes
        -----
        The telemetry API uses a different auth token (Bearer) and returns
        drone state data (position, heading, battery, gimbal target).
        Coordinate values may use European number formatting (commas as
        decimal separators, scientific notation).
        """
        if self._telem_token is None:
            log.warning('Telemetry token not configured.')
            return None

        filt = 'latest' if count == 1 else str(count)
        url = f'{self._telem_base}/getDJIData/{self._telem_project}/{filt}'
        headers = {'Authorization': f'Bearer {self._telem_token}'}
        return self._get_json(url, headers=headers)

    @staticmethod
    def parse_telemetry_coord(value: str) -> Optional[float]:
        """Best-effort parse of a telemetry coordinate value.

        The getDJIData endpoint returns coordinates in inconsistent formats:
        European comma-decimal, scientific notation, or plain integers.
        Some values are scaled by large factors (÷ 10^13 or 10^14).

        This method attempts to normalise the value to degrees. Returns
        None if the value cannot be interpreted.
        """
        if not value or value == '0':
            return None

        # Replace European comma-decimal
        cleaned = value.replace(',', '.')

        try:
            num = float(cleaned)
        except ValueError:
            return None

        abs_val = abs(num)

        # Already looks like degrees (plausible lat/lon range)
        if abs_val < 180:
            return num

        # Try common DJI scaling factors
        for divisor in (1e14, 1e13, 1e7):
            candidate = num / divisor
            if abs(candidate) < 180:
                return candidate

        return None

    # ── HTTP helpers ────────────────────────────────────────────

    def _get_json(self, url: str,
                  headers: Optional[Dict[str, str]] = None) -> Optional[object]:
        """HTTP GET returning parsed JSON, or None on error."""
        req = urllib.request.Request(url)
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        try:
            resp = urllib.request.urlopen(
                req, context=self._ssl_ctx, timeout=_HTTP_TIMEOUT,
            )
            return json.loads(resp.read())
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            log.error(f'API request failed: {url!r} → {exc}')
            return None
