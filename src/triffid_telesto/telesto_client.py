"""TELESTO Map Manager API client.

Provides ``TelestoClient`` for CRUD operations on GeoJSON features hosted
at the TELESTO backend (WordPress/map-manager REST API).

Endpoints
---------
- GET    /features           → FeatureCollection
- PUT    /features           → create a new Feature (returns with server ID)
- PATCH  /features/{id}      → update an existing Feature
- DELETE /features/{id}      → remove a Feature

Observer (sync status):
- GET    /observer-sync/v1/status
- PATCH  /observer-sync/v1/status

Uses stdlib only (no ``requests`` dependency).
"""

from __future__ import annotations

import json
import logging
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Spatial helpers (used by accumulate_collection)

def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    import math as _math
    r = 6378137.0
    d_lat = _math.radians(lat2 - lat1)
    d_lon = _math.radians(lon2 - lon1)
    a = (_math.sin(d_lat / 2) ** 2
         + _math.cos(_math.radians(lat1)) * _math.cos(_math.radians(lat2))
         * _math.sin(d_lon / 2) ** 2)
    return r * 2.0 * _math.asin(_math.sqrt(a))


def _feature_centroid(feature: dict):
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


# Defaults
_DEFAULT_BASE = 'https://crispres.com/wp-json/map-manager/v1'
_DEFAULT_OBSERVER = 'https://crispres.com/wp-json/observer-sync/v1'
_TIMEOUT = 15  # seconds


def _make_ssl_ctx() -> ssl.SSLContext:
    """Create a default SSL context for HTTPS requests."""
    ctx = ssl.create_default_context()
    return ctx


def _request(
    url: str,
    method: str = 'GET',
    body: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = _TIMEOUT,
) -> dict:
    """Execute an HTTP request and return parsed JSON response.

    Raises ``TelestoError`` on HTTP errors or JSON parse failures.
    """
    hdrs = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    if headers:
        hdrs.update(headers)

    data = None
    if body is not None:
        data = json.dumps(body).encode('utf-8')

    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_make_ssl_ctx()) as resp:
            raw = resp.read().decode('utf-8')
            if not raw.strip():
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        body_text = ''
        try:
            body_text = e.read().decode('utf-8', errors='replace')
        except Exception:
            pass
        raise TelestoError(
            f'{method} {url} → HTTP {e.code}: {body_text[:300]}'
        ) from e
    except urllib.error.URLError as e:
        raise TelestoError(f'{method} {url} → {e.reason}') from e
    except json.JSONDecodeError as e:
        raise TelestoError(f'{method} {url} → invalid JSON: {e}') from e


class TelestoError(Exception):
    """Raised on TELESTO API errors."""


class TelestoClient:
    """Client for the TELESTO Map Manager REST API.

    Parameters
    ----------
    base_url : str
        Map Manager endpoint (default: ``https://crispres.com/wp-json/map-manager/v1``).
    observer_url : str
        Observer Sync endpoint (default: ``https://crispres.com/wp-json/observer-sync/v1``).
    timeout : int
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE,
        observer_url: str = _DEFAULT_OBSERVER,
        timeout: int = _TIMEOUT,
    ):
        self.base_url = base_url.rstrip('/')
        self.observer_url = observer_url.rstrip('/')
        self.timeout = timeout
        # Track remote feature IDs we've created, keyed by (source, local_id)
        self._remote_ids: Dict[tuple, str] = {}

    # Feature CRUD

    def get_features(self) -> dict:
        """GET all features.  Returns the full FeatureCollection."""
        return _request(
            f'{self.base_url}/features',
            timeout=self.timeout,
        )

    def put_feature(self, feature: dict) -> dict:
        """PUT (create) a single feature.

        Parameters
        ----------
        feature : dict
            A GeoJSON Feature dict (must have ``geometry`` and ``properties``).

        Returns
        -------
        dict
            The created Feature as returned by the server (includes server ``id``).
        """
        payload = {
            'geometry': feature['geometry'],
            'properties': feature.get('properties', {}),
        }
        return _request(
            f'{self.base_url}/features',
            method='PUT',
            body=payload,
            timeout=self.timeout,
        )

    def patch_feature(self, feature_id: str, feature: dict) -> dict:
        """PATCH (update) an existing feature by its server-assigned ID.

        Parameters
        ----------
        feature_id : str
            The TELESTO feature ID (e.g. ``feature_6994dac99fc5a3.42572521``).
        feature : dict
            Updated geometry/properties.
        """
        payload = {
            'geometry': feature['geometry'],
            'properties': feature.get('properties', {}),
        }
        return _request(
            f'{self.base_url}/features/{feature_id}',
            method='PATCH',
            body=payload,
            timeout=self.timeout,
        )

    def delete_feature(self, feature_id: str) -> dict:
        """DELETE a feature by its server-assigned ID."""
        return _request(
            f'{self.base_url}/features/{feature_id}',
            method='DELETE',
            timeout=self.timeout,
        )

    # Bulk operations

    def upload_collection(self, collection: dict) -> List[dict]:
        """Upload every feature in a FeatureCollection via PUT.

        Returns
        -------
        list[dict]
            List of server responses for each created feature.
        """
        results = []
        for feature in collection.get('features', []):
            try:
                resp = self.put_feature(feature)
                source = feature.get('properties', {}).get('source', '')
                local_id = feature.get('properties', {}).get('id', '')
                if resp.get('id'):
                    self._remote_ids[(source, local_id)] = resp['id']
                results.append(resp)
            except TelestoError as e:
                log.error(f'PUT failed for feature: {e}')
                results.append({'error': str(e)})
        return results

    def sync_collection(
        self,
        collection: dict,
        source: Optional[str] = None,
    ) -> dict:
        """Smart sync: upload features and remove stale remote ones.

        1. GET current remote features
        2. Filter by ``source`` (if given) to find our previously-uploaded set
        3. PUT new features / PATCH changed features
        4. DELETE remote features that are no longer in the local set

        Parameters
        ----------
        collection : dict
            GeoJSON FeatureCollection to sync.
        source : str, optional
            Filter remote features by ``properties.source`` (e.g. ``"ugv"``).
            If None, uses the source from the first feature in the collection.

        Returns
        -------
        dict
            Summary: ``{"created": N, "updated": N, "deleted": N, "errors": N}``.
        """
        stats = {'created': 0, 'updated': 0, 'deleted': 0, 'errors': 0}
        local_features = collection.get('features', [])

        if source is None and local_features:
            source = local_features[0].get('properties', {}).get('source')

        local_ids = set()
        for f in local_features:
            lid = f.get('properties', {}).get('id', '')
            local_ids.add(lid)

        # GET remote features
        try:
            remote = self.get_features()
        except TelestoError as e:
            log.error(f'Failed to GET remote features: {e}')
            results = self.upload_collection(collection)
            stats['created'] = sum(1 for r in results if 'error' not in r)
            stats['errors'] = sum(1 for r in results if 'error' in r)
            return stats

        # Index remote features by local ID (from properties.id)
        remote_by_local_id: Dict[str, dict] = {}
        remote_our_ids: Dict[str, str] = {}  # local_id → remote server ID
        for rf in remote.get('features', []):
            rf_source = rf.get('properties', {}).get('source', '')
            if source and rf_source != source:
                continue
            rf_local_id = rf.get('properties', {}).get('id', '')
            rf_server_id = rf.get('id', '')
            if rf_local_id:
                remote_by_local_id[rf_local_id] = rf
                remote_our_ids[rf_local_id] = rf_server_id

        # Create or update
        for f in local_features:
            lid = f.get('properties', {}).get('id', '')
            try:
                if lid in remote_our_ids:
                    self.patch_feature(remote_our_ids[lid], f)
                    stats['updated'] += 1
                else:
                    resp = self.put_feature(f)
                    if resp.get('id'):
                        self._remote_ids[(source or '', lid)] = resp['id']
                    stats['created'] += 1
            except TelestoError as e:
                log.error(f'Sync failed for feature {lid}: {e}')
                stats['errors'] += 1

        # Delete stale features
        for remote_lid, remote_sid in remote_our_ids.items():
            if remote_lid not in local_ids:
                try:
                    self.delete_feature(remote_sid)
                    stats['deleted'] += 1
                except TelestoError as e:
                    log.error(f'DELETE failed for {remote_sid}: {e}')
                    stats['errors'] += 1

        return stats

    def accumulate_collection(
        self,
        collection: dict,
        radius_m: float = 10.0,
    ) -> dict:
        """Upgrade-or-insert sync: never delete remote features.

        For each local feature:
        - ``local_frame=True``: always PUT (body-frame coords, no geo-compare)
        - No nearby remote within *radius_m* metres: PUT (new discovery)
        - Nearby remote with lower confidence: PATCH (better reading)
        - Nearby remote with same/higher confidence: skip

        Returns
        -------
        dict
            Summary: ``{"created": N, "updated": N, "skipped": N, "errors": N}``.
        """
        stats: Dict[str, int] = {
            'created': 0, 'updated': 0, 'skipped': 0, 'errors': 0,
        }
        local_features = collection.get('features', [])
        if not local_features:
            return stats

        try:
            remote_coll = self.get_features()
            remote_features = remote_coll.get('features', [])
        except TelestoError as e:
            log.error(f'GET remote features failed, falling back to blind PUT: {e}')
            results = self.upload_collection(collection)
            stats['created'] = sum(1 for r in results if 'error' not in r)
            stats['errors'] = sum(1 for r in results if 'error' in r)
            return stats

        for local_f in local_features:
            props = local_f.get('properties', {})
            local_conf = float(props.get('confidence', 0.0))
            local_centroid = _feature_centroid(local_f)

            try:
                if props.get('local_frame', False) or local_centroid is None:
                    resp = self.put_feature(local_f)
                    if resp.get('id'):
                        self._remote_ids[
                            (props.get('source', ''), props.get('id', ''))
                        ] = resp['id']
                    stats['created'] += 1
                    continue

                # Find nearest remote feature (any class)
                best_remote = None
                best_dist = float('inf')
                for rf in remote_features:
                    rc = _feature_centroid(rf)
                    if rc is None:
                        continue
                    d = _haversine_m(
                        local_centroid[0], local_centroid[1], rc[0], rc[1],
                    )
                    if d < radius_m and d < best_dist:
                        best_dist = d
                        best_remote = rf

                if best_remote is None:
                    resp = self.put_feature(local_f)
                    if resp.get('id'):
                        self._remote_ids[
                            (props.get('source', ''), props.get('id', ''))
                        ] = resp['id']
                    stats['created'] += 1
                else:
                    remote_conf = float(
                        best_remote.get('properties', {}).get('confidence', 0.0)
                    )
                    if local_conf > remote_conf:
                        remote_id = best_remote.get('id', '')
                        if remote_id:
                            self.patch_feature(remote_id, local_f)
                            stats['updated'] += 1
                        else:
                            self.put_feature(local_f)
                            stats['created'] += 1
                    else:
                        stats['skipped'] += 1

            except TelestoError as e:
                log.error(f'accumulate_collection: feature error: {e}')
                stats['errors'] += 1

        return stats

    def clear_source(self, source: str) -> int:
        """Delete all remote features with the given source.

        Returns the number of successfully deleted features.
        """
        try:
            remote = self.get_features()
        except TelestoError:
            return 0

        deleted = 0
        for rf in remote.get('features', []):
            if rf.get('properties', {}).get('source') == source:
                try:
                    self.delete_feature(rf['id'])
                    deleted += 1
                except TelestoError as e:
                    log.error(f'DELETE failed: {e}')
        return deleted

    # Observer API

    def get_observer_status(self) -> dict:
        """GET observer sync status."""
        return _request(
            f'{self.observer_url}/status',
            timeout=self.timeout,
        )

    def notify_observer(self, **kwargs) -> dict:
        """PATCH observer status to signal data has been updated.

        Common fields: ``fe_updated``, ``mobile_updated``, ``ar_updated``
        (set to 1 to notify, 0 to clear).

        Example::

            client.notify_observer(fe_updated=1)
        """
        return _request(
            f'{self.observer_url}/status',
            method='PATCH',
            body=kwargs,
            timeout=self.timeout,
        )
