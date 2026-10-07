"""Utilities for project configuration and SAR data workflow scaffolding."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

TRISHULI_AOI = {
    "min_lat": 27.8,
    "max_lat": 28.3,
    "min_lon": 85.1,
    "max_lon": 85.8,
    "event_date": "2026-08-26",
    "before_date": "2026-08-14",
    "after_date": "2026-09-05",
}


def ensure_repo_layout(root: str | Path | None = None) -> dict[str, Path]:
    """Create the expected project structure and return paths keyed by purpose."""
    root_path = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    directories = {
        "root": root_path,
        "data_raw": root_path / "data" / "raw",
        "sentinel1_before": root_path / "data" / "raw" / "sentinel1" / "before",
        "sentinel1_after": root_path / "data" / "raw" / "sentinel1" / "after",
        "sentinel2": root_path / "data" / "raw" / "sentinel2",
        "dem": root_path / "data" / "raw" / "dem",
        "osm": root_path / "data" / "raw" / "osm",
        "processed": root_path / "data" / "processed",
        "training": root_path / "data" / "training",
        "outputs_maps": root_path / "outputs" / "maps",
        "outputs_geojson": root_path / "outputs" / "geojson",
        "outputs_reports": root_path / "outputs" / "reports",
        "src": root_path / "src",
        "app": root_path / "app",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def download_ohsome_snapshot(
    bbox: tuple[float, float, float, float],
    snapshot_date: str,
    output_dir: str | Path,
    timeout: int = 120,
) -> dict[str, Path]:
    """Download historical OSM building, road, and bridge layers from ohsome."""
    west, south, east, north = bbox
    if west >= east or south >= north:
        raise ValueError("bbox must be ordered west, south, east, north.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    endpoint = "https://api.ohsome.org/v1/elements/geometry"
    queries = {
        "buildings": "building=*",
        "roads": "highway=*",
        "bridges": "bridge=*",
    }
    saved: dict[str, Path] = {}
    for layer, element_filter in queries.items():
        params = urlencode({
            "bboxes": f"{west},{south},{east},{north}",
            "time": snapshot_date,
            "filter": element_filter,
            "properties": "tags",
            "format": "geojson",
        })
        request = Request(f"{endpoint}?{params}", headers={"User-Agent": "FloodLens/1.0"})
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"ohsome {layer} snapshot request failed: {exc}") from exc
        destination = root / f"osm_{layer}_{snapshot_date}.geojson"
        destination.write_text(json.dumps(payload), encoding="utf-8")
        saved[layer] = destination
    return saved
