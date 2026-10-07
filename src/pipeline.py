"""Orchestration for the file-based FloodLens analysis pipeline."""

from __future__ import annotations

from pathlib import Path
import warnings

import rasterio

from src.flood_mapping.sar import create_flood_products
from src.infrastructure.damage import analyze_infrastructure
from src.network_analysis.cutoff import analyze_cutoff_settlements
from src.reporting.summary import write_situation_report


def _orbit_number(path: str | Path) -> str | None:
    with rasterio.open(path) as source:
        tags = {key.casefold(): value for key, value in source.tags().items()}
    for key in ("relativeorbitnumber", "relative_orbit_number", "relative_orbit"):
        if key in tags:
            return tags[key]
    return None


def run_pipeline(
    *,
    before_sar: str | Path,
    after_sar: str | Path,
    output_dir: str | Path,
    event_date: str | None = None,
    inputs_are_db: bool = False,
    optical_before: str | Path | None = None,
    optical_after: str | Path | None = None,
    buildings: str | Path | None = None,
    roads: str | Path | None = None,
    bridges: str | Path | None = None,
    settlements: str | Path | None = None,
    hospitals: str | Path | None = None,
) -> dict[str, object]:
    """Run available analysis stages; missing optional vector layers are not fabricated."""
    output = Path(output_dir)
    processed = output / "processed"
    vectors = output / "geojson"
    reports = output / "reports"
    before_orbit = _orbit_number(before_sar)
    after_orbit = _orbit_number(after_sar)
    if before_orbit and after_orbit and before_orbit != after_orbit:
        raise ValueError(
            f"Sentinel-1 relative orbit mismatch ({before_orbit} vs {after_orbit}); "
            "these rasters must not be compared."
        )
    if not before_orbit or not after_orbit:
        warnings.warn(
            "Relative orbit metadata was not found in both SAR rasters; verify the pair manually.",
            RuntimeWarning,
            stacklevel=2,
        )
    products = create_flood_products(
        before_sar,
        after_sar,
        processed,
        inputs_are_db=inputs_are_db,
        optical_before_path=optical_before,
        optical_after_path=optical_after,
    )
    is_synthetic_fixture = any("demo" in Path(path).stem.casefold() for path in (before_sar, after_sar))

    result: dict[str, object] = {
        "products": products,
        "synthetic_fixture": is_synthetic_fixture,
        "orbit": {"before": before_orbit, "after": after_orbit},
        "infrastructure": None,
        "cutoff": None,
        "report": None,
        "warnings": [],
    }
    if buildings and roads:
        infrastructure = analyze_infrastructure(
            products["flood_mask"],
            buildings,
            roads,
            vectors,
            bridges_path=bridges,
        )
        result["infrastructure"] = infrastructure
    else:
        result["warnings"].append("Provide pre-event OSM buildings and roads GeoJSON to calculate infrastructure impacts.")

    if settlements and hospitals and roads:
        polygon_path = vectors / "flood_polygons.geojson"
        if not polygon_path.exists():
            from src.infrastructure.damage import flood_polygons_from_mask

            polygons = flood_polygons_from_mask(products["flood_mask"])
            polygon_path.parent.mkdir(parents=True, exist_ok=True)
            polygons.to_file(polygon_path, driver="GeoJSON")
        result["cutoff"] = analyze_cutoff_settlements(
            roads,
            settlements,
            hospitals,
            polygon_path,
            vectors / "cutoff_settlements.geojson",
        )
    else:
        result["warnings"].append("Provide pre-event roads, settlement points, and hospital points for cut-off analysis.")

    summary: dict[str, object] = {
        "Event date": event_date or "not specified",
        "Input status": (
            "SYNTHETIC SOFTWARE-TEST DATA — not observed satellite imagery."
            if is_synthetic_fixture
            else "User-supplied satellite rasters; source validity is not independently verified."
        ),
        "SAR inputs are in dB": inputs_are_db,
        "Sentinel-1 before relative orbit": before_orbit or "not embedded; manual verification required",
        "Sentinel-1 after relative orbit": after_orbit or "not embedded; manual verification required",
        "Final flood mask": str(products["flood_mask"]),
    }
    if result["infrastructure"] is not None:
        summary.update(result["infrastructure"]["summary"])
    else:
        summary["Infrastructure analysis"] = "Not run: pre-event OSM layers were not supplied."
    if result["cutoff"] is not None:
        summary["Cut-off settlements"] = result["cutoff"]["cutoff_count"]
        summary["Flood-intersecting OSM road IDs removed"] = ", ".join(result["cutoff"]["blocked_road_ids"]) or "none"
    else:
        summary["Cut-off analysis"] = "Not run: roads, settlements, and hospitals were not all supplied."
    summary["Limitations"] = (
        "Potential impact is inferred from raster overlap, not ground truth. "
        "Sentinel revisit timing, cloud cover, terrain effects, and OSM completeness constrain results."
    )
    report_path = write_situation_report(reports, **summary)
    result["report"] = report_path
    return result
