"""Flood-mask intersection analysis for pre-event OSM infrastructure."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union


def flood_polygons_from_mask(mask_path: str | Path) -> gpd.GeoDataFrame:
    """Polygonize valid flooded cells from a 0/1 raster mask."""
    with rasterio.open(mask_path) as source:
        mask = source.read(1)
        flooded = (mask == 1) & (mask != source.nodata)
        if not flooded.any():
            return gpd.GeoDataFrame({"value": []}, geometry=[], crs=source.crs)
        geometries = [
            shape(geometry)
            for geometry, value in shapes(mask, mask=flooded, transform=source.transform)
            if int(value) == 1
        ]
        return gpd.GeoDataFrame({"value": [1] * len(geometries)}, geometry=geometries, crs=source.crs)


def analyze_infrastructure(
    mask_path: str | Path,
    buildings_path: str | Path,
    roads_path: str | Path,
    output_dir: str | Path,
    bridges_path: str | Path | None = None,
) -> dict[str, object]:
    """Export flood polygons and OSM features intersecting the final flood mask."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    flood = flood_polygons_from_mask(mask_path)
    flood_path = output / "flood_polygons.geojson"
    flood.to_file(flood_path, driver="GeoJSON")

    buildings = gpd.read_file(buildings_path)
    roads = gpd.read_file(roads_path)
    bridges_provided = bridges_path is not None
    bridges = gpd.read_file(bridges_path) if bridges_provided else gpd.GeoDataFrame(geometry=[], crs=roads.crs)
    if flood.crs is None:
        raise ValueError("Flood mask has no CRS; geospatial intersections cannot be computed.")
    for name, frame in (("buildings", buildings), ("roads", roads), ("bridges", bridges)):
        if not frame.empty and frame.crs is None:
            raise ValueError(f"The {name} input has no CRS.")

    flooded_geometry = unary_union(flood.geometry) if not flood.empty else None
    def intersecting(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if frame.empty or flooded_geometry is None:
            result = frame.iloc[0:0].copy()
            result["potentially_damaged"] = pd.Series(dtype=bool)
            return result
        target = frame.to_crs(flood.crs) if frame.crs != flood.crs else frame.copy()
        result = target.loc[target.geometry.intersects(flooded_geometry)].copy()
        result["potentially_damaged"] = True
        return result.to_crs(frame.crs) if frame.crs != flood.crs else result

    damaged_buildings = intersecting(buildings)
    damaged_roads = intersecting(roads)
    damaged_bridges = intersecting(bridges)
    buildings_output = output / "damaged_buildings.geojson"
    roads_output = output / "damaged_roads.geojson"
    bridges_output = output / "damaged_bridges.geojson"
    damaged_buildings.to_file(buildings_output, driver="GeoJSON")
    damaged_roads.to_file(roads_output, driver="GeoJSON")
    damaged_bridges.to_file(bridges_output, driver="GeoJSON")

    road_km = 0.0
    if not damaged_roads.empty and flooded_geometry is not None:
        metric = damaged_roads.to_crs(damaged_roads.estimate_utm_crs())
        flood_metric = gpd.GeoSeries([flooded_geometry], crs=flood.crs).to_crs(metric.crs).iloc[0]
        road_km = float(metric.geometry.intersection(flood_metric).length.sum() / 1000)
    summary = {
        "potentially_damaged_buildings": int(len(damaged_buildings)),
        "potentially_disrupted_roads": int(len(damaged_roads)),
        "potentially_disrupted_road_km": round(road_km, 3),
        "bridges_intersecting_flood_mask": int(len(damaged_bridges)) if bridges_provided else None,
        "bridge_layer_provided": bridges_provided,
        "flood_polygon_count": int(len(flood)),
    }
    return {
        "flood_polygons": flood_path,
        "buildings": buildings_output,
        "roads": roads_output,
        "bridges": bridges_output,
        "summary": summary,
    }


def create_demo_geodata(output_dir: str | Path | None = None) -> dict[str, Path]:
    """Retained helper for generating clearly labeled, synthetic development fixtures."""
    from shapely.geometry import LineString, Point

    root = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[2] / "outputs" / "geojson"
    root.mkdir(parents=True, exist_ok=True)
    buildings = gpd.GeoDataFrame(
        {"id": ["b1", "b2", "b3"]},
        geometry=[Point(85.5, 28.1), Point(85.6, 28.0), Point(85.7, 28.2)],
        crs="EPSG:4326",
    )
    roads = gpd.GeoDataFrame(
        {"id": ["r1", "r2"], "highway": ["primary", "track"]},
        geometry=[
            LineString([(85.46, 28.12), (85.52, 28.11), (85.58, 28.10)]),
            LineString([(85.64, 28.22), (85.68, 28.20)]),
        ],
        crs="EPSG:4326",
    )
    buildings_path = root / "damaged_buildings_demo.geojson"
    roads_path = root / "damaged_roads_demo.geojson"
    buildings.to_file(buildings_path, driver="GeoJSON")
    roads.to_file(roads_path, driver="GeoJSON")
    return {"buildings": buildings_path, "roads": roads_path}


def summarize_damage(buildings: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return factual counts for already-intersected OSM layers."""
    road_km = 0.0
    if not roads.empty:
        metric = roads.to_crs(roads.estimate_utm_crs())
        road_km = float(metric.geometry.length.sum() / 1000)
    return pd.DataFrame([{
        "potentially_damaged_buildings": len(buildings),
        "potentially_disrupted_roads": len(roads),
        "potentially_disrupted_road_km": round(road_km, 3),
    }])
