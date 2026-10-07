"""Settlement accessibility analysis on pre-event road geometries."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import split, unary_union


def _line_parts(geometry: object) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type in ("MultiLineString", "GeometryCollection"):
        return [part for item in geometry.geoms for part in _line_parts(item)]
    return []


def _build_graph(roads: gpd.GeoDataFrame, blocked_geometry: object | None) -> tuple[nx.Graph, list[dict[str, object]]]:
    graph = nx.Graph()
    blocked: list[dict[str, object]] = []
    segments: list[LineString] = []
    road_id_field = next((field for field in ("id", "osmid", "@osmId", "osm_id") if field in roads.columns), None)
    for index, row in roads.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        if blocked_geometry is not None and geometry.intersects(blocked_geometry):
            road_id = row[road_id_field] if road_id_field else index
            blocked.append({"road_id": str(road_id), "geometry": geometry})
        segments.extend(_line_parts(geometry))

    # Node line intersections, then remove only the individual edges touched by flood.
    noded = unary_union(segments) if segments else None
    for line in _line_parts(noded):
        pieces = [line]
        if blocked_geometry is not None and line.intersects(blocked_geometry.boundary):
            pieces = _line_parts(split(line, blocked_geometry.boundary))
        for piece in pieces:
            if blocked_geometry is not None and piece.intersects(blocked_geometry):
                continue
            coordinates = list(piece.coords)
            for start, end in zip(coordinates, coordinates[1:]):
                u = (round(float(start[0]), 3), round(float(start[1]), 3))
                v = (round(float(end[0]), 3), round(float(end[1]), 3))
                if u != v:
                    graph.add_edge(u, v, length=float(Point(u).distance(Point(v))))
    return graph, blocked


def analyze_cutoff_settlements(
    roads_path: str | Path,
    settlements_path: str | Path,
    hospitals_path: str | Path,
    flood_mask_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Remove flood-intersecting roads and export settlements disconnected from hospitals."""
    roads = gpd.read_file(roads_path)
    settlements = gpd.read_file(settlements_path)
    hospitals = gpd.read_file(hospitals_path)
    flood = gpd.read_file(Path(flood_mask_path))
    if roads.empty or settlements.empty or hospitals.empty:
        raise ValueError("Road, settlement, and hospital layers must each contain features.")
    if any(frame.crs is None for frame in (roads, settlements, hospitals, flood)):
        raise ValueError("All input vector layers must have a defined CRS.")
    target_crs = roads.estimate_utm_crs()
    roads = roads.to_crs(target_crs)
    settlements = settlements.to_crs(target_crs)
    hospitals = hospitals.to_crs(target_crs)
    flood = flood.to_crs(target_crs)
    flooded_geometry = unary_union(flood.geometry) if not flood.empty else None
    graph, blocked_roads = _build_graph(roads, flooded_geometry)
    if graph.number_of_nodes() == 0:
        raise ValueError("No routable road edges remain after removing flooded segments.")

    graph_nodes = list(graph.nodes)
    node_tree = cKDTree(np.asarray(graph_nodes, dtype=np.float64))
    hospital_nodes: list[tuple[tuple[float, float], Point]] = []
    for point in hospitals.geometry:
        if point is not None and not point.is_empty:
            _, nearest = node_tree.query((point.x, point.y))
            hospital_nodes.append((graph_nodes[int(nearest)], point))
    if not hospital_nodes:
        raise ValueError("Hospital layer has no usable point geometries.")

    records: list[dict[str, object]] = []
    name_field = next((field for field in ("name", "name:en", "ref") if field in settlements.columns), None)
    blocked_ids = sorted({road["road_id"] for road in blocked_roads})
    for index, row in settlements.iterrows():
        point = row.geometry
        if point is None or point.is_empty:
            continue
        _, nearest = node_tree.query((point.x, point.y))
        settlement_node = graph_nodes[int(nearest)]
        nearest_hospital, _ = min(hospital_nodes, key=lambda item: point.distance(item[1]))
        try:
            nx.shortest_path(graph, settlement_node, nearest_hospital, weight="length")
            can_reach_hospital = True
        except nx.NetworkXNoPath:
            can_reach_hospital = False
        if not can_reach_hospital:
            settlement_name = str(row[name_field]) if name_field and row[name_field] else f"Settlement {index}"
            why = ", ".join(blocked_ids) if blocked_ids else "no flood-intersecting road segment identified"
            records.append({
                "name": settlement_name,
                "reason": f"No route to nearest mapped hospital after removing potentially disrupted road(s): {why}",
                "geometry": point,
            })

    result = gpd.GeoDataFrame(records, geometry="geometry", crs=target_crs)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_crs("EPSG:4326").to_file(destination, driver="GeoJSON")
    return {
        "path": destination,
        "cutoff_count": len(result),
        "blocked_road_ids": blocked_ids,
    }


def identify_cutoff_settlements(settlement_nodes: Iterable[tuple[str, float, float]], disrupted_edges: list[tuple[str, str]]) -> list[dict[str, object]]:
    """Compatibility helper for simple in-memory graphs."""
    graph = nx.Graph()
    nodes = list(settlement_nodes)
    graph.add_nodes_from((name, {"x": x, "y": y}) for name, x, y in nodes)
    graph.add_edges_from(disrupted_edges)
    hospitals = [name for name, _, _ in nodes if name.lower().startswith("hospital")]
    result: list[dict[str, object]] = []
    for name, x, y in nodes:
        if name in hospitals:
            continue
        if not any(nx.has_path(graph, name, hospital) for hospital in hospitals):
            result.append({"settlement": name, "x": x, "y": y, "reason": "no road route to mapped hospital"})
    return result
