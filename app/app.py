"""Streamlit interface for the real-file FloodLens pipeline."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import folium
import numpy as np
import rasterio
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import HeatMap
from rasterio.warp import transform as reproject_coordinates

from src.data_ingestion.sentinel import TRISHULI_AOI, download_ohsome_snapshot, ensure_repo_layout
from src.infrastructure.damage import flood_polygons_from_mask
from src.pipeline import run_pipeline
from src.reporting.summary import create_pdf_report, translate_report_to_nepali


def _uncertainty_heat_points(path: str) -> list[list[float]]:
    with rasterio.open(path) as uncertainty_src:
        if uncertainty_src.crs is None:
            raise ValueError("Uncertainty raster must have a defined CRS.")
        uncertainty = uncertainty_src.read(1, masked=True).astype(np.float32).filled(0)
        stride = max(1, int(np.ceil(max(uncertainty.shape) / 100)))
        rows, columns = np.mgrid[0:uncertainty.shape[0]:stride, 0:uncertainty.shape[1]:stride]
        weights = uncertainty[::stride, ::stride]
        valid_uncertainty = np.isfinite(weights) & (weights > 0)
        x_values, y_values = rasterio.transform.xy(
            uncertainty_src.transform,
            rows[valid_uncertainty].ravel(),
            columns[valid_uncertainty].ravel(),
            offset="center",
        )
        latitudes, longitudes = reproject_coordinates(
            uncertainty_src.crs,
            "EPSG:4326",
            list(x_values),
            list(y_values),
        )
        return [
            [latitude, longitude, float(weight)]
            for latitude, longitude, weight in zip(
                latitudes,
                longitudes,
                weights[valid_uncertainty].ravel(),
            )
        ]


st.set_page_config(page_title="FloodLens", layout="wide")
st.title("FloodLens")
st.caption("Evidence-led flood extent and accessibility analysis. Outputs are estimates, not confirmed damage.")

paths = ensure_repo_layout()
sidebar = st.sidebar
sidebar.header("Analysis inputs")
min_lat = sidebar.number_input("Minimum latitude", value=TRISHULI_AOI["min_lat"], step=0.01)
max_lat = sidebar.number_input("Maximum latitude", value=TRISHULI_AOI["max_lat"], step=0.01)
min_lon = sidebar.number_input("Minimum longitude", value=TRISHULI_AOI["min_lon"], step=0.01)
max_lon = sidebar.number_input("Maximum longitude", value=TRISHULI_AOI["max_lon"], step=0.01)
event_date = sidebar.date_input("Flood event date", value=date.fromisoformat(TRISHULI_AOI["event_date"]))
before_path = sidebar.text_input("Sentinel-1 before GeoTIFF")
after_path = sidebar.text_input("Sentinel-1 after GeoTIFF")
inputs_are_db = sidebar.checkbox("SAR rasters contain dB values", value=False)
optical_before = sidebar.text_input("Sentinel-2 before GeoTIFF (optional)")
optical_after = sidebar.text_input("Sentinel-2 after GeoTIFF (optional)")
osm_buildings = sidebar.text_input("Pre-event OSM buildings GeoJSON")
osm_roads = sidebar.text_input("Pre-event OSM roads GeoJSON")
osm_bridges = sidebar.text_input("Pre-event OSM bridges GeoJSON (optional)")
osm_settlements = sidebar.text_input("Pre-event settlement points GeoJSON (optional)")
osm_hospitals = sidebar.text_input("Pre-event hospital points GeoJSON (optional)")
uncertainty_path = sidebar.text_input("Model uncertainty GeoTIFF (optional)")
uncertainty_opacity = sidebar.slider("Uncertainty heatmap opacity", min_value=0.1, max_value=1.0, value=0.65)

if sidebar.button("Download pre-event OSM snapshot"):
    if min_lat >= max_lat or min_lon >= max_lon:
        sidebar.error("Bounding box coordinates are invalid.")
    else:
        try:
            downloaded = download_ohsome_snapshot(
                (min_lon, min_lat, max_lon, max_lat),
                "2026-07-27",
                paths["osm"],
            )
            for layer, file_path in downloaded.items():
                sidebar.success(f"{layer}: {file_path}")
        except (RuntimeError, ValueError) as exc:
            sidebar.error(str(exc))

if sidebar.button("Run Pipeline", type="primary"):
    if not before_path.strip() or not after_path.strip():
        st.error("Select both processed Sentinel-1 before and after GeoTIFFs.")
    elif bool(optical_before.strip()) != bool(optical_after.strip()):
        st.error("Select both Sentinel-2 dates, or leave both optical inputs empty.")
    else:
        try:
            result = run_pipeline(
                before_sar=before_path.strip(),
                after_sar=after_path.strip(),
                output_dir=Path(__file__).resolve().parents[1] / "outputs",
                event_date=event_date.isoformat(),
                inputs_are_db=inputs_are_db,
                optical_before=optical_before.strip() or None,
                optical_after=optical_after.strip() or None,
                buildings=osm_buildings.strip() or None,
                roads=osm_roads.strip() or None,
                bridges=osm_bridges.strip() or None,
                settlements=osm_settlements.strip() or None,
                hospitals=osm_hospitals.strip() or None,
            )
            st.session_state["pipeline_result"] = result
            st.success(f"Pipeline completed. Report: {result['report']}")
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Pipeline failed: {exc}")

result = st.session_state.get("pipeline_result")
if result:
    products = result["products"]
    mask_path = products["flood_mask"]
    polygon_path = Path(result["report"]).parents[1] / "geojson" / "flood_polygons.geojson"
    if not polygon_path.exists():
        polygons = flood_polygons_from_mask(mask_path)
        polygon_path.parent.mkdir(parents=True, exist_ok=True)
        polygons.to_file(polygon_path, driver="GeoJSON")

    left, right = st.columns([2, 1])
    with left:
        flood = json.loads(polygon_path.read_text(encoding="utf-8"))
        map_view = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=10)
        folium.GeoJson(
            flood,
            name="Flood mask",
            style_function=lambda _: {"color": "#0875d1", "fillColor": "#168be1", "fillOpacity": 0.45, "weight": 1},
        ).add_to(map_view)
        if uncertainty_path.strip():
            try:
                heat_points = _uncertainty_heat_points(uncertainty_path.strip())
            except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
                st.warning(f"Could not add the uncertainty layer: {exc}")
                heat_points = []
            if heat_points:
                HeatMap(
                    heat_points,
                    name="Model uncertainty — requires ground verification",
                    min_opacity=uncertainty_opacity,
                    max_zoom=16,
                    radius=10,
                    blur=12,
                ).add_to(map_view)
        infrastructure = result.get("infrastructure")
        if infrastructure:
            for layer_name, key, color in (
                ("Potentially affected buildings", "buildings", "#f28e2b"),
                ("Potentially disrupted roads", "roads", "#d62728"),
                ("Bridges intersecting mask", "bridges", "#7b2cbf"),
            ):
                layer = json.loads(Path(infrastructure[key]).read_text(encoding="utf-8"))
                if layer.get("features"):
                    folium.GeoJson(
                        layer,
                        name=layer_name,
                        style_function=lambda _, layer_color=color: {"color": layer_color, "fillColor": layer_color, "fillOpacity": 0.45, "weight": 3},
                        marker=folium.CircleMarker(radius=5, color=color, fill=True),
                    ).add_to(map_view)
        cutoff = result.get("cutoff")
        if cutoff:
            cutoff_layer = json.loads(Path(cutoff["path"]).read_text(encoding="utf-8"))
            folium.GeoJson(
                cutoff_layer,
                name="Cut-off settlements",
                marker=folium.Marker(icon=folium.Icon(color="red")),
                tooltip=folium.GeoJsonTooltip(fields=["name", "reason"]),
            ).add_to(map_view)
        folium.LayerControl().add_to(map_view)
        components.html(map_view.get_root().render(), height=620, scrolling=True)
    with right:
        st.subheader("Pipeline results")
        if infrastructure:
            st.json(infrastructure["summary"])
        if cutoff:
            st.metric("Settlements without a route to a mapped hospital", cutoff["cutoff_count"])
        for warning in result["warnings"]:
            st.warning(warning)
        report_path = Path(result["report"])
        st.download_button(
            "Download situation report",
            report_path.read_bytes(),
            file_name="situation_report.txt",
            mime="text/plain",
        )
        if st.button("Create PDF report"):
            try:
                pdf_path = create_pdf_report(report_path)
                st.session_state["pdf_report"] = str(pdf_path)
            except RuntimeError as exc:
                st.error(str(exc))
        pdf_path = st.session_state.get("pdf_report")
        if pdf_path and Path(pdf_path).exists():
            st.download_button("Download PDF", Path(pdf_path).read_bytes(), file_name=Path(pdf_path).name, mime="application/pdf")
        if st.button("Translate report to Nepali"):
            try:
                nepali_path = translate_report_to_nepali(report_path)
                st.session_state["nepali_report"] = str(nepali_path)
            except RuntimeError as exc:
                st.error(str(exc))
        nepali_path = st.session_state.get("nepali_report")
        if nepali_path and Path(nepali_path).exists():
            st.download_button(
                "Download Nepali report",
                Path(nepali_path).read_bytes(),
                file_name=Path(nepali_path).name,
                mime="text/plain",
            )

with sidebar.expander("System limitations", expanded=True):
    st.write(
        "Sentinel revisit timing can miss peak flooding; cloud, terrain shadow and layover affect observations; "
        "rural OSM may omit roads or facilities. Overlap indicates potentially affected infrastructure, not verified damage. "
        "This educational prototype is not a real-time warning or rescue-routing system."
    )
