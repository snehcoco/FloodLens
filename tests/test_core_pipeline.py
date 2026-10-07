from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from src.pipeline import run_pipeline
from src.ai_model.segmentation import segmentation_metrics, validate_event_split


class CorePipelineTests(unittest.TestCase):
    def test_segmentation_split_guard_and_metrics(self) -> None:
        validate_event_split(["flood-a"], ["flood-b"], ["flood-c"])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_event_split(["flood-a"], ["flood-a"], ["flood-c"])
        with self.assertRaisesRegex(ValueError, "Trishuli"):
            validate_event_split(["flood-a"], ["flood-b"], ["Trishuli"])
        metrics = segmentation_metrics(
            np.array([[0.9, 0.8], [0.1, 0.2]]),
            np.array([[1, 0], [1, 0]]),
        )
        self.assertEqual(metrics["iou"], 1 / 3)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)

    def test_end_to_end_masks_infrastructure_and_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before_path = root / "before.tif"
            after_path = root / "after.tif"
            transform = from_origin(85.0, 28.0, 0.001, 0.001)
            before = np.full((20, 20), 10, dtype=np.float32)
            after = before.copy()
            after[8:12, 8:12] = 30
            profile = {
                "driver": "GTiff",
                "height": 20,
                "width": 20,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:4326",
                "transform": transform,
                "nodata": -9999,
            }
            for path, array in ((before_path, before), (after_path, after)):
                with rasterio.open(path, "w", **profile) as destination:
                    destination.write(array, 1)
                    destination.update_tags(RELATIVE_ORBIT_NUMBER="123")

            buildings_path = root / "buildings.geojson"
            roads_path = root / "roads.geojson"
            settlements_path = root / "settlements.geojson"
            hospitals_path = root / "hospitals.geojson"
            gpd.GeoDataFrame(
                {"name": ["Floodplain house", "Other house"]},
                geometry=[Point(85.0095, 27.9905), Point(85.002, 27.998)],
                crs="EPSG:4326",
            ).to_file(buildings_path, driver="GeoJSON")
            gpd.GeoDataFrame(
                {"id": ["road-main"]},
                geometry=[LineString([(85.0, 27.9905), (85.02, 27.9905)])],
                crs="EPSG:4326",
            ).to_file(roads_path, driver="GeoJSON")
            gpd.GeoDataFrame(
                {"name": ["Village West", "Village East"]},
                geometry=[Point(85.002, 27.9905), Point(85.018, 27.9905)],
                crs="EPSG:4326",
            ).to_file(settlements_path, driver="GeoJSON")
            gpd.GeoDataFrame(
                {"name": ["District Hospital"]},
                geometry=[Point(85.018, 27.9905)],
                crs="EPSG:4326",
            ).to_file(hospitals_path, driver="GeoJSON")

            result = run_pipeline(
                before_sar=before_path,
                after_sar=after_path,
                output_dir=root / "outputs",
                event_date="2026-08-26",
                bbox=(85.0, 27.99, 85.01, 28.0),
                buildings=buildings_path,
                roads=roads_path,
                settlements=settlements_path,
                hospitals=hospitals_path,
            )

            with rasterio.open(result["products"]["flood_mask"]) as mask:
                final_mask = mask.read(1)
                self.assertEqual(mask.crs.to_string(), "EPSG:4326")
                self.assertTrue(np.any(final_mask == 1))
                self.assertEqual(final_mask[0, 19], 255)
            self.assertGreater(result["infrastructure"]["summary"]["potentially_damaged_buildings"], 0)
            self.assertEqual(result["cutoff"]["cutoff_count"], 1)
            cutoff = gpd.read_file(result["cutoff"]["path"])
            self.assertEqual(cutoff.iloc[0]["name"], "Village West")
            report = Path(result["report"]).read_text(encoding="utf-8")
            self.assertIn("Contains modified Copernicus Sentinel data 2026.", report)
            self.assertIn("potentially_damaged_buildings", report)

    def test_orbit_mismatch_fails_before_raster_products_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raster = root / "input.tif"
            profile = {
                "driver": "GTiff",
                "height": 2,
                "width": 2,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:4326",
                "transform": from_origin(85.0, 28.0, 0.01, 0.01),
            }
            for orbit in ("10", "11"):
                with rasterio.open(root / f"{orbit}.tif", "w", **profile) as destination:
                    destination.write(np.ones((2, 2), dtype=np.float32), 1)
                    destination.update_tags(RELATIVE_ORBIT_NUMBER=orbit)
            with self.assertRaisesRegex(ValueError, "relative orbit mismatch"):
                run_pipeline(
                    before_sar=root / "10.tif",
                    after_sar=root / "11.tif",
                    output_dir=root / "outputs",
                )
            self.assertFalse((root / "outputs" / "processed").exists())


if __name__ == "__main__":
    unittest.main()
