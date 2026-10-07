"""Raster processing for SAR change detection and evidence fusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage
from skimage.filters import threshold_otsu


def compute_sar_change(before: np.ndarray, after: np.ndarray, dB_mode: bool = False) -> np.ndarray:
    """Return log(after / before) for linear intensity or after - before for dB."""
    before_arr = np.asarray(before, dtype=np.float32)
    after_arr = np.asarray(after, dtype=np.float32)
    if before_arr.shape != after_arr.shape:
        raise ValueError(f"Before and after arrays must have the same shape, got {before_arr.shape} and {after_arr.shape}.")
    valid = np.isfinite(before_arr) & np.isfinite(after_arr)
    result = np.full(before_arr.shape, np.nan, dtype=np.float32)
    if dB_mode:
        result[valid] = after_arr[valid] - before_arr[valid]
    else:
        positive = valid & (before_arr > 0) & (after_arr > 0)
        result[positive] = np.log(after_arr[positive] / before_arr[positive])
    return result


def create_flood_mask_from_change(
    change: np.ndarray,
    threshold: float | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Use Otsu thresholding on change magnitude to retain increases and decreases."""
    values = np.asarray(change, dtype=np.float32)
    valid = np.isfinite(values)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    magnitudes = np.abs(values[valid])
    if magnitudes.size == 0:
        raise ValueError("SAR change image contains no valid pixels.")
    if threshold is None:
        threshold = float(threshold_otsu(magnitudes)) if np.unique(magnitudes).size > 1 else float(magnitudes[0])
    return (valid & (np.abs(values) > threshold)).astype(np.uint8)


def align_raster(source_path: str | Path, reference_path: str | Path, output_path: str | Path) -> Path:
    """Reproject a single-band raster to the CRS, transform, and shape of a reference."""
    output = Path(output_path)
    with rasterio.open(reference_path) as reference, rasterio.open(source_path) as source:
        if source.count < 1 or reference.count < 1:
            raise ValueError("Input rasters must contain at least one band.")
        if source.crs is None or reference.crs is None:
            raise ValueError("Both source and reference rasters must have a defined CRS.")
        profile = reference.profile.copy()
        profile.update(count=1, dtype="float32", compress="deflate", nodata=np.nan)
        aligned = np.full((reference.height, reference.width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(source, 1),
            destination=aligned,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as destination:
        destination.write(aligned, 1)
    return output


def _write_mask(path: Path, values: np.ndarray, profile: dict) -> None:
    output_profile = profile.copy()
    output_profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(values.astype(np.uint8), 1)


def create_flood_products(
    before_path: str | Path,
    after_path: str | Path,
    output_dir: str | Path,
    *,
    inputs_are_db: bool = False,
    optical_before_path: str | Path | None = None,
    optical_after_path: str | Path | None = None,
    green_band: int = 3,
    nir_band: int = 8,
    min_component_pixels: int = 4,
) -> dict[str, Path]:
    """Create aligned SAR change, SAR candidate, and final evidence-fused masks.

    Sentinel-2 files must be co-registered multispectral rasters. The optical branch
    adds newly observed water only when both dates are supplied and valid.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with rasterio.open(before_path) as before_src, rasterio.open(after_path) as after_src:
        if before_src.count < 1 or after_src.count < 1:
            raise ValueError("Sentinel-1 input rasters must have at least one band.")
        if before_src.crs is None or after_src.crs is None:
            raise ValueError("Both Sentinel-1 rasters must have a defined CRS.")
        before = before_src.read(1, masked=True).astype(np.float32).filled(np.nan)
        after = np.full((before_src.height, before_src.width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(after_src, 1),
            destination=after,
            src_transform=after_src.transform,
            src_crs=after_src.crs,
            src_nodata=after_src.nodata,
            dst_transform=before_src.transform,
            dst_crs=before_src.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        profile = before_src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")

    change = compute_sar_change(before, after, dB_mode=inputs_are_db)
    valid = np.isfinite(before) & np.isfinite(after) & np.isfinite(change)
    sar_mask = create_flood_mask_from_change(change, valid_mask=valid)
    final_mask = sar_mask.copy()
    final_valid = valid.copy()
    products: dict[str, Path] = {}

    if (optical_before_path is None) != (optical_after_path is None):
        raise ValueError("Provide both pre-event and post-event Sentinel-2 rasters, or neither.")
    if optical_before_path is not None and optical_after_path is not None:
        optical_paths = (Path(optical_before_path), Path(optical_after_path))
        ndwi_arrays: list[np.ndarray] = []
        for optical_path in optical_paths:
            with rasterio.open(optical_path) as optical:
                if optical.count < max(green_band, nir_band):
                    raise ValueError(f"{optical_path} does not contain requested Green/NIR bands.")
                if optical.crs is None:
                    raise ValueError(f"{optical_path} has no CRS.")
                green = optical.read(green_band, masked=True).astype(np.float32).filled(np.nan)
                nir = optical.read(nir_band, masked=True).astype(np.float32).filled(np.nan)
                ndwi = np.full(green.shape, np.nan, dtype=np.float32)
                denominator = green + nir
                good = np.isfinite(green) & np.isfinite(nir) & (np.abs(denominator) > 1e-8)
                ndwi[good] = (green[good] - nir[good]) / denominator[good]
                if (
                    optical.crs != profile["crs"]
                    or optical.transform != profile["transform"]
                    or ndwi.shape != (profile["height"], profile["width"])
                ):
                    aligned_ndwi = np.full((profile["height"], profile["width"]), np.nan, dtype=np.float32)
                    reproject(
                        source=ndwi,
                        destination=aligned_ndwi,
                        src_transform=optical.transform,
                        src_crs=optical.crs,
                        src_nodata=np.nan,
                        dst_transform=profile["transform"],
                        dst_crs=profile["crs"],
                        dst_nodata=np.nan,
                        resampling=Resampling.bilinear,
                    )
                    ndwi = aligned_ndwi
                ndwi_arrays.append(ndwi)
        optical_before, optical_after = ndwi_arrays
        reliable = np.isfinite(optical_before) & np.isfinite(optical_after)
        newly_wet = reliable & (optical_after >= 0.3) & ((optical_after - optical_before) >= 0.1)
        final_mask = (
            np.where(valid, sar_mask.astype(bool), False) | newly_wet
        ).astype(np.uint8)
        final_valid |= reliable
        optical_mask = (newly_wet & valid).astype(np.uint8)
        optical_change = optical_after - optical_before
        optical_change[~reliable] = np.nan
        optical_mask_path = output / "flood_mask_optical.tif"
        optical_change_path = output / "ndwi_change.tif"
        _write_mask(optical_mask_path, np.where(reliable, optical_mask, 255), profile)
        with rasterio.open(optical_change_path, "w", **profile) as destination:
            destination.write(optical_change, 1)
        products["optical_mask"] = optical_mask_path
        products["ndwi_change"] = optical_change_path

    if min_component_pixels > 1:
        labels, _ = ndimage.label(final_mask)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_component_pixels
        keep[0] = False
        final_mask = keep[labels].astype(np.uint8)
    final_mask[~final_valid] = 255

    change_path = output / "sar_change.tif"
    sar_path = output / "flood_mask_sar.tif"
    final_path = output / "flood_mask.tif"
    with rasterio.open(change_path, "w", **profile) as destination:
        destination.write(change, 1)
    _write_mask(sar_path, np.where(valid, sar_mask, 255), profile)
    _write_mask(final_path, final_mask, profile)
    products.update({"change": change_path, "sar_mask": sar_path, "flood_mask": final_path})
    return products


def generate_demo_sar_rasters(output_dir: str | Path | None = None) -> dict[str, Path]:
    """Create synthetic rasters for software tests only; never use as case-study results."""
    root = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[2] / "data" / "processed"
    root.mkdir(parents=True, exist_ok=True)
    before = np.full((64, 64), 0.12, dtype=np.float32)
    after = before.copy()
    after[12:52, 20:44] = 0.25
    transform = rasterio.transform.from_origin(85.0, 28.5, 0.01, 0.01)
    profile = {
        "driver": "GTiff",
        "height": 64,
        "width": 64,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
    }
    before_path = root / "before_demo.tif"
    after_path = root / "after_demo.tif"
    for path, values in ((before_path, before), (after_path, after)):
        with rasterio.open(path, "w", **profile) as destination:
            destination.write(values, 1)
    products = create_flood_products(before_path, after_path, root / "demo_products")
    return {"before": before_path, "after": after_path, **products}
