# FloodLens

FloodLens is an educational prototype for mapping potential flood/debris change, infrastructure overlap, and settlement access disruption from satellite and pre-event OpenStreetMap data. It is not an operational warning or rescue-routing system.

## Run

Use Python 3.10+ and install dependencies:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

The dashboard accepts preprocessed GeoTIFFs and pre-event GeoJSON files. Sentinel-1 before/after rasters must use the same polarization and processing scale. The app checks embedded relative-orbit metadata when available, aligns the after raster to the before raster, computes log(after/before) for linear intensity or after-before for dB, and thresholds absolute change with Otsu. The optional Sentinel-2 branch computes NDWI from bands 3 and 8 and adds pixels with a reliable new-water signal.

For pre-event OSM, use **Download pre-event OSM snapshot** in the app. It requests building, highway, and bridge geometries from ohsome for 27 July 2026. Choose the downloaded layers in the respective input fields. Settlement and hospital point layers may be supplied separately to run accessibility analysis.

Outputs are written under `outputs/`:

- `processed/flood_mask_sar.tif`, `processed/flood_mask_optical.tif` (when optical data is supplied), and `processed/flood_mask.tif`
- `geojson/flood_polygons.geojson`, `damaged_buildings.geojson`, `damaged_roads.geojson`, `damaged_bridges.geojson`, and `cutoff_settlements.geojson` (when corresponding OSM layers are supplied)
- `reports/situation_report.txt` with pipeline-derived values, plus optional PDF and Nepali translations

The SAR GeoTIFFs must be preprocessed before loading: apply orbit correction, radiometric calibration, and terrain correction consistently to both dates in ESA SNAP; export aligned, single-band VV or VH GeoTIFFs. For optical input, provide co-registered Sentinel-2 L2A GeoTIFFs with bands 3 (Green) and 8 (NIR). This repository does not yet automate Copernicus Dataspace authentication or SNAP processing of raw SAFE products.

## Model utilities

`src/ai_model/segmentation.py` provides an EfficientNet-B0 U-Net builder, BCE + Dice training, event-split guards, held-out metrics (IoU, F1, precision, recall, and ECE), Monte Carlo Dropout inference (20 passes by default), and prediction/uncertainty GeoTIFF export. Kuro Siwo event-level loaders and train/validation/test event lists must be prepared separately; keep Trishuli out of all model-selection splits and use EMSR927 only after predictions are frozen.

The Nepali translation uses Helsinki-NLP MarianMT (`Helsinki-NLP/opus-mt-en-ne`). It may download/cache model weights on first use; translated labels retain the source report's values verbatim. PDF generation needs ReportLab.

## Data integrity and limitations

- Never use EMSR927, UNOSAT, other published damage maps, or post-event OSM edits as model inputs or thresholds. EMSR927 is for checking results only.
- Use the same Sentinel-1 relative orbit for pixelwise comparison. If orbit tags are absent from the rasters, verify the orbit pair manually.
- OSM overlap is labelled **potentially damaged/disrupted**, not ground-truth damage. The cut-off result depends on the completeness and topology of mapped roads and facilities.
- Sentinel revisit timing may miss peak flood extent; SAR terrain effects, clouds in optical imagery, and incomplete rural OSM all affect results. This system cannot provide real-time warnings.
- Do not include images of victims in the dashboard, report, or demo.
- A final case-study accuracy comparison with EMSR927, full raw-scene acquisition/preprocessing, and validation on the actual Kuro Siwo held-out events require those source data and must be performed separately.

## Required submission attributions

Include applicable attributions in every submission and generated report:

- “Contains modified Copernicus Sentinel data 2026.”
- “Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.”
- “© OpenStreetMap contributors.”
- Kuro Siwo: cite Bountos et al., 2024, if used for training.
- EMSR927 checking reference: credit “European Union, Copernicus Emergency Management Service data”.
