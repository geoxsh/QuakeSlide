# QuakeSlide

**Near-real-time forecasting of earthquake-triggered landslide hazards and impacts**

Author: Shihao Xiao  
Contact: sxiaoai@connect.ust.hk  
Date: August 2026

QuakeSlide is an end-to-end tool designed to assess the earthquake-triggered landslide hazards and impacts in near-real-time. It provides functionalities for earthquake event search, landslide hazard and impact prediction, uncertainty quantification, and report generation.

## Framework

QuakeSlide contains four sequential modules:

1. **Data** — delineates the landslide-affected area and prepares predictor and impact layers.
2. **Hazard** — generates posterior ensembles of landslide count and landslide areal coverage using pre-fitted Bayesian models.
3. **Impact** — propagates the hazard ensembles to population exposure and potential road and river interruptions.
4. **Summary** — aggregates event-level results and generates a one-page report.

The hazard models use four predictors on a 250 m grid:

- peak ground velocity (PGV);
- slope;
- land cover; and
- lithology.

Event-specific PGA and PGV are obtained from USGS ShakeMap or supplied by the user. Major-road data are downloaded from OpenStreetMap, while other global datasets are stored locally.

## Requirements

Recommended hardware:

- 32GB RAM, Intel i7-11700 (2.50 GHz).

Required software:

- Python 3.8 or a compatible environment;
- GDAL, Rasterio, GeoPandas, Pyogrio, Shapely, PyProj and OSMnx;
- NumPy, Pandas, PyArrow, NetworkX, Matplotlib and Requests;
- R and `Rscript`; and
- R packages `brms`, `arrow` and `matrixStats`.

A dedicated Conda environment is recommended, particularly on Windows, to keep the geospatial libraries compatible. Update `RscriptExe` in `Input.txt` if R is installed at a different location.

## Configuration

Edit `Input.txt` before running QuakeSlide. The main settings are:

| Setting | Description |
| --- | --- |
| `PGAThreshold` | Minimum PGA used to delineate the affected area, in g. |
| `SlopeThreshold` | Minimum slope used to delineate the affected area, in degrees. |
| `IS_rectangle_boundary` | Use a rectangular affected-area boundary when `True`. |
| `IS_save_PointMappingUnit` | Save the point mapping-unit Shapefile when `True`. |
| `LandslideInventoryFile` | Optional observed landslide vector file; use `None` if unavailable. |
| `N_DRAWS` | Number of posterior predictive draws. |
| `CountBatchSize` | Batch size for the landslide-count model. |
| `ArealCoverageBatchSize` | Batch size for the areal-coverage model. |
| `buffer_LS_runout` | Road and river impact buffer, in metres. |

Static global dataset paths and model paths are also defined in `Input.txt`.

## Usage

The simplest way to run QuakeSlide is through `QuakeSlide.ipynb`.

The notebook has four sections:

1. **Search for earthquake events:** Find USGS event IDs.
2. **Single-earthquake assessment:** Run the full workflow for one event ID.
3. **Composite-earthquake assessment:** Combine ShakeMaps from multiple events and run one assessment.
4. **Demo:** Run an example for the 5 August 2018 Mw 6.9 earthquake using the cropped data in `demo/data/` (see `demo/README.md`).

### Single earthquake

```python
from src.QuakeSlide_main import QuakeSlide

lithology_data, road_network_data = QuakeSlide(
    event_id="us6000t7zc",
)
```

### Composite ShakeMap

```python
lithology_data, road_network_data = QuakeSlide(
    composite_event_ids=["us6000t7zc", "us6000t7zp"],
)
```

PGA and PGV from the component earthquakes are combined using the pixelwise maximum.

## Outputs

Each new run writes to `Results/<run_id>/`:

```text
Results/<run_id>/
|-- assessment_results/
|   |-- counts/
|   |-- areal_coverage/
|   |-- population_exposure/
|   |-- river_interruptions/
|   `-- road_interruptions/
|-- report/
|   `-- plot_components/
`-- intermediate/
    |-- ShakeMap/
    |-- feature_maps/
    |-- landslide_affected_area/
    |-- point_mapping_unit/     
    `-- cache/
```

`assessment_results/` contains the final hazard and impact rasters, arrays, vector outputs and summary statistics. `report/` contains the one-page report in PNG and PDF formats.

`intermediate/` contains processing files. Cached road networks are kept as GraphML files so they can be reused.

## Project layout

```text
QuakeSlide/
|-- README.md                          # Project overview, setup and usage
|-- QuakeSlide.ipynb                   # Interactive notebook entry point
|-- Input.txt                          # Dataset paths and runtime settings
|-- Model/
|   |-- ZIP_count_model.rds            # Pre-fitted count model
|   |-- ZOIB_areal_coverage_model.rds  # Pre-fitted areal-coverage model
|   |-- std_params.json                # Predictor standardization parameters
|   |-- land_cover_mapping.csv         # Land-cover class mapping
|   `-- lithology_mapping.csv          # Lithology class mapping
|-- Data/
|   |-- Data Availability Note.txt     # Dataset sources and expected paths
|   |-- DEM/                           # Global elevation data
|   |-- Global cities/                 # Global city locations
|   |-- Global ocean boundary/         # Global water-boundary polygons
|   |-- land cover-Copernicus LC100/   # Global land-cover raster
|   |-- Landform/                      # Optional global landform data
|   |-- Lithology/                     # Global lithology data
|   |-- Population density/            # Global population-density raster
|   |-- River network/                 # Global river-network vectors
|   `-- Slope/                         # Global slope raster
|-- src/
|   |-- QuakeSlide_main.py             # End-to-end workflow orchestration
|   |-- shakemap_product.py            # USGS ShakeMap product selection
|   |-- composite_event.py             # Composite-event ShakeMap handling
|   |-- download_PGA_PGV.py            # PGA and PGV acquisition and preparation
|   |-- create_landslide_affected_area.py # Affected-area delineation
|   |-- preload_data.py                # Static-data loading and road download
|   |-- feature_extraction.py          # Mapping-unit predictor extraction
|   |-- hazard_module.py               # Bayesian hazard prediction
|   |-- impacts_module.py              # Population, road and river impacts
|   |-- report_generation.py           # One-page report generation
|   |-- plot_utils.py                  # Shared plotting utilities
|   |-- define_file_path.py            # Input parsing and output paths
|   |-- run_logging.py                 # Workflow logging and progress output
|   `-- utils.py                       # Shared geospatial utilities
`-- Results/                           # Created at runtime; event outputs
```

The large datasets inside `Data/` are not distributed with the repository; see
`Data/Data Availability Note.txt` for download sources. 

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions, suggestions or collaboration, contact Shihao Xiao at sxiaoai@connect.ust.hk.
