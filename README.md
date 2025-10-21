# QuakeSlide: Prompt Earthquake-Triggered Landslide Impact Assessment

*Author: Shihao Xiao*  
*Email: sxiaoai@connect.ust.hk*  
*Date: October 2025*

---

## Overview
QuakeSlide is an end-to-end tool designed to assess the impacts of earthquake-triggered landslides. It provides functionalities for earthquake event search, landslide hazard and impact prediction, uncertainty quantification, and report generation.

---

## System Requirements

### Hardware
- **Recommended**: 32GB RAM, Intel i7-11700 (2.50 GHz).

### Software
- **Dependencies**: Listed in `environment.yml` and `requirements.txt`.

To set up the environment:
```bash
# Create the environment from environment.yml
conda env create -f "path_to_file/environment.yml"

# Activate the environment
conda activate QuakeSlide
```
The created environment includes Python 3.8.18 and all required packages.

---

## Usage
1. Open `QuakeSlide.ipynb` and follow the steps outlined in the notebook.
2. Select a specific earthquake event and run the analysis.
3. View the generated report in the output directory.

---

## File Structure
```
```text
QuakeSlide/
├── QuakeSlide.ipynb                             # Main workflow notebook
├── Input.txt                                    # Configuration file
├── Results/                                     # Output directory
│   └── {event_id}/                              # Event-specific results (reports, maps, logs)
├── src/
│   ├── QuakeSlide_main.py                       # Main entry point
│   ├── define_file_path.py                      # File path management & configuration
│   ├── download_PGA_raster.py                   # USGS ShakeMap data acquisition
│   ├── create_landslide_affected_area.py        # Affected-area delineation
│   ├── preload_data.py                          # Global datasets loading
│   ├── feature_extraction.py                    # Spatial feature extraction & processing
│   ├── predict_landslide_intensity.py           # Landslide intensity prediction
│   ├── uncertainty_propagation_hazards.py       # Hazard uncertainty analysis
│   ├── uncertainty_propagation_impacts.py       # Impact uncertainty analysis
│   ├── report_generation.py                     # Automated report generation
│   ├── plot_utils.py                            # Plotting utilities
│   ├── utils.py                                 # Utility functions
│   ├── environment.yml                          # Conda environment specification
├── Data/                                        # Global datasets (Some files are large for this repo. If you need these datasets, please contact Shihao Xiao)
│   ├── DEM/                                     # Digital Elevation Model (DEM) raster files
│   ├── Global cities/                           # Major cities
│   ├── Global ocean boundary/                   # Ocean boundary shapefile
│   ├── Landform/                                # Landform types
│   ├── Lithology/                               # Lithological maps
│   ├── Population density/                      # Population density rasters
│   ├── River network/                           # River network datasets
│   └── Slope/                                   # Slope rasters
├── Model/                                       
│   ├── GAM_model_20250727.pkl                   # Trained model
│   ├── GAM_model_features_20250727.pkl          # Selected features
│   └── GAM_model_label_encoders_20250727.pkl    # Label encoders
└── README.md                                    # Project README
```

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact
For questions, suggestions, or collaboration, contact Shihao Xiao at sxiaoai@connect.ust.hk.