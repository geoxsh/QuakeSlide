"""
File path configuration module for QuakeSlide HBM system.

Date: August 2026
"""

import os


def define_file_path(input_file_path, run_id):
    input_path = readInputFiles(input_file_path)
    if not input_path:
        print('ERROR: Input configuration file not found.')
        return None, None

    output_path = defineOutputFile(run_id)
    return input_path, output_path


def readInputFiles(inputFilePath):
    inputPath = {}

    if not os.path.exists(inputFilePath):
        print(f'ERROR: Input file not found at "{inputFilePath}"')
        return None

    with open(inputFilePath, 'r') as f:
        line = f.readline()
        while line:
            # ---------- data file paths ----------
            if 'GlobalSlopeFile' in line:
                line = f.readline()
                inputPath['GlobalSlopeFile'] = line.strip()
            elif 'GlobalLithologyFile' in line:
                line = f.readline()
                inputPath['GlobalLithologyFile'] = line.strip()
            elif 'GlobalLC100File' in line:
                line = f.readline()
                inputPath['GlobalLC100File'] = line.strip()
            elif 'GlobalOceanBoundaryFile' in line:
                line = f.readline()
                inputPath['GlobalOceanBoundaryFile'] = line.strip()
            elif 'GlobalPopulationDensityFile' in line:
                line = f.readline()
                inputPath['GlobalPopulationDensityFile'] = line.strip()
            elif 'GlobalRiverNetwork' in line:
                line = f.readline()
                inputPath['GlobalRiverNetwork'] = line.strip()
            elif 'GlobalDEMFile' in line:
                line = f.readline()
                inputPath['GlobalDEMFile'] = line.strip()
            elif 'GlobalCityFile' in line:
                line = f.readline()
                inputPath['GlobalCityFile'] = line.strip()

            # ---------- thresholds / flags ----------
            elif 'PGAThreshold' in line:
                line = f.readline()
                inputPath['PGAThreshold'] = line.strip()
            elif 'SlopeThreshold' in line:
                line = f.readline()
                inputPath['SlopeThreshold'] = line.strip()
            elif 'IS_save_PointMappingUnit' in line:
                line = f.readline()
                inputPath['IS_save_PointMappingUnit'] = line.strip()
            elif 'LandslideInventoryFile' in line:
                line = f.readline()
                inputPath['LandslideInventoryFile'] = line.strip()
            elif 'IS_LS_PolygonAvailable' in line:
                line = f.readline()
                inputPath['IS_LS_PolygonAvailable'] = line.strip()
            elif 'IS_rectangle_boundary' in line:
                line = f.readline()
                inputPath['IS_rectangle_boundary'] = line.strip()

            # ---------- HBM model files ----------
            elif 'CountModelFile' in line:
                line = f.readline()
                inputPath['CountModelFile'] = line.strip()
            elif 'ArealCoverageModelFile' in line:
                line = f.readline()
                inputPath['ArealCoverageModelFile'] = line.strip()
            elif 'StdParamsFile' in line:
                line = f.readline()
                inputPath['StdParamsFile'] = line.strip()
            elif 'RscriptExe' in line:
                line = f.readline()
                inputPath['RscriptExe'] = line.strip()
            elif 'N_DRAWS' in line:
                line = f.readline()
                inputPath['N_DRAWS'] = line.strip()
            elif 'CountBatchSize' in line:
                line = f.readline()
                inputPath['CountBatchSize'] = line.strip()
            elif 'ArealCoverageBatchSize' in line:
                line = f.readline()
                inputPath['ArealCoverageBatchSize'] = line.strip()

            # ---------- impact assessment ----------
            elif 'buffer_LS_runout' in line:
                line = f.readline()
                inputPath['buffer_LS_runout'] = line.strip()

            line = f.readline()

    legacy_inventory_flag = inputPath.pop('IS_LS_PolygonAvailable', None)
    if legacy_inventory_flag and legacy_inventory_flag.strip().lower() == 'true':
        raise ValueError(
            "Input.txt option IS_LS_PolygonAvailable=True is no longer supported. "
            "Replace it with LandslideInventoryFile and provide the path to the "
            "observed landslide inventory vector file."
        )

    inventory_file = inputPath.get('LandslideInventoryFile')
    if inventory_file is None or inventory_file.strip().lower() in {
        '', 'none', 'null', 'false'
    }:
        inputPath['LandslideInventoryFile'] = None
    else:
        inventory_file = os.path.normpath(inventory_file.strip())
        if not os.path.isfile(inventory_file):
            raise FileNotFoundError(
                f'LandslideInventoryFile does not exist or is not a file: '
                f'"{inventory_file}"'
            )
        inputPath['LandslideInventoryFile'] = inventory_file

    return inputPath


def defineOutputFile(run_id):
    base_dir = f'Results/{run_id}'
    intermediate_dir = f'{base_dir}/intermediate'

    subdirs = [
        'assessment_results',
        'report',
        'intermediate/ShakeMap',
        'intermediate/feature_maps',
        'intermediate/landslide_affected_area',
        'intermediate/cache',
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

    assessment_subdirs = [
        'counts', 'areal_coverage',
        'population_exposure', 'river_interruptions', 'road_interruptions',
    ]
    for subdir in assessment_subdirs:
        os.makedirs(os.path.join(base_dir, 'assessment_results', subdir), exist_ok=True)

    op = {}

    op['RunID'] = run_id
    op['BaseDir'] = base_dir

    # directories
    op['Intermediate']       = intermediate_dir
    op['ShakeMap']           = f'{intermediate_dir}/ShakeMap'
    op['feature_maps']       = f'{intermediate_dir}/feature_maps'
    op['CacheDir']           = f'{intermediate_dir}/cache'

    # ShakeMap rasters
    op['PGAFile'] = f'{intermediate_dir}/ShakeMap/PGA.tif'
    op['PGVFile'] = f'{intermediate_dir}/ShakeMap/PGV.tif'
    op['ShakeMapSourceFile'] = f'{intermediate_dir}/ShakeMap/shakemap_source.json'
    op['ResampledPGVFile'] = f'{intermediate_dir}/feature_maps/Resampled_PGV.tif'

    # Clipped terrain rasters
    op['ClippedSlopeFile']   = f'{intermediate_dir}/feature_maps/Clipped_Slope.tif'
    op['LandslideAffectedArea'] = f'{intermediate_dir}/landslide_affected_area/landslide_affected_area.shp'
    op['save_PointMappingUnitFile'] = f'{intermediate_dir}/point_mapping_unit/point_mapping_unit.shp'

    # HBM prediction output files
    op['LSCountStatsFile']     = f'{intermediate_dir}/cache/ls_count_stats.parquet'
    op['LSCountTotalsFile']    = f'{intermediate_dir}/cache/ls_count_totals.csv'
    op['LSCountDrawsFile']     = f'{intermediate_dir}/cache/ls_count_draws.parquet'
    op['LSCoverageStatsFile']  = f'{intermediate_dir}/cache/ls_coverage_stats.parquet'
    op['LSCoverageTotalsFile'] = f'{intermediate_dir}/cache/ls_coverage_totals.csv'
    op['LSCoverageDrawsFile']  = f'{intermediate_dir}/cache/ls_coverage_draws.parquet'

    # Quantile rasters (count)
    op['LSCountQ025File'] = f'{base_dir}/assessment_results/counts/ls_count_q025.tif'
    op['LSCountQ500File'] = f'{base_dir}/assessment_results/counts/ls_count_q500.tif'
    op['LSCountQ975File'] = f'{base_dir}/assessment_results/counts/ls_count_q975.tif'

    # Quantile rasters (areal coverage)
    op['LSCoverageQ025File'] = f'{base_dir}/assessment_results/areal_coverage/ls_coverage_q025.tif'
    op['LSCoverageQ500File'] = f'{base_dir}/assessment_results/areal_coverage/ls_coverage_q500.tif'
    op['LSCoverageQ975File'] = f'{base_dir}/assessment_results/areal_coverage/ls_coverage_q975.tif'

    # LSIntensityFile points to q500 count raster (used by impact module for raster meta)
    op['LSIntensityFile'] = op['LSCountQ500File']

    # Summary statistics arrays
    op['TotalCountsFile']             = f'{base_dir}/assessment_results/total_landslides_counts.npy'
    op['TotalAreasFile']              = f'{base_dir}/assessment_results/total_landslide_areas.npy'
    op['TotalPopulationExposureFile'] = f'{base_dir}/assessment_results/total_population_exposure.npy'

    # River interruption files
    op['RiverInterruptionsNoDataFile'] = f'{base_dir}/assessment_results/river_interruptions/No river network found.txt'
    op['TotalRiverInterruptionsFile'] = f'{base_dir}/assessment_results/total_river_interruptions.npy'
    op['MeanRiverInterruptionsDensityFile'] = f'{base_dir}/assessment_results/river_interruptions/river_interruptions_density.shp'
    op['ObservedTotalRiverInterruptionsFile'] = f'{base_dir}/assessment_results/observed_total_river_interruptions_FromLS_Polygon.npy'
    op['ObservedRiverInterruptionsFile'] = f'{base_dir}/assessment_results/river_interruptions/observed_river_interruptions.shp'

    # Road interruption files
    op['RoadInterruptionsNoDataFile'] = f'{base_dir}/assessment_results/road_interruptions/No road network found.txt'
    op['TotalRoadInterruptionsFile'] = f'{base_dir}/assessment_results/total_road_interruptions.npy'
    op['MeanRoadInterruptionsDensityFile'] = f'{base_dir}/assessment_results/road_interruptions/road_interruptions_density.shp'
    op['ObservedTotalRoadInterruptionsFile'] = f'{base_dir}/assessment_results/observed_total_road_interruptions_FromLS_Polygon.npy'
    op['ObservedRoadInterruptionsFile'] = f'{base_dir}/assessment_results/road_interruptions/observed_road_interruptions.shp'

    # Mean spatial rasters
    op['MeanCountsDensityRasterFile']       = f'{base_dir}/assessment_results/counts/counts_density_mean_map.tif'
    op['MeanArealCoveragesRasterFile']      = f'{base_dir}/assessment_results/areal_coverage/coverages_mean_map.tif'

    # Population exposure files
    op['PopulationExposure']                        = f'{base_dir}/assessment_results/population_exposure'
    op['ClippedPopulationDensityFile']              = f'{base_dir}/assessment_results/population_exposure/Clipped_Population_density.tif'
    op['ResampledClippedPopulationDensityFile']     = f'{base_dir}/assessment_results/population_exposure/resampled_Clipped_Population_density.tif'
    op['MeanPopulationExposureDensityRasterFile']   = f'{base_dir}/assessment_results/population_exposure/exposure_density_mean_map.tif'

    # Assessment results directory
    op['AssessmentResults'] = f'{base_dir}/assessment_results'

    # Report files
    op['Report']          = f'{base_dir}/report'
    op['plot_components'] = f'{base_dir}/report/plot_components'
    os.makedirs(op['plot_components'], exist_ok=True)

    op['ClippedElevationFile']  = f'{base_dir}/report/plot_components/Clipped_Elevation.tif'
    op['ClippedHillshadeFile']  = f'{base_dir}/report/plot_components/Clipped_Hillshade.tif'
    op['PGAContourShpFile']     = f'{base_dir}/report/plot_components/pga_contours.shp'
    op['PGAContourJsonFile']    = f'{base_dir}/report/plot_components/pga_contours.json'

    return op
