"""
File path configuration module for QuakeSlide system.

This module handles input parameter loading and output directory structure creation
for earthquake-triggered landslide impact assessment.

Author: Shihao Xiao
Date: Oct 2025
"""

import os


def define_file_path(input_file_path, event_id):
    """
    Main function to define input parameters and output directories.
    
    Args:
        input_file_path (str): Path to input configuration file
        event_id (str): Earthquake event ID
        
    Returns:
        tuple: (input_path_dict, output_path_dict) or (None, None) if input file not found
    """
    input_path = readInputFiles(input_file_path)
    if not input_path:
        print('ERROR: Input configuration file not found.')
        return None, None
    else:
        print('✓ Input configuration file loaded successfully.')
        output_path = defineOutputFile(event_id)
        return input_path, output_path

def readInputFiles(inputFilePath):
    """
    Parse input configuration file and extract all parameter settings.
    
    Args:
        inputFilePath (str): Path to input configuration file
        
    Returns:
        dict: Dictionary containing all parsed parameters, or None if file not found
    """
    inputPath = {}  # Dictionary to store all input parameters
    
    if not os.path.exists(inputFilePath):
        print(f'ERROR: Input file not found at "{inputFilePath}"')
        return None
        
    with open(inputFilePath, 'r') as f:
        line = f.readline()
        while line:
            # Data file paths
            if 'GlobalSlopeFile' in line:
                line = f.readline()
                inputPath['GlobalSlopeFile'] = line.strip()
            elif 'GlobalLandformFile' in line:
                line = f.readline()
                inputPath['GlobalLandformFile'] = line.strip()
            elif 'GlobalLithologyFile' in line:
                line = f.readline()
                inputPath['GlobalLithologyFile'] = line.strip()
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

            # Thresholds and flags
            elif 'PGAThreshold' in line:
                line = f.readline()
                inputPath['PGAThreshold'] = line.strip()
            elif 'SlopeThreshold' in line:
                line = f.readline()
                inputPath['SlopeThreshold'] = line.strip()
            elif 'IS_save_PointMappingUnit' in line:
                line = f.readline()
                inputPath['IS_save_PointMappingUnit'] = line.strip()
            elif 'IS_LS_PolygonAvailable' in line:
                line = f.readline()
                inputPath['IS_LS_PolygonAvailable'] = line.strip()
            elif 'IS_rectangle_boundary' in line:
                line = f.readline()
                inputPath['IS_rectangle_boundary'] = line.strip()
                
            # Model files
            elif 'LSIntensityModelFile' in line:
                line = f.readline()
                inputPath['LSIntensityModelFile'] = line.strip()
            elif 'SelectedModelFeatureFile' in line:
                line = f.readline()
                inputPath['SelectedModelFeatureFile'] = line.strip()
            elif 'FeatureLabelEncodersFile' in line:
                line = f.readline()
                inputPath['FeatureLabelEncodersFile'] = line.strip()
                
            # Uncertainty propagation parameters
            elif 'log_size_mean' in line:
                line = f.readline()
                inputPath['log_size_mean'] = line.strip()
            elif 'log_size_std' in line:
                line = f.readline()
                inputPath['log_size_std'] = line.strip()
            elif 'sigma_ln_lambda' in line:
                line = f.readline()
                inputPath['sigma_ln_lambda'] = line.strip()
            elif 'mu_mu_ln_lambda' in line:
                line = f.readline()
                inputPath['mu_mu_ln_lambda'] = line.strip()
            elif 'sigma_mu_ln_lambda' in line:
                line = f.readline()
                inputPath['sigma_mu_ln_lambda'] = line.strip()
            elif 'area_percentage_threshold' in line:
                line = f.readline()
                inputPath['area_percentage_threshold'] = line.strip()
            elif 'count_threshold' in line:
                line = f.readline()
                inputPath['count_threshold'] = line.strip()
                
 
            elif 'n_inter_event_simulations' in line:
                line = f.readline()
                inputPath['n_inter_event_simulations'] = line.strip()
            elif 'n_intra_event_simulations' in line:
                line = f.readline()
                inputPath['n_intra_event_simulations'] = line.strip()
                
            # Landslide runout buffer
            elif 'buffer_LS_runout' in line:
                line = f.readline()
                inputPath['buffer_LS_runout'] = line.strip()

            line = f.readline()  # Read next line
                
    return inputPath

def defineOutputFile(event_id):
    base_dir = f'Results/{event_id}' 

    # Create additional subdirectories
    subdirs = [
        'ShakeMap', 
        'feature_maps', 
        'landslide_affected_area', 
        'point_mapping_unit', 
        'assessment_results', 
        'report', 
        'mapped_LS_Polygon'
        ]
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)

    # Create five additional subdirectories under Results/{event_id}/assessment_results
    assessment_subdirs = [
        'counts',
        'areas',
        'areal_coverage',
        'population_exposure',
        'river_blockages',
        'road_blockages'
    ]
    assessment_results_dir = os.path.join(base_dir, 'assessment_results')
    for subdir in assessment_subdirs:
        subdir_path = os.path.join(assessment_results_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)

    outputPath = dict()


    # Directory paths
    outputPath['ShakeMap'] = f'{base_dir}/ShakeMap'
    outputPath['feature_maps'] = f'{base_dir}/feature_maps'
    outputPath['mapped_LS_Polygon'] = f'{base_dir}/mapped_LS_Polygon'
    
    # Basic output files
    outputPath['save_PointMappingUnitFile'] = f'{base_dir}/point_mapping_unit/point_mapping_unit.shp'
    outputPath['ClippedSlopeFile'] = f'{base_dir}/feature_maps/Clipped_Slope.tif'
    outputPath['ClippedLandformFile'] = f'{base_dir}/feature_maps/Clipped_Geomorphon.tif'
    outputPath['LandslideAffectedArea'] = f'{base_dir}/landslide_affected_area/landslide_affected_area.shp'
    outputPath['PointMappingUnit'] = f'{base_dir}/point_mapping_unit/point_mapping_unit.shp'
    outputPath['PGAFile'] = f'{base_dir}/ShakeMap/PGA.tif'

    # Assessment result files
    outputPath['LSIntensityFile'] = f'{base_dir}/assessment_results/Predicted_LSIntensity_without_uncertainty.tif'
    outputPath['TotalCountsFile'] = f'{base_dir}/assessment_results/total_landslides_counts.npy'
    outputPath['TotalAreasFile'] = f'{base_dir}/assessment_results/total_landslide_areas.npy'
    outputPath['TotalPopulationExposureFile'] = f'{base_dir}/assessment_results/total_population_exposure.npy'
    
    # River blockage files
    outputPath['RiverBlockages_NonDataFile'] = f'{base_dir}/assessment_results/river_blockages/No river network found.txt'
    outputPath['TotalRiverBlockagesFile'] = f'{base_dir}/assessment_results/total_river_blockages.npy'
    outputPath['MeanRiverBlockagesDensityFile'] = f'{base_dir}/assessment_results/river_blockages/river_blockages_density.shp'
    outputPath['ObservedTotalRiverBlockagesFile'] = f'{base_dir}/assessment_results/observed_total_river_blockages_FromLS_Polygon.npy'
    outputPath['ObservedRiverBlocksFile'] = f'{base_dir}/assessment_results/river_blockages/observed_riverblocks.shp'
    
    # Road blockage files
    outputPath['TotalRoadBlockagesFile'] = f'{base_dir}/assessment_results/total_road_blockages.npy'
    outputPath['MeanRoadBlockagesDensityFile'] = f'{base_dir}/assessment_results/road_blockages/road_blockages_density.shp'
    outputPath['ObservedTotalRoadBlockagesFile'] = f'{base_dir}/assessment_results/observed_total_road_blockages_FromLS_Polygon.npy'
    outputPath['ObservedRoadBlocksFile'] = f'{base_dir}/assessment_results/road_blockages/observed_roadblocks.shp'

    # Count and area raster files
    outputPath['MeanCountsDensityRasterFile'] = f'{base_dir}/assessment_results/counts/counts_density_mean_map.tif'
    outputPath['MeanArealCoveragesRasterFile'] = f'{base_dir}/assessment_results/areal_coverage/coverages_mean_map.tif'
    
    # Population exposure files
    outputPath['PopulationExposure'] = f'{base_dir}/assessment_results/population_exposure'
    outputPath['ClippedPopulationDensityFile'] = f'{base_dir}/assessment_results/population_exposure/Clipped_Population_density.tif'
    outputPath['ResampledClippedPopulationDensityFile'] = f'{base_dir}/assessment_results/population_exposure/resampled_Clipped_Population_density.tif'
    outputPath['MeanPopulationExposureDensityRasterFile'] = f'{base_dir}/assessment_results/population_exposure/exposure_density_mean_map.tif'

    # Report generation files
    outputPath['Report'] = f'{base_dir}/report'
    outputPath['plot_components'] = f'{base_dir}/report/plot_components'
    os.makedirs(outputPath['plot_components'], exist_ok=True)
    
    outputPath['ClippedElevationFile'] = f'{base_dir}/report/plot_components/Clipped_Elevation.tif'
    outputPath['ClippedHillshadeFile'] = f'{base_dir}/report/plot_components/Clipped_Hillshade.tif'
    outputPath['PGAContourShpFile'] = f'{base_dir}/report/plot_components/pga_contours.shp'
    outputPath['PGAContourJsonFile'] = f'{base_dir}/report/plot_components/pga_contours.json'
    
    return outputPath
