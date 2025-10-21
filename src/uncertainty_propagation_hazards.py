"""
Uncertainty Propagation Module for Landslide Hazards.

This module implements Monte Carlo simulation for propagating uncertainties.

Author: Shihao Xiao
Date: Oct 2025
"""

# Standard library imports
import time

# Third-party imports
import numpy as np
import rasterio
from scipy import stats
from tqdm import tqdm

# Local imports
from src.utils import save_raster

def uncertainty_propagation_hazards(input_path, output_path):
    """
    Perform uncertainty propagation analysis for landslide hazards.
    
    This function conducts Monte Carlo simulations to propagate uncertainties
    in landslide hazard predictions, accounting for both inter-event and
    intra-event variabilities.
    
    Args:
        input_path (dict): Dictionary containing input parameters and file paths
        output_path (dict): Dictionary containing output file paths
        
    Returns:
        None: Results are saved as raster files and numpy arrays
    """
    print("\nStarting uncertainty propagation...")
    
    # Step 1: Determine grid area for spatial calculations (unit: m²)
    lon_len, lat_len, grid_area = get_grid_area(output_path['ClippedSlopeFile'])
    # print(f"✓ Grid analysis - Longitude: {lon_len:.2f}m, Latitude: {lat_len:.2f}m, Area: {grid_area:.2f}m²")


    # Step 2: Generate Monte Carlo simulations for landslide hazards
    # print("✓ Running Monte Carlo simulations...")
    
    # Perform uncertainty propagation with Monte Carlo simulation
    (
        simulated_counts_store, 
        simulated_areas_store, 
        total_landslides_counts, 
        total_landslide_areas
    ) = generate_maps(
        log_size_mean=float(input_path['log_size_mean']), 
        log_size_std=float(input_path['log_size_std']), 
        grid_area=grid_area, 
        area_percentage_threshold=float(input_path['area_percentage_threshold']), 
        count_threshold=float(input_path['count_threshold']), 
        predicted_counts_raster=output_path['LSIntensityFile'], 
        sigma_ln_lambda=float(input_path['sigma_ln_lambda']),
        mu_mu_ln_lambda=float(input_path['mu_mu_ln_lambda']), 
        sigma_mu_ln_lambda=float(input_path['sigma_mu_ln_lambda']),
        n_inter_event_simulations=int(input_path['n_inter_event_simulations']), 
        n_intra_event_simulations=int(input_path['n_intra_event_simulations'])
    )

    # Step 3: Calculate statistical measures for landslide counts
    # print("✓ Computing statistical measures...")
    
    # Calculate mean landslide counts per grid cell
    counts_mean_map = np.mean(simulated_counts_store, axis=(0, 1))
    
    # Optional: Calculate additional statistics (currently commented out)
    # counts_quantile_2_5 = np.percentile(simulated_counts_store, 2.5, axis=(0, 1))   # 2.5% quantile
    # counts_quantile_97_5 = np.percentile(simulated_counts_store, 97.5, axis=(0, 1)) # 97.5% quantile
    # counts_quantile_50_0 = np.percentile(simulated_counts_store, 50, axis=(0, 1))   # Median
    # counts_std_map = np.std(simulated_counts_store, axis=(0, 1))                     # Standard deviation

    # Step 4: Calculate areal coverage statistics (unit: m²/m²)
    simulated_area_coverages_store = simulated_areas_store / grid_area
    
    # Free up memory by deleting intermediate arrays
    del simulated_areas_store
    # print("✓ Memory optimization: Removed intermediate area arrays")
    # coverages_std_map = np.std(simulated_area_coverages_store, axis=(0, 1))  # 标准差

    coverages_mean_map = np.mean(simulated_area_coverages_store, axis=(0, 1))  # 均值


    # Step 5: Calculate and display summary statistics
    counts_mean = np.mean(total_landslides_counts)
    counts_2_5 = np.percentile(total_landslides_counts, 2.5)
    counts_50_0 = np.percentile(total_landslides_counts, 50.0)
    counts_97_5 = np.percentile(total_landslides_counts, 97.5)
    
    areas_mean = np.mean(total_landslide_areas)
    areas_2_5 = np.percentile(total_landslide_areas, 2.5)
    areas_50_0 = np.percentile(total_landslide_areas, 50.0)
    areas_97_5 = np.percentile(total_landslide_areas, 97.5)

    print(f"Total landslide counts: median={int(round(counts_50_0))}, 95% prediction interval=({int(round(counts_2_5))}, {int(round(counts_97_5))})")
    print(f"Total landslide areas: median={int(round(areas_50_0 / 1e6))} km², 95% prediction interval=({int(round(areas_2_5 / 1e6))}, {int(round(areas_97_5 / 1e6))}) km²")

    # Step 6: Save simulation results
    # print("✓ Saving simulation results...")
    
    # Save total counts and areas as numpy arrays
    np.save(output_path['TotalCountsFile'], total_landslides_counts)
    np.save(output_path['TotalAreasFile'], total_landslide_areas)

    # Read reference raster metadata for output rasters
    with rasterio.open(output_path['ClippedSlopeFile']) as src:
        slope_raster_meta = src.meta.copy()
        slope_raster_transform = src.transform

    # Calculate and save landslide density map (unit: counts/km²)
    counts_density_mean_map = counts_mean_map / grid_area * 1e6
    save_raster(counts_density_mean_map, output_path['MeanCountsDensityRasterFile'], slope_raster_meta, slope_raster_transform)

    # Save areal coverage map (dimensionless ratio)
    save_raster(coverages_mean_map, output_path['MeanArealCoveragesRasterFile'], slope_raster_meta, slope_raster_transform)

    return grid_area, simulated_counts_store, simulated_area_coverages_store




def get_grid_area(raster_path):
    """
    Calculate grid cell area from raster file.
    
    This function analyzes a raster file to determine the actual physical dimensions
    of grid cells at the center of the raster.
    
    Parameters:
    raster_path : str
        Path to the input raster file
    
    Returns:
    tuple
        - lon_length (float): Grid cell width in longitude direction (meters)
        - lat_length (float): Grid cell height in latitude direction (meters)  
        - area (float): Grid cell area (square meters)
    
    Notes:
    ------
    The calculation uses the WGS84 ellipsoid radius and accounts for latitude-dependent
    longitude scaling using the cosine of latitude at the raster center.
    """
    # Earth's radius in meters (WGS84 ellipsoid semi-major axis)
    EARTH_RADIUS_M = 6378137
    
    with rasterio.open(raster_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        
        # Calculate center pixel coordinates
        center_row = height // 2
        center_col = width // 2
        center_lon, center_lat = transform * (center_col + 0.5, center_row + 0.5)
        
        # Extract pixel resolution from transform matrix
        dlon = abs(transform.a)  # Longitude resolution (degrees)
        dlat = abs(transform.e)  # Latitude resolution (degrees)

    # Calculate physical dimensions
    # Latitude: approximately constant at 111,320 meters per degree
    lat_length = 111320 * dlat
    
    # Longitude: varies with latitude due to Earth's curvature
    lon_length = 111320 * dlon * np.cos(np.radians(center_lat))
    
    # Alternative precise calculation using Earth's radius:
    # lat_length = EARTH_RADIUS_M * np.radians(dlat)
    # lon_length = EARTH_RADIUS_M * np.radians(dlon) * np.cos(np.radians(center_lat))

    # Calculate grid cell area
    area = lat_length * lon_length
    
    return lon_length, lat_length, area



def generate_maps(log_size_mean, log_size_std, 
                  grid_area, area_percentage_threshold=None, count_threshold=None,
                  predicted_counts_raster=None, 
                  sigma_ln_lambda=None, 
                  mu_mu_ln_lambda=None,
                  sigma_mu_ln_lambda=None,
                  n_inter_event_simulations=20, n_intra_event_simulations=20):
    """
    Generate landslide count and area maps using nested Monte Carlo simulation.
    
    This function implements a two-level uncertainty propagation:
    1. Inter-event variability: Samples different mu_epsilon values
    2. Intra-event variability: For each mu_epsilon, generates multiple realizations
    
    Parameters:
    -----------
    log_size_mean : float
        Mean of log-normal landslide size distribution
    log_size_std : float
        Standard deviation of log-normal landslide size distribution
    grid_area : float
        Area of each grid cell in square meters
    area_percentage_threshold : float, optional
        Maximum allowed landslide area as fraction of grid cell area
    count_threshold : float, optional
        Maximum allowed landslide count per grid cell
    predicted_counts_raster : str
        Path to raster file containing predicted landslide counts (lambda values)
    sigma_ln_lambda : float
        Standard deviation for intra-event variability in ln(lambda)
    mu_mu_ln_lambda : float
        Mean of mu_epsilon for inter-event variability
    sigma_mu_ln_lambda : float
        Standard deviation of mu_epsilon for inter-event variability
    n_inter_event_simulations : int, default=20
        Number of outer loop simulations (inter-event variability)
    n_intra_event_simulations : int, default=20
        Number of inner loop simulations per mu_epsilon (intra-event variability)
    
    Returns:
    --------
    tuple
        - simulated_counts_store : ndarray
            4D array of simulated landslide counts [outer, inner, height, width]
        - simulated_areas_store : ndarray
            4D array of simulated landslide areas [outer, inner, height, width]
        - total_landslides_counts : list
            Total landslide counts for each simulation realization
        - total_landslide_areas : list
            Total landslide areas for each simulation realization
            
    Notes:
    ------
    The function uses quantile-based sampling for mu_epsilon to ensure better
    representation of the distribution compared to random sampling.
    """
    start_time = time.time()
    
    # Set fixed random seed for reproducible results
    np.random.seed(42)

    # Step 1: Load and validate predicted counts raster
    with rasterio.open(predicted_counts_raster) as src:
        predicted_counts_raster = src.read(1)

    # Create mask for valid pixels (non-NaN and positive values)
    valid_mask = ~np.isnan(predicted_counts_raster) & (predicted_counts_raster > 0)

    # Extract valid lambda values and calculate ln(lambda)
    valid_lambda = predicted_counts_raster[valid_mask]
    ln_lambda = np.log(valid_lambda)

    # Step 2: Initialize result storage arrays
    height, width = predicted_counts_raster.shape
    
    simulated_counts_store = np.zeros((n_inter_event_simulations, n_intra_event_simulations, height, width), dtype=np.float32)
    simulated_areas_store = np.zeros((n_inter_event_simulations, n_intra_event_simulations, height, width), dtype=np.float32)
    total_landslides_counts = []
    total_landslide_areas = []

    # Step 3: Generate mu_epsilon samples using quantile-based approach
    # This provides better distribution coverage than pure random sampling
    from scipy.stats import norm
    percentiles = np.linspace(0.5/n_inter_event_simulations, 1 - 0.5/n_inter_event_simulations, n_inter_event_simulations)
    mu_epsilon_samples = norm.ppf(percentiles, loc=mu_mu_ln_lambda, scale=sigma_mu_ln_lambda)


    # Step 4: Execute Monte Carlo simulation
    total_iterations = n_inter_event_simulations * n_intra_event_simulations
    with tqdm(total=total_iterations, desc="Generating maps") as pbar:
        
        # Outer loop: Inter-event variability (different mu_epsilon values)
        for i, mu_epsilon in enumerate(mu_epsilon_samples):
            # Adjust ln_lambda by current mu_epsilon for this event scenario
            mean_ln_lambda = ln_lambda + mu_epsilon

            # Inner loop: Intra-event variability (multiple realizations per scenario)
            for j in range(n_intra_event_simulations):
                
                # Step 4.1: Simulate lambda values from log-normal distribution
                simulated_lambda = np.full_like(predicted_counts_raster, np.nan, dtype=np.float32)
                simulated_lambda[valid_mask] = np.random.lognormal(mean=mean_ln_lambda, sigma=sigma_ln_lambda)

                # Step 4.2: Generate landslide counts from Poisson distribution
                simulated_landslides = np.full_like(predicted_counts_raster, np.nan, dtype=np.float32)
                simulated_landslides[valid_mask] = np.random.poisson(simulated_lambda[valid_mask])

                # Step 4.3: Apply count threshold constraint
                simulated_landslides[valid_mask] = np.clip(simulated_landslides[valid_mask], None, count_threshold)
                simulated_counts_store[i, j] = simulated_landslides

                # Step 4.4: Generate landslide areas efficiently (vectorized approach)
                counts_flat = simulated_landslides[valid_mask].astype(int)
                nonzero_counts = counts_flat > 0
                areas_flat = np.zeros_like(counts_flat, dtype=np.float32)
                
                if np.any(nonzero_counts):
                    # Calculate total number of random samples needed
                    counts_nonzero = counts_flat[nonzero_counts]
                    total_random = np.sum(counts_nonzero)
                    
                    # Generate all landslide areas at once from log-normal distribution
                    all_areas = np.random.lognormal(mean=log_size_mean, sigma=log_size_std, size=total_random)
                    
                    # Sum areas for each grid cell using efficient reduceat operation
                    idx = np.cumsum(counts_nonzero)
                    areas_list = np.add.reduceat(all_areas, np.concatenate(([0], idx[:-1])))
                    
                    # Apply area percentage threshold if specified
                    if area_percentage_threshold is not None:
                        areas_list = np.minimum(areas_list, area_percentage_threshold * grid_area)
                    
                    areas_flat[nonzero_counts] = areas_list

                # Step 4.5: Assign computed areas back to the spatial grid
                simulated_areas = np.full_like(predicted_counts_raster, np.nan, dtype=np.float32)
                simulated_areas[valid_mask] = areas_flat
                
                # Final clipping to ensure area constraints
                simulated_areas[valid_mask] = np.clip(simulated_areas[valid_mask], None, area_percentage_threshold * grid_area)
                simulated_areas_store[i, j] = simulated_areas

                # Step 4.6: Calculate summary statistics for this realization
                total_landslides_counts.append(np.nansum(simulated_landslides))
                total_landslide_areas.append(np.nansum(simulated_areas))

                # Step 4.7: Update progress with elapsed time
                elapsed_time = (time.time() - start_time) / 60
                pbar.set_postfix(elapsed_time=f"{elapsed_time:.2f} min")
                pbar.update(1)
                
    return simulated_counts_store, simulated_areas_store, total_landslides_counts, total_landslide_areas
