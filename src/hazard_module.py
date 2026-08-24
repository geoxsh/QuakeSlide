"""
Bayesian HBM prediction module for QuakeSlide system.

Runs count and areal coverage models via R subprocess
(brms::posterior_predict) and returns draws as 4-D numpy arrays compatible
with the downstream uncertainty_propagation_impacts module.

Interface:
    simulated_counts_store           shape (1, N_DRAWS, H, W)
    simulated_area_coverages_store   shape (1, N_DRAWS, H, W)

Date: August 2026
"""

import json
import os
import pathlib
import re
import subprocess
import textwrap
import time

import numpy as np
import pandas as pd
import rasterio


PREDICTION_STAGE_FILE_KEYS = (
    'LSCountDrawsFile',
    'LSCountStatsFile',
    'LSCountTotalsFile',
    'LSCoverageDrawsFile',
    'LSCoverageStatsFile',
    'LSCoverageTotalsFile',
)


def cleanup_prediction_stage_files(output_path):
    """Delete HBM staging files after all final outputs and the report exist."""
    removed = []
    for key in PREDICTION_STAGE_FILE_KEYS:
        file_path = output_path[key]
        if os.path.isfile(file_path):
            os.remove(file_path)
            removed.append(file_path)
    return removed


# =============================================================================
# Public entry point
# =============================================================================

def predict_hbm(mapping_unit, grid_area_m2, input_path, output_path):
    """
    Run count and areal coverage posterior prediction and return draws as 4-D arrays.

    Args:
        mapping_unit (GeoDataFrame): Cleaned feature table from feature_extraction
        grid_area_m2 (float): Area of each grid cell in m²
        input_path (dict): Parsed Input.txt parameters
        output_path (dict): Parsed output paths

    Returns:
        tuple:
            simulated_counts_store (ndarray)  shape (1, N_DRAWS, H, W) – integer counts
            simulated_area_coverages_store (ndarray) shape (1, N_DRAWS, H, W) – 0-1 fractions
            grid_area_m2 (float)
    """
    n_draws     = int(input_path['N_DRAWS'])
    rscript_exe = input_path['RscriptExe']
    cache_dir   = pathlib.Path(output_path['CacheDir'])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load standardization parameters
    with open(input_path['StdParamsFile'], 'r') as fh:
        std_params = json.load(fh)

    count_rds = input_path['CountModelFile']
    areal_coverage_rds = input_path['ArealCoverageModelFile']
    ls_count_batch = int(input_path.get('CountBatchSize', 20000))
    ls_coverage_batch = int(input_path.get('ArealCoverageBatchSize', 10000))

    # Reference raster for spatial reconstruction (clipped Slope raster)
    ref_raster = output_path['ClippedSlopeFile']

    # ------------------------------------------------------------------
    # 1. Count prediction
    # ------------------------------------------------------------------
    print("\nRunning count prediction...")
    ls_count_df = _prepare_ls_count_input(mapping_unit, std_params)
    ls_count_draws_path = output_path['LSCountDrawsFile']
    ls_count_stats_path = output_path['LSCountStatsFile']
    ls_count_totals_path = output_path['LSCountTotalsFile']

    _run_ls_count_r(
        ls_count_df, count_rds, ls_count_draws_path, ls_count_stats_path, ls_count_totals_path,
        n_draws, ls_count_batch, rscript_exe, cache_dir
    )

    # ------------------------------------------------------------------
    # 2. Areal coverage prediction
    # ------------------------------------------------------------------
    print("\nRunning areal coverage prediction...")
    ls_coverage_df = _prepare_ls_coverage_input(mapping_unit, std_params)
    ls_coverage_draws_path = output_path['LSCoverageDrawsFile']
    ls_coverage_stats_path = output_path['LSCoverageStatsFile']
    ls_coverage_totals_path = output_path['LSCoverageTotalsFile']

    _run_ls_coverage_r(
        ls_coverage_df, areal_coverage_rds,
        ls_coverage_draws_path, ls_coverage_stats_path, ls_coverage_totals_path,
        n_draws, ls_coverage_batch, rscript_exe, cache_dir
    )

    # ------------------------------------------------------------------
    # 3. Load draws and convert to rasters
    # ------------------------------------------------------------------
    print("\nLoading draws and building raster arrays...")

    ls_count_stats = pd.read_parquet(ls_count_stats_path)
    ls_coverage_stats = pd.read_parquet(ls_coverage_stats_path)
    # Q50 per pixel is typically 0 for zero-inflated coverage models;
    # Q975 and mean are more informative diagnostics.
    ls_count_draws = pd.read_parquet(ls_count_draws_path)       # n_pixels x N_DRAWS
    ls_coverage_draws = pd.read_parquet(ls_coverage_draws_path) # n_pixels x N_DRAWS

    with rasterio.open(ref_raster) as src:
        H, W      = src.height, src.width
        raster_meta = src.meta.copy()
        raster_transform = src.transform

    pix_row = mapping_unit['pix_row'].values
    pix_col = mapping_unit['pix_col'].values

    # Build (N_DRAWS, H, W) arrays
    ls_count_draws_np = ls_count_draws.values.T.astype(np.float32)       # (N_DRAWS, n_pixels)
    ls_coverage_draws_np = ls_coverage_draws.values.T.astype(np.float32) # (N_DRAWS, n_pixels)

    counts_3d    = np.zeros((n_draws, H, W), dtype=np.float32)
    coverage_3d  = np.zeros((n_draws, H, W), dtype=np.float32)

    counts_3d[:, pix_row, pix_col] = ls_count_draws_np
    coverage_3d[:, pix_row, pix_col] = ls_coverage_draws_np

    # ------------------------------------------------------------------
    # 4. Save quantile rasters (q025, q500, q975)
    # ------------------------------------------------------------------
    _save_quantile_rasters(
        ls_count_stats['q025'].values, ls_count_stats['q500'].values, ls_count_stats['q975'].values,
        pix_row, pix_col, H, W, raster_meta, raster_transform,
        output_path['LSCountQ025File'],
        output_path['LSCountQ500File'],
        output_path['LSCountQ975File'],
    )
    _save_quantile_rasters(
        ls_coverage_stats['q025'].values, ls_coverage_stats['q500'].values, ls_coverage_stats['q975'].values,
        pix_row, pix_col, H, W, raster_meta, raster_transform,
        output_path['LSCoverageQ025File'],
        output_path['LSCoverageQ500File'],
        output_path['LSCoverageQ975File'],
    )


    # ------------------------------------------------------------------
    # 4b. Save mean rasters (required by report_generation.py)
    #   - counts_density_mean_map  : mean predicted count density (counts / km²)
    #   - coverages_mean_map       : mean predicted areal coverage (dimensionless 0–1)
    # ------------------------------------------------------------------
    _meta = raster_meta.copy()
    _meta.update(dtype='float32', count=1, nodata=-9999.0)

    counts_density_mean = np.full((H, W), -9999.0, dtype=np.float32)
    counts_density_mean[pix_row, pix_col] = (
        np.mean(counts_3d[:, pix_row, pix_col], axis=0) / grid_area_m2 * 1e6
    ).astype(np.float32)
    os.makedirs(os.path.dirname(output_path['MeanCountsDensityRasterFile']), exist_ok=True)
    with rasterio.open(output_path['MeanCountsDensityRasterFile'], 'w', **_meta) as dst:
        dst.write(counts_density_mean, 1)

    coverages_mean = np.full((H, W), -9999.0, dtype=np.float32)
    _cov_mean_px = np.mean(coverage_3d[:, pix_row, pix_col], axis=0).astype(np.float32)
    coverages_mean[pix_row, pix_col] = _cov_mean_px
    os.makedirs(os.path.dirname(output_path['MeanArealCoveragesRasterFile']), exist_ok=True)
    with rasterio.open(output_path['MeanArealCoveragesRasterFile'], 'w', **_meta) as dst:
        dst.write(coverages_mean, 1)


    # ------------------------------------------------------------------
    # 5. Save total counts/areas distributions
    # ------------------------------------------------------------------
    ls_count_totals = pd.read_csv(ls_count_totals_path)['pt'].values.astype(np.float32)
    ls_coverage_totals = pd.read_csv(ls_coverage_totals_path)['pt'].values.astype(np.float32)
    total_areas = ls_coverage_totals * grid_area_m2

    np.save(output_path['TotalCountsFile'], ls_count_totals)
    np.save(output_path['TotalAreasFile'],  total_areas)

    total_counts_median = float(np.percentile(ls_count_totals, 50))
    total_counts_q025   = float(np.percentile(ls_count_totals, 2.5))
    total_counts_q975   = float(np.percentile(ls_count_totals, 97.5))
    total_area_median   = float(np.percentile(total_areas, 50))
    total_area_q025     = float(np.percentile(total_areas, 2.5))
    total_area_q975     = float(np.percentile(total_areas, 97.5))

    print(
        f"\n  Predicted counts:  median={total_counts_median:.0f}  "
        f"95% CI=({total_counts_q025:.0f}, {total_counts_q975:.0f})"
    )
    print(
        f"  Predicted area:    median={total_area_median/1e6:.2f} km2  "
        f"95% CI=({total_area_q025/1e6:.2f}, {total_area_q975/1e6:.2f} km2)"
    )

    # ------------------------------------------------------------------
    # 6. Return 4-D arrays for impact assessment
    # np.newaxis expands (N_DRAWS, H, W) → (1, N_DRAWS, H, W)
    # so that uncertainty_propagation_impacts sees [outer, inner, H, W]
    # ------------------------------------------------------------------
    simulated_counts_store           = counts_3d[np.newaxis, ...]    # (1, N_DRAWS, H, W)
    simulated_area_coverages_store   = coverage_3d[np.newaxis, ...]  # (1, N_DRAWS, H, W)

    return simulated_counts_store, simulated_area_coverages_store, grid_area_m2


# =============================================================================
# Input preparation
# =============================================================================

def _prepare_ls_count_input(mapping_unit, std_params):
    """Build the count-model prediction dataframe (PGV_s, Slope_s, LC100, Lithology)."""
    df = mapping_unit.copy()

    # Standardize continuous predictors used by v9 norain model
    for col in ['PGV', 'Slope']:
        mu = std_params[col]['mean']
        sd = std_params[col]['sd']
        df[f'{col}_s'] = (df[col] - mu) / sd

    # Dummy Counts column (required by brms newdata)
    df['Counts'] = 0

    cols = ['Counts', 'PGV_s', 'Slope_s', 'LC100', 'Lithology']
    return df[cols].copy()


def _prepare_ls_coverage_input(mapping_unit, std_params):
    """Build the areal-coverage-model prediction dataframe (PGV_s, Slope_s, LC100, Lithology)."""
    df = mapping_unit.copy()
    df['LsCoverage'] = 0.0  # placeholder

    # Standardize continuous predictors used by v14 norain model
    for col in ['PGV', 'Slope']:
        mu = std_params[col]['mean']
        sd = std_params[col]['sd']
        df[f'{col}_s'] = (df[col] - mu) / sd

    cols = ['LsCoverage', 'PGV_s', 'Slope_s', 'LC100', 'Lithology']
    return df[cols].copy()


# =============================================================================
# R script generation and execution
# =============================================================================

def _run_ls_count_r(df, model_rds, draws_path, stats_path, totals_path,
                n_draws, batch_size, rscript_exe, tmp_dir):
    """Write count-model input parquet, generate R script, run it, verify output."""
    tmp_dir = pathlib.Path(tmp_dir)
    chunk_path = str(tmp_dir / 'ls_count_input.parquet')

    # Write LC100/Lithology as plain strings so R can set factor levels
    df_out = df.copy()
    df_out['LC100']     = df_out['LC100'].astype(str)
    df_out['Lithology'] = df_out['Lithology'].astype(str)
    df_out.to_parquet(chunk_path, index=False)

    rds   = model_rds.replace('\\', '/')
    chunk = chunk_path.replace('\\', '/')
    dpath = draws_path.replace('\\', '/')
    spath = stats_path.replace('\\', '/')
    tpath = totals_path.replace('\\', '/')

    r_script = textwrap.dedent(f"""\
suppressPackageStartupMessages({{
  library(brms); library(arrow); library(matrixStats)
}})
options(mc.cores = 1L)
cat("Loading count model ...\\n", flush=TRUE)
fit <- readRDS("{rds}")
cat("Model loaded.\\n", flush=TRUE)

lc100_levels <- levels(fit$data$LC100)
lith_levels  <- levels(fit$data$Lithology)

df <- arrow::read_parquet("{chunk}")
lc100_raw <- as.character(df$LC100)
lith_raw  <- as.character(df$Lithology)
if (anyNA(lc100_raw)) {{
  stop("LC100 contains NA before prediction. Check land-cover reclassification.")
}}
if (anyNA(lith_raw)) {{
  stop("Lithology contains NA before prediction. Check lithology reclassification.")
}}
bad_lc100 <- setdiff(unique(lc100_raw), lc100_levels)
bad_lith  <- setdiff(unique(lith_raw),  lith_levels)
if (length(bad_lc100) > 0L) {{
  stop("Unknown LC100 levels in prediction data: ", paste(bad_lc100, collapse=", "))
}}
if (length(bad_lith) > 0L) {{
  stop("Unknown Lithology levels in prediction data: ", paste(bad_lith, collapse=", "))
}}
df$LC100     <- factor(lc100_raw, levels = lc100_levels)
df$Lithology <- factor(lith_raw,  levels = lith_levels)
df$earthquake  <- factor("new_event")
df$case_weight <- 1.0

n_rows    <- nrow(df)
BATCH_SIZE <- {batch_size}L
NDRAWS     <- {n_draws}L
n_batches  <- ceiling(n_rows / BATCH_SIZE)
cat(sprintf("n_rows=%d  n_batches=%d  NDRAWS=%d\\n", n_rows, n_batches, NDRAWS), flush=TRUE)

# Pre-allocate full draws matrix (NDRAWS x n_rows)
draws_mat <- matrix(0.0, nrow=NDRAWS, ncol=n_rows)
all_q025  <- numeric(n_rows)
all_q500  <- numeric(n_rows)
all_q975  <- numeric(n_rows)

for (b in seq_len(n_batches)) {{
  i_from <- (b - 1L) * BATCH_SIZE + 1L
  i_to   <- min(b * BATCH_SIZE, n_rows)
  idx    <- i_from:i_to
  if (n_batches > 1L) {{
    cat(sprintf("  count batch %d/%d  rows %d-%d\\n", b, n_batches, i_from, i_to), flush=TRUE)
  }}
  set.seed(42L)
  pred_b <- posterior_predict(
    fit, newdata = df[idx, , drop=FALSE],
    allow_new_levels  = TRUE,
    sample_new_levels = "gaussian",
    ndraws = NDRAWS)
  storage.mode(pred_b) <- "double"
  draws_mat[, idx] <- pred_b
  qs_b <- colQuantiles(pred_b, probs = c(0.025, 0.5, 0.975))
  all_q025[idx] <- qs_b[, 1L]
  all_q500[idx] <- qs_b[, 2L]
  all_q975[idx] <- qs_b[, 3L]
  rm(pred_b, qs_b)
}}
gc(verbose=FALSE)

total_sums <- matrixStats::rowSums2(draws_mat)

# Save draws as n_pixels x NDRAWS parquet (rows=pixels, cols=draws)
draws_df <- as.data.frame(t(draws_mat))
colnames(draws_df) <- paste0("d", seq_len(NDRAWS))
storage.mode(draws_df[[1]]) <- "double"
arrow::write_parquet(draws_df, "{dpath}", compression="snappy")

arrow::write_parquet(data.frame(q025=all_q025, q500=all_q500, q975=all_q975), "{spath}")
write.csv(data.frame(pt=total_sums), "{tpath}", row.names=FALSE)
rm(draws_mat, draws_df)
gc(verbose=FALSE)
cat("Count model done.\\n", flush=TRUE)
""")

    _execute_r_script(r_script, "ls_count_predict.R", rscript_exe, tmp_dir, label="count")
    # Clean up chunk
    try:
        os.remove(chunk_path)
    except Exception:
        pass


def _run_ls_coverage_r(df, model_rds, draws_path, stats_path, totals_path,
                n_draws, batch_size, rscript_exe, tmp_dir):
    """Write areal-coverage-model input parquet, generate R script, run it, verify output."""
    tmp_dir = pathlib.Path(tmp_dir)
    chunk_path = str(tmp_dir / 'ls_coverage_input.parquet')

    df_out = df.copy()
    df_out['LC100']     = df_out['LC100'].astype(str)
    df_out['Lithology'] = df_out['Lithology'].astype(str)
    df_out.to_parquet(chunk_path, index=False)

    rds   = model_rds.replace('\\', '/')
    chunk = chunk_path.replace('\\', '/')
    dpath = draws_path.replace('\\', '/')
    spath = stats_path.replace('\\', '/')
    tpath = totals_path.replace('\\', '/')

    r_script = textwrap.dedent(f"""\
suppressPackageStartupMessages({{
  library(brms); library(arrow); library(matrixStats)
}})
options(mc.cores = 1L)
cat("Loading areal coverage model ...\\n", flush=TRUE)
fit <- readRDS("{rds}")
cat("Model loaded.\\n", flush=TRUE)

lc100_levels <- levels(fit$data$LC100)
lith_levels  <- levels(fit$data$Lithology)

df <- arrow::read_parquet("{chunk}")
lc100_raw <- as.character(df$LC100)
lith_raw  <- as.character(df$Lithology)
if (anyNA(lc100_raw)) {{
  stop("LC100 contains NA before prediction. Check land-cover reclassification.")
}}
if (anyNA(lith_raw)) {{
  stop("Lithology contains NA before prediction. Check lithology reclassification.")
}}
bad_lc100 <- setdiff(unique(lc100_raw), lc100_levels)
bad_lith  <- setdiff(unique(lith_raw),  lith_levels)
if (length(bad_lc100) > 0L) {{
  stop("Unknown LC100 levels in prediction data: ", paste(bad_lc100, collapse=", "))
}}
if (length(bad_lith) > 0L) {{
  stop("Unknown Lithology levels in prediction data: ", paste(bad_lith, collapse=", "))
}}
df$LC100     <- factor(lc100_raw, levels = lc100_levels)
df$Lithology <- factor(lith_raw,  levels = lith_levels)
df$earthquake  <- factor("new_event")
df$case_weight <- 1.0

n_rows     <- nrow(df)
BATCH_SIZE <- {batch_size}L
NDRAWS     <- {n_draws}L
n_batches  <- ceiling(n_rows / BATCH_SIZE)
cat(sprintf("n_rows=%d  n_batches=%d  NDRAWS=%d\\n", n_rows, n_batches, NDRAWS), flush=TRUE)

draws_mat <- matrix(0.0, nrow=NDRAWS, ncol=n_rows)
all_q025  <- numeric(n_rows)
all_q500  <- numeric(n_rows)
all_q975  <- numeric(n_rows)

for (b in seq_len(n_batches)) {{
  i_from <- (b - 1L) * BATCH_SIZE + 1L
  i_to   <- min(b * BATCH_SIZE, n_rows)
  idx    <- i_from:i_to
  if (n_batches > 1L) {{
    cat(sprintf("  areal coverage batch %d/%d  rows %d-%d\\n", b, n_batches, i_from, i_to), flush=TRUE)
  }}
  set.seed(42L)
  pred_b <- posterior_predict(
    fit, newdata = df[idx, , drop=FALSE],
    allow_new_levels  = TRUE,
    sample_new_levels = "gaussian",
    ndraws = NDRAWS)
  storage.mode(pred_b) <- "double"
  draws_mat[, idx] <- pred_b
  qs_b <- colQuantiles(pred_b, probs = c(0.025, 0.5, 0.975))
  all_q025[idx] <- qs_b[, 1L]
  all_q500[idx] <- qs_b[, 2L]
  all_q975[idx] <- qs_b[, 3L]
  rm(pred_b, qs_b)
}}
gc(verbose=FALSE)

total_sums <- matrixStats::rowSums2(draws_mat)

draws_df <- as.data.frame(t(draws_mat))
colnames(draws_df) <- paste0("d", seq_len(NDRAWS))
storage.mode(draws_df[[1]]) <- "double"
arrow::write_parquet(draws_df, "{dpath}", compression="snappy")

arrow::write_parquet(data.frame(q025=all_q025, q500=all_q500, q975=all_q975), "{spath}")
write.csv(data.frame(pt=total_sums), "{tpath}", row.names=FALSE)
rm(draws_mat, draws_df)
gc(verbose=FALSE)
cat("Areal coverage model done.\\n", flush=TRUE)
""")

    _execute_r_script(r_script, "ls_coverage_predict.R", rscript_exe, tmp_dir, label="areal coverage")
    try:
        os.remove(chunk_path)
    except Exception:
        pass


def _execute_r_script(r_script_str, script_name, rscript_exe, tmp_dir, label="R"):
    """Write R script to disk and execute via subprocess.
    Batch-progress lines update a single-line text progress bar (\\r);
    gc memory lines are filtered; other output printed with [R] prefix.
    """
    _batch_re = re.compile(r'batch\s+(\d+)/(\d+)')
    _gc_re    = re.compile(r'\(Mb\)|Ncells\s+\d|Vcells\s+\d')

    # Remove any accidental flush=TRUE from cat() calls (not a valid R cat() param)
    r_script_str = r_script_str.replace(', flush=TRUE)', ')')
    # Suppress verbose gc() output that some packages enable globally
    r_script_str = r_script_str.replace(
        'options(mc.cores = 1L)',
        'options(mc.cores = 1L)\noptions(verbose=FALSE)'
    )

    script_path = pathlib.Path(tmp_dir) / script_name
    script_path.write_text(r_script_str, encoding='utf-8')

    t0   = time.time()
    proc = subprocess.Popen(
        [rscript_exe, '--vanilla', str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
    )

    in_progress = False
    for line in iter(proc.stdout.readline, ''):
        stripped = line.rstrip()
        if not stripped:
            continue
        # Filter R gc() memory stats lines
        if _gc_re.search(stripped):
            continue
        m = _batch_re.search(stripped)
        if m:
            current = int(m.group(1))
            total   = int(m.group(2))
            pct     = current / total
            filled  = int(pct * 30)
            bar     = '\u2588' * filled + ' ' * (30 - filled)
            print(f"\r  {pct*100:.0f}%|{bar}| {current}/{total}",
                  end='', flush=True)
            in_progress = True
            if current == total:
                print()
                in_progress = False
        else:
            if in_progress:
                print()
                in_progress = False
            # suppress routine R cat() output (model load/done messages)

    if in_progress:
        print()

    proc.stdout.close()
    stderr = proc.stderr.read()
    proc.stderr.close()
    proc.wait()

    if proc.returncode != 0:
        print(f"R stderr (last 3000 chars):\n{stderr[-3000:]}")
        raise RuntimeError(f"R script '{script_name}' exited with code {proc.returncode}")

    print(f"  done in {time.time()-t0:.0f}s")
    try:
        script_path.unlink()
    except Exception:
        pass


# =============================================================================
# Raster helpers
# =============================================================================

def _save_quantile_rasters(q025_vals, q500_vals, q975_vals,
                            pix_row, pix_col, H, W,
                            raster_meta, raster_transform,
                            path_q025, path_q500, path_q975):
    """Fill flat per-pixel arrays back into rasters and save three GeoTIFFs."""
    meta = raster_meta.copy()
    meta.update(dtype='float32', count=1, nodata=-9999.0)

    for vals, out_path in [
        (q025_vals, path_q025),
        (q500_vals, path_q500),
        (q975_vals, path_q975),
    ]:
        grid = np.full((H, W), -9999.0, dtype=np.float32)
        grid[pix_row, pix_col] = vals.astype(np.float32)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(grid, 1)
