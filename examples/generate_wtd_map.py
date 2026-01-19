#!/usr/bin/env python3
"""
Generate high-resolution Water Table Depth (WTD) map.

This script generates a high-resolution WTD map using a trained Random Forest model
and environmental covariate rasters. Follows the approach from HydroFrame-ML/high-res-WTD-static.

Usage:
    python generate_wtd_map.py \
        --model wtd_rf_model.pkl \
        --covariates covariate_config.json \
        --extent 170.0 -45.0 175.0 -42.0 \
        --resolution 30 \
        --output wtd_map_30m.tif \
        --uncertainty wtd_uncertainty_30m.tif

Author: Groundwater Interpolation System
Date: 2024
"""

import argparse
import json
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window
import joblib
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_model(model_path):
    """Load trained Random Forest model."""
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print(f"Model loaded: {type(model).__name__}")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Number of features: {model.n_features_in_}")
    return model


def load_covariate_config(config_path):
    """
    Load covariate configuration file.
    
    Expected JSON format:
    {
        "elevation": "path/to/elevation.tif",
        "slope": "path/to/slope.tif",
        "aspect": "path/to/aspect.tif",
        ...
    }
    """
    print(f"\nLoading covariate configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Found {len(config)} covariates:")
    for name in config.keys():
        print(f"  - {name}")
    
    return config


def create_output_grid(extent, resolution, crs_epsg=2193):
    """
    Create output grid parameters.
    
    Parameters:
    -----------
    extent : tuple
        (min_lon, min_lat, max_lon, max_lat) in WGS84
    resolution : float
        Grid resolution in meters
    crs_epsg : int
        EPSG code for metric CRS (default: 2193 for NZTM2000)
    """
    import pyproj
    
    min_lon, min_lat, max_lon, max_lat = extent
    
    # Transform extent to metric CRS
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326",  # WGS84
        f"EPSG:{crs_epsg}",
        always_xy=True
    )
    
    min_x, min_y = transformer.transform(min_lon, min_lat)
    max_x, max_y = transformer.transform(max_lon, max_lat)
    
    # Create grid
    x_coords = np.arange(min_x, max_x, resolution)
    y_coords = np.arange(min_y, max_y, resolution)
    
    height = len(y_coords)
    width = len(x_coords)
    
    # Create transform
    transform = from_origin(min_x, max_y, resolution, resolution)
    
    print(f"\nOutput grid:")
    print(f"  CRS: EPSG:{crs_epsg}")
    print(f"  Resolution: {resolution}m")
    print(f"  Extent (metric): ({min_x:.0f}, {min_y:.0f}, {max_x:.0f}, {max_y:.0f})")
    print(f"  Grid size: {width} x {height} = {width*height:,} pixels")
    print(f"  Memory estimate: ~{(width*height*4)/(1024**2):.1f} MB per array")
    
    return {
        'width': width,
        'height': height,
        'transform': transform,
        'crs': CRS.from_epsg(crs_epsg),
        'x_coords': x_coords,
        'y_coords': y_coords
    }


def load_covariates_tiled(covariate_config, grid_params, tile_size=1000):
    """
    Load covariate rasters in tiles to manage memory.
    
    Parameters:
    -----------
    covariate_config : dict
        Mapping of covariate names to file paths
    grid_params : dict
        Output grid parameters
    tile_size : int
        Size of tiles for processing (pixels)
    
    Yields:
    -------
    tile_data : dict
        Contains 'X_tile', 'row_start', 'col_start', 'tile_height', 'tile_width'
    """
    width = grid_params['width']
    height = grid_params['height']
    n_features = len(covariate_config)
    
    # Calculate number of tiles
    n_tiles_x = int(np.ceil(width / tile_size))
    n_tiles_y = int(np.ceil(height / tile_size))
    total_tiles = n_tiles_x * n_tiles_y
    
    print(f"\nProcessing in tiles:")
    print(f"  Tile size: {tile_size} x {tile_size}")
    print(f"  Number of tiles: {n_tiles_x} x {n_tiles_y} = {total_tiles}")
    
    # Open all covariate rasters
    covariate_datasets = {}
    for name, filepath in covariate_config.items():
        covariate_datasets[name] = rasterio.open(filepath)
    
    covariate_names = list(covariate_config.keys())
    
    try:
        # Process each tile
        for tile_y in range(n_tiles_y):
            for tile_x in range(n_tiles_x):
                # Calculate tile bounds
                col_start = tile_x * tile_size
                row_start = tile_y * tile_size
                tile_width = min(tile_size, width - col_start)
                tile_height = min(tile_size, height - row_start)
                
                # Create window
                window = Window(col_start, row_start, tile_width, tile_height)
                
                # Load covariate data for this tile
                X_tile = np.zeros((tile_height * tile_width, n_features))
                
                for i, name in enumerate(covariate_names):
                    src = covariate_datasets[name]
                    data = src.read(1, window=window)
                    X_tile[:, i] = data.flatten()
                
                yield {
                    'X_tile': X_tile,
                    'row_start': row_start,
                    'col_start': col_start,
                    'tile_height': tile_height,
                    'tile_width': tile_width,
                    'window': window
                }
    
    finally:
        # Close all datasets
        for src in covariate_datasets.values():
            src.close()


def predict_wtd(model, X_tile):
    """
    Make WTD predictions and estimate uncertainty.
    
    Returns:
    --------
    predictions : ndarray
        WTD predictions
    uncertainty : ndarray
        Prediction uncertainty (std of tree predictions)
    """
    # Identify valid pixels (no NaN values)
    valid_mask = ~np.isnan(X_tile).any(axis=1)
    
    # Initialize output arrays
    predictions = np.full(X_tile.shape[0], np.nan)
    uncertainty = np.full(X_tile.shape[0], np.nan)
    
    if valid_mask.sum() > 0:
        X_valid = X_tile[valid_mask]
        
        # Mean prediction from all trees
        predictions[valid_mask] = model.predict(X_valid)
        
        # Uncertainty: standard deviation across trees
        tree_predictions = np.array([tree.predict(X_valid) for tree in model.estimators_])
        uncertainty[valid_mask] = tree_predictions.std(axis=0)
    
    return predictions, uncertainty


def main():
    parser = argparse.ArgumentParser(
        description='Generate high-resolution WTD map',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', required=True,
                        help='Path to trained Random Forest model (.pkl)')
    parser.add_argument('--covariates', required=True,
                        help='Path to covariate configuration JSON file')
    parser.add_argument('--extent', nargs=4, type=float, required=True,
                        metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
                        help='Geographic extent in WGS84 degrees')
    parser.add_argument('--resolution', type=float, default=30,
                        help='Grid resolution in meters (default: 30)')
    parser.add_argument('--crs-epsg', type=int, default=2193,
                        help='EPSG code for output CRS (default: 2193 for NZTM2000)')
    parser.add_argument('--output', required=True,
                        help='Output WTD map GeoTIFF file')
    parser.add_argument('--uncertainty', default=None,
                        help='Output uncertainty map GeoTIFF file (optional)')
    parser.add_argument('--tile-size', type=int, default=1000,
                        help='Tile size for processing (default: 1000)')
    parser.add_argument('--nodata', type=float, default=-9999,
                        help='NoData value for output rasters (default: -9999)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("HIGH-RESOLUTION WATER TABLE DEPTH MAP GENERATION")
    print("="*70)
    
    # Load model
    model = load_model(args.model)
    
    # Load covariate configuration
    covariate_config = load_covariate_config(args.covariates)
    
    # Create output grid
    grid_params = create_output_grid(args.extent, args.resolution, args.crs_epsg)
    
    # Initialize output arrays
    wtd_map = np.full((grid_params['height'], grid_params['width']), args.nodata, dtype=np.float32)
    
    if args.uncertainty:
        uncertainty_map = np.full((grid_params['height'], grid_params['width']), args.nodata, dtype=np.float32)
    
    # Process tiles
    print("\nGenerating predictions...")
    n_tiles_x = int(np.ceil(grid_params['width'] / args.tile_size))
    n_tiles_y = int(np.ceil(grid_params['height'] / args.tile_size))
    total_tiles = n_tiles_x * n_tiles_y
    
    with tqdm(total=total_tiles, desc="Processing tiles") as pbar:
        for tile_data in load_covariates_tiled(covariate_config, grid_params, args.tile_size):
            # Make predictions
            predictions, uncertainty = predict_wtd(model, tile_data['X_tile'])
            
            # Reshape and insert into output arrays
            row_start = tile_data['row_start']
            col_start = tile_data['col_start']
            tile_height = tile_data['tile_height']
            tile_width = tile_data['tile_width']
            
            pred_grid = predictions.reshape(tile_height, tile_width)
            wtd_map[row_start:row_start+tile_height, col_start:col_start+tile_width] = pred_grid
            
            if args.uncertainty:
                unc_grid = uncertainty.reshape(tile_height, tile_width)
                uncertainty_map[row_start:row_start+tile_height, col_start:col_start+tile_width] = unc_grid
            
            pbar.update(1)
    
    # Write WTD map
    print(f"\nWriting WTD map to {args.output}...")
    with rasterio.open(
        args.output,
        'w',
        driver='GTiff',
        height=grid_params['height'],
        width=grid_params['width'],
        count=1,
        dtype=np.float32,
        crs=grid_params['crs'],
        transform=grid_params['transform'],
        nodata=args.nodata,
        compress='lzw',
        tiled=True,
        blockxsize=256,
        blockysize=256
    ) as dst:
        dst.write(wtd_map, 1)
        dst.set_band_description(1, 'Water Table Depth (m)')
    
    print(f"✓ WTD map written successfully")
    
    # Write uncertainty map
    if args.uncertainty:
        print(f"\nWriting uncertainty map to {args.uncertainty}...")
        with rasterio.open(
            args.uncertainty,
            'w',
            driver='GTiff',
            height=grid_params['height'],
            width=grid_params['width'],
            count=1,
            dtype=np.float32,
            crs=grid_params['crs'],
            transform=grid_params['transform'],
            nodata=args.nodata,
            compress='lzw',
            tiled=True,
            blockxsize=256,
            blockysize=256
        ) as dst:
            dst.write(uncertainty_map, 1)
            dst.set_band_description(1, 'WTD Prediction Uncertainty (m)')
        
        print(f"✓ Uncertainty map written successfully")
    
    # Summary statistics
    valid_pixels = wtd_map[wtd_map != args.nodata]
    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    print(f"\nOutput: {args.output}")
    print(f"Valid pixels: {len(valid_pixels):,} ({100*len(valid_pixels)/(grid_params['width']*grid_params['height']):.1f}%)")
    print(f"\nWTD Statistics:")
    print(f"  Mean: {valid_pixels.mean():.2f} m")
    print(f"  Std:  {valid_pixels.std():.2f} m")
    print(f"  Min:  {valid_pixels.min():.2f} m")
    print(f"  Max:  {valid_pixels.max():.2f} m")
    
    if args.uncertainty:
        valid_unc = uncertainty_map[uncertainty_map != args.nodata]
        print(f"\nUncertainty Statistics:")
        print(f"  Mean: {valid_unc.mean():.2f} m")
        print(f"  Std:  {valid_unc.std():.2f} m")


if __name__ == '__main__':
    main()
