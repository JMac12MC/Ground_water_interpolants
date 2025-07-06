"""
Tiled Regional Heatmap Generator

This module creates regional heatmaps by dividing the area into manageable tiles
and processing each tile individually to avoid memory issues with large datasets.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from typing import List, Tuple, Dict, Optional
import time

class TiledRegionalHeatmapGenerator:
    def __init__(self, tile_size_km=20, max_wells_per_tile=1000):
        """
        Initialize the tiled regional heatmap generator
        
        Parameters:
        -----------
        tile_size_km : float
            Size of each tile in kilometers
        max_wells_per_tile : int
            Maximum number of wells to use per tile for performance
        """
        self.tile_size_km = tile_size_km
        self.max_wells_per_tile = max_wells_per_tile
        
        # Canterbury region bounds
        self.bounds = {
            'min_lat': -44.5,  # Southern Canterbury
            'max_lat': -42.5,  # Northern Canterbury
            'min_lng': 170.0,  # Western boundary
            'max_lng': 173.0   # Eastern boundary (coast)
        }
        
        # Convert km to approximate degree offsets (rough approximation for NZ)
        self.lat_km_degree = 1 / 111.0  # ~111 km per degree latitude
        self.lng_km_degree = 1 / 85.0   # ~85 km per degree longitude at NZ latitude
        
        self.tile_cache = {}
        
    def generate_tile_grid(self) -> List[Dict]:
        """
        Generate a grid of tiles covering the Canterbury region
        
        Returns:
        --------
        list
            List of tile definitions with bounds
        """
        tiles = []
        
        # Calculate tile size in degrees
        tile_lat_size = self.tile_size_km * self.lat_km_degree
        tile_lng_size = self.tile_size_km * self.lng_km_degree
        
        # Generate tile grid
        lat = self.bounds['min_lat']
        tile_id = 0
        
        while lat < self.bounds['max_lat']:
            lng = self.bounds['min_lng']
            
            while lng < self.bounds['max_lng']:
                tile = {
                    'id': tile_id,
                    'bounds': {
                        'min_lat': lat,
                        'max_lat': min(lat + tile_lat_size, self.bounds['max_lat']),
                        'min_lng': lng,
                        'max_lng': min(lng + tile_lng_size, self.bounds['max_lng'])
                    },
                    'center_lat': lat + tile_lat_size / 2,
                    'center_lng': lng + tile_lng_size / 2
                }
                tiles.append(tile)
                tile_id += 1
                lng += tile_lng_size
                
            lat += tile_lat_size
            
        return tiles
    
    def get_wells_for_tile(self, wells_df: pd.DataFrame, tile: Dict, buffer_km: float = 5.0) -> pd.DataFrame:
        """
        Get wells within a tile plus buffer area
        
        Parameters:
        -----------
        wells_df : DataFrame
            Complete wells dataset
        tile : dict
            Tile definition with bounds
        buffer_km : float
            Buffer distance in km to include wells outside tile bounds
            
        Returns:
        --------
        DataFrame
            Wells within the tile area
        """
        # Add buffer to tile bounds
        buffer_lat = buffer_km * self.lat_km_degree
        buffer_lng = buffer_km * self.lng_km_degree
        
        bounds = tile['bounds']
        min_lat = bounds['min_lat'] - buffer_lat
        max_lat = bounds['max_lat'] + buffer_lat
        min_lng = bounds['min_lng'] - buffer_lng
        max_lng = bounds['max_lng'] + buffer_lng
        
        # Filter wells within buffered tile bounds
        tile_wells = wells_df[
            (wells_df['latitude'] >= min_lat) &
            (wells_df['latitude'] <= max_lat) &
            (wells_df['longitude'] >= min_lng) &
            (wells_df['longitude'] <= max_lng)
        ].copy()
        
        # Limit number of wells for performance
        if len(tile_wells) > self.max_wells_per_tile:
            tile_wells = tile_wells.sample(n=self.max_wells_per_tile, random_state=42).copy()
            
        return tile_wells
    
    def interpolate_tile(self, tile_wells: pd.DataFrame, tile: Dict, 
                        variable: str = 'depth_to_groundwater', 
                        resolution: int = 50, soil_polygons=None) -> List[List[float]]:
        """
        Interpolate values for a single tile using Ordinary Kriging (same as local interpolations)
        
        Parameters:
        -----------
        tile_wells : DataFrame
            Wells within the tile area
        tile : dict
            Tile definition
        variable : str
            Variable to interpolate
        resolution : int
            Grid resolution for the tile (increased for better quality)
        soil_polygons : GeoDataFrame, optional
            Soil polygons for clipping
            
        Returns:
        --------
        list
            Heatmap data points [[lat, lng, value], ...]
        """
        if len(tile_wells) < 5:  # Need minimum 5 wells for kriging
            return []
            
        # Filter wells with valid data
        valid_wells = tile_wells.dropna(subset=[variable])
        if len(valid_wells) < 5:
            return []
            
        try:
            # Import kriging here to avoid import errors
            from pykrige.ok import OrdinaryKriging
            
            # Create interpolation grid for this tile
            bounds = tile['bounds']
            lat_grid = np.linspace(bounds['min_lat'], bounds['max_lat'], resolution)
            lng_grid = np.linspace(bounds['min_lng'], bounds['max_lng'], resolution)
            grid_lats, grid_lngs = np.meshgrid(lat_grid, lng_grid, indexing='ij')
            
            # Get well coordinates and values  
            well_lngs = valid_wells['longitude'].values
            well_lats = valid_wells['latitude'].values
            values = valid_wells[variable].values
            
            # Create Ordinary Kriging model with spherical variogram (same as local interpolations)
            OK = OrdinaryKriging(
                well_lngs, well_lats, values,
                variogram_model='spherical',
                verbose=False,
                enable_plotting=False,
                coordinates_type='geographic'
            )
            
            # Perform kriging interpolation
            interpolated_grid, variance_grid = OK.execute('grid', lng_grid, lat_grid)
            
            # Convert to heatmap points format
            heatmap_points = []
            for i in range(len(lat_grid)):
                for j in range(len(lng_grid)):
                    lat = lat_grid[i]
                    lng = lng_grid[j]
                    value = interpolated_grid[i, j]
                    
                    # Skip masked/invalid values
                    if not np.isnan(value) and not np.ma.is_masked(value):
                        heatmap_points.append([lat, lng, float(value)])
            
            # Apply soil polygon clipping if available (same as local interpolations)
            if soil_polygons is not None and len(heatmap_points) > 0:
                heatmap_points = self._apply_soil_polygon_clipping(heatmap_points, soil_polygons, tile)
                        
            return heatmap_points
            
        except Exception as e:
            # Fallback to IDW if kriging fails
            return self._idw_fallback(valid_wells, tile, variable, resolution)
    
    def _apply_soil_polygon_clipping(self, heatmap_points, soil_polygons, tile):
        """
        Apply soil polygon clipping to heatmap points (same logic as local interpolations)
        """
        try:
            from shapely.geometry import Point
            import geopandas as gpd
            
            # Create points from heatmap data
            points = [Point(lng, lat) for lat, lng, _ in heatmap_points]
            points_gdf = gpd.GeoDataFrame(
                {'value': [val for _, _, val in heatmap_points]}, 
                geometry=points, 
                crs='EPSG:4326'
            )
            
            # Spatial join with soil polygons
            clipped_points = gpd.sjoin(points_gdf, soil_polygons, how='inner', predicate='within')
            
            # Convert back to heatmap format
            filtered_points = []
            for idx, row in clipped_points.iterrows():
                point = row.geometry
                filtered_points.append([point.y, point.x, row['value']])
                
            return filtered_points
            
        except Exception as e:
            # Return original points if clipping fails
            return heatmap_points
    
    def _idw_fallback(self, valid_wells, tile, variable, resolution):
        """
        IDW fallback interpolation method
        """
        bounds = tile['bounds']
        lat_grid = np.linspace(bounds['min_lat'], bounds['max_lat'], resolution)
        lng_grid = np.linspace(bounds['min_lng'], bounds['max_lng'], resolution)
        
        well_coords = valid_wells[['latitude', 'longitude']].values
        values = valid_wells[variable].values
        
        heatmap_points = []
        
        for lat in lat_grid:
            for lng in lng_grid:
                # Calculate distances to all wells
                point_coords = np.array([[lat, lng]])
                try:
                    distances = np.sqrt(np.sum((point_coords - well_coords) ** 2, axis=1))
                    
                    # Avoid division by zero
                    distances = np.maximum(distances, 1e-6)
                    
                    # IDW interpolation (power = 2)
                    weights = 1 / (distances ** 2)
                    weights_sum = np.sum(weights)
                    
                    if weights_sum > 0:
                        interpolated_value = np.sum(weights * values) / weights_sum
                        heatmap_points.append([lat, lng, float(interpolated_value)])
                except:
                    continue
                    
        return heatmap_points
    
    def process_tile(self, wells_df: pd.DataFrame, tile: Dict, 
                    variable: str = 'depth_to_groundwater', soil_polygons=None) -> Tuple[int, List[List[float]]]:
        """
        Process a single tile (for parallel execution)
        
        Returns:
        --------
        tuple
            (tile_id, heatmap_points)
        """
        try:
            tile_wells = self.get_wells_for_tile(wells_df, tile)
            heatmap_points = self.interpolate_tile(tile_wells, tile, variable, soil_polygons=soil_polygons)
            return tile['id'], heatmap_points
        except Exception as e:
            print(f"Error processing tile {tile['id']}: {e}")
            return tile['id'], []
    
    def generate_regional_heatmap(self, wells_df: pd.DataFrame, 
                                variable: str = 'depth_to_groundwater',
                                max_workers: int = 4, soil_polygons=None) -> List[List[float]]:
        """
        Generate regional heatmap using tiled approach
        
        Parameters:
        -----------
        wells_df : DataFrame
            Complete wells dataset
        variable : str
            Variable to interpolate
        max_workers : int
            Number of parallel workers
            
        Returns:
        --------
        list
            Complete regional heatmap data
        """
        print(f"Generating tiled regional heatmap for {variable}...")
        
        # Filter wells with valid data
        valid_wells = wells_df.dropna(subset=[variable])
        if len(valid_wells) < 10:
            print(f"Insufficient data: only {len(valid_wells)} wells with {variable}")
            return []
            
        print(f"Using {len(valid_wells)} wells with valid {variable} data")
        
        # Generate tile grid
        tiles = self.generate_tile_grid()
        print(f"Processing {len(tiles)} tiles with {max_workers} workers...")
        
        # Process tiles in parallel
        all_heatmap_points = []
        completed_tiles = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tile processing tasks
            future_to_tile = {
                executor.submit(self.process_tile, valid_wells, tile, variable, soil_polygons): tile
                for tile in tiles
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_tile):
                tile_id, heatmap_points = future.result()
                all_heatmap_points.extend(heatmap_points)
                completed_tiles += 1
                
                if completed_tiles % 5 == 0:
                    print(f"Completed {completed_tiles}/{len(tiles)} tiles")
        
        print(f"Generated {len(all_heatmap_points)} heatmap points across {len(tiles)} tiles")
        return all_heatmap_points
    
    def save_heatmap_cache(self, heatmap_data: List[List[float]], 
                          variable: str = 'depth_to_groundwater'):
        """
        Save heatmap data to cache file
        """
        cache_filename = f"tiled_regional_{variable}_heatmap.json"
        try:
            with open(cache_filename, 'w') as f:
                json.dump(heatmap_data, f)
            print(f"Saved heatmap cache to {cache_filename}")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def load_heatmap_cache(self, variable: str = 'depth_to_groundwater') -> Optional[List[List[float]]]:
        """
        Load heatmap data from cache file
        """
        cache_filename = f"tiled_regional_{variable}_heatmap.json"
        try:
            if os.path.exists(cache_filename):
                with open(cache_filename, 'r') as f:
                    data = json.load(f)
                print(f"Loaded cached heatmap from {cache_filename}")
                return data
        except Exception as e:
            print(f"Failed to load cache: {e}")
        return None

def generate_tiled_regional_heatmap(wells_df: pd.DataFrame, 
                                  variable: str = 'depth_to_groundwater',
                                  soil_polygons=None) -> List[List[float]]:
    """
    Convenience function to generate tiled regional heatmap
    
    Parameters:
    -----------
    wells_df : DataFrame
        Complete wells dataset
    variable : str
        Variable to interpolate ('depth_to_groundwater' or 'yield_rate')
    soil_polygons : GeoDataFrame, optional
        Soil polygons for filtering (not used in tiled approach)
        
    Returns:
    --------
    list
        Regional heatmap data
    """
    generator = TiledRegionalHeatmapGenerator()
    
    # Try to load from cache first
    cached_data = generator.load_heatmap_cache(variable)
    if cached_data:
        return cached_data
    
    # Generate new heatmap with soil polygon support
    heatmap_data = generator.generate_regional_heatmap(wells_df, variable, soil_polygons=soil_polygons)
    
    # Save to cache
    if heatmap_data:
        generator.save_heatmap_cache(heatmap_data, variable)
    
    return heatmap_data