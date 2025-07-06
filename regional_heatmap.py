"""
Regional Groundwater Depth Heatmap Generator

This module creates a comprehensive, high-resolution heatmap covering the entire
Canterbury region using all available well data. It provides the same level of 
detail as 20km search areas but covers the full region for default display.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pykrige.ok import OrdinaryKriging
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import folium
from folium.plugins import HeatMap
import json
from shapely.geometry import Point, Polygon
from utils import get_distance
import os

class RegionalHeatmapGenerator:
    def __init__(self):
        """Initialize the regional heatmap generator"""
        self.bounds = {
            'min_lat': -44.5,  # Southern Canterbury
            'max_lat': -42.5,  # Northern Canterbury
            'min_lng': 170.0,  # Western boundary
            'max_lng': 173.0   # Eastern boundary (coast)
        }
        self.resolution = 200  # High resolution grid (200x200)
        self.regional_heatmap_data = None
        self.grid_lats = None
        self.grid_lngs = None
        
    def generate_regional_heatmap(self, wells_df, variable='depth_to_groundwater'):
        """
        Generate a comprehensive regional heatmap using all available wells
        
        Parameters:
        -----------
        wells_df : DataFrame
            Complete wells dataset with all available wells
        variable : str
            Variable to interpolate ('depth_to_groundwater' or 'yield_rate')
        
        Returns:
        --------
        list
            Heatmap data in format [[lat, lng, intensity], ...]
        """
        print(f"Generating regional {variable} heatmap using {len(wells_df)} wells...")
        
        # Filter wells that have valid data for the specified variable
        if variable == 'depth_to_groundwater':
            valid_wells = wells_df.dropna(subset=['depth_to_groundwater'])
            values = valid_wells['depth_to_groundwater'].values
        else:  # yield_rate
            valid_wells = wells_df.dropna(subset=['yield_rate'])
            values = valid_wells['yield_rate'].values
        
        if len(valid_wells) < 10:
            print(f"Insufficient data for {variable} heatmap: only {len(valid_wells)} wells")
            return []
        
        print(f"Using {len(valid_wells)} wells with valid {variable} data")
        print(f"Value range: {values.min():.2f} to {values.max():.2f}")
        
        # Create high-resolution grid
        lat_grid = np.linspace(self.bounds['min_lat'], self.bounds['max_lat'], self.resolution)
        lng_grid = np.linspace(self.bounds['min_lng'], self.bounds['max_lng'], self.resolution)
        grid_lats, grid_lngs = np.meshgrid(lat_grid, lng_grid, indexing='ij')
        
        # Store grid for later use
        self.grid_lats = grid_lats
        self.grid_lngs = grid_lngs
        
        # Get well coordinates and values
        well_coords = np.column_stack([
            valid_wells['latitude'].values,
            valid_wells['longitude'].values
        ])
        
        # Create grid points for interpolation
        grid_points = np.column_stack([grid_lats.ravel(), grid_lngs.ravel()])
        
        try:
            # Use Ordinary Kriging for high-quality interpolation
            print("Performing Ordinary Kriging interpolation...")
            
            # Create kriging model
            ok_model = OrdinaryKriging(
                valid_wells['longitude'].values,
                valid_wells['latitude'].values,
                values,
                variogram_model='spherical',
                enable_plotting=False,
                coordinates_type='geographic'
            )
            
            # Perform interpolation in chunks to manage memory
            chunk_size = 5000
            interpolated_values = np.zeros(len(grid_points))
            
            for i in range(0, len(grid_points), chunk_size):
                end_idx = min(i + chunk_size, len(grid_points))
                chunk_points = grid_points[i:end_idx]
                
                # Interpolate chunk
                z_chunk, _ = ok_model.execute('points', 
                                            chunk_points[:, 1],  # longitude
                                            chunk_points[:, 0])  # latitude
                interpolated_values[i:end_idx] = z_chunk
                
                if (i // chunk_size + 1) % 10 == 0:
                    print(f"Processed {i + len(chunk_points)}/{len(grid_points)} grid points")
            
        except Exception as e:
            print(f"Kriging failed: {e}, using IDW interpolation")
            # Fallback to Inverse Distance Weighting
            interpolated_values = self._idw_interpolation(well_coords, values, grid_points)
        
        # Reshape to grid
        interpolated_grid = interpolated_values.reshape(grid_lats.shape)
        
        # Convert to heatmap format
        heatmap_data = self._grid_to_heatmap_data(grid_lats, grid_lngs, interpolated_grid)
        
        print(f"Generated regional heatmap with {len(heatmap_data)} points")
        
        # Cache the result
        self.regional_heatmap_data = heatmap_data
        
        return heatmap_data
    
    def _idw_interpolation(self, well_coords, values, grid_points, power=2):
        """
        Inverse Distance Weighting interpolation as fallback method
        """
        print("Performing IDW interpolation...")
        
        # Calculate distances from each grid point to all wells
        distances = cdist(grid_points, well_coords)
        
        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)
        
        # Calculate weights (inverse distance with power)
        weights = 1.0 / (distances ** power)
        
        # Normalize weights
        weights_sum = np.sum(weights, axis=1)
        weights_normalized = weights / weights_sum[:, np.newaxis]
        
        # Calculate interpolated values
        interpolated_values = np.sum(weights_normalized * values, axis=1)
        
        return interpolated_values
    
    def _grid_to_heatmap_data(self, grid_lats, grid_lngs, interpolated_grid, 
                             min_percentile=5, max_percentile=95):
        """
        Convert interpolated grid to heatmap data format with outlier filtering
        """
        # Filter outliers using percentiles
        valid_mask = ~np.isnan(interpolated_grid)
        if np.sum(valid_mask) == 0:
            return []
        
        valid_values = interpolated_grid[valid_mask]
        min_val = np.percentile(valid_values, min_percentile)
        max_val = np.percentile(valid_values, max_percentile)
        
        # Normalize values to 0-1 range for heatmap intensity
        normalized_grid = np.clip(
            (interpolated_grid - min_val) / (max_val - min_val), 
            0, 1
        )
        
        # Convert to heatmap format: [[lat, lng, intensity], ...]
        heatmap_data = []
        for i in range(grid_lats.shape[0]):
            for j in range(grid_lats.shape[1]):
                if valid_mask[i, j] and normalized_grid[i, j] > 0.01:  # Skip very low values
                    heatmap_data.append([
                        float(grid_lats[i, j]),
                        float(grid_lngs[i, j]),
                        float(normalized_grid[i, j])
                    ])
        
        return heatmap_data
    
    def apply_soil_polygon_mask(self, heatmap_data, soil_polygons):
        """
        Filter heatmap data to only include points within soil drainage polygons
        
        Parameters:
        -----------
        heatmap_data : list
            Heatmap data points [[lat, lng, intensity], ...]
        soil_polygons : GeoDataFrame
            Soil drainage polygons for masking
        
        Returns:
        --------
        list
            Filtered heatmap data
        """
        if soil_polygons is None or len(soil_polygons) == 0:
            return heatmap_data
        
        print("Applying soil polygon mask to regional heatmap...")
        
        # Create unified geometry for faster point-in-polygon testing
        try:
            unified_geometry = soil_polygons.geometry.unary_union
        except Exception as e:
            print(f"Failed to create unified geometry: {e}")
            return heatmap_data
        
        # Filter heatmap points
        filtered_data = []
        for lat, lng, intensity in heatmap_data:
            point = Point(lng, lat)
            if unified_geometry.contains(point) or unified_geometry.intersects(point):
                filtered_data.append([lat, lng, intensity])
        
        print(f"Filtered heatmap: {len(filtered_data)}/{len(heatmap_data)} points within soil polygons")
        return filtered_data
    
    def save_regional_heatmap(self, heatmap_data, filename='regional_groundwater_heatmap.json'):
        """
        Save regional heatmap data to file for quick loading
        """
        try:
            heatmap_dict = {
                'bounds': self.bounds,
                'resolution': self.resolution,
                'data': heatmap_data,
                'metadata': {
                    'total_points': len(heatmap_data),
                    'generated_at': pd.Timestamp.now().isoformat()
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(heatmap_dict, f, indent=2)
            
            print(f"Saved regional heatmap to {filename}")
            return True
            
        except Exception as e:
            print(f"Failed to save regional heatmap: {e}")
            return False
    
    def load_regional_heatmap(self, filename='regional_groundwater_heatmap.json'):
        """
        Load pre-computed regional heatmap data
        """
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    heatmap_dict = json.load(f)
                
                self.regional_heatmap_data = heatmap_dict['data']
                print(f"Loaded regional heatmap with {len(self.regional_heatmap_data)} points")
                return self.regional_heatmap_data
            else:
                print(f"Regional heatmap file {filename} not found")
                return None
                
        except Exception as e:
            print(f"Failed to load regional heatmap: {e}")
            return None
    
    def get_cached_heatmap(self):
        """
        Get cached regional heatmap data
        """
        return self.regional_heatmap_data


def generate_default_regional_heatmap(wells_df, soil_polygons=None):
    """
    Convenience function to generate regional heatmap for default display
    
    Parameters:
    -----------
    wells_df : DataFrame
        Complete wells dataset
    soil_polygons : GeoDataFrame, optional
        Soil drainage polygons for masking
    
    Returns:
    --------
    list
        Regional heatmap data for default display
    """
    generator = RegionalHeatmapGenerator()
    
    # Try to load existing heatmap first
    cached_data = generator.load_regional_heatmap()
    if cached_data is not None:
        # Apply soil polygon mask if provided
        if soil_polygons is not None:
            cached_data = generator.apply_soil_polygon_mask(cached_data, soil_polygons)
        return cached_data
    
    # Generate new heatmap
    heatmap_data = generator.generate_regional_heatmap(wells_df, 'depth_to_groundwater')
    
    # Apply soil polygon mask
    if soil_polygons is not None:
        heatmap_data = generator.apply_soil_polygon_mask(heatmap_data, soil_polygons)
    
    # Save for future use
    generator.save_regional_heatmap(heatmap_data)
    
    return heatmap_data