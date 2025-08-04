"""
Unified Raster Generation for Seamless Multi-Heatmap Display

This module creates a single continuous raster overlay from multiple heatmap areas,
eliminating gaps and overlaps between adjacent sections.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import folium
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

def create_unified_raster_overlay(stored_heatmaps, heatmap_style='smooth_raster', opacity=0.7):
    """
    Create a single unified raster overlay from multiple stored heatmaps
    
    Parameters:
    -----------
    stored_heatmaps : list
        List of stored heatmap dictionaries with geojson_data
    heatmap_style : str
        Display style ('smooth_raster' or 'triangle_mesh')
    opacity : float
        Opacity of the overlay (0.0 to 1.0)
    
    Returns:
    --------
    folium overlay or None
        Unified raster overlay for the map
    """
    
    if not stored_heatmaps or heatmap_style != 'smooth_raster':
        return None
    
    try:
        print(f"🔄 UNIFIED RASTER: Creating seamless overlay from {len(stored_heatmaps)} heatmap sections")
        
        # Step 1: Collect all data points from all heatmaps
        all_points = []
        all_values = []
        all_bounds = {'min_lat': float('inf'), 'max_lat': float('-inf'), 
                     'min_lon': float('inf'), 'max_lon': float('-inf')}
        
        for i, heatmap in enumerate(stored_heatmaps):
            geojson_data = heatmap.get('geojson_data')
            if not geojson_data or not geojson_data.get('features'):
                continue
                
            heatmap_name = heatmap.get('heatmap_name', f'heatmap_{i}')
            feature_count = len(geojson_data['features'])
            
            # Extract triangle centroids and values
            section_points = []
            section_values = []
            
            for feature in geojson_data['features']:
                try:
                    coords = feature['geometry']['coordinates'][0]
                    value = feature['properties'].get('yield', feature['properties'].get('yield_rate', feature['properties'].get('value', 0)))
                    
                    if len(coords) >= 3 and value is not None:
                        # Calculate triangle centroid
                        lats = [coord[1] for coord in coords[:-1]]  # Remove duplicate closing point
                        lons = [coord[0] for coord in coords[:-1]]
                        
                        if len(lats) >= 3:
                            centroid_lat = sum(lats) / len(lats)
                            centroid_lon = sum(lons) / len(lons)
                            
                            section_points.append([centroid_lat, centroid_lon])
                            section_values.append(float(value))
                            
                            # Update bounds
                            all_bounds['min_lat'] = min(all_bounds['min_lat'], centroid_lat)
                            all_bounds['max_lat'] = max(all_bounds['max_lat'], centroid_lat)
                            all_bounds['min_lon'] = min(all_bounds['min_lon'], centroid_lon)
                            all_bounds['max_lon'] = max(all_bounds['max_lon'], centroid_lon)
                            
                except Exception as e:
                    continue
            
            if section_points:
                all_points.extend(section_points)
                all_values.extend(section_values)
                print(f"  📊 {heatmap_name}: {len(section_points)} data points extracted")
        
        if not all_points:
            print("❌ UNIFIED RASTER: No valid data points found")
            return None
            
        all_points = np.array(all_points)
        all_values = np.array(all_values)
        
        print(f"  🎯 TOTAL DATA: {len(all_points)} points from {len(stored_heatmaps)} sections")
        print(f"  🗺️  BOUNDS: lat {all_bounds['min_lat']:.4f} to {all_bounds['max_lat']:.4f}, "
              f"lon {all_bounds['min_lon']:.4f} to {all_bounds['max_lon']:.4f}")
        print(f"  📈 VALUE RANGE: {all_values.min():.2f} to {all_values.max():.2f}")
        
        # Step 2: Create high-resolution unified grid
        resolution = 1024  # Higher resolution for seamless appearance
        lat_range = all_bounds['max_lat'] - all_bounds['min_lat']
        lon_range = all_bounds['max_lon'] - all_bounds['min_lon']
        
        # Add small padding to ensure complete coverage
        padding = 0.01  # ~1km padding
        grid_lats = np.linspace(all_bounds['min_lat'] - padding, 
                               all_bounds['max_lat'] + padding, resolution)
        grid_lons = np.linspace(all_bounds['min_lon'] - padding, 
                               all_bounds['max_lon'] + padding, resolution)
        
        # Step 3: Perform cubic interpolation across entire area
        print(f"  🔄 INTERPOLATING: {resolution}×{resolution} grid using cubic method")
        
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lons, grid_lats)
        
        # Use cubic interpolation for smoothness
        try:
            interpolated_values = griddata(
                all_points, all_values,
                (grid_lat_2d, grid_lon_2d),
                method='cubic',
                fill_value=np.nan
            )
            
            # Fill NaN areas with linear interpolation
            nan_mask = np.isnan(interpolated_values)
            if np.any(nan_mask):
                linear_fill = griddata(
                    all_points, all_values,
                    (grid_lat_2d, grid_lon_2d),
                    method='linear',
                    fill_value=0
                )
                interpolated_values[nan_mask] = linear_fill[nan_mask]
                
        except Exception as e:
            print(f"  ⚠️  Cubic interpolation failed, using linear: {e}")
            interpolated_values = griddata(
                all_points, all_values,
                (grid_lat_2d, grid_lon_2d),
                method='linear',
                fill_value=0
            )
        
        # Step 4: Apply Gaussian smoothing for professional appearance
        sigma = 1.5  # Smoothing level
        interpolated_values = gaussian_filter(interpolated_values, sigma=sigma)
        print(f"  ✨ SMOOTHING: Applied Gaussian filter (σ={sigma})")
        
        # Step 5: Generate seamless raster image using the same global colormap
        # Import the global colormap function from the main app
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        try:
            # Try to get the global colormap function from app.py
            from app import get_global_unified_color, global_min_value, global_max_value
            
            print(f"  🎨 USING GLOBAL COLORMAP: Range {global_min_value:.2f} to {global_max_value:.2f}")
            
            # Use the exact same crest colormap that the triangular mesh uses
            from matplotlib.colors import to_rgba
            import matplotlib.pyplot as plt
            
            # Get the crest colormap that matches the triangular mesh
            crest_colors = ['#0c2c84', '#225ea8', '#1d91c0', '#41b6c4', '#7fcdbb', 
                           '#c7e9b4', '#edf8b1', '#ffffd9']
            cmap = LinearSegmentedColormap.from_list('crest', crest_colors, N=256)
            
            # Normalize using the SAME global range as triangular mesh
            if global_max_value > global_min_value:
                norm_values = (interpolated_values - global_min_value) / (global_max_value - global_min_value)
            else:
                norm_values = np.zeros_like(interpolated_values)
            norm_values = np.clip(norm_values, 0, 1)
            
            # Generate RGBA image with the same colormap
            rgba_image = cmap(norm_values)
            rgba_image[..., 3] = opacity  # Set opacity
            
            # Make NaN areas transparent
            nan_mask = np.isnan(interpolated_values)
            rgba_image[nan_mask, 3] = 0
                        
        except ImportError:
            # Fallback to simplified colormap if import fails
            print("  ⚠️  Could not import global colormap, using fallback")
            # Use the existing triangular mesh colormap (crest-style)
            colors = ['#0c2c84', '#1e3a8a', '#1d4ed8', '#2563eb', '#3b82f6',
                     '#60a5fa', '#93c5fd', '#bfdbfe', '#dbeafe', '#eff6ff',
                     '#f0f9ff', '#ecfdf5', '#d1fae5', '#a7f3d0', '#6ee7b7',
                     '#34d399', '#10b981', '#059669', '#047857', '#065f46',
                     '#064e3b', '#fbbf24', '#f59e0b', '#d97706', '#b45309']
            
            cmap = LinearSegmentedColormap.from_list('crest', colors, N=256)
            
            # Normalize values for coloring using global range
            vmin, vmax = all_values.min(), all_values.max()
            norm_values = (interpolated_values - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(interpolated_values)
            norm_values = np.clip(norm_values, 0, 1)
            
            # Generate RGBA image
            rgba_image = cmap(norm_values)
            rgba_image[..., 3] = opacity  # Set opacity
        
        # Convert to PIL Image
        rgba_uint8 = (rgba_image * 255).astype(np.uint8)
        pil_image = Image.fromarray(rgba_uint8, mode='RGBA')
        
        # Step 6: Create folium ImageOverlay
        # Convert to base64 for folium
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        img_url = f"data:image/png;base64,{img_base64}"
        
        # Create bounds for the overlay
        bounds = [
            [all_bounds['min_lat'] - padding, all_bounds['min_lon'] - padding],
            [all_bounds['max_lat'] + padding, all_bounds['max_lon'] + padding]
        ]
        
        # Create unified raster overlay
        overlay = folium.raster_layers.ImageOverlay(
            image=img_url,
            bounds=bounds,
            opacity=opacity,
            interactive=False,
            cross_origin=False,
            name="Unified Seamless Heatmap"
        )
        
        print(f"  ✅ UNIFIED RASTER: Created seamless {resolution}×{resolution} overlay")
        print(f"  🎨 COLORMAP: {vmin:.2f} to {vmax:.2f} with {len(colors)} color gradients")
        
        return overlay
        
    except Exception as e:
        print(f"❌ UNIFIED RASTER ERROR: {e}")
        return None

def get_unified_colormap_info(stored_heatmaps):
    """
    Extract colormap information from unified heatmap data
    
    Returns:
    --------
    dict with colormap metadata
    """
    try:
        all_values = []
        
        for heatmap in stored_heatmaps:
            geojson_data = heatmap.get('geojson_data')
            if not geojson_data or not geojson_data.get('features'):
                continue
                
            for feature in geojson_data['features']:
                try:
                    value = feature['properties'].get('yield', feature['properties'].get('yield_rate', feature['properties'].get('value', 0)))
                    if value is not None:
                        all_values.append(float(value))
                except:
                    continue
        
        if not all_values:
            return None
            
        all_values = np.array(all_values)
        
        return {
            'min_value': all_values.min(),
            'max_value': all_values.max(),
            'percentiles': {
                '25th': np.percentile(all_values, 25),
                '50th': np.percentile(all_values, 50),
                '75th': np.percentile(all_values, 75)
            },
            'total_points': len(all_values)
        }
        
    except Exception as e:
        print(f"Error getting unified colormap info: {e}")
        return None