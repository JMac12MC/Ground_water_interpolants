# Composite Heatmap System
# Combines multiple heatmaps into a single optimized raster overlay to prevent WebSocket crashes

import numpy as np
import folium
from PIL import Image, ImageDraw
import io
import base64
from interpolation import generate_smooth_raster_overlay

def create_composite_heatmap_overlay(stored_heatmaps, bounds=None, raster_size=(1024, 1024), opacity=0.7):
    """
    Combine all stored heatmaps into a single composite raster overlay.
    
    This solves the WebSocket payload explosion problem by:
    1. Converting each heatmap from vector to raster
    2. Compositing all rasters into one image
    3. Displaying as single ImageOverlay instead of 98 vector layers
    
    Parameters:
    -----------
    stored_heatmaps : list
        List of stored heatmap dictionaries with GeoJSON data
    bounds : dict
        Overall geographic bounds for the composite
    raster_size : tuple
        Size of the output raster (width, height)
    opacity : float
        Transparency level for the composite overlay
        
    Returns:
    --------
    dict
        Dictionary containing base64 encoded composite image and bounds
    """
    print(f"🎯 COMPOSITE RASTER: Starting composite generation for {len(stored_heatmaps)} heatmaps")
    
    if not stored_heatmaps:
        return None
        
    # Calculate overall bounds if not provided
    if not bounds:
        bounds = calculate_overall_bounds(stored_heatmaps)
        
    # Initialize composite image with the actual dimensions that generate_smooth_raster_overlay produces
    # Based on 100m sampling distance, the actual output size is ~3120×3381 regardless of raster_size parameter
    actual_size = (3120, 3381)  # Height × Width from 100m grid spacing
    composite_array = np.zeros((*actual_size, 4), dtype=np.uint8)  # RGBA
    
    # Process each heatmap and add to composite
    processed_count = 0
    for i, heatmap in enumerate(stored_heatmaps):
        try:
            geojson_data = heatmap.get('geojson_data')
            if not geojson_data or not geojson_data.get('features'):
                continue
                
            # Generate raster for this individual heatmap
            # Force consistent bounds and dimensions for all heatmaps
            standard_bounds = {
                'north': -42.108505,
                'south': -44.916557, 
                'east': 173.904300,
                'west': 169.706745
            }
            
            raster_overlay = generate_smooth_raster_overlay(
                geojson_data=geojson_data,
                bounds=standard_bounds,  # Use consistent bounds
                raster_size=raster_size,  # Force consistent size
                opacity=opacity
            )
            
            if raster_overlay:
                # Decode the base64 image
                image_data = base64.b64decode(raster_overlay['image_base64'])
                heatmap_image = Image.open(io.BytesIO(image_data)).convert('RGBA')
                
                # Resize to ensure consistent dimensions for compositing
                target_size = (actual_size[1], actual_size[0])  # PIL expects (width, height)
                if heatmap_image.size != target_size:
                    print(f"🔧 RESIZE: Resizing heatmap from {heatmap_image.size} to {target_size}")
                    heatmap_image = heatmap_image.resize(target_size, Image.Resampling.LANCZOS)
                
                heatmap_array = np.array(heatmap_image)
                
                # Verify dimensions match before compositing
                if heatmap_array.shape[:2] != composite_array.shape[:2]:
                    print(f"⚠️ DIMENSION MISMATCH: Skipping heatmap with shape {heatmap_array.shape} (expected {composite_array.shape})")
                    continue
                    
                # Composite using alpha blending
                composite_array = alpha_blend(composite_array, heatmap_array)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"🎨 COMPOSITE: Processed {processed_count}/{len(stored_heatmaps)} heatmaps")
                    
        except Exception as e:
            print(f"⚠️ COMPOSITE WARNING: Error processing heatmap {i}: {e}")
            continue
    
    if processed_count == 0:
        print("❌ COMPOSITE ERROR: No heatmaps could be processed")
        return None
        
    # Convert composite array back to image
    composite_image = Image.fromarray(composite_array, 'RGBA')
    
    # Encode to base64
    buffer = io.BytesIO()
    composite_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    print(f"✅ COMPOSITE SUCCESS: Combined {processed_count} heatmaps into single overlay")
    
    return {
        'image_base64': image_base64,
        'bounds': [[bounds['south'], bounds['west']], [bounds['north'], bounds['east']]],
        'opacity': opacity,
        'processed_count': processed_count
    }

def calculate_overall_bounds(stored_heatmaps):
    """Calculate the overall geographic bounds for all heatmaps"""
    north = south = east = west = None
    
    for heatmap in stored_heatmaps:
        geojson_data = heatmap.get('geojson_data')
        if not geojson_data or not geojson_data.get('features'):
            continue
            
        for feature in geojson_data['features']:
            coords = feature['geometry']['coordinates'][0]  # Assuming polygon
            for coord in coords:
                lon, lat = coord[:2]
                
                if north is None:
                    north = south = lat
                    east = west = lon
                else:
                    north = max(north, lat)
                    south = min(south, lat)
                    east = max(east, lon)
                    west = min(west, lon)
    
    return {
        'north': north,
        'south': south,
        'east': east,
        'west': west
    } if north is not None else {'north': -43.0, 'south': -45.0, 'east': 172.0, 'west': 170.0}

def alpha_blend(background, foreground):
    """Alpha blend two RGBA arrays"""
    # Normalize alpha channels
    bg_alpha = background[:, :, 3] / 255.0
    fg_alpha = foreground[:, :, 3] / 255.0
    
    # Calculate output alpha
    out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
    
    # Avoid division by zero
    out_alpha_safe = np.where(out_alpha == 0, 1, out_alpha)
    
    # Blend RGB channels
    result = np.zeros_like(background)
    for i in range(3):  # RGB channels
        result[:, :, i] = (
            foreground[:, :, i] * fg_alpha + 
            background[:, :, i] * bg_alpha * (1 - fg_alpha)
        ) / out_alpha_safe
    
    # Set alpha channel
    result[:, :, 3] = out_alpha * 255
    
    return result.astype(np.uint8)