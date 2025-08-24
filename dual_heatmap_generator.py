"""
Dual Heatmap Generator for Depth to Groundwater with Indicator Kriging Clipping

This module implements the correct workflow:
1. Generate and store Depth to Groundwater heatmaps
2. Generate and store Indicator Kriging heatmaps  
3. Apply indicator ≥0.7 clipping to depth heatmaps when displaying
"""

import streamlit as st
import numpy as np
from interpolation import generate_geo_json_grid
from database import PolygonDB


def generate_depth_and_indicator_heatmaps(wells_data, click_point, search_radius, polygon_db, soil_polygons=None, new_clipping_polygon=None, grid_size=(2, 3)):
    """
    Generate both depth to groundwater and indicator kriging heatmaps for all grid positions.
    
    Returns:
        tuple: (depth_success_count, indicator_success_count, stored_depth_ids, stored_indicator_ids, error_messages)
    """
    
    print(f"🔄 DUAL HEATMAP GENERATION: Starting depth + indicator kriging workflow")
    
    # Define grid positions (same as existing sequential generation)
    if grid_size == (10, 10):
        # 10x10 grid for comprehensive coverage
        grid_points = []
        base_lat, base_lon = click_point
        for i in range(10):
            for j in range(10):
                # Create 10x10 grid with 17.8km spacing
                lat_offset = (i - 4.5) * 0.16  # Approximately 17.8km spacing
                lon_offset = (j - 4.5) * 0.245
                grid_points.append((base_lat + lat_offset, base_lon + lon_offset))
    else:
        # 2x3 grid (existing 6-heatmap layout)
        center_lat, center_lon = click_point
        
        grid_points = [
            (center_lat, center_lon),                                # Original
            (center_lat, center_lon + 0.245),                      # East
            (center_lat, center_lon + 0.49),                       # Northeast  
            (center_lat - 0.178, center_lon),                      # South
            (center_lat - 0.178, center_lon + 0.245),              # Southeast
            (center_lat - 0.178, center_lon + 0.49),               # Far Southeast
        ]
    
    location_names = [f"location_{i+1}" for i in range(len(grid_points))]
    
    depth_success_count = 0
    indicator_success_count = 0 
    stored_depth_ids = []
    stored_indicator_ids = []
    error_messages = []
    
    print(f"🎯 GRID SETUP: {len(grid_points)} locations for dual generation")
    
    for i, (center_point, location_name) in enumerate(zip(grid_points, location_names)):
        st.write(f"🔄 Building dual heatmaps {i+1}/{len(grid_points)}: {location_name} ({center_point[0]:.6f}, {center_point[1]:.6f})")
        
        # Filter wells for this location
        import pandas as pd
        wells_df = pd.DataFrame(wells_data)
        
        # Apply square-based filtering
        wells_df['within_square'] = wells_df.apply(
            lambda row: (
                abs(row['latitude'] - center_point[0]) <= search_radius * 0.009 and
                abs(row['longitude'] - center_point[1]) <= search_radius * 0.009
            ), 
            axis=1
        )
        
        filtered_wells = wells_df[wells_df['within_square']]
        
        if len(filtered_wells) == 0:
            error_messages.append(f"No wells found for {location_name}")
            continue
            
        print(f"  {location_name.upper()}: {len(filtered_wells)} wells found")
        
        # STEP 1: Generate Depth to Groundwater heatmap
        try:
            print(f"  🌊 DEPTH KRIGING: Generating depth to groundwater heatmap...")
            depth_geojson = generate_geo_json_grid(
                filtered_wells.copy(),
                center_point,
                search_radius,
                resolution=100,
                method='depth_kriging',
                show_variance=False,
                auto_fit_variogram=True,
                variogram_model='spherical',
                soil_polygons=soil_polygons,
                indicator_mask=None,  # No clipping during generation
                new_clipping_polygon=new_clipping_polygon
            )
            
            if depth_geojson and len(depth_geojson.get('features', [])) > 0:
                print(f"  ✅ DEPTH KRIGING: Generated {len(depth_geojson.get('features', []))} features")
                
                # Store depth heatmap in database
                center_lat, center_lon = center_point
                depth_heatmap_name = f"depth_kriging_{location_name}_{center_lat:.3f}_{center_lon:.3f}"
                
                # Convert to heatmap data format
                depth_heatmap_data = []
                for feature in depth_geojson.get('features', []):
                    if 'geometry' in feature and 'properties' in feature:
                        geom = feature['geometry']
                        if geom['type'] == 'Polygon' and len(geom['coordinates']) > 0:
                            coords = geom['coordinates'][0]
                            if len(coords) >= 3:
                                lat = sum(coord[1] for coord in coords) / len(coords)
                                lon = sum(coord[0] for coord in coords) / len(coords)
                                value = feature['properties'].get('depth', 0)
                                depth_heatmap_data.append([lat, lon, value])
                
                # Store in database
                stored_depth_id = polygon_db.store_heatmap(
                    heatmap_name=depth_heatmap_name,
                    center_lat=float(center_lat),
                    center_lon=float(center_lon),
                    radius_km=search_radius,
                    interpolation_method='depth_kriging',
                    heatmap_data=depth_heatmap_data,
                    geojson_data=depth_geojson,
                    global_min=0.0,
                    global_max=50.0,
                    percentiles={'25th': 2.0, '50th': 5.0, '75th': 15.0},
                    total_values=len(depth_heatmap_data)
                )
                
                stored_depth_ids.append(stored_depth_id)
                depth_success_count += 1
                print(f"  💾 DEPTH STORED: ID {stored_depth_id}")
                
            else:
                error_messages.append(f"Failed to generate depth heatmap for {location_name}")
                
        except Exception as e:
            error_messages.append(f"Error generating depth heatmap for {location_name}: {e}")
            print(f"  ❌ DEPTH ERROR: {e}")
        
        # STEP 2: Generate Indicator Kriging heatmap
        try:
            print(f"  🎯 INDICATOR KRIGING: Generating yield suitability heatmap...")
            indicator_geojson = generate_geo_json_grid(
                filtered_wells.copy(),
                center_point,
                search_radius,
                resolution=100,
                method='indicator_kriging',
                show_variance=False,
                auto_fit_variogram=True,
                variogram_model='linear',
                soil_polygons=soil_polygons,
                indicator_mask=None,  # Indicator kriging doesn't use masks
                new_clipping_polygon=new_clipping_polygon
            )
            
            if indicator_geojson and len(indicator_geojson.get('features', [])) > 0:
                print(f"  ✅ INDICATOR KRIGING: Generated {len(indicator_geojson.get('features', []))} features")
                
                # Store indicator heatmap in database
                center_lat, center_lon = center_point  
                indicator_heatmap_name = f"indicator_kriging_{location_name}_{center_lat:.3f}_{center_lon:.3f}"
                
                # Convert to heatmap data format
                indicator_heatmap_data = []
                for feature in indicator_geojson.get('features', []):
                    if 'geometry' in feature and 'properties' in feature:
                        geom = feature['geometry']
                        if geom['type'] == 'Polygon' and len(geom['coordinates']) > 0:
                            coords = geom['coordinates'][0]
                            if len(coords) >= 3:
                                lat = sum(coord[1] for coord in coords) / len(coords)
                                lon = sum(coord[0] for coord in coords) / len(coords)
                                value = feature['properties'].get('probability', 0)
                                indicator_heatmap_data.append([lat, lon, value])
                
                # Store in database
                stored_indicator_id = polygon_db.store_heatmap(
                    heatmap_name=indicator_heatmap_name,
                    center_lat=float(center_lat),
                    center_lon=float(center_lon),
                    radius_km=search_radius,
                    interpolation_method='indicator_kriging',
                    heatmap_data=indicator_heatmap_data,
                    geojson_data=indicator_geojson,
                    global_min=0.0,
                    global_max=1.0,
                    percentiles={'25th': 0.3, '50th': 0.5, '75th': 0.8},
                    total_values=len(indicator_heatmap_data)
                )
                
                stored_indicator_ids.append(stored_indicator_id)
                indicator_success_count += 1
                print(f"  💾 INDICATOR STORED: ID {stored_indicator_id}")
                
            else:
                error_messages.append(f"Failed to generate indicator heatmap for {location_name}")
                
        except Exception as e:
            error_messages.append(f"Error generating indicator heatmap for {location_name}: {e}")
            print(f"  ❌ INDICATOR ERROR: {e}")
    
    print(f"🏁 DUAL GENERATION COMPLETE: {depth_success_count} depth + {indicator_success_count} indicator heatmaps")
    
    return depth_success_count, indicator_success_count, stored_depth_ids, stored_indicator_ids, error_messages


def apply_indicator_clipping_to_depth_heatmaps(polygon_db, depth_heatmaps, indicator_heatmaps, threshold=0.7):
    """
    Apply indicator kriging clipping to depth to groundwater heatmaps.
    
    Args:
        polygon_db: Database instance
        depth_heatmaps: List of stored depth heatmap records
        indicator_heatmaps: List of stored indicator heatmap records  
        threshold: Probability threshold for clipping (default 0.7)
        
    Returns:
        list: Clipped depth heatmaps ready for display
    """
    
    print(f"🎭 INDICATOR CLIPPING: Applying ≥{threshold} threshold to {len(depth_heatmaps)} depth heatmaps")
    
    clipped_heatmaps = []
    
    for depth_heatmap in depth_heatmaps:
        # Find matching indicator heatmap for the same location
        depth_name = depth_heatmap.get('heatmap_name', '')
        location_part = depth_name.replace('depth_kriging_', '').replace('indicator_kriging_', '')
        
        matching_indicator = None
        for indicator_heatmap in indicator_heatmaps:
            indicator_name = indicator_heatmap.get('heatmap_name', '')
            if location_part in indicator_name:
                matching_indicator = indicator_heatmap
                break
        
        if matching_indicator is None:
            print(f"  ⚠️ NO INDICATOR MATCH: {depth_name} - displaying unclipped")
            clipped_heatmaps.append(depth_heatmap)
            continue
        
        try:
            # Apply clipping using indicator data
            depth_geojson = depth_heatmap.get('geojson_data', {})
            indicator_geojson = matching_indicator.get('geojson_data', {})
            
            if not depth_geojson.get('features') or not indicator_geojson.get('features'):
                print(f"  ⚠️ EMPTY DATA: {depth_name} - displaying unclipped")
                clipped_heatmaps.append(depth_heatmap)
                continue
            
            # Create indicator mask from probability data
            indicator_mask = {}
            for feature in indicator_geojson.get('features', []):
                if 'geometry' in feature and 'properties' in feature:
                    geom = feature['geometry']
                    if geom['type'] == 'Polygon' and len(geom['coordinates']) > 0:
                        coords = geom['coordinates'][0]
                        if len(coords) >= 3:
                            lat = sum(coord[1] for coord in coords) / len(coords)
                            lon = sum(coord[0] for coord in coords) / len(coords)
                            prob = feature['properties'].get('probability', 0)
                            indicator_mask[f"{lat:.6f},{lon:.6f}"] = prob
            
            # Filter depth features based on indicator threshold
            clipped_features = []
            original_count = len(depth_geojson.get('features', []))
            
            for feature in depth_geojson.get('features', []):
                if 'geometry' in feature and 'properties' in feature:
                    geom = feature['geometry']
                    if geom['type'] == 'Polygon' and len(geom['coordinates']) > 0:
                        coords = geom['coordinates'][0]
                        if len(coords) >= 3:
                            lat = sum(coord[1] for coord in coords) / len(coords)
                            lon = sum(coord[0] for coord in coords) / len(coords)
                            coord_key = f"{lat:.6f},{lon:.6f}"
                            
                            # Check if this location has high probability (≥threshold)
                            probability = indicator_mask.get(coord_key, 0)
                            if probability >= threshold:
                                clipped_features.append(feature)
            
            # Create clipped heatmap
            clipped_geojson = {
                "type": "FeatureCollection",
                "features": clipped_features
            }
            
            clipped_heatmap = depth_heatmap.copy()
            clipped_heatmap['geojson_data'] = clipped_geojson
            
            clipped_count = len(clipped_features)
            print(f"  ✅ CLIPPED: {depth_name} - {clipped_count}/{original_count} features kept (≥{threshold})")
            
            clipped_heatmaps.append(clipped_heatmap)
            
        except Exception as e:
            print(f"  ❌ CLIPPING ERROR: {depth_name} - {e}")
            clipped_heatmaps.append(depth_heatmap)  # Use unclipped as fallback
    
    print(f"🎭 CLIPPING COMPLETE: {len(clipped_heatmaps)} heatmaps processed")
    
    return clipped_heatmaps