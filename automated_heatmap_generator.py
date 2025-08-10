"""
Automated Heatmap Generator
Leverages the proven sequential heatmap system to generate comprehensive coverage
"""

import streamlit as st
import pandas as pd
import numpy as np

def test_automated_generation(wells_data, interpolation_method, polygon_db, soil_polygons=None, new_clipping_polygon=None, num_tiles=5):
    """
    Generate heatmaps automatically using the exact sequential system logic but extended to cover all well data.
    Uses the proven 19.82km spacing and coordinate conversion from sequential_heatmap.py.
    """
    
    from sequential_heatmap import generate_quad_heatmaps_sequential
    
    print(f"🚀 AUTOMATED GENERATION: Using proven sequential system to cover all well data")
    print(f"📋 Available columns: {list(wells_data.columns)}")
    
    # Find the southwest corner of wells data to start our grid from
    # Check for different possible coordinate column names
    if 'latitude' in wells_data.columns and 'longitude' in wells_data.columns:
        # Data already has lat/lon columns
        valid_wells = wells_data.dropna(subset=['latitude', 'longitude'])
        
        if len(valid_wells) == 0:
            return {"success": False, "error": "No valid well coordinates found"}
        
        lat_coords = valid_wells['latitude'].astype(float)
        lon_coords = valid_wells['longitude'].astype(float)
        
        # Get bounds directly from lat/lon
        sw_lat, ne_lat = lat_coords.min(), lat_coords.max()
        sw_lon, ne_lon = lon_coords.min(), lon_coords.max()
        
        print(f"📍 Well data bounds: SW({sw_lat:.6f}, {sw_lon:.6f}) to NE({ne_lat:.6f}, {ne_lon:.6f})")
        print(f"📊 Processing {len(valid_wells)} wells")
        
    elif 'nztm_x' in wells_data.columns and 'nztm_y' in wells_data.columns:
        # Data has NZTM coordinates, convert to lat/lon
        valid_wells = wells_data.dropna(subset=['nztm_x', 'nztm_y'])
        
        if len(valid_wells) == 0:
            return {"success": False, "error": "No valid well coordinates found"}
        
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
        
        x_coords = valid_wells['nztm_x'].astype(float)
        y_coords = valid_wells['nztm_y'].astype(float)
        
        # Get bounds and convert to lat/lon
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()
        
        # Convert corners to lat/lon
        sw_lon, sw_lat = transformer.transform(min_x, min_y)
        ne_lon, ne_lat = transformer.transform(max_x, max_y)
        
        print(f"📍 Well data bounds: SW({sw_lat:.6f}, {sw_lon:.6f}) to NE({ne_lat:.6f}, {ne_lon:.6f})")
        print(f"📊 Processing {len(valid_wells)} wells")
        
    elif 'X' in wells_data.columns and 'Y' in wells_data.columns:
        # Legacy data format
        valid_wells = wells_data.dropna(subset=['X', 'Y'])
        
        if len(valid_wells) == 0:
            return {"success": False, "error": "No valid well coordinates found"}
        
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
        
        x_coords = valid_wells['X'].astype(float)
        y_coords = valid_wells['Y'].astype(float)
        
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()
        
        sw_lon, sw_lat = transformer.transform(min_x, min_y)
        ne_lon, ne_lat = transformer.transform(max_x, max_y)
        
        print(f"📍 Well data bounds: SW({sw_lat:.6f}, {sw_lon:.6f}) to NE({ne_lat:.6f}, {ne_lon:.6f})")
        print(f"📊 Processing {len(valid_wells)} wells")
        
    else:
        return {"success": False, "error": f"Could not find coordinate columns. Available: {list(wells_data.columns)}"}
    
    # Calculate how many grid positions we need using 19.82km spacing
    # Each heatmap covers 40km diameter, centers are 19.82km apart
    from utils import get_distance
    
    # Calculate grid dimensions needed
    lat_span = ne_lat - sw_lat
    lon_span = ne_lon - sw_lon
    
    # Convert to approximate km using center point
    center_lat = (sw_lat + ne_lat) / 2
    center_lon = (sw_lon + ne_lon) / 2
    
    lat_km = lat_span * 111.0  # degrees to km
    lon_km = lon_span * 111.0 * np.cos(np.radians(center_lat))
    
    # Calculate grid size needed (with 19.82km spacing)
    grid_spacing_km = 19.82
    rows_needed = max(1, int(np.ceil(lat_km / grid_spacing_km)) + 1)
    cols_needed = max(1, int(np.ceil(lon_km / grid_spacing_km)) + 1)
    
    print(f"📐 Data extent: {lat_km:.1f}km × {lon_km:.1f}km")
    print(f"📐 Grid needed: {rows_needed} × {cols_needed} = {rows_needed * cols_needed} heatmaps")
    print(f"📐 Using {min(num_tiles, rows_needed * cols_needed)} tiles for this test")
    
    # Use the existing sequential system but start from a position that will cover the well data
    # Position the starting point so that the grid will encompass the well data area
    center_lat = (sw_lat + ne_lat) / 2
    center_lon = (sw_lon + ne_lon) / 2
    
    # Start from center of data area for better coverage
    start_point = [center_lat, center_lon]
    
    # Call the proven sequential generation function with appropriate grid size
    # For test, use a smaller grid that will fit within the data area
    if num_tiles <= 4:
        actual_grid = (2, 2)  # 2x2 grid for small tests
    elif num_tiles <= 6:
        actual_grid = (2, 3)  # 2x3 grid (original default)
    else:
        test_grid_size = int(np.ceil(np.sqrt(num_tiles)))
        actual_grid = (min(test_grid_size, 3), min(test_grid_size, 3))  # Cap at 3x3 for testing
    
    print(f"📐 Test grid size: {actual_grid[0]} × {actual_grid[1]} (center-positioned for data coverage)")
    
    try:
        result = generate_quad_heatmaps_sequential(
            wells_data, 
            start_point, 
            20,  # search_radius in km
            interpolation_method, 
            polygon_db, 
            soil_polygons, 
            new_clipping_polygon, 
            actual_grid
        )
        
        # Extract results
        if isinstance(result, tuple) and len(result) >= 2:
            success_count, stored_heatmap_ids = result[0], result[1]
            
            print(f"📋 GENERATION RESULTS:")
            print(f"   Grid processed: {actual_grid[0]} × {actual_grid[1]}")
            print(f"   Successful heatmaps: {success_count}")
            print(f"   Stored heatmap IDs: {len(stored_heatmap_ids)}")
            
            return {
                "success": success_count > 0,
                "success_count": success_count,
                "total_heatmaps": len(stored_heatmap_ids),
                "heatmap_ids": stored_heatmap_ids,
                "grid_size": actual_grid,
                "errors": []
            }
        else:
            error_msg = "Unexpected result format from sequential generation"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
            
    except Exception as e:
        error_msg = f"Error in sequential generation: {str(e)}"
        print(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}


def generate_automated_heatmaps(wells_data, interpolation_method, polygon_db, soil_polygons=None, new_clipping_polygon=None, search_radius_km=20, max_tiles=50):
    """
    Generate comprehensive heatmap coverage using the proven sequential system.
    This is the main function called by app.py for full automated generation.
    """
    
    print(f"🚀 FULL AUTOMATED GENERATION: Covering all well data with up to {max_tiles} heatmaps")
    print(f"📋 Available columns: {list(wells_data.columns)}")
    
    from sequential_heatmap import generate_quad_heatmaps_sequential
    
    # Find the bounds of wells data 
    if 'latitude' in wells_data.columns and 'longitude' in wells_data.columns:
        valid_wells = wells_data.dropna(subset=['latitude', 'longitude'])
        
        if len(valid_wells) == 0:
            return 0, [], ["No valid well coordinates found"]
        
        lat_coords = valid_wells['latitude'].astype(float)
        lon_coords = valid_wells['longitude'].astype(float)
        
        sw_lat, ne_lat = lat_coords.min(), lat_coords.max()
        sw_lon, ne_lon = lon_coords.min(), lon_coords.max()
        
        print(f"📍 Well data bounds: SW({sw_lat:.6f}, {sw_lon:.6f}) to NE({ne_lat:.6f}, {ne_lon:.6f})")
        print(f"📊 Processing {len(valid_wells)} wells")
        
    else:
        return 0, [], [f"Could not find coordinate columns. Available: {list(wells_data.columns)}"]
    
    # SMART WELL-DENSITY BASED POSITIONING
    # Instead of rigid grid, find areas with highest well density
    from utils import get_distance
    from sklearn.cluster import KMeans
    
    print(f"🧠 SMART POSITIONING: Analyzing well distribution for optimal heatmap placement...")
    
    # Create coordinate pairs for clustering
    well_coords = valid_wells[['latitude', 'longitude']].values
    
    # Use K-means clustering to find optimal heatmap centers
    n_clusters = min(max_tiles, len(valid_wells) // 50)  # At least 50 wells per heatmap
    n_clusters = max(1, n_clusters)  # At least 1 cluster
    
    print(f"📊 Clustering {len(valid_wells)} wells into {n_clusters} optimal heatmap locations...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_centers = kmeans.fit(well_coords).cluster_centers_
    
    # Verify each cluster center has enough wells within search radius
    optimal_centers = []
    for i, center in enumerate(cluster_centers):
        center_lat, center_lon = center[0], center[1]
        
        # Count wells within search radius of this center
        wells_in_range = 0
        for _, well in valid_wells.iterrows():
            distance = get_distance(center_lat, center_lon, well['latitude'], well['longitude'])
            if distance <= search_radius_km:
                wells_in_range += 1
        
        if wells_in_range >= 10:  # Minimum 10 wells for viable heatmap
            optimal_centers.append([center_lat, center_lon])
            print(f"  📍 Center {i+1}: ({center_lat:.3f}, {center_lon:.3f}) - {wells_in_range} wells within {search_radius_km}km")
        else:
            print(f"  ❌ Center {i+1}: ({center_lat:.3f}, {center_lon:.3f}) - Only {wells_in_range} wells (insufficient)")
    
    if not optimal_centers:
        # Fallback to center-based approach if clustering fails
        center_lat = (sw_lat + ne_lat) / 2
        center_lon = (sw_lon + ne_lon) / 2
        optimal_centers = [[center_lat, center_lon]]
        print(f"⚠️  Clustering produced no viable centers, using fallback center approach")
    
    print(f"✅ Selected {len(optimal_centers)} optimal heatmap locations based on well density")
    
    # Convert to individual heatmap generation calls instead of grid-based
    # This ensures each heatmap is placed where wells actually exist
    
    try:
        # Generate heatmaps individually at optimal locations instead of grid-based
        from interpolation import generate_geo_json_grid, generate_indicator_kriging_mask
        
        generated_heatmaps = []
        stored_heatmap_ids = []
        error_messages = []
        
        total_centers = len(optimal_centers)
        
        for i, center_point in enumerate(optimal_centers):
            try:
                center_lat, center_lon = center_point
                print(f"🔄 Building heatmap {i+1}/{total_centers}: Location ({center_lat:.3f}, {center_lon:.3f})...")
                
                # Filter wells for this location
                wells_df = wells_data.copy()
                wells_df['within_range'] = wells_df.apply(
                    lambda row: get_distance(
                        row['latitude'], row['longitude'], 
                        center_lat, center_lon
                    ) <= search_radius_km, 
                    axis=1
                )
                
                filtered_wells = wells_df[wells_df['within_range']]
                
                if len(filtered_wells) < 10:
                    error_messages.append(f"Location {i+1}: Insufficient wells ({len(filtered_wells)}) within range")
                    continue
                
                print(f"  ✅ Location {i+1}: {len(filtered_wells)} wells found within {search_radius_km}km")
                
                # Generate indicator mask if needed
                indicator_mask = None
                methods_requiring_mask = [
                    'kriging', 'yield_kriging_spherical', 'specific_capacity_kriging', 
                    'depth_kriging', 'depth_kriging_auto', 'ground_water_level_kriging'
                ]
                
                if interpolation_method in methods_requiring_mask:
                    try:
                        indicator_mask = generate_indicator_kriging_mask(
                            filtered_wells.copy(),
                            center_point,
                            search_radius_km,
                            resolution=100,
                            soil_polygons=soil_polygons,
                            threshold=0.7
                        )
                    except Exception as e:
                        print(f"  Warning: Could not generate indicator mask for location {i+1}: {e}")
                
                # Generate heatmap
                geojson_data = generate_geo_json_grid(
                    filtered_wells.copy(),
                    center_point,
                    search_radius_km,
                    resolution=100,
                    method=interpolation_method,
                    show_variance=False,
                    auto_fit_variogram=True,
                    variogram_model='spherical',
                    soil_polygons=soil_polygons,
                    indicator_mask=indicator_mask,
                    new_clipping_polygon=new_clipping_polygon
                )
                
                if geojson_data and len(geojson_data.get('features', [])) > 0:
                    print(f"  ✅ Location {i+1}: Generated {len(geojson_data.get('features', []))} features")
                    
                    # Store in database
                    heatmap_name = f"{interpolation_method}_smart_{center_lat:.3f}_{center_lon:.3f}"
                    
                    # Convert to heatmap data format
                    heatmap_data = []
                    for feature in geojson_data.get('features', []):
                        if 'geometry' in feature and 'properties' in feature:
                            geom = feature['geometry']
                            if geom['type'] == 'Polygon' and len(geom['coordinates']) > 0:
                                coords = geom['coordinates'][0]
                                if len(coords) >= 3:
                                    lat = sum(coord[1] for coord in coords) / len(coords)
                                    lon = sum(coord[0] for coord in coords) / len(coords)
                                    value = feature['properties'].get('yield', 0)
                                    heatmap_data.append([lat, lon, value])
                    
                    # Store in database
                    stored_heatmap_id = polygon_db.store_heatmap(
                        heatmap_name=heatmap_name,
                        center_lat=float(center_lat),
                        center_lon=float(center_lon),
                        radius_km=search_radius_km,
                        interpolation_method=interpolation_method,
                        heatmap_data=heatmap_data,
                        geojson_data=geojson_data,
                        well_count=len(filtered_wells),
                        colormap_metadata=None  # Will be calculated globally later
                    )
                    
                    if stored_heatmap_id:
                        stored_heatmap_ids.append((f"smart_location_{i+1}", stored_heatmap_id))
                        print(f"  💾 Location {i+1}: Stored as ID {stored_heatmap_id}")
                    else:
                        print(f"  ⚠️  Location {i+1}: Already exists in database")
                        
                else:
                    error_messages.append(f"Location {i+1}: Failed to generate heatmap features")
                    print(f"  ❌ Location {i+1}: Generation failed")
                    
            except Exception as e:
                error_messages.append(f"Location {i+1}: {str(e)}")
                print(f"  ❌ Location {i+1}: Exception - {e}")
                continue
        
        success_count = len(stored_heatmap_ids)
        
        print(f"📋 SMART GENERATION RESULTS:")
        print(f"   Optimal locations processed: {total_centers}")
        print(f"   Successful heatmaps: {success_count}")
        print(f"   Stored heatmap IDs: {len(stored_heatmap_ids)}")
        print(f"   Errors: {len(error_messages)}")
        
        return success_count, stored_heatmap_ids, error_messages
        
    except Exception as e:
        error_msg = f"Error in smart automated generation: {str(e)}"
        print(f"❌ {error_msg}")
        return 0, [], [error_msg]


def full_automated_generation(wells_data, interpolation_method, polygon_db, soil_polygons=None, new_clipping_polygon=None):
    """
    Legacy function name - redirect to generate_automated_heatmaps
    """
    return generate_automated_heatmaps(
        wells_data=wells_data,
        interpolation_method=interpolation_method, 
        polygon_db=polygon_db,
        soil_polygons=soil_polygons,
        new_clipping_polygon=new_clipping_polygon,
        max_tiles=100
    )