# Sequential Heatmap Generation System
# Generates, stores, and displays heatmaps one at a time to prevent crashes

def generate_quad_heatmaps_sequential(wells_data, click_point, search_radius, interpolation_method, polygon_db, soil_polygons=None):
    """
    Generate six heatmaps sequentially (original, east, south, southeast, northeast, far_southeast) to avoid memory issues.
    
    Layout:
    [Original] [East] [Northeast]
    [South] [Southeast] [Far_Southeast]
    
    Returns:
        tuple: (success_count, stored_heatmap_ids, error_messages)
    """
    import streamlit as st
    from interpolation import generate_geo_json_grid, generate_indicator_kriging_mask
    from utils import is_within_square, get_distance
    import numpy as np
    
    # Calculate positions for all four heatmaps using proven 19.79km offset
    clicked_lat, clicked_lng = click_point
    
    # Calculate offset using spherical distance
    km_per_degree_lon = 111.0 * np.cos(np.radians(clicked_lat))
    gap_to_close_km = 0.21  # Measured gap from previous testing
    base_offset_km = 20.0
    east_offset_km = base_offset_km - gap_to_close_km  # 19.79km
    east_offset_degrees = east_offset_km / km_per_degree_lon
    
    # South offset (same distance)
    km_per_degree_lat = 111.0
    south_offset_degrees = east_offset_km / km_per_degree_lat
    
    # Define all six locations in 2x3 grid
    locations = [
        ('original', [clicked_lat, clicked_lng]),
        ('east', [clicked_lat, clicked_lng + east_offset_degrees]),
        ('northeast', [clicked_lat, clicked_lng + (2 * east_offset_degrees)]),  # 5th heatmap: 19.79km east of northeast
        ('south', [clicked_lat - south_offset_degrees, clicked_lng]),
        ('southeast', [clicked_lat - south_offset_degrees, clicked_lng + east_offset_degrees]),
        ('far_southeast', [clicked_lat - south_offset_degrees, clicked_lng + (2 * east_offset_degrees)])  # 6th heatmap: 19.79km south of northeast
    ]
    
    print(f"SEQUENTIAL HEATMAP GENERATION: Starting {len(locations)} heatmaps in 2x3 grid")
    for i, (name, coords) in enumerate(locations):
        print(f"  {i+1}. {name.upper()}: ({coords[0]:.6f}, {coords[1]:.6f})")
    
    # Process each location sequentially
    generated_heatmaps = []
    stored_heatmap_ids = []
    error_messages = []
    
    for i, (location_name, center_point) in enumerate(locations):
        try:
            st.write(f"🔄 Building heatmap {i+1}/6: {location_name.title()} location...")
            
            # Filter wells for this location
            wells_df = wells_data.copy()
            wells_df['within_square'] = wells_df.apply(
                lambda row: is_within_square(
                    row['latitude'], 
                    row['longitude'],
                    center_point[0],
                    center_point[1],
                    search_radius
                ), 
                axis=1
            )
            
            filtered_wells = wells_df[wells_df['within_square']]
            
            if len(filtered_wells) == 0:
                error_messages.append(f"No wells found for {location_name} location")
                continue
                
            print(f"  {location_name.upper()}: {len(filtered_wells)} wells found")
            
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
                        search_radius,
                        resolution=100,
                        soil_polygons=soil_polygons,
                        threshold=0.7
                    )
                except Exception as e:
                    print(f"  Warning: Could not generate indicator mask for {location_name}: {e}")
            
            # Generate heatmap
            geojson_data = generate_geo_json_grid(
                filtered_wells.copy(),
                center_point,
                search_radius,
                resolution=100,
                method=interpolation_method,
                show_variance=False,
                auto_fit_variogram=True,
                variogram_model='spherical',
                soil_polygons=soil_polygons,
                indicator_mask=indicator_mask
            )
            
            if geojson_data and len(geojson_data.get('features', [])) > 0:
                print(f"  ✅ {location_name.upper()}: Generated {len(geojson_data.get('features', []))} features")
                generated_heatmaps.append((location_name, center_point, geojson_data, len(filtered_wells)))
                
                # Store immediately in database
                st.write(f"💾 Storing {location_name} heatmap in database...")
                
                if polygon_db:
                    center_lat, center_lon = center_point
                    center_lat = float(center_lat)
                    center_lon = float(center_lon)
                    
                    # Create unique name
                    if location_name == 'original':
                        heatmap_name = f"{interpolation_method}_{center_lat:.3f}_{center_lon:.3f}"
                    else:
                        heatmap_name = f"{interpolation_method}_{location_name}_{center_lat:.3f}_{center_lon:.3f}"
                    
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
                        center_lat=center_lat,
                        center_lon=center_lon,
                        radius_km=search_radius,
                        interpolation_method=interpolation_method,
                        heatmap_data=heatmap_data,
                        geojson_data=geojson_data,
                        well_count=len(filtered_wells)
                    )
                    
                    if stored_heatmap_id:
                        stored_heatmap_ids.append((location_name, stored_heatmap_id))
                        print(f"  ✅ {location_name.upper()}: Stored as ID {stored_heatmap_id}")
                    else:
                        print(f"  ⚠️  {location_name.upper()}: Already exists in database")
                        
            else:
                error_messages.append(f"Failed to generate {location_name} heatmap - no features")
                print(f"  ❌ {location_name.upper()}: Generation failed")
                
        except Exception as e:
            error_messages.append(f"Error processing {location_name}: {str(e)}")
            print(f"  ❌ {location_name.upper()}: Exception - {e}")
            continue
    
    print(f"SEQUENTIAL PROCESSING COMPLETE: {len(generated_heatmaps)} heatmaps successful, {len(error_messages)} errors")
    
    return len(generated_heatmaps), stored_heatmap_ids, error_messages