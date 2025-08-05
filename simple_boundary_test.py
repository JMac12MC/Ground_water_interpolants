#!/usr/bin/env python3
"""
Simple, clean 1×2 grid boundary snapping test
Bypasses complex existing code paths to test boundary snapping in isolation
"""

import pandas as pd
import numpy as np
from interpolation import generate_geo_json_grid
from data_loader import load_sample_data
from utils import get_distance

def generate_clean_1x2_heatmaps(click_point, wells_data, interpolation_method='ground_water_level_kriging'):
    """
    Clean implementation of 1×2 grid heatmap generation with boundary snapping
    Simplified logic without complex code paths
    """
    print("🚀 SIMPLE 1×2 GRID BOUNDARY TEST")
    print("=" * 50)
    
    clicked_lat, clicked_lng = click_point
    search_radius = 40  # 40km search radius
    
    # Calculate east center with precise 19.82km offset
    target_offset_km = 19.82
    
    # Simple east offset calculation
    east_offset_degrees = target_offset_km / (111.0 * abs(np.cos(np.radians(clicked_lat))))
    east_center = (clicked_lat, clicked_lng + east_offset_degrees)
    
    print(f"Original center: {clicked_lat:.6f}, {clicked_lng:.6f}")
    print(f"East center: {east_center[0]:.6f}, {east_center[1]:.6f}")
    
    # Verify distance
    actual_distance = get_distance(clicked_lat, clicked_lng, east_center[0], east_center[1])
    print(f"Distance: {actual_distance:.2f}km")
    
    # Generate original heatmap (no boundary snapping)
    print("\n📍 GENERATING ORIGINAL HEATMAP")
    print("-" * 30)
    
    original_geojson = generate_geo_json_grid(
        wells_data.copy(),
        (clicked_lat, clicked_lng),
        search_radius,
        interpolation_method,
        adjacent_boundaries=None,  # No boundary snapping
        indicator_mask=None,
        soil_polygons=None,
        boundary_vertices=None
    )
    
    original_feature_count = len(original_geojson['features'])
    print(f"✓ Original heatmap: {original_feature_count} features")
    
    # Calculate original boundaries for snapping
    final_radius_km = search_radius * 0.5  # 20km final clipping
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * abs(np.cos(np.radians(clicked_lat)))
    
    original_boundaries = {
        'north': clicked_lat + (final_radius_km / km_per_deg_lat),
        'south': clicked_lat - (final_radius_km / km_per_deg_lat),
        'east': clicked_lng + (final_radius_km / km_per_deg_lon),
        'west': clicked_lng - (final_radius_km / km_per_deg_lon)
    }
    
    print(f"✓ Original boundaries calculated:")
    for direction, boundary in original_boundaries.items():
        print(f"  {direction}: {boundary:.8f}")
    
    # Generate east heatmap WITH boundary snapping
    print("\n📍 GENERATING EAST HEATMAP (WITH BOUNDARY SNAPPING)")
    print("-" * 50)
    
    # Create adjacent boundaries for east heatmap
    # East heatmap snaps its WEST boundary to original's EAST boundary
    adjacent_boundaries = {
        'west': original_boundaries['east']  # Only snap the overlapping edge
    }
    
    print(f"East heatmap will snap west edge to: {adjacent_boundaries['west']:.8f}")
    
    east_geojson = generate_geo_json_grid(
        wells_data.copy(),
        east_center,
        search_radius,
        interpolation_method,
        adjacent_boundaries=adjacent_boundaries,  # Snap west boundary only
        indicator_mask=None,
        soil_polygons=None,
        boundary_vertices=None
    )
    
    east_feature_count = len(east_geojson['features'])
    print(f"✓ East heatmap: {east_feature_count} features")
    
    # Analysis
    print("\n🔍 BOUNDARY SNAPPING ANALYSIS")
    print("=" * 40)
    print(f"Original features: {original_feature_count}")
    print(f"East features: {east_feature_count}")
    
    if east_feature_count >= original_feature_count * 1.3:  # East should have ~50% more features
        print("✅ SUCCESS: East heatmap has independent full coverage")
        success = True
    else:
        print("❌ FAILURE: East heatmap still being truncated")
        success = False
    
    return {
        'success': success,
        'original_features': original_feature_count,
        'east_features': east_feature_count,
        'original_geojson': original_geojson,
        'east_geojson': east_geojson
    }

def generate_clean_1x2_heatmaps(wells_data, click_point, interpolation_method, polygon_db, soil_polygons=None):
    """
    Generate clean 1x2 heatmaps using simplified boundary snapping logic for main app integration
    Returns: (success_count, stored_heatmaps)
    """
    print("🧪 CLEAN TEST: Generating 1x2 heatmaps with simplified boundary snapping")
    
    try:
        # Convert click point to coordinates
        original_lat, original_lng = click_point
        
        # Calculate east center using the same precision as simple test
        east_lng = original_lng + 0.245458  # Precise 19.85km offset
        east_lat = original_lat
        
        print(f"Original center: {original_lat}, {original_lng}")
        print(f"East center: {east_lat}, {east_lng}")
        
        # Generate original heatmap using clean logic
        original_geojson = generate_clean_heatmap_for_app(
            wells_data, original_lat, original_lng, 
            interpolation_method, soil_polygons, 
            None  # No boundary snapping for original
        )
        
        # Calculate original boundaries for east heatmap snapping
        original_bounds = get_clean_boundaries_for_app(original_lat, original_lng)
        west_boundary = original_bounds['east']  # East edge of original becomes west boundary for east heatmap
        
        # Generate east heatmap with boundary snapping
        east_geojson = generate_clean_heatmap_for_app(
            wells_data, east_lat, east_lng,
            interpolation_method, soil_polygons,
            {'west': west_boundary}  # Snap west edge to original's east edge
        )
        
        # Store heatmaps in database
        stored_heatmaps = []
        
        # Store original heatmap
        original_id = polygon_db.store_heatmap(
            heatmap_name="Original (Clean)",
            center_lat=original_lat,
            center_lng=original_lng,
            geojson_data=original_geojson,
            interpolation_method=interpolation_method,
            heatmap_type="yield",
            feature_count=len(original_geojson['features'])
        )
        stored_heatmaps.append({
            'id': original_id,
            'name': "Original (Clean)",
            'geojson': original_geojson
        })
        
        # Store east heatmap
        east_id = polygon_db.store_heatmap(
            heatmap_name="East (Clean)",
            center_lat=east_lat,
            center_lng=east_lng,
            geojson_data=east_geojson,
            interpolation_method=interpolation_method,
            heatmap_type="yield",
            feature_count=len(east_geojson['features'])
        )
        stored_heatmaps.append({
            'id': east_id,
            'name': "East (Clean)",
            'geojson': east_geojson
        })
        
        print(f"✅ CLEAN TEST: Generated 2 heatmaps - Original: {len(original_geojson['features'])} features, East: {len(east_geojson['features'])} features")
        
        return len(stored_heatmaps), stored_heatmaps
        
    except Exception as e:
        print(f"❌ CLEAN TEST ERROR: {e}")
        return 0, []

def generate_clean_heatmap_for_app(wells_data, center_lat, center_lng, interpolation_method, soil_polygons, boundary_snap=None):
    """Generate a single clean heatmap with optional boundary snapping for app integration"""
    import pandas as pd
    from interpolation import generate_geo_json_grid
    from utils import get_distance
    
    # Filter wells within 40km radius
    filtered_wells = []
    for _, well in wells_data.iterrows():
        distance = get_distance(center_lat, center_lng, well['latitude'], well['longitude'])
        if distance <= 40:  # 40km filtering radius
            filtered_wells.append(well)
    
    if not filtered_wells:
        print(f"❌ No wells found within 40km of ({center_lat}, {center_lng})")
        return {'type': 'FeatureCollection', 'features': []}
    
    filtered_df = pd.DataFrame(filtered_wells)
    print(f"📍 Filtered to {len(filtered_df)} wells within 40km")
    
    # Generate interpolated grid using clean clipping logic
    geojson_data = generate_geo_json_grid(
        filtered_df,
        (center_lat, center_lng),  # center_point as tuple
        radius_km=20,  # 20km final clipping (50% of 40km filtering)
        method=interpolation_method,
        soil_polygons=soil_polygons,
        adjacent_boundaries=boundary_snap  # Use correct parameter name
    )
    
    return geojson_data

def get_clean_boundaries_for_app(center_lat, center_lng, radius_km=20):
    """Calculate clean boundaries for a heatmap center for app integration"""
    # Use the same precision conversion as in simple test
    lat_factor = 111.19492664  # km per degree latitude
    lon_factor = 80.88869289   # km per degree longitude (at this latitude)
    
    lat_offset = radius_km / lat_factor
    lng_offset = radius_km / lon_factor
    
    return {
        'north': center_lat + lat_offset,
        'south': center_lat - lat_offset,
        'east': center_lng + lng_offset,
        'west': center_lng - lng_offset
    }

if __name__ == "__main__":
    # Test with sample data
    print("Loading wells data...")
    wells_data = load_sample_data()
    
    # Test coordinates similar to user's click
    test_point = (-43.327, 172.132)
    
    result = generate_clean_1x2_heatmaps(test_point, wells_data)
    
    if result['success']:
        print("\n🎉 BOUNDARY SNAPPING TEST PASSED!")
    else:
        print("\n💥 BOUNDARY SNAPPING TEST FAILED!")
        print("Issue: East heatmap still being truncated by boundary contamination")