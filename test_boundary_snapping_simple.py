#!/usr/bin/env python3
"""
Simplified Boundary Snapping Test
=================================

Direct test of boundary snapping logic without complex dependencies.
Tests the core interpolation function to verify boundary coordinates change.
"""

import sys
import numpy as np
import pandas as pd
import json

# Add current directory to path
sys.path.append('.')

def create_test_wells():
    """Create test wells data for boundary snapping verification"""
    # Create wells in a grid pattern around two adjacent heatmap centers
    
    # Heatmap centers (19.82km apart)
    center_lat = -43.5321
    center_lon = 171.7622
    
    # East offset calculation 
    east_offset_km = 19.82
    km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat))
    east_offset_degrees = east_offset_km / km_per_deg_lon
    
    original_center = [center_lat, center_lon]
    east_center = [center_lat, center_lon + east_offset_degrees]
    
    print(f"Original center: {original_center[0]:.6f}, {original_center[1]:.6f}")
    print(f"East center: {east_center[0]:.6f}, {east_center[1]:.6f}")
    print(f"Distance: {east_offset_km:.2f}km ({east_offset_degrees:.8f} degrees)")
    
    # Create wells around both centers
    wells = []
    well_id = 1000
    
    # Grid of wells around original center
    for lat_offset in np.linspace(-0.1, 0.1, 10):
        for lon_offset in np.linspace(-0.1, 0.1, 10):
            lat = original_center[0] + lat_offset
            lon = original_center[1] + lon_offset
            yield_rate = max(0.1, np.random.exponential(3.0))
            
            wells.append({
                'well_id': f'W-{well_id}',
                'latitude': lat,
                'longitude': lon,
                'yield_rate': yield_rate,
                'depth': np.random.uniform(10, 100),
                'status': 'Active'
            })
            well_id += 1
    
    # Grid of wells around east center
    for lat_offset in np.linspace(-0.1, 0.1, 10):
        for lon_offset in np.linspace(-0.1, 0.1, 10):
            lat = east_center[0] + lat_offset
            lon = east_center[1] + lon_offset
            yield_rate = max(0.1, np.random.exponential(3.0))
            
            wells.append({
                'well_id': f'W-{well_id}',
                'latitude': lat,
                'longitude': lon,
                'yield_rate': yield_rate,
                'depth': np.random.uniform(10, 100),
                'status': 'Active'
            })
            well_id += 1
    
    return pd.DataFrame(wells), original_center, east_center

def filter_wells_for_heatmap(wells_df, center_point, radius_km=12.5):
    """Filter wells within square area around center point"""
    center_lat, center_lon = center_point
    
    # Convert radius to degrees
    lat_offset = radius_km / 111.0
    lon_offset = radius_km / (111.0 * np.cos(np.radians(center_lat)))
    
    # Filter wells within square
    filtered = wells_df[
        (abs(wells_df['latitude'] - center_lat) <= lat_offset) &
        (abs(wells_df['longitude'] - center_lon) <= lon_offset)
    ].copy()
    
    return filtered

def extract_boundary_coords(geojson_data):
    """Extract boundary coordinates from GeoJSON triangular mesh"""
    if not geojson_data or 'features' not in geojson_data:
        return None
        
    features = geojson_data['features']
    if not features:
        return None
        
    # Extract all coordinates from triangular features
    all_coords = []
    for feature in features:
        if feature.get('geometry', {}).get('type') == 'Polygon':
            coords = feature['geometry']['coordinates'][0]  # Outer ring
            all_coords.extend(coords)
            
    if not all_coords:
        return None
        
    # Calculate boundary extents
    lons = [coord[0] for coord in all_coords]
    lats = [coord[1] for coord in all_coords]
    
    return {
        'north': max(lats),
        'south': min(lats),
        'east': max(lons),
        'west': min(lons),
        'vertex_count': len(all_coords)
    }

def test_boundary_snapping():
    """Test boundary snapping with simplified direct calls"""
    print("🚀 TESTING BOUNDARY SNAPPING")
    print("=" * 50)
    
    # Import the interpolation function
    try:
        from interpolation import generate_geo_json_grid
        print("✓ Imported interpolation module")
    except Exception as e:
        print(f"✗ Failed to import interpolation: {e}")
        return False
    
    # Create test data
    wells_df, original_center, east_center = create_test_wells()
    print(f"✓ Created {len(wells_df)} test wells")
    
    # Test 1: Generate original heatmap (no adjacent boundaries)
    print(f"\n📍 STEP 1: ORIGINAL HEATMAP (no boundary snapping)")
    print("-" * 40)
    
    original_wells = filter_wells_for_heatmap(wells_df, original_center)
    print(f"Wells for original: {len(original_wells)}")
    
    try:
        original_geojson = generate_geo_json_grid(
            original_wells,
            original_center,
            radius_km=12.5,
            resolution=30,  # Low resolution for speed
            method='indicator_kriging',
            show_variance=False,
            auto_fit_variogram=True,
            variogram_model='spherical',
            soil_polygons=None,
            banks_peninsula_coords=None,
            adjacent_boundaries=None  # No snapping
        )
        
        if original_geojson and 'features' in original_geojson:
            print(f"✓ Original heatmap: {len(original_geojson['features'])} features")
            original_boundaries = extract_boundary_coords(original_geojson)
            print(f"✓ Original boundaries: {original_boundaries}")
        else:
            print("✗ Failed to generate original heatmap")
            return False
            
    except Exception as e:
        print(f"✗ Error generating original heatmap: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Generate east heatmap WITH boundary snapping
    print(f"\n📍 STEP 2: EAST HEATMAP (WITH boundary snapping)")
    print("-" * 40)
    
    east_wells = filter_wells_for_heatmap(wells_df, east_center)
    print(f"Wells for east: {len(east_wells)}")
    
    # Calculate expected boundary before snapping
    center_lat, center_lon = east_center
    radius_km = 12.5
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat))
    
    expected_west = center_lon - (radius_km / km_per_deg_lon)
    target_west = original_boundaries['east']  # Should snap to this
    
    print(f"Expected west (no snap): {expected_west:.8f}")
    print(f"Target west (snap to):   {target_west:.8f}")
    print(f"Difference: {abs(expected_west - target_west):.8f} degrees ({abs(expected_west - target_west) * 111000:.1f}m)")
    
    # Set up adjacent boundaries for snapping
    adjacent_boundaries = {
        'west': original_boundaries['east']
    }
    
    try:
        east_geojson = generate_geo_json_grid(
            east_wells,
            east_center,
            radius_km=12.5,
            resolution=30,  # Low resolution for speed
            method='indicator_kriging',
            show_variance=False,
            auto_fit_variogram=True,
            variogram_model='spherical',
            soil_polygons=None,
            banks_peninsula_coords=None,
            adjacent_boundaries=adjacent_boundaries  # ENABLE SNAPPING
        )
        
        if east_geojson and 'features' in east_geojson:
            print(f"✓ East heatmap: {len(east_geojson['features'])} features")
            east_boundaries = extract_boundary_coords(east_geojson)
            print(f"✓ East boundaries: {east_boundaries}")
        else:
            print("✗ Failed to generate east heatmap")
            return False
            
    except Exception as e:
        print(f"✗ Error generating east heatmap: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Analyze boundary snapping results
    print(f"\n🔍 BOUNDARY SNAPPING ANALYSIS")
    print("=" * 40)
    
    # Check if snapping occurred
    actual_west = east_boundaries['west']
    distance_to_target = abs(actual_west - target_west)
    distance_from_expected = abs(actual_west - expected_west)
    
    print(f"WEST BOUNDARY ANALYSIS:")
    print(f"  Expected (no snap): {expected_west:.8f}")
    print(f"  Target (snap to):   {target_west:.8f}")
    print(f"  Actual result:      {actual_west:.8f}")
    print(f"  ")
    print(f"  Distance moved:     {distance_from_expected:.8f} degrees ({distance_from_expected * 111000:.1f}m)")
    print(f"  Distance to target: {distance_to_target:.8f} degrees ({distance_to_target * 111000:.1f}m)")
    
    # Determine if snapping worked
    SNAP_THRESHOLD = 0.0001  # 10m in degrees
    ACCURACY_THRESHOLD = 0.0001  # 10m accuracy
    
    snapping_detected = distance_from_expected > SNAP_THRESHOLD
    snapping_accurate = distance_to_target < ACCURACY_THRESHOLD
    
    print(f"\n🎯 SNAPPING VERDICT:")
    if snapping_detected:
        print(f"✅ SNAPPING DETECTED: Boundary moved {distance_from_expected * 111000:.1f}m")
        if snapping_accurate:
            print(f"✅ SNAPPING ACCURATE: Within {distance_to_target * 111000:.1f}m of target")
            print(f"🏆 SUCCESS: Boundary snapping is working correctly!")
            return True
        else:
            print(f"⚠️ SNAPPING INACCURATE: {distance_to_target * 111000:.1f}m from target")
            print(f"🔧 PARTIAL SUCCESS: Snapping detected but needs precision improvement")
            return False
    else:
        print(f"❌ NO SNAPPING: Boundary only moved {distance_from_expected * 111000:.1f}m")
        print(f"💥 FAILURE: Boundary snapping is NOT working!")
        return False

if __name__ == "__main__":
    success = test_boundary_snapping()
    if success:
        print(f"\n🎉 TEST PASSED: Boundary snapping is working!")
        sys.exit(0)
    else:
        print(f"\n💥 TEST FAILED: Boundary snapping has issues!")
        sys.exit(1)