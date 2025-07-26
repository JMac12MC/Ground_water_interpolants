#!/usr/bin/env python3

import numpy as np
import math

def calculate_seamless_spacing():
    """Calculate the exact spacing needed for seamless heatmap joining"""
    
    # Last generated heatmap centers from logs
    last_center_lat = -44.036
    last_center_lon = 170.810
    
    print("=== SEAMLESS HEATMAP SPACING CALCULATOR ===")
    print(f"Reference center: ({last_center_lat}, {last_center_lon})")
    print()
    
    # Each heatmap has radius_km = 20, so total width = 40km
    radius_km = 20.0
    heatmap_width_km = radius_km * 2  # 40km total width/height
    
    print(f"Heatmap coverage: {heatmap_width_km}km × {heatmap_width_km}km each")
    print()
    
    # For seamless joining, center-to-center distance must equal heatmap width
    # So adjacent centers should be exactly 40km apart
    
    # Calculate degree conversions at this latitude
    km_per_degree_lat = 111.0  # Constant
    km_per_degree_lon = 111.0 * abs(math.cos(math.radians(last_center_lat)))
    
    print(f"At latitude {last_center_lat}°:")
    print(f"  1° latitude = {km_per_degree_lat:.3f} km")
    print(f"  1° longitude = {km_per_degree_lon:.3f} km")
    print()
    
    # Calculate exact degree offsets for seamless joining
    lat_offset_degrees = heatmap_width_km / km_per_degree_lat
    lon_offset_degrees = heatmap_width_km / km_per_degree_lon
    
    print("EXACT OFFSETS FOR SEAMLESS JOINING:")
    print(f"  Latitude offset: {lat_offset_degrees:.8f}° ({heatmap_width_km}km)")
    print(f"  Longitude offset: {lon_offset_degrees:.8f}° ({heatmap_width_km}km)")
    print()
    
    # Calculate all 6 heatmap centers for perfect grid
    centers = {
        'original': (last_center_lat, last_center_lon),
        'east': (last_center_lat, last_center_lon + lon_offset_degrees),
        'northeast': (last_center_lat, last_center_lon + 2 * lon_offset_degrees),
        'south': (last_center_lat - lat_offset_degrees, last_center_lon),
        'southeast': (last_center_lat - lat_offset_degrees, last_center_lon + lon_offset_degrees),
        'far_southeast': (last_center_lat - lat_offset_degrees, last_center_lon + 2 * lon_offset_degrees)
    }
    
    print("PERFECT SEAMLESS GRID CENTERS:")
    for name, (lat, lon) in centers.items():
        print(f"  {name.upper()}: ({lat:.8f}, {lon:.8f})")
    
    print()
    
    # Verify distances
    from utils import get_distance
    
    print("VERIFICATION - CENTER-TO-CENTER DISTANCES:")
    
    # Horizontal distances
    dist_orig_east = get_distance(centers['original'][0], centers['original'][1],
                                 centers['east'][0], centers['east'][1])
    dist_east_northeast = get_distance(centers['east'][0], centers['east'][1],
                                      centers['northeast'][0], centers['northeast'][1])
    dist_south_southeast = get_distance(centers['south'][0], centers['south'][1],
                                       centers['southeast'][0], centers['southeast'][1])
    dist_southeast_far = get_distance(centers['southeast'][0], centers['southeast'][1],
                                     centers['far_southeast'][0], centers['far_southeast'][1])
    
    print("  Horizontal (should all be 40.00 km):")
    print(f"    ORIGINAL ↔ EAST: {dist_orig_east:.2f} km")
    print(f"    EAST ↔ NORTHEAST: {dist_east_northeast:.2f} km")
    print(f"    SOUTH ↔ SOUTHEAST: {dist_south_southeast:.2f} km")
    print(f"    SOUTHEAST ↔ FAR_SOUTHEAST: {dist_southeast_far:.2f} km")
    
    # Vertical distances
    dist_orig_south = get_distance(centers['original'][0], centers['original'][1],
                                  centers['south'][0], centers['south'][1])
    dist_east_southeast = get_distance(centers['east'][0], centers['east'][1],
                                      centers['southeast'][0], centers['southeast'][1])
    dist_northeast_far = get_distance(centers['northeast'][0], centers['northeast'][1],
                                     centers['far_southeast'][0], centers['far_southeast'][1])
    
    print("  Vertical (should all be 40.00 km):")
    print(f"    ORIGINAL ↕ SOUTH: {dist_orig_south:.2f} km")
    print(f"    EAST ↕ SOUTHEAST: {dist_east_southeast:.2f} km")
    print(f"    NORTHEAST ↕ FAR_SOUTHEAST: {dist_northeast_far:.2f} km")
    
    print()
    
    # Compare with current implementation
    print("COMPARISON WITH CURRENT SEQUENTIAL_HEATMAP.PY:")
    print("Current implementation uses:")
    print("  - Fixed 19.79km offsets")
    print("  - No latitude-dependent longitude correction")
    print()
    print("RECOMMENDED FIXES:")
    print(f"1. Update latitude offset to: {lat_offset_degrees:.8f}°")
    print(f"2. Update longitude offset to: {lon_offset_degrees:.8f}°")
    print("3. These values ensure exact 40km spacing = seamless joining")
    
    return lat_offset_degrees, lon_offset_degrees

if __name__ == "__main__":
    calculate_seamless_spacing()