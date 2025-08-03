#!/usr/bin/env python3
"""
Analyze the accuracy of the 0.5 clipping zone boundary system
"""

import math
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

def get_distance_km(lat1, lon1, lat2, lon2):
    """Calculate exact distance between two points using Haversine formula"""
    R = 6371.0  # Earth radius in km
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def precise_offset_for_distance(center_lat, center_lon, target_distance_km, direction):
    """Calculate precise coordinate offset for exact distance using iterative refinement"""
    if direction in ['north', 'south']:
        # Start with approximation
        offset = target_distance_km / 111.0
        test_lat = center_lat + (offset if direction == 'north' else -offset)
        
        # Iterative refinement
        for _ in range(10):  # Maximum 10 iterations
            actual_distance = get_distance_km(center_lat, center_lon, test_lat, center_lon)
            error = target_distance_km - actual_distance
            if abs(error) < 0.001:  # 1 meter precision
                break
            # Adjust offset based on error
            offset_correction = error / 111.0
            offset += offset_correction
            test_lat = center_lat + (offset if direction == 'north' else -offset)
        
        return offset if direction == 'north' else -offset, 0
    else:
        # Start with approximation
        offset = target_distance_km / (111.0 * math.cos(math.radians(center_lat)))
        test_lon = center_lon + (offset if direction == 'east' else -offset)
        
        # Iterative refinement
        for _ in range(10):  # Maximum 10 iterations
            actual_distance = get_distance_km(center_lat, center_lon, center_lat, test_lon)
            error = target_distance_km - actual_distance
            if abs(error) < 0.001:  # 1 meter precision
                break
            # Adjust offset based on error
            offset_correction = error / (111.0 * math.cos(math.radians(center_lat)))
            offset += offset_correction
            test_lon = center_lon + (offset if direction == 'east' else -offset)
        
        return 0, offset if direction == 'east' else -offset

def analyze_clipping_boundary(center_lat, center_lon, search_radius_km=20.0):
    """Analyze the accuracy of the 0.5 clipping boundary system"""
    print(f"🔍 ANALYZING CLIPPING BOUNDARY ACCURACY")
    print(f"=" * 60)
    print(f"Center: ({center_lat:.6f}, {center_lon:.6f})")
    print(f"Original search radius: {search_radius_km} km")
    
    # Step 1: Create the clipping boundary (same as interpolation.py)
    final_clip_factor = 0.5
    final_radius_km = search_radius_km * final_clip_factor
    print(f"Target final radius: {final_radius_km} km (factor: {final_clip_factor})")
    
    # Calculate precise offsets using iterative refinement
    lat_offset_north, _ = precise_offset_for_distance(center_lat, center_lon, final_radius_km, 'north')
    lat_offset_south, _ = precise_offset_for_distance(center_lat, center_lon, final_radius_km, 'south')
    _, lon_offset_east = precise_offset_for_distance(center_lat, center_lon, final_radius_km, 'east')
    _, lon_offset_west = precise_offset_for_distance(center_lat, center_lon, final_radius_km, 'west')
    
    final_clip_lat_radius = lat_offset_north
    final_clip_lon_radius = lon_offset_east
    
    print(f"\n📐 CALCULATED COORDINATE OFFSETS:")
    print(f"   Latitude radius:  {final_clip_lat_radius:.8f}°")
    print(f"   Longitude radius: {final_clip_lon_radius:.8f}°")
    
    # Create the clipping polygon coordinates
    final_clip_polygon_coords = [
        [center_lon - final_clip_lon_radius, center_lat - final_clip_lat_radius],  # SW
        [center_lon + final_clip_lon_radius, center_lat - final_clip_lat_radius],  # SE
        [center_lon + final_clip_lon_radius, center_lat + final_clip_lat_radius],  # NE
        [center_lon - final_clip_lon_radius, center_lat + final_clip_lat_radius],  # NW
        [center_lon - final_clip_lon_radius, center_lat - final_clip_lat_radius]   # Close
    ]
    
    final_clip_geometry = ShapelyPolygon(final_clip_polygon_coords)
    
    print(f"\n🔲 CLIPPING POLYGON CORNERS:")
    for i, corner in enumerate(['SW', 'SE', 'NE', 'NW']):
        coord = final_clip_polygon_coords[i]
        print(f"   {corner}: ({coord[1]:.8f}, {coord[0]:.8f})")
    
    # Step 2: Verify accuracy by measuring actual distances
    print(f"\n📏 BOUNDARY ACCURACY VERIFICATION:")
    
    # Check distances from center to each edge
    north_edge_lat = center_lat + final_clip_lat_radius
    south_edge_lat = center_lat - final_clip_lat_radius
    east_edge_lon = center_lon + final_clip_lon_radius
    west_edge_lon = center_lon - final_clip_lon_radius
    
    north_dist = get_distance_km(center_lat, center_lon, north_edge_lat, center_lon)
    south_dist = get_distance_km(center_lat, center_lon, south_edge_lat, center_lon)
    east_dist = get_distance_km(center_lat, center_lon, center_lat, east_edge_lon)
    west_dist = get_distance_km(center_lat, center_lon, center_lat, west_edge_lon)
    
    print(f"   North edge: {north_dist:.6f} km = {int(north_dist * 1000)} meters")
    print(f"   South edge: {south_dist:.6f} km = {int(south_dist * 1000)} meters")
    print(f"   East edge:  {east_dist:.6f} km = {int(east_dist * 1000)} meters")
    print(f"   West edge:  {west_dist:.6f} km = {int(west_dist * 1000)} meters")
    
    # Check diagonal distances (corners)
    corners = [
        ('SW', south_edge_lat, west_edge_lon),
        ('SE', south_edge_lat, east_edge_lon),
        ('NE', north_edge_lat, east_edge_lon),
        ('NW', north_edge_lat, west_edge_lon)
    ]
    
    print(f"\n🔄 CORNER DISTANCES (should be ~14.14 km for 10km square):")
    expected_diagonal = final_radius_km * math.sqrt(2)  # Diagonal of square
    for corner_name, corner_lat, corner_lon in corners:
        corner_dist = get_distance_km(center_lat, center_lon, corner_lat, corner_lon)
        diagonal_error = abs(corner_dist - expected_diagonal) * 1000
        print(f"   {corner_name}: {corner_dist:.6f} km = {int(corner_dist * 1000)} meters (error: {diagonal_error:.0f}m)")
    
    # Step 3: Test triangle intersection accuracy
    print(f"\n🔺 TRIANGLE INTERSECTION TESTING:")
    
    # Test triangles at various positions relative to boundary
    test_cases = [
        ("Center", center_lat, center_lon, "INSIDE"),
        ("Edge (just inside)", center_lat, center_lon + final_clip_lon_radius * 0.95, "INSIDE"),
        ("Edge (just outside)", center_lat, center_lon + final_clip_lon_radius * 1.05, "OUTSIDE"),
        ("Corner (inside)", center_lat + final_clip_lat_radius * 0.9, center_lon + final_clip_lon_radius * 0.9, "INSIDE"),
        ("Corner (outside)", center_lat + final_clip_lat_radius * 1.1, center_lon + final_clip_lon_radius * 1.1, "OUTSIDE")
    ]
    
    for test_name, test_lat, test_lon, expected in test_cases:
        # Create a small triangle around the test point
        triangle_size = 0.01  # Small triangle in degrees
        triangle_coords = [
            (test_lon, test_lat),
            (test_lon + triangle_size, test_lat),
            (test_lon, test_lat + triangle_size),
            (test_lon, test_lat)  # Close
        ]
        
        triangle_polygon = ShapelyPolygon(triangle_coords)
        intersects = final_clip_geometry.intersects(triangle_polygon)
        contains = final_clip_geometry.contains(triangle_polygon)
        
        distance_to_center = get_distance_km(center_lat, center_lon, test_lat, test_lon)
        
        status = "✅" if (intersects and expected == "INSIDE") or (not intersects and expected == "OUTSIDE") else "❌"
        
        print(f"   {test_name}: {distance_to_center:.2f}km from center")
        print(f"      Intersects: {intersects}, Contains: {contains}, Expected: {expected} {status}")
    
    # Summary
    avg_edge_dist = (north_dist + south_dist + east_dist + west_dist) / 4
    max_edge_error = max(abs(north_dist - 10.0), abs(south_dist - 10.0), 
                        abs(east_dist - 10.0), abs(west_dist - 10.0)) * 1000
    
    print(f"\n📊 ACCURACY SUMMARY:")
    print(f"   Target radius: 10.000 km")
    print(f"   Average radius: {avg_edge_dist:.6f} km")
    print(f"   Maximum error: {max_edge_error:.1f} meters")
    print(f"   Precision level: {'SUB-METER' if max_edge_error < 1 else 'METER' if max_edge_error < 10 else 'DECAMETER'}")
    
    return {
        'target_radius': 10.0,
        'actual_radius': avg_edge_dist,
        'max_error_meters': max_edge_error,
        'clipping_geometry': final_clip_geometry
    }

def main():
    # Test with typical Canterbury coordinates
    test_locations = [
        (-43.560491, 171.915436, "Canterbury Plains"),
        (-43.345155, 171.606445, "North Canterbury"),
        (-44.000000, 171.000000, "South Canterbury")
    ]
    
    for lat, lon, name in test_locations:
        print(f"\n{'='*80}")
        print(f"TESTING LOCATION: {name}")
        print(f"{'='*80}")
        
        result = analyze_clipping_boundary(lat, lon)
        print(f"\n🎯 RESULT: {result['max_error_meters']:.1f}m maximum error for {name}")

if __name__ == "__main__":
    main()