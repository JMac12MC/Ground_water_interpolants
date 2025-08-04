"""
Final Boundary Snapping for Triangular Heatmaps

This module performs the final step of extending triangle vertices to snap
exactly to adjacent heatmap boundaries, eliminating gaps between sections.
"""

import numpy as np
import json

def snap_final_triangular_boundaries(stored_heatmaps):
    """
    Final boundary snapping that extends triangle vertices to match adjacent boundaries.
    
    This is the last processing step that ensures seamless connections between
    triangulated heatmap sections by extending boundary vertices to exact matches.
    """
    
    if not stored_heatmaps or len(stored_heatmaps) < 2:
        return stored_heatmaps
    
    print(f"🔧 FINAL BOUNDARY SNAP: Processing {len(stored_heatmaps)} heatmaps")
    
    # Define the 2x3 grid adjacency (simplified)
    adjacencies = [
        (0, 1, 'east'),   # Far SE → SE
        (1, 2, 'east'),   # SE → South  
        (0, 3, 'south'),  # Far SE → NE
        (1, 4, 'south'),  # SE → East
        (2, 5, 'south'),  # South → Original
        (3, 4, 'east'),   # NE → East
        (4, 5, 'east'),   # East → Original
    ]
    
    # Create working copies
    result_heatmaps = []
    for heatmap in stored_heatmaps:
        result_heatmap = heatmap.copy()
        if 'geojson_data' in heatmap and heatmap['geojson_data']:
            result_heatmap['geojson_data'] = json.loads(json.dumps(heatmap['geojson_data']))
        result_heatmaps.append(result_heatmap)
    
    # Process each adjacency relationship
    for idx1, idx2, direction in adjacencies:
        if idx1 >= len(result_heatmaps) or idx2 >= len(result_heatmaps):
            continue
            
        heatmap1 = result_heatmaps[idx1]
        heatmap2 = result_heatmaps[idx2]
        
        if not heatmap1.get('geojson_data') or not heatmap2.get('geojson_data'):
            continue
            
        # Snap boundaries between these two heatmaps
        snap_count = snap_adjacent_boundaries(
            heatmap1['geojson_data'], 
            heatmap2['geojson_data'], 
            direction,
            heatmap1.get('heatmap_name', f'heatmap_{idx1}'),
            heatmap2.get('heatmap_name', f'heatmap_{idx2}')
        )
        
        if snap_count > 0:
            print(f"  ✅ Snapped {snap_count} vertices between {direction} adjacency")
    
    print(f"🔧 FINAL BOUNDARY SNAP: Completed")
    return result_heatmaps

def snap_adjacent_boundaries(geojson1, geojson2, direction, name1, name2):
    """
    Snap boundaries between two adjacent GeoJSON heatmaps.
    
    This extends vertices from the first heatmap to match the boundary of the second.
    """
    
    # Get boundary lines for both heatmaps
    boundary1 = get_boundary_line(geojson1, direction)
    boundary2 = get_boundary_line(geojson2, get_opposite_direction(direction))
    
    if not boundary1 or not boundary2:
        print(f"    ❌ No boundaries found: {len(boundary1)} vs {len(boundary2)}")
        return 0
    
    print(f"    📍 Snapping {direction}: {name1} → {name2}")
    print(f"    🔍 Boundaries: {len(boundary1)} from {name1}, {len(boundary2)} from {name2}")
    
    # Debug: Print boundary sample
    if boundary1:
        print(f"    📌 {name1} {direction} boundary sample: {boundary1[0]} to {boundary1[-1] if len(boundary1) > 1 else 'single point'}")
    if boundary2:
        print(f"    📌 {name2} {get_opposite_direction(direction)} boundary sample: {boundary2[0]} to {boundary2[-1] if len(boundary2) > 1 else 'single point'}")
    
    snap_count = 0
    extension_threshold = 0.05  # 5km - very generous for debugging
    candidates_found = 0
    
    # For each triangle in the first heatmap
    for feature in geojson1['features']:
        if feature['geometry']['type'] != 'Polygon':
            continue
            
        coords = feature['geometry']['coordinates'][0]
        modified = False
        
        # Check each vertex of the triangle
        for i in range(len(coords) - 1):  # Skip duplicate closing coordinate
            vertex = coords[i]
            
            # Check if this vertex is near the boundary that needs snapping
            if is_near_boundary(vertex, boundary1, extension_threshold):
                candidates_found += 1
                # Find the closest point on the adjacent boundary
                snapped_point = find_closest_boundary_point(vertex, boundary2, extension_threshold)
                
                if snapped_point and snapped_point != vertex:
                    # Calculate distance moved
                    distance_moved = np.sqrt((vertex[0] - snapped_point[0])**2 + (vertex[1] - snapped_point[1])**2)
                    print(f"      🎯 Snapping vertex {vertex} → {snapped_point} (moved {distance_moved*111:.0f}m)")
                    
                    coords[i] = snapped_point
                    snap_count += 1
                    modified = True
                    
                    # Update closing coordinate if this is the first vertex
                    if i == 0:
                        coords[-1] = snapped_point
        
        # Update feature if modified
        if modified:
            feature['geometry']['coordinates'][0] = coords
    
    print(f"    ✅ Found {candidates_found} boundary candidates, snapped {snap_count} vertices")
    return snap_count

def get_boundary_line(geojson_data, direction):
    """Get the boundary line coordinates for a specific direction."""
    if not geojson_data or not geojson_data.get('features'):
        return []
    
    all_coords = []
    for feature in geojson_data['features']:
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0]
            all_coords.extend(coords[:-1])  # Remove duplicate closing point
    
    if not all_coords:
        return []
    
    # Get extremes for boundary detection
    lats = [coord[1] for coord in all_coords]
    lons = [coord[0] for coord in all_coords]
    
    print(f"      🔍 Analyzing {len(all_coords)} coordinates for {direction} boundary")
    print(f"      📊 Lat range: {min(lats):.6f} to {max(lats):.6f}")
    print(f"      📊 Lon range: {min(lons):.6f} to {max(lons):.6f}")
    
    tolerance = 0.005  # Larger tolerance for debugging - ~500m
    
    if direction == 'east':
        max_lon = max(lons)
        boundary_coords = [coord for coord in all_coords if coord[0] >= max_lon - tolerance]
        print(f"      🎯 East boundary: max_lon={max_lon:.6f}, found {len(boundary_coords)} coords >= {max_lon-tolerance:.6f}")
    elif direction == 'west':
        min_lon = min(lons)
        boundary_coords = [coord for coord in all_coords if coord[0] <= min_lon + tolerance]
        print(f"      🎯 West boundary: min_lon={min_lon:.6f}, found {len(boundary_coords)} coords <= {min_lon+tolerance:.6f}")
    elif direction == 'north':
        max_lat = max(lats)
        boundary_coords = [coord for coord in all_coords if coord[1] >= max_lat - tolerance]
        print(f"      🎯 North boundary: max_lat={max_lat:.6f}, found {len(boundary_coords)} coords >= {max_lat-tolerance:.6f}")
    elif direction == 'south':
        min_lat = min(lats)
        boundary_coords = [coord for coord in all_coords if coord[1] <= min_lat + tolerance]
        print(f"      🎯 South boundary: min_lat={min_lat:.6f}, found {len(boundary_coords)} coords <= {min_lat+tolerance:.6f}")
    else:
        return []
    
    # Remove duplicates and sort
    unique_coords = list(set(tuple(coord) for coord in boundary_coords))
    if direction in ['east', 'west']:
        unique_coords.sort(key=lambda x: x[1])  # Sort by latitude for vertical boundaries
    else:
        unique_coords.sort(key=lambda x: x[0])  # Sort by longitude for horizontal boundaries
    
    result = [list(coord) for coord in unique_coords]
    print(f"      ✅ Final boundary: {len(result)} unique coordinates")
    return result

def is_near_boundary(vertex, boundary_coords, threshold):
    """Check if a vertex is near any boundary coordinate."""
    for boundary_coord in boundary_coords:
        distance = np.sqrt((vertex[0] - boundary_coord[0])**2 + (vertex[1] - boundary_coord[1])**2)
        if distance <= threshold:
            return True
    return False

def find_closest_boundary_point(vertex, boundary_coords, threshold):
    """Find the closest boundary point to snap to."""
    min_distance = float('inf')
    closest_point = None
    
    for boundary_coord in boundary_coords:
        distance = np.sqrt((vertex[0] - boundary_coord[0])**2 + (vertex[1] - boundary_coord[1])**2)
        
        if distance <= threshold and distance < min_distance:
            min_distance = distance
            closest_point = boundary_coord
    
    return closest_point

def get_opposite_direction(direction):
    """Get the opposite direction for boundary matching."""
    opposites = {
        'north': 'south',
        'south': 'north',
        'east': 'west',
        'west': 'east'
    }
    return opposites.get(direction, direction)