"""
Boundary Alignment System for Seamless Triangular Heatmaps

This module implements geometric boundary snapping to eliminate gaps between
adjacent triangulated heatmap sections while maintaining scientific accuracy.
"""

import numpy as np
import json
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points
import geopandas as gpd

def align_heatmap_boundaries(stored_heatmaps, heatmap_order=None):
    """
    Align boundaries between adjacent triangulated heatmaps to eliminate gaps.
    
    This function modifies the GeoJSON coordinates of triangular features to ensure
    that adjacent heatmaps share exact boundary coordinates, creating seamless
    transitions without gaps or overlaps.
    
    Parameters:
    -----------
    stored_heatmaps : list
        List of stored heatmap dictionaries with geojson_data
    heatmap_order : list, optional
        Specific order for processing (default: sequential order)
    
    Returns:
    --------
    list : Modified heatmaps with aligned boundaries
    """
    
    if not stored_heatmaps or len(stored_heatmaps) < 2:
        return stored_heatmaps
    
    print(f"🔧 BOUNDARY ALIGNMENT: Processing {len(stored_heatmaps)} heatmaps for seamless boundaries")
    
    # Define the 2x3 grid layout and adjacency relationships
    grid_adjacency = {
        0: {'east': 1, 'south': 3},           # Original (top-left)
        1: {'west': 0, 'east': 2, 'south': 4}, # East (top-center)  
        2: {'west': 1, 'south': 5},           # Northeast (top-right)
        3: {'north': 0, 'east': 4},           # South (bottom-left)
        4: {'west': 3, 'north': 1, 'east': 5}, # Southeast (bottom-center)
        5: {'west': 4, 'north': 2}            # Far Southeast (bottom-right)
    }
    
    # Create working copies of heatmaps
    aligned_heatmaps = []
    for heatmap in stored_heatmaps:
        aligned_heatmap = heatmap.copy()
        if 'geojson_data' in heatmap and heatmap['geojson_data']:
            # Deep copy the GeoJSON data for modification
            aligned_heatmap['geojson_data'] = json.loads(json.dumps(heatmap['geojson_data']))
        aligned_heatmaps.append(aligned_heatmap)
    
    # Process each heatmap and align its boundaries with already-processed neighbors
    for i, current_heatmap in enumerate(aligned_heatmaps):
        current_geojson = current_heatmap.get('geojson_data')
        if not current_geojson or not current_geojson.get('features'):
            continue
            
        heatmap_name = current_heatmap.get('heatmap_name', f'heatmap_{i}')
        print(f"  🎯 Processing {heatmap_name} (index {i})")
        
        # Get adjacency relationships for current heatmap
        if i not in grid_adjacency:
            continue
            
        adjacencies = grid_adjacency[i]
        
        # For each adjacent direction, align boundaries with already-processed heatmaps
        for direction, adjacent_index in adjacencies.items():
            if adjacent_index >= len(aligned_heatmaps) or adjacent_index >= i:
                # Only align with previously processed heatmaps
                continue
                
            adjacent_heatmap = aligned_heatmaps[adjacent_index]
            adjacent_geojson = adjacent_heatmap.get('geojson_data')
            
            if not adjacent_geojson or not adjacent_geojson.get('features'):
                continue
                
            # Perform boundary alignment between current and adjacent heatmap
            alignment_count = align_boundary_edge(
                current_geojson, adjacent_geojson, direction, 
                current_heatmap.get('heatmap_name', f'heatmap_{i}'),
                adjacent_heatmap.get('heatmap_name', f'heatmap_{adjacent_index}')
            )
            
            if alignment_count > 0:
                print(f"    ✅ Aligned {alignment_count} boundary points with {direction} neighbor")
    
    print(f"🔧 BOUNDARY ALIGNMENT: Completed seamless boundary processing")
    return aligned_heatmaps

def align_boundary_edge(current_geojson, adjacent_geojson, direction, current_name, adjacent_name):
    """
    Align the boundary edge between two adjacent heatmaps using precise coordinate snapping.
    """
    
    # Get the exact boundary line for both heatmaps
    current_boundary_line = get_exact_boundary_line(current_geojson, direction)
    adjacent_boundary_line = get_exact_boundary_line(adjacent_geojson, get_opposite_direction(direction))
    
    if not current_boundary_line or not adjacent_boundary_line:
        return 0
    
    print(f"    🔍 Aligning {direction} boundary: {len(current_boundary_line)} current vs {len(adjacent_boundary_line)} adjacent points")
    
    alignment_count = 0
    snap_threshold = 0.01  # 1km tolerance
    
    # Process every triangle and snap boundary vertices to the adjacent boundary
    for feature in current_geojson['features']:
        if feature['geometry']['type'] != 'Polygon':
            continue
            
        coords = feature['geometry']['coordinates'][0]
        modified = False
        
        # Check each vertex of the triangle
        for j in range(len(coords) - 1):  # Skip last duplicate coordinate
            vertex = coords[j]
            
            # Check if this vertex is on the boundary that needs alignment
            if is_on_boundary(vertex, direction):
                # Find the closest point on the adjacent boundary line
                snapped_vertex = find_closest_boundary_point(vertex, adjacent_boundary_line, snap_threshold)
                
                if snapped_vertex and snapped_vertex != vertex:
                    coords[j] = snapped_vertex
                    alignment_count += 1
                    modified = True
                    
                    # Also update the closing coordinate if this is the last vertex
                    if j == 0:
                        coords[-1] = snapped_vertex
        
        # Update the feature coordinates if modified
        if modified:
            feature['geometry']['coordinates'][0] = coords
    
    print(f"    ✅ Successfully aligned {alignment_count} boundary vertices")
    return alignment_count

def get_exact_boundary_line(geojson_data, direction):
    """Get the exact boundary line coordinates for a specific direction."""
    if not geojson_data or not geojson_data.get('features'):
        return []
    
    all_coords = []
    for feature in geojson_data['features']:
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0]
            all_coords.extend(coords[:-1])  # Remove duplicate closing point
    
    if not all_coords:
        return []
    
    # Find the exact boundary line based on direction
    lats = [coord[1] for coord in all_coords]
    lons = [coord[0] for coord in all_coords]
    
    if direction == 'north':
        boundary_lat = max(lats)
        boundary_coords = [coord for coord in all_coords if abs(coord[1] - boundary_lat) < 0.0001]
    elif direction == 'south':
        boundary_lat = min(lats)
        boundary_coords = [coord for coord in all_coords if abs(coord[1] - boundary_lat) < 0.0001]
    elif direction == 'east':
        boundary_lon = max(lons)
        boundary_coords = [coord for coord in all_coords if abs(coord[0] - boundary_lon) < 0.0001]
    elif direction == 'west':
        boundary_lon = min(lons)
        boundary_coords = [coord for coord in all_coords if abs(coord[0] - boundary_lon) < 0.0001]
    else:
        return []
    
    # Remove duplicates and sort appropriately
    unique_coords = list(set(tuple(coord) for coord in boundary_coords))
    if direction in ['north', 'south']:
        # Sort by longitude for horizontal boundaries
        unique_coords.sort(key=lambda x: x[0])
    else:
        # Sort by latitude for vertical boundaries
        unique_coords.sort(key=lambda x: x[1])
    
    return [list(coord) for coord in unique_coords]

def is_on_boundary(vertex, direction):
    """Check if a vertex is potentially on the specified boundary."""
    # This is a simplified check - in practice, we'll determine this during processing
    return True  # We'll check proximity to actual boundary during snapping

def find_closest_boundary_point(vertex, boundary_line, threshold):
    """Find the closest point on the boundary line to snap to."""
    if not boundary_line:
        return None
    
    min_distance = float('inf')
    closest_point = None
    
    # Check distance to each point on the boundary line
    for boundary_point in boundary_line:
        distance = np.sqrt((vertex[0] - boundary_point[0])**2 + (vertex[1] - boundary_point[1])**2)
        
        if distance < min_distance and distance <= threshold:
            min_distance = distance
            closest_point = boundary_point
    
    # If no exact point match, try linear interpolation along boundary segments
    if not closest_point and len(boundary_line) > 1:
        for i in range(len(boundary_line) - 1):
            p1 = boundary_line[i]
            p2 = boundary_line[i + 1]
            
            # Find closest point on line segment p1-p2 to vertex
            interpolated_point = closest_point_on_segment(vertex, p1, p2)
            if interpolated_point:
                distance = np.sqrt((vertex[0] - interpolated_point[0])**2 + (vertex[1] - interpolated_point[1])**2)
                if distance <= threshold and distance < min_distance:
                    min_distance = distance
                    closest_point = interpolated_point
    
    return closest_point

def closest_point_on_segment(point, seg_start, seg_end):
    """Find the closest point on a line segment to a given point."""
    # Vector from seg_start to seg_end
    seg_vec = [seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]]
    # Vector from seg_start to point
    point_vec = [point[0] - seg_start[0], point[1] - seg_start[1]]
    
    # Calculate segment length squared
    seg_len_sq = seg_vec[0]**2 + seg_vec[1]**2
    if seg_len_sq == 0:
        return seg_start  # Segment is a point
    
    # Calculate the projection parameter t
    t = max(0, min(1, (point_vec[0] * seg_vec[0] + point_vec[1] * seg_vec[1]) / seg_len_sq))
    
    # Calculate the closest point on the segment
    closest = [seg_start[0] + t * seg_vec[0], seg_start[1] + t * seg_vec[1]]
    return closest

def get_opposite_direction(direction):
    """Get the opposite direction for boundary matching."""
    opposites = {
        'north': 'south',
        'south': 'north', 
        'east': 'west',
        'west': 'east'
    }
    return opposites.get(direction, direction)

def apply_boundary_alignment_to_stored_heatmaps(stored_heatmaps):
    """
    Apply boundary alignment to stored heatmaps and return modified versions.
    
    This is the main entry point for the boundary alignment system.
    """
    if not stored_heatmaps:
        return stored_heatmaps
        
    # Apply boundary alignment
    aligned_heatmaps = align_heatmap_boundaries(stored_heatmaps)
    
    return aligned_heatmaps