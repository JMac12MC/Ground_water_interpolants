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
    Align the boundary edge between two adjacent heatmaps.
    
    Parameters:
    -----------
    current_geojson : dict
        GeoJSON data for current heatmap (will be modified)
    adjacent_geojson : dict  
        GeoJSON data for adjacent heatmap (reference for alignment)
    direction : str
        Direction of adjacency ('north', 'south', 'east', 'west')
    current_name : str
        Name of current heatmap for logging
    adjacent_name : str
        Name of adjacent heatmap for logging
    
    Returns:
    --------
    int : Number of boundary points aligned
    """
    
    # Extract boundary coordinates from both heatmaps
    current_boundary = extract_boundary_coords(current_geojson, direction, 'current')
    adjacent_boundary = extract_boundary_coords(adjacent_geojson, get_opposite_direction(direction), 'adjacent')
    
    if not current_boundary or not adjacent_boundary:
        return 0
    
    print(f"    🔍 Aligning {direction} boundary: {len(current_boundary)} current vs {len(adjacent_boundary)} adjacent points")
    
    # Find matching boundary segments and snap coordinates
    alignment_count = 0
    snap_distance_threshold = 0.005  # ~500m tolerance for boundary matching
    
    # For each triangle in current heatmap, check if it has boundary edges that need alignment
    for feature in current_geojson['features']:
        if feature['geometry']['type'] != 'Polygon':
            continue
            
        coords = feature['geometry']['coordinates'][0]
        modified = False
        
        # Check each edge of the triangle
        for j in range(len(coords) - 1):  # -1 because last coord duplicates first
            point1 = coords[j]
            point2 = coords[(j + 1) % (len(coords) - 1)]
            
            # Check if this edge is on the boundary that needs alignment
            if is_boundary_edge(point1, point2, direction, current_boundary):
                # Find nearest points on adjacent boundary and snap to them
                snapped_p1 = snap_to_boundary(point1, adjacent_boundary, snap_distance_threshold)
                snapped_p2 = snap_to_boundary(point2, adjacent_boundary, snap_distance_threshold)
                
                if snapped_p1 or snapped_p2:
                    if snapped_p1:
                        coords[j] = snapped_p1
                        alignment_count += 1
                    if snapped_p2:
                        coords[(j + 1) % (len(coords) - 1)] = snapped_p2
                        alignment_count += 1
                        # Also update the closing coordinate if it matches
                        if j + 1 == len(coords) - 1:
                            coords[-1] = snapped_p2
                    modified = True
        
        # Update the feature if coordinates were modified
        if modified:
            feature['geometry']['coordinates'][0] = coords
    
    return alignment_count

def extract_boundary_coords(geojson_data, direction, role):
    """Extract boundary coordinates for a specific direction."""
    if not geojson_data or not geojson_data.get('features'):
        return []
    
    all_coords = []
    for feature in geojson_data['features']:
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0]
            all_coords.extend(coords[:-1])  # Remove duplicate closing point
    
    if not all_coords:
        return []
    
    # Find boundary coordinates based on direction
    lats = [coord[1] for coord in all_coords]
    lons = [coord[0] for coord in all_coords]
    
    boundary_coords = []
    tolerance = 0.002  # ~200m tolerance for boundary detection
    
    if direction == 'north':
        max_lat = max(lats)
        boundary_coords = [coord for coord in all_coords if coord[1] >= max_lat - tolerance]
    elif direction == 'south':
        min_lat = min(lats)
        boundary_coords = [coord for coord in all_coords if coord[1] <= min_lat + tolerance]
    elif direction == 'east':
        max_lon = max(lons)
        boundary_coords = [coord for coord in all_coords if coord[0] >= max_lon - tolerance]
    elif direction == 'west':
        min_lon = min(lons)
        boundary_coords = [coord for coord in all_coords if coord[0] <= min_lon + tolerance]
    
    # Remove duplicates and sort
    unique_coords = list(set(tuple(coord) for coord in boundary_coords))
    return [list(coord) for coord in unique_coords]

def is_boundary_edge(point1, point2, direction, boundary_coords):
    """Check if an edge is on the specified boundary."""
    boundary_set = set(tuple(coord) for coord in boundary_coords)
    return tuple(point1) in boundary_set or tuple(point2) in boundary_set

def snap_to_boundary(point, boundary_coords, threshold):
    """Snap a point to the nearest boundary coordinate within threshold."""
    min_distance = float('inf')
    nearest_coord = None
    
    for boundary_coord in boundary_coords:
        # Calculate distance between point and boundary coordinate
        distance = np.sqrt((point[0] - boundary_coord[0])**2 + (point[1] - boundary_coord[1])**2)
        
        if distance < min_distance and distance <= threshold:
            min_distance = distance
            nearest_coord = boundary_coord
    
    return nearest_coord if nearest_coord else None

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