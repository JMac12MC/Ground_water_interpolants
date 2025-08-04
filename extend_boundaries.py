"""
Boundary Extension System for Triangular Heatmaps

This module extends boundary triangles to create overlapping coverage,
eliminating gaps between heatmap sections by extending boundary coordinates
towards adjacent sections.
"""

import numpy as np
import json

def extend_triangular_boundaries(stored_heatmaps):
    """
    Extend boundaries of triangular heatmaps to eliminate gaps.
    
    Instead of snapping to distant boundaries, this extends the boundary
    coordinates of each heatmap section to overlap with adjacent areas.
    """
    
    if not stored_heatmaps or len(stored_heatmaps) < 2:
        return stored_heatmaps
    
    print(f"🔧 BOUNDARY EXTENSION: Processing {len(stored_heatmaps)} heatmaps for gap elimination")
    
    # Define extension directions and amounts for 2x3 grid
    extensions = {
        0: {'east': 0.01, 'south': 0.01},     # Far SE: extend east and south
        1: {'west': 0.01, 'east': 0.01, 'south': 0.01},  # SE: extend all directions
        2: {'west': 0.01, 'south': 0.01},     # South: extend west and south  
        3: {'east': 0.01, 'north': 0.01},     # NE: extend east and north
        4: {'west': 0.01, 'east': 0.01, 'north': 0.01},  # East: extend all directions
        5: {'west': 0.01, 'north': 0.01}      # Original: extend west and north
    }
    
    # Create working copies
    result_heatmaps = []
    for heatmap in stored_heatmaps:
        result_heatmap = heatmap.copy()
        if 'geojson_data' in heatmap and heatmap['geojson_data']:
            result_heatmap['geojson_data'] = json.loads(json.dumps(heatmap['geojson_data']))
        result_heatmaps.append(result_heatmap)
    
    # Extend boundaries for each heatmap
    for i, heatmap in enumerate(result_heatmaps):
        if i not in extensions or not heatmap.get('geojson_data'):
            continue
            
        heatmap_name = heatmap.get('heatmap_name', f'heatmap_{i}')
        print(f"  🎯 Extending {heatmap_name} boundaries")
        
        extension_directions = extensions[i]
        extended_count = extend_heatmap_boundaries(
            heatmap['geojson_data'], 
            extension_directions,
            heatmap_name
        )
        
        if extended_count > 0:
            print(f"    ✅ Extended {extended_count} boundary vertices")
    
    print(f"🔧 BOUNDARY EXTENSION: Completed gap elimination")
    return result_heatmaps

def extend_heatmap_boundaries(geojson_data, directions, heatmap_name):
    """
    Extend the boundaries of a single heatmap in specified directions.
    """
    
    if not geojson_data or not geojson_data.get('features'):
        return 0
    
    # Find boundary extremes
    all_coords = []
    for feature in geojson_data['features']:
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0]
            all_coords.extend(coords[:-1])
    
    if not all_coords:
        return 0
    
    lats = [coord[1] for coord in all_coords]
    lons = [coord[0] for coord in all_coords]
    
    # Calculate boundary thresholds
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    # Use tighter tolerance for boundary detection
    boundary_tolerance = 0.002  # ~200m
    
    print(f"    📊 Coordinate ranges: lat {min_lat:.6f} to {max_lat:.6f}, lon {min_lon:.6f} to {max_lon:.6f}")
    
    extended_count = 0
    
    # Process each triangle
    for feature in geojson_data['features']:
        if feature['geometry']['type'] != 'Polygon':
            continue
            
        coords = feature['geometry']['coordinates'][0]
        modified = False
        
        # Check each vertex for boundary extension
        for i in range(len(coords) - 1):  # Skip duplicate closing coordinate
            vertex = coords[i]
            lat, lon = vertex[1], vertex[0]
            
            # Check if vertex needs extension in any direction
            for direction, extension_amount in directions.items():
                should_extend = False
                new_coord = None
                
                if direction == 'north' and lat >= max_lat - boundary_tolerance:
                    # Extend northward
                    new_coord = [lon, lat + extension_amount]
                    should_extend = True
                elif direction == 'south' and lat <= min_lat + boundary_tolerance:
                    # Extend southward  
                    new_coord = [lon, lat - extension_amount]
                    should_extend = True
                elif direction == 'east' and lon >= max_lon - boundary_tolerance:
                    # Extend eastward
                    new_coord = [lon + extension_amount, lat]
                    should_extend = True
                elif direction == 'west' and lon <= min_lon + boundary_tolerance:
                    # Extend westward
                    new_coord = [lon - extension_amount, lat]
                    should_extend = True
                
                if should_extend and new_coord:
                    coords[i] = new_coord
                    extended_count += 1
                    modified = True
                    
                    # Update closing coordinate if this is the first vertex
                    if i == 0:
                        coords[-1] = new_coord
                    
                    print(f"      🎯 Extended {direction}: [{lon:.6f}, {lat:.6f}] → [{new_coord[0]:.6f}, {new_coord[1]:.6f}]")
                    break  # Only extend in one direction per vertex
        
        # Update feature if modified
        if modified:
            feature['geometry']['coordinates'][0] = coords
    
    return extended_count

def apply_boundary_extension_to_stored_heatmaps(stored_heatmaps):
    """
    Apply boundary extension to stored heatmaps and return modified versions.
    
    This is the main entry point for the boundary extension system.
    """
    if not stored_heatmaps:
        return stored_heatmaps
        
    # Apply boundary extension
    extended_heatmaps = extend_triangular_boundaries(stored_heatmaps)
    
    return extended_heatmaps