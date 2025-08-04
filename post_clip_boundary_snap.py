"""
Post-Clipping Boundary Snapping Module

This module implements boundary snapping AFTER soil polygon clipping (0.5km buffer).
It finds vertices near adjacent heatmap boundaries and snaps them to create seamless connections.
"""

import numpy as np
from shapely.geometry import Point, LineString
from utils import get_distance

def snap_vertices_to_adjacent_boundaries(geojson_features, adjacent_boundaries, snap_threshold_km=1.0):
    """
    Snap vertices of clipped triangular mesh to adjacent heatmap boundaries.
    
    This function operates AFTER soil polygon clipping and finds vertices within
    snap_threshold_km of adjacent boundaries, then snaps them for seamless joining.
    
    Parameters:
    -----------
    geojson_features : list
        List of GeoJSON triangle features (already clipped by soil polygons)
    adjacent_boundaries : dict
        Dictionary with boundary coordinates {'west': lon, 'east': lon, 'north': lat, 'south': lat}
    snap_threshold_km : float
        Maximum distance in km to search for vertices to snap
        
    Returns:
    --------
    list
        Modified GeoJSON features with snapped vertices
    """
    
    if not adjacent_boundaries or not geojson_features:
        print("POST-CLIP SNAP: No boundaries or features to snap")
        return geojson_features
    
    print(f"🔧 POST-CLIP BOUNDARY SNAP: Processing {len(geojson_features)} triangular features")
    print(f"🔧 SNAP THRESHOLD: {snap_threshold_km}km")
    print(f"🔧 AVAILABLE BOUNDARIES: {list(adjacent_boundaries.keys())}")
    
    snapped_count = 0
    vertex_modifications = []
    
    # Process each triangle feature
    for feature_idx, feature in enumerate(geojson_features):
        if feature['geometry']['type'] != 'Polygon':
            continue
            
        coordinates = feature['geometry']['coordinates'][0]  # First ring of polygon
        modified = False
        
        # Check each vertex of the triangle
        for vertex_idx, vertex in enumerate(coordinates[:-1]):  # Skip last vertex (duplicate of first)
            lon, lat = vertex
            vertex_point = Point(lon, lat)
            
            # Check against each available boundary
            for boundary_type, boundary_value in adjacent_boundaries.items():
                
                if boundary_type == 'west' and boundary_value is not None:
                    # Snap to west boundary (vertical line at fixed longitude)
                    boundary_line = LineString([(boundary_value, lat - 0.01), (boundary_value, lat + 0.01)])
                    distance_km = get_distance(lat, lon, lat, boundary_value)
                    
                    if distance_km <= snap_threshold_km:
                        # Snap to west boundary
                        coordinates[vertex_idx] = [boundary_value, lat]
                        if vertex_idx == 0:  # Also update last vertex if it's the closing vertex
                            coordinates[-1] = [boundary_value, lat]
                        modified = True
                        snapped_count += 1
                        vertex_modifications.append({
                            'feature': feature_idx,
                            'vertex': vertex_idx, 
                            'boundary': 'west',
                            'original': [lon, lat],
                            'snapped': [boundary_value, lat],
                            'distance_km': distance_km
                        })
                        print(f"  🎯 WEST SNAP: Feature {feature_idx}, vertex {vertex_idx} → {boundary_value:.6f} (was {lon:.6f}, distance: {distance_km:.3f}km)")
                        break
                        
                elif boundary_type == 'east' and boundary_value is not None:
                    # Snap to east boundary (vertical line at fixed longitude)
                    distance_km = get_distance(lat, lon, lat, boundary_value)
                    
                    if distance_km <= snap_threshold_km:
                        # Snap to east boundary
                        coordinates[vertex_idx] = [boundary_value, lat]
                        if vertex_idx == 0:  # Also update last vertex if it's the closing vertex
                            coordinates[-1] = [boundary_value, lat]
                        modified = True
                        snapped_count += 1
                        vertex_modifications.append({
                            'feature': feature_idx,
                            'vertex': vertex_idx,
                            'boundary': 'east', 
                            'original': [lon, lat],
                            'snapped': [boundary_value, lat],
                            'distance_km': distance_km
                        })
                        print(f"  🎯 EAST SNAP: Feature {feature_idx}, vertex {vertex_idx} → {boundary_value:.6f} (was {lon:.6f}, distance: {distance_km:.3f}km)")
                        break
                        
                elif boundary_type == 'north' and boundary_value is not None:
                    # Snap to north boundary (horizontal line at fixed latitude)
                    distance_km = get_distance(lat, lon, boundary_value, lon)
                    
                    if distance_km <= snap_threshold_km:
                        # Snap to north boundary
                        coordinates[vertex_idx] = [lon, boundary_value]
                        if vertex_idx == 0:  # Also update last vertex if it's the closing vertex
                            coordinates[-1] = [lon, boundary_value]
                        modified = True
                        snapped_count += 1
                        vertex_modifications.append({
                            'feature': feature_idx,
                            'vertex': vertex_idx,
                            'boundary': 'north',
                            'original': [lon, lat],
                            'snapped': [lon, boundary_value],
                            'distance_km': distance_km
                        })
                        print(f"  🎯 NORTH SNAP: Feature {feature_idx}, vertex {vertex_idx} → {boundary_value:.6f} (was {lat:.6f}, distance: {distance_km:.3f}km)")
                        break
                        
                elif boundary_type == 'south' and boundary_value is not None:
                    # Snap to south boundary (horizontal line at fixed latitude)
                    distance_km = get_distance(lat, lon, boundary_value, lon)
                    
                    if distance_km <= snap_threshold_km:
                        # Snap to south boundary
                        coordinates[vertex_idx] = [lon, boundary_value]
                        if vertex_idx == 0:  # Also update last vertex if it's the closing vertex
                            coordinates[-1] = [lon, boundary_value]
                        modified = True
                        snapped_count += 1
                        vertex_modifications.append({
                            'feature': feature_idx,
                            'vertex': vertex_idx,
                            'boundary': 'south',
                            'original': [lon, lat],
                            'snapped': [lon, boundary_value],
                            'distance_km': distance_km
                        })
                        print(f"  🎯 SOUTH SNAP: Feature {feature_idx}, vertex {vertex_idx} → {boundary_value:.6f} (was {lat:.6f}, distance: {distance_km:.3f}km)")
                        break
        
        # Update the feature with modified coordinates
        if modified:
            feature['geometry']['coordinates'][0] = coordinates
    
    # Summary reporting
    if snapped_count > 0:
        print(f"🔧 POST-CLIP SNAP COMPLETE: {snapped_count} vertices snapped to adjacent boundaries")
        
        # Group by boundary type for summary
        boundary_counts = {}
        for mod in vertex_modifications:
            boundary = mod['boundary']
            boundary_counts[boundary] = boundary_counts.get(boundary, 0) + 1
            
        for boundary, count in boundary_counts.items():
            print(f"  📊 {boundary.upper()} boundary: {count} vertices snapped")
            
        # Show distance statistics
        distances = [mod['distance_km'] for mod in vertex_modifications]
        if distances:
            avg_distance = sum(distances) / len(distances)
            max_distance = max(distances)
            print(f"  📏 SNAP DISTANCES: avg={avg_distance:.3f}km, max={max_distance:.3f}km")
    else:
        print("🔧 POST-CLIP SNAP: No vertices were close enough to adjacent boundaries for snapping")
    
    return geojson_features


def apply_post_clip_boundary_snapping(geojson_data, adjacent_boundaries, snap_threshold_km=1.0):
    """
    Apply post-clipping boundary snapping to a complete GeoJSON dataset.
    
    This is the main entry point for post-clipping boundary snapping.
    
    Parameters:
    -----------
    geojson_data : dict
        GeoJSON FeatureCollection with triangular features
    adjacent_boundaries : dict  
        Dictionary with boundary coordinates
    snap_threshold_km : float
        Maximum distance in km to search for vertices to snap
        
    Returns:
    --------
    dict
        Modified GeoJSON FeatureCollection with snapped boundaries
    """
    
    if not geojson_data or 'features' not in geojson_data:
        print("POST-CLIP SNAP: No GeoJSON data to process")
        return geojson_data
    
    print(f"🔧 APPLYING POST-CLIP BOUNDARY SNAPPING")
    print(f"🔧 INPUT: {len(geojson_data['features'])} features")
    
    # Apply snapping to features
    snapped_features = snap_vertices_to_adjacent_boundaries(
        geojson_data['features'],
        adjacent_boundaries,
        snap_threshold_km
    )
    
    # Return modified GeoJSON
    result = {
        "type": "FeatureCollection",
        "features": snapped_features
    }
    
    print(f"🔧 OUTPUT: {len(result['features'])} features with boundary snapping applied")
    return result