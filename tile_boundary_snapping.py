"""
Tile Boundary Snapping for Stored Heatmap Grids

This module fixes gaps and overlaps between adjacent heatmap tiles by snapping
vertices that are within 100 meters of each other to the same location.
"""

import numpy as np
from scipy.spatial import KDTree
import json
import sqlite3
try:
    from utils import get_distance
    from database import get_database_connection
except ImportError:
    # Fallback for when running standalone
    def get_distance(lat1, lon1, lat2, lon2):
        """Simple haversine distance calculation"""
        from math import radians, cos, sin, asin, sqrt
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 2 * asin(sqrt(a)) * 6371  # Earth radius in km
    
    def get_database_connection():
        """Simple database connection"""
        return sqlite3.connect('heatmaps.db')

SNAP_DISTANCE_METERS = 500  # Snap vertices within 500 meters
MIN_TRIANGLE_AREA = 1000    # Minimum triangle area in square meters to avoid degenerate triangles

def extract_vertices_from_geojson(geojson_data):
    """
    Extract unique vertices from GeoJSON triangle features.
    
    Args:
        geojson_data: GeoJSON FeatureCollection with triangle polygons
        
    Returns:
        tuple: (vertices_array, vertex_to_triangles_map)
            - vertices_array: np.array of [lon, lat] coordinates
            - vertex_to_triangles_map: dict mapping vertex_idx to list of triangle indices
    """
    vertices = []
    vertex_map = {}  # (lon, lat) -> vertex_index
    vertex_to_triangles = {}  # vertex_index -> [triangle_indices]
    
    for triangle_idx, feature in enumerate(geojson_data.get('features', [])):
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0][:-1]  # Remove duplicate last point
            
            for lon, lat in coords:
                coord_key = (round(lon, 8), round(lat, 8))  # Round to avoid floating point issues
                
                if coord_key not in vertex_map:
                    vertex_idx = len(vertices)
                    vertices.append([lon, lat])
                    vertex_map[coord_key] = vertex_idx
                    vertex_to_triangles[vertex_idx] = []
                
                vertex_idx = vertex_map[coord_key]
                if triangle_idx not in vertex_to_triangles[vertex_idx]:
                    vertex_to_triangles[vertex_idx].append(triangle_idx)
    
    return np.array(vertices), vertex_to_triangles

def vertices_to_geojson(vertices, vertex_to_triangles, original_geojson):
    """
    Reconstruct GeoJSON from modified vertices.
    
    Args:
        vertices: np.array of [lon, lat] coordinates
        vertex_to_triangles: dict mapping vertex_idx to triangle indices
        original_geojson: Original GeoJSON for reference
        
    Returns:
        Updated GeoJSON FeatureCollection
    """
    features = original_geojson.get('features', [])
    vertex_coords = {i: [vertices[i][0], vertices[i][1]] for i in range(len(vertices))}
    
    # Rebuild triangle coordinates
    triangle_vertices = {}  # triangle_idx -> [vertex_indices]
    for vertex_idx, triangle_indices in vertex_to_triangles.items():
        for triangle_idx in triangle_indices:
            if triangle_idx not in triangle_vertices:
                triangle_vertices[triangle_idx] = []
            triangle_vertices[triangle_idx].append(vertex_idx)
    
    # Update feature coordinates
    for triangle_idx, feature in enumerate(features):
        if triangle_idx in triangle_vertices:
            vertex_indices = triangle_vertices[triangle_idx]
            if len(vertex_indices) >= 3:
                # Take first 3 vertices to form triangle
                coords = [vertex_coords[v_idx] for v_idx in vertex_indices[:3]]
                coords.append(coords[0])  # Close the polygon
                feature['geometry']['coordinates'] = [coords]
    
    return original_geojson

def convert_to_meters(vertices, reference_lat):
    """
    Convert lat/lon vertices to local meter coordinates for distance calculations.
    
    Args:
        vertices: np.array of [lon, lat] coordinates
        reference_lat: Reference latitude for projection
        
    Returns:
        np.array of [x_meters, y_meters] coordinates
    """
    if len(vertices) == 0:
        return np.array([])
    
    # Use reference point as origin
    ref_lon, ref_lat = np.mean(vertices, axis=0)
    
    # Convert to meters using local projection
    km_per_degree_lat = 111.0
    km_per_degree_lon = 111.0 * np.cos(np.radians(reference_lat))
    
    x_meters = (vertices[:, 0] - ref_lon) * km_per_degree_lon * 1000
    y_meters = (vertices[:, 1] - ref_lat) * km_per_degree_lat * 1000
    
    return np.column_stack([x_meters, y_meters])

def convert_from_meters(vertices_meters, reference_coords, reference_lat):
    """
    Convert meter coordinates back to lat/lon.
    
    Args:
        vertices_meters: np.array of [x_meters, y_meters] coordinates
        reference_coords: [ref_lon, ref_lat] origin point
        reference_lat: Reference latitude for projection
        
    Returns:
        np.array of [lon, lat] coordinates
    """
    if len(vertices_meters) == 0:
        return np.array([])
    
    ref_lon, ref_lat = reference_coords
    
    km_per_degree_lat = 111.0
    km_per_degree_lon = 111.0 * np.cos(np.radians(reference_lat))
    
    lons = ref_lon + (vertices_meters[:, 0] / 1000) / km_per_degree_lon
    lats = ref_lat + (vertices_meters[:, 1] / 1000) / km_per_degree_lat
    
    return np.column_stack([lons, lats])

def snap_tile_vertices(tiles, snap_distance=SNAP_DISTANCE_METERS):
    """
    Snap vertices between overlapping/adjacent tiles to fix gaps and overlaps.
    
    Args:
        tiles: dict with tile_id as key and tile data as values
            {
                'tile_id': {
                    'geojson': GeoJSON FeatureCollection,
                    'center_lat': latitude,
                    'center_lon': longitude
                }
            }
        snap_distance: Maximum distance in meters for snapping
        
    Returns:
        Updated tiles with snapped vertices
    """
    tile_ids = list(tiles.keys())
    snapped_tiles = {}
    snap_stats = {'total_snaps': 0, 'tiles_modified': 0}
    
    print(f"\n=== TILE BOUNDARY SNAPPING ===")
    print(f"Processing {len(tile_ids)} tiles with {snap_distance}m snap distance")
    
    # Extract vertices from all tiles
    tile_vertices = {}
    tile_vertex_maps = {}
    
    for tile_id in tile_ids:
        tile_data = tiles[tile_id]
        vertices, vertex_to_triangles = extract_vertices_from_geojson(tile_data['geojson'])
        
        if len(vertices) > 0:
            # Convert to meter coordinates for accurate distance calculations
            reference_lat = tile_data.get('center_lat', -43.5)  # Canterbury region default
            vertices_meters = convert_to_meters(vertices, reference_lat)
            
            tile_vertices[tile_id] = {
                'vertices_latlon': vertices,
                'vertices_meters': vertices_meters,
                'vertex_to_triangles': vertex_to_triangles,
                'reference_lat': reference_lat,
                'reference_coords': np.mean(vertices, axis=0),
                'modified': False
            }
    
    # Snap vertices between tiles
    for i, tile_id in enumerate(tile_ids):
        if tile_id not in tile_vertices:
            continue
            
        current_tile = tile_vertices[tile_id]
        current_vertices_meters = current_tile['vertices_meters']
        
        tile_snaps = 0
        
        # Compare with all other tiles (neighboring tiles)
        for j, neighbor_tile_id in enumerate(tile_ids):
            if i == j or neighbor_tile_id not in tile_vertices:
                continue
                
            neighbor_tile = tile_vertices[neighbor_tile_id]
            neighbor_vertices_meters = neighbor_tile['vertices_meters']
            
            if len(neighbor_vertices_meters) == 0:
                continue
            
            # Build KDTree for efficient neighbor search
            tree = KDTree(neighbor_vertices_meters)
            
            # Find nearby vertices and snap them
            for vertex_idx, vertex_meters in enumerate(current_vertices_meters):
                distances, neighbor_indices = tree.query(vertex_meters, k=1)
                
                if distances <= snap_distance:
                    # Snap current vertex to the neighbor vertex
                    neighbor_vertex_meters = neighbor_vertices_meters[neighbor_indices]
                    current_vertices_meters[vertex_idx] = neighbor_vertex_meters
                    
                    tile_snaps += 1
                    current_tile['modified'] = True
        
        if tile_snaps > 0:
            print(f"  Tile {tile_id}: {tile_snaps} vertices snapped")
            snap_stats['total_snaps'] += tile_snaps
            snap_stats['tiles_modified'] += 1
    
    # Convert back to lat/lon and reconstruct GeoJSON
    for tile_id in tile_ids:
        if tile_id not in tile_vertices:
            snapped_tiles[tile_id] = tiles[tile_id]
            continue
            
        tile_data = tile_vertices[tile_id]
        
        if tile_data['modified']:
            # Convert snapped vertices back to lat/lon
            snapped_vertices_latlon = convert_from_meters(
                tile_data['vertices_meters'],
                tile_data['reference_coords'],
                tile_data['reference_lat']
            )
            
            # Reconstruct GeoJSON with snapped vertices
            updated_geojson = vertices_to_geojson(
                snapped_vertices_latlon,
                tile_data['vertex_to_triangles'],
                tiles[tile_id]['geojson']
            )
            
            snapped_tiles[tile_id] = {
                **tiles[tile_id],
                'geojson': updated_geojson
            }
        else:
            snapped_tiles[tile_id] = tiles[tile_id]
    
    print(f"SNAPPING COMPLETE: {snap_stats['total_snaps']} vertices snapped across {snap_stats['tiles_modified']} tiles")
    return snapped_tiles, snap_stats

def load_stored_heatmaps_for_snapping():
    """
    Load stored heatmaps from database for boundary snapping.
    
    Returns:
        dict: tiles data ready for snapping
    """
    tiles = {}
    
    try:
        # Use the same database system as the main app
        from database import PolygonDatabase
        
        db = PolygonDatabase()
        stored_heatmaps = db.get_all_stored_heatmaps()
        
        print(f"Found {len(stored_heatmaps)} stored heatmaps in database")
        
        for heatmap in stored_heatmaps:
            heatmap_id = heatmap['id']
            geojson_data = heatmap.get('geojson_data')
            
            if geojson_data and isinstance(geojson_data, dict):
                feature_count = len(geojson_data.get('features', []))
                
                if feature_count > 0:
                    tiles[heatmap_id] = {
                        'geojson': geojson_data,
                        'center_lat': heatmap['center_lat'],
                        'center_lon': heatmap['center_lon'],
                        'method_name': heatmap['interpolation_method'],
                        'feature_count': feature_count
                    }
                    
        print(f"Loaded {len(tiles)} tiles with GeoJSON data for boundary snapping")
        return tiles
        
    except Exception as e:
        print(f"Error loading stored heatmaps: {e}")
        import traceback
        print(traceback.format_exc())
        return {}

def save_snapped_heatmaps(snapped_tiles):
    """
    Save snapped heatmaps back to database.
    
    Args:
        snapped_tiles: dict of tiles with snapped vertices
    """
    try:
        from database import PolygonDatabase
        from sqlalchemy import text
        
        db = PolygonDatabase()
        updated_count = 0
        
        with db.engine.connect() as conn:
            for tile_id, tile_data in snapped_tiles.items():
                geojson_str = json.dumps(tile_data['geojson'])
                
                result = conn.execute(text("""
                    UPDATE stored_heatmaps 
                    SET geojson_data = :geojson_data
                    WHERE id = :heatmap_id
                """), {
                    'geojson_data': geojson_str,
                    'heatmap_id': tile_id
                })
                
                if result.rowcount > 0:
                    updated_count += 1
            
            conn.commit()
        
        print(f"Updated {updated_count} heatmaps in database with snapped vertices")
        
    except Exception as e:
        print(f"Error saving snapped heatmaps: {e}")
        import traceback
        print(traceback.format_exc())

def run_boundary_snapping():
    """
    Main function to run the boundary snapping process on all stored heatmaps.
    """
    print("\n🔧 STARTING TILE BOUNDARY SNAPPING PROCESS")
    
    # Load tiles from database
    tiles = load_stored_heatmaps_for_snapping()
    
    if not tiles:
        print("No tiles found for snapping")
        return
    
    # Perform snapping
    snapped_tiles, stats = snap_tile_vertices(tiles, SNAP_DISTANCE_METERS)
    
    # Save back to database
    if stats['tiles_modified'] > 0:
        save_snapped_heatmaps(snapped_tiles)
        print(f"\n✅ BOUNDARY SNAPPING COMPLETE")
        print(f"   Snapped {stats['total_snaps']} vertices across {stats['tiles_modified']} tiles")
        print(f"   Gaps and overlaps reduced within {SNAP_DISTANCE_METERS}m tolerance")
    else:
        print(f"\n✅ NO SNAPPING NEEDED - All tiles already aligned within {SNAP_DISTANCE_METERS}m tolerance")

if __name__ == "__main__":
    run_boundary_snapping()