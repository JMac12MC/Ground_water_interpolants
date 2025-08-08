"""
Boundary-only tile snapping algorithm
Only snaps vertices that are on the actual boundaries between different heatmap tiles
"""

import json
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
import geopandas as gpd

SNAP_DISTANCE_METERS = 100

def find_heatmap_boundaries(tiles):
    """
    Find the actual boundary lines between different heatmap tiles.
    
    Args:
        tiles: dict of tile data with GeoJSON
        
    Returns:
        dict: {tile_id: boundary_vertices} where boundary_vertices are vertices on tile edges
    """
    print("🔍 ANALYZING HEATMAP BOUNDARIES...")
    
    tile_boundaries = {}
    tile_polygons = {}
    
    # Create overall polygons for each tile
    for tile_id, tile_data in tiles.items():
        geojson = tile_data['geojson']
        
        # Extract all triangular polygons from this tile
        polygons = []
        for feature in geojson.get('features', []):
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0][:-1]  # Remove duplicate last point
                if len(coords) >= 3:
                    polygons.append(Polygon(coords))
        
        if polygons:
            # Create the overall boundary of this heatmap tile
            tile_boundary = unary_union(polygons)
            tile_polygons[tile_id] = tile_boundary
            print(f"  Tile {tile_id}: {len(polygons)} triangles → boundary polygon")
    
    print(f"📊 CREATED BOUNDARIES: {len(tile_polygons)} tile boundaries")
    
    # Find vertices that are on boundaries between different tiles
    for tile_id, tile_data in tiles.items():
        if tile_id not in tile_polygons:
            continue
            
        boundary_vertices = []
        geojson = tile_data['geojson']
        
        # Check each vertex in each triangle
        for feature in geojson.get('features', []):
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0][:-1]  # Remove duplicate last point
                
                for vertex_coord in coords:
                    vertex_point = Point(vertex_coord)
                    
                    # Check if this vertex is on the boundary with any other tile
                    is_boundary_vertex = False
                    
                    for other_tile_id, other_boundary in tile_polygons.items():
                        if other_tile_id == tile_id:
                            continue
                            
                        # Check if vertex is very close to the other tile's boundary
                        distance_to_other = vertex_point.distance(other_boundary.boundary)
                        
                        # Convert degrees to meters (rough approximation)
                        distance_meters = distance_to_other * 111000  # Very rough conversion
                        
                        if distance_meters < SNAP_DISTANCE_METERS * 2:  # Within 200m of another tile
                            is_boundary_vertex = True
                            break
                    
                    if is_boundary_vertex:
                        boundary_vertices.append(vertex_coord)
        
        # Remove duplicates
        boundary_vertices = list(set(tuple(v) for v in boundary_vertices))
        tile_boundaries[tile_id] = [list(v) for v in boundary_vertices]
        
        print(f"  Tile {tile_id}: {len(boundary_vertices)} boundary vertices identified")
    
    return tile_boundaries

def snap_boundary_vertices_only(tiles, snap_distance=SNAP_DISTANCE_METERS):
    """
    Snap only vertices that are on boundaries between different heatmap tiles.
    
    Args:
        tiles: dict with tile_id as key and tile data as values
        snap_distance: Maximum distance in meters for snapping
        
    Returns:
        Updated tiles with snapped boundary vertices only
    """
    print(f"\n=== BOUNDARY-ONLY VERTEX SNAPPING ===")
    print(f"Processing {len(tiles)} tiles with {snap_distance}m snap distance")
    
    # Step 1: Identify boundary vertices
    tile_boundaries = find_heatmap_boundaries(tiles)
    
    # Step 2: Convert boundary vertices to meters for accurate distance calculations
    tile_boundary_meters = {}
    for tile_id, boundary_vertices in tile_boundaries.items():
        if len(boundary_vertices) > 0:
            vertices_array = np.array(boundary_vertices)
            reference_lat = tiles[tile_id].get('center_lat', -43.5)
            
            # Convert to meters
            vertices_meters = convert_to_meters(vertices_array, reference_lat)
            tile_boundary_meters[tile_id] = {
                'vertices_latlon': vertices_array,
                'vertices_meters': vertices_meters,
                'reference_lat': reference_lat
            }
    
    # Step 3: Snap boundary vertices between tiles
    snap_stats = {'total_snaps': 0, 'tiles_modified': 0}
    vertex_mappings = {}  # Maps (tile_id, original_vertex) -> snapped_vertex
    
    for tile_id in tile_boundaries.keys():
        if tile_id not in tile_boundary_meters:
            continue
            
        current_boundary = tile_boundary_meters[tile_id]
        current_vertices_meters = current_boundary['vertices_meters']
        tile_snaps = 0
        
        # Compare with all other tiles
        for other_tile_id in tile_boundaries.keys():
            if tile_id == other_tile_id or other_tile_id not in tile_boundary_meters:
                continue
            
            other_boundary = tile_boundary_meters[other_tile_id]
            other_vertices_meters = other_boundary['vertices_meters']
            
            if len(other_vertices_meters) == 0:
                continue
            
            # Build KDTree for efficient neighbor search
            tree = KDTree(other_vertices_meters)
            
            # Find nearby vertices and snap them
            for vertex_idx, vertex_meters in enumerate(current_vertices_meters):
                distances, neighbor_indices = tree.query(vertex_meters, k=1)
                
                if distances <= snap_distance:
                    # Create mapping from original vertex to snapped vertex
                    original_vertex = tuple(current_boundary['vertices_latlon'][vertex_idx])
                    snapped_vertex_meters = other_vertices_meters[neighbor_indices]
                    
                    # Convert back to lat/lon
                    snapped_vertex_latlon = convert_from_meters(
                        snapped_vertex_meters.reshape(1, -1),
                        np.mean(current_boundary['vertices_latlon'], axis=0),
                        current_boundary['reference_lat']
                    )[0]
                    
                    vertex_mappings[(tile_id, original_vertex)] = snapped_vertex_latlon
                    tile_snaps += 1
        
        if tile_snaps > 0:
            print(f"  Tile {tile_id}: {tile_snaps} boundary vertices snapped")
            snap_stats['total_snaps'] += tile_snaps
            snap_stats['tiles_modified'] += 1
    
    # Step 4: Apply vertex mappings to original GeoJSON data
    snapped_tiles = {}
    for tile_id, tile_data in tiles.items():
        original_geojson = tile_data['geojson']
        updated_features = []
        
        for feature in original_geojson.get('features', []):
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0]
                updated_coords = []
                
                for coord in coords:
                    coord_tuple = tuple(coord)
                    mapping_key = (tile_id, coord_tuple)
                    
                    if mapping_key in vertex_mappings:
                        # Use snapped vertex
                        snapped_coord = vertex_mappings[mapping_key]
                        updated_coords.append([float(snapped_coord[0]), float(snapped_coord[1])])
                    else:
                        # Keep original vertex
                        updated_coords.append(coord)
                
                # Update feature geometry
                updated_feature = feature.copy()
                updated_feature['geometry']['coordinates'] = [updated_coords]
                updated_features.append(updated_feature)
            else:
                updated_features.append(feature)
        
        # Create updated GeoJSON
        updated_geojson = {
            'type': 'FeatureCollection',
            'features': updated_features
        }
        
        snapped_tiles[tile_id] = {
            **tile_data,
            'geojson': updated_geojson
        }
    
    print(f"BOUNDARY SNAPPING COMPLETE: {snap_stats['total_snaps']} boundary vertices snapped across {snap_stats['tiles_modified']} tiles")
    return snapped_tiles, snap_stats

def convert_to_meters(vertices, reference_lat):
    """Convert lat/lon vertices to meter coordinates."""
    if len(vertices) == 0:
        return np.array([])
    
    km_per_degree_lat = 111.0
    km_per_degree_lon = 111.0 * np.cos(np.radians(reference_lat))
    
    # Convert to kilometers, then to meters
    x_meters = vertices[:, 0] * km_per_degree_lon * 1000
    y_meters = vertices[:, 1] * km_per_degree_lat * 1000
    
    return np.column_stack([x_meters, y_meters])

def convert_from_meters(vertices_meters, reference_coords, reference_lat):
    """Convert meter coordinates back to lat/lon."""
    if len(vertices_meters) == 0:
        return np.array([])
    
    ref_lon, ref_lat = reference_coords
    
    km_per_degree_lat = 111.0
    km_per_degree_lon = 111.0 * np.cos(np.radians(reference_lat))
    
    lons = ref_lon + (vertices_meters[:, 0] / 1000) / km_per_degree_lon
    lats = ref_lat + (vertices_meters[:, 1] / 1000) / km_per_degree_lat
    
    return np.column_stack([lons, lats])

def run_boundary_only_snapping():
    """
    Main function to run boundary-only tile snapping on stored heatmaps.
    """
    print("🔧 STARTING BOUNDARY-ONLY TILE SNAPPING PROCESS")
    
    # Load stored heatmaps from database
    from database import PolygonDatabase
    
    db = PolygonDatabase()
    stored_heatmaps = db.get_all_stored_heatmaps()
    
    if not stored_heatmaps:
        print("❌ No stored heatmaps found in database")
        return
    
    print(f"📊 Found {len(stored_heatmaps)} stored heatmaps in database")
    
    # Prepare tiles data
    tiles = {}
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
    
    print(f"📊 Loaded {len(tiles)} tiles with GeoJSON data for boundary snapping")
    
    if len(tiles) < 2:
        print("⚠️ Need at least 2 tiles for boundary snapping")
        return
    
    # Perform boundary-only snapping
    snapped_tiles, snap_stats = snap_boundary_vertices_only(tiles)
    
    # Save results back to database
    if snap_stats['total_snaps'] > 0:
        save_snapped_heatmaps(snapped_tiles, db)
        print(f"\n✅ BOUNDARY-ONLY SNAPPING COMPLETE")
        print(f"   Snapped {snap_stats['total_snaps']} boundary vertices across {snap_stats['tiles_modified']} tiles")
        print(f"   Internal triangle vertices unchanged")
    else:
        print(f"\n📊 NO BOUNDARY SNAPPING NEEDED")
        print(f"   All tile boundaries are already well-aligned")

def save_snapped_heatmaps(snapped_tiles, db):
    """Save snapped heatmaps back to database."""
    try:
        from sqlalchemy import text
        
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
        
        print(f"📊 Updated {updated_count} heatmaps in database with boundary-snapped vertices")
        
    except Exception as e:
        print(f"❌ Error saving boundary-snapped heatmaps: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    run_boundary_only_snapping()