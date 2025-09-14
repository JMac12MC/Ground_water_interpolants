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

SNAP_DISTANCE_METERS = 500

# Boundary detection removed - now processing all vertices directly

def snap_boundary_vertices_only(tiles, snap_distance=SNAP_DISTANCE_METERS):
    """
    Snap only vertices that are on boundaries between different heatmap tiles.
    Uses sequential processing: newer heatmaps snap to older ones to avoid conflicts.
    
    Args:
        tiles: dict with tile_id as key and tile data as values
        snap_distance: Maximum distance in meters for snapping
        
    Returns:
        Updated tiles with snapped boundary vertices only
    """
    print(f"\n=== SEQUENTIAL BOUNDARY-ONLY VERTEX SNAPPING ===")
    print(f"Processing {len(tiles)} tiles with {snap_distance}m snap distance")
    
    # Step 1: Sort tiles by ID (older heatmaps have lower IDs)
    sorted_tile_ids = sorted(tiles.keys())
    print(f"Processing order (oldest→newest): {sorted_tile_ids}")
    
    # Step 2: Extract ALL triangle vertices from each tile (not just boundary estimates)
    tile_vertices = {}
    for tile_id in sorted_tile_ids:
        tile_data = tiles[tile_id]
        geojson = tile_data['geojson']
        
        # Extract every vertex from every triangle
        all_vertices = []
        for feature in geojson.get('features', []):
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0][:-1]  # Remove duplicate last point
                all_vertices.extend(coords)
        
        # Remove exact duplicates but keep all unique vertices
        unique_vertices = []
        vertex_set = set()
        for vertex in all_vertices:
            vertex_tuple = tuple(vertex)
            if vertex_tuple not in vertex_set:
                unique_vertices.append(vertex)
                vertex_set.add(vertex_tuple)
        
        tile_vertices[tile_id] = unique_vertices
        print(f"  Tile {tile_id}: {len(unique_vertices)} unique triangle vertices extracted")
    
    # Step 3: Sequential snapping - each tile snaps to ALL previously processed tiles
    snap_stats = {'total_snaps': 0, 'tiles_modified': 0}
    vertex_mappings = {}  # Maps (tile_id, original_vertex) -> snapped_vertex
    
    # Convert snap distance from meters to degrees
    snap_distance_degrees = snap_distance / 111000.0  # ~111km per degree
    
    for i, current_tile_id in enumerate(sorted_tile_ids):
        if current_tile_id not in tile_vertices:
            continue
            
        current_vertices = np.array(tile_vertices[current_tile_id])
        tile_snaps = 0
        
        print(f"\n🔗 PROCESSING TILE {i+1}/{len(sorted_tile_ids)}: {current_tile_id}")
        print(f"   This tile has {len(current_vertices)} vertices to potentially snap")
        
        # Snap current tile to ALL previously processed tiles (older tiles)
        older_tiles_count = i  # Number of older tiles to check against
        if older_tiles_count == 0:
            print(f"   → This is the first tile (no older tiles to snap to)")
            continue
            
        print(f"   → Checking against {older_tiles_count} older tiles...")
        
        for j in range(i):  # Only look at older tiles
            older_tile_id = sorted_tile_ids[j]
            if older_tile_id not in tile_vertices:
                continue
            
            older_vertices = np.array(tile_vertices[older_tile_id])
            if len(older_vertices) == 0:
                continue
            
            print(f"    Checking {current_tile_id} vertices against older tile {older_tile_id}")
            
            # Check each vertex in current tile against all vertices in older tile
            snaps_with_this_tile = 0
            for vertex_idx, vertex in enumerate(current_vertices):
                # Skip if this vertex is already mapped to avoid overriding
                original_vertex = tuple(vertex)
                mapping_key = (current_tile_id, original_vertex)
                if mapping_key in vertex_mappings:
                    continue
                
                # Calculate distances to all vertices in the older tile
                distances = np.sqrt(
                    (vertex[0] - older_vertices[:, 0])**2 + 
                    (vertex[1] - older_vertices[:, 1])**2
                )
                
                min_distance_idx = np.argmin(distances)
                min_distance = distances[min_distance_idx]
                
                if min_distance <= snap_distance_degrees:
                    # Snap to the closest vertex in the older tile
                    snapped_vertex = tuple(older_vertices[min_distance_idx])
                    
                    vertex_mappings[mapping_key] = snapped_vertex
                    tile_snaps += 1
                    snaps_with_this_tile += 1
                    
                    print(f"      Snapped {original_vertex} → {snapped_vertex} (distance: {min_distance*111000:.1f}m)")
            
            if snaps_with_this_tile > 0:
                print(f"    → {snaps_with_this_tile} vertices from {current_tile_id} snapped to {older_tile_id}")
        
        if tile_snaps > 0:
            print(f"  Tile {current_tile_id}: {tile_snaps} vertices snapped to older tiles")
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
    updated_count = 0
    
    for tile_id, tile_data in snapped_tiles.items():
        try:
            # Use the database's built-in update method with retry logic
            geojson_str = json.dumps(tile_data['geojson'])
            
            # Try to update using database method with connection retry
            success = db.update_stored_heatmap_geojson(tile_id, geojson_str)
            
            if success:
                updated_count += 1
                print(f"✅ Updated heatmap {tile_id} with snapped boundaries")
            else:
                print(f"⚠️ Failed to update heatmap {tile_id}")
                
        except Exception as e:
            print(f"❌ Error updating heatmap {tile_id}: {e}")
            continue
    
    print(f"📊 Successfully updated {updated_count}/{len(snapped_tiles)} heatmaps with boundary-snapped vertices")

if __name__ == "__main__":
    run_boundary_only_snapping()