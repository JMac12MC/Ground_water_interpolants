"""
Debug tile boundary snapping to understand why it's not working
"""

import json
import sqlite3
import numpy as np
from scipy.spatial import KDTree

def debug_stored_heatmaps():
    """Debug stored heatmaps to understand structure and find snapping issues"""
    
    try:
        # Connect to database
        conn = sqlite3.connect('heatmaps.db')
        cursor = conn.cursor()
        
        # Get basic info about stored heatmaps
        cursor.execute("SELECT COUNT(*) FROM stored_heatmaps")
        total_count = cursor.fetchone()[0]
        print(f"Total stored heatmaps: {total_count}")
        
        # Get heatmaps with GeoJSON data
        cursor.execute("""
            SELECT heatmap_id, method_name, center_lat, center_lon, 
                   LENGTH(geojson_data) as json_size, geojson_data
            FROM stored_heatmaps 
            WHERE geojson_data IS NOT NULL
            ORDER BY heatmap_id
            LIMIT 3
        """)
        
        sample_heatmaps = cursor.fetchall()
        print(f"\nSample heatmaps with GeoJSON data: {len(sample_heatmaps)}")
        
        for i, (heatmap_id, method_name, center_lat, center_lon, json_size, geojson_data) in enumerate(sample_heatmaps):
            print(f"\nHeatmap {i+1}: {heatmap_id}")
            print(f"  Method: {method_name}")
            print(f"  Center: {center_lat:.4f}, {center_lon:.4f}")
            print(f"  JSON size: {json_size:,} characters")
            
            # Parse GeoJSON and analyze structure
            try:
                geojson = json.loads(geojson_data)
                features = geojson.get('features', [])
                print(f"  Features: {len(features)}")
                
                if features:
                    # Analyze first few features
                    sample_feature = features[0]
                    print(f"  Sample feature type: {sample_feature.get('geometry', {}).get('type')}")
                    
                    # Extract vertices to understand spacing
                    vertices = []
                    for feature in features[:100]:  # Sample first 100 features
                        if feature['geometry']['type'] == 'Polygon':
                            coords = feature['geometry']['coordinates'][0][:-1]  # Remove duplicate last point
                            for lon, lat in coords:
                                vertices.append([lon, lat])
                    
                    if len(vertices) > 1:
                        vertices = np.array(vertices)
                        print(f"  Sample vertices: {len(vertices)}")
                        print(f"  Vertex range - Lat: {vertices[:, 1].min():.6f} to {vertices[:, 1].max():.6f}")
                        print(f"  Vertex range - Lon: {vertices[:, 0].min():.6f} to {vertices[:, 0].max():.6f}")
                        
                        # Calculate some distances between vertices to understand spacing
                        if len(vertices) >= 10:
                            sample_distances = []
                            for j in range(min(10, len(vertices))):
                                for k in range(j+1, min(j+5, len(vertices))):
                                    dist = np.sqrt((vertices[j, 0] - vertices[k, 0])**2 + 
                                                 (vertices[j, 1] - vertices[k, 1])**2)
                                    # Convert to approximate meters (rough estimate)
                                    dist_meters = dist * 111000  # Very rough conversion
                                    sample_distances.append(dist_meters)
                            
                            if sample_distances:
                                print(f"  Sample vertex distances: {np.mean(sample_distances):.1f}m avg, {np.min(sample_distances):.1f}m min")
                        
            except json.JSONDecodeError as e:
                print(f"  ERROR parsing GeoJSON: {e}")
                continue
        
        # Check for potential overlapping areas
        print(f"\n=== OVERLAP ANALYSIS ===")
        cursor.execute("""
            SELECT h1.heatmap_id as id1, h2.heatmap_id as id2,
                   h1.center_lat as lat1, h1.center_lon as lon1,
                   h2.center_lat as lat2, h2.center_lon as lon2,
                   ABS(h1.center_lat - h2.center_lat) as lat_diff,
                   ABS(h1.center_lon - h2.center_lon) as lon_diff
            FROM stored_heatmaps h1, stored_heatmaps h2
            WHERE h1.heatmap_id < h2.heatmap_id
              AND h1.geojson_data IS NOT NULL 
              AND h2.geojson_data IS NOT NULL
              AND ABS(h1.center_lat - h2.center_lat) < 0.3  -- Within ~33km
              AND ABS(h1.center_lon - h2.center_lon) < 0.3  -- Within ~33km
            ORDER BY lat_diff + lon_diff
            LIMIT 10
        """)
        
        overlaps = cursor.fetchall()
        print(f"Found {len(overlaps)} potentially overlapping heatmap pairs:")
        
        for id1, id2, lat1, lon1, lat2, lon2, lat_diff, lon_diff in overlaps:
            # Rough distance calculation
            distance_km = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Very rough
            print(f"  {id1} ↔ {id2}: {distance_km:.1f}km apart")
            print(f"    {id1}: {lat1:.4f}, {lon1:.4f}")
            print(f"    {id2}: {lat2:.4f}, {lon2:.4f}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error in debug analysis: {e}")
        import traceback
        print(traceback.format_exc())

def test_vertex_snapping():
    """Test the vertex snapping algorithm with simple data"""
    
    print(f"\n=== TESTING VERTEX SNAPPING ALGORITHM ===")
    
    # Create test data - two overlapping triangular grids
    # Grid 1: centered at (0, 0)
    grid1_vertices = np.array([
        [0.0, 0.0],      # Center
        [0.001, 0.0],    # ~111m east
        [0.0, 0.001],    # ~111m north
        [0.001, 0.001]   # ~111m northeast
    ])
    
    # Grid 2: slightly offset (simulating small overlap/gap)
    offset = 0.0005  # ~55m offset
    grid2_vertices = np.array([
        [offset, 0.0],          # Slightly offset center
        [0.001 + offset, 0.0],  # Slightly offset east
        [offset, 0.001],        # Slightly offset north
        [0.001 + offset, 0.001] # Slightly offset northeast
    ])
    
    print(f"Grid 1 vertices: {len(grid1_vertices)}")
    print(f"Grid 2 vertices: {len(grid2_vertices)}")
    
    # Convert to meters for distance calculation
    def deg_to_meters(vertices, ref_lat=-43.5):
        """Convert lat/lon degrees to meters"""
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(ref_lat))
        
        x_meters = vertices[:, 0] * km_per_deg_lon * 1000
        y_meters = vertices[:, 1] * km_per_deg_lat * 1000
        return np.column_stack([x_meters, y_meters])
    
    grid1_meters = deg_to_meters(grid1_vertices)
    grid2_meters = deg_to_meters(grid2_vertices)
    
    print(f"Grid 1 in meters: {grid1_meters}")
    print(f"Grid 2 in meters: {grid2_meters}")
    
    # Test snapping with 100m threshold
    snap_distance = 100  # meters
    tree = KDTree(grid2_meters)
    
    snapped_count = 0
    for i, vertex in enumerate(grid1_meters):
        distances, neighbor_indices = tree.query(vertex, k=1)
        
        if distances <= snap_distance:
            print(f"  Vertex {i}: {distances:.1f}m → SNAP")
            snapped_count += 1
        else:
            print(f"  Vertex {i}: {distances:.1f}m → no snap")
    
    print(f"Total vertices snapped: {snapped_count}/{len(grid1_vertices)}")

if __name__ == "__main__":
    debug_stored_heatmaps()
    test_vertex_snapping()