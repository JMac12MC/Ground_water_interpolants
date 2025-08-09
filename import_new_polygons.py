#!/usr/bin/env python3
"""
Import and process new polygon data to replace Banks Peninsula and soil drainage polygons
"""
import json
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import pyproj
from pyproj import Transformer
import pandas as pd

def load_esri_json(file_path):
    """Load Esri-style JSON polygon data"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def convert_esri_to_shapely_polygons(esri_data):
    """Convert Esri JSON rings to Shapely polygons"""
    polygons = []
    
    for feature in esri_data.get('features', []):
        geometry = feature.get('geometry', {})
        rings = geometry.get('rings', [])
        
        if rings:
            # First ring is exterior, subsequent rings are holes
            exterior = rings[0]
            holes = rings[1:] if len(rings) > 1 else []
            
            try:
                # Create polygon with holes
                polygon = Polygon(exterior, holes)
                if polygon.is_valid:
                    polygons.append(polygon)
                else:
                    print(f"Invalid polygon found, attempting to fix...")
                    # Try to fix invalid geometry
                    fixed_polygon = polygon.buffer(0)
                    if fixed_polygon.is_valid:
                        polygons.append(fixed_polygon)
            except Exception as e:
                print(f"Error creating polygon: {e}")
                continue
    
    return polygons

def convert_nztm_to_wgs84(polygons):
    """Convert from NZTM (EPSG:2193) to WGS84 (EPSG:4326)"""
    # New Zealand Transverse Mercator to WGS84
    transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
    
    converted_polygons = []
    for polygon in polygons:
        if isinstance(polygon, Polygon):
            # Convert exterior coordinates
            exterior_coords = [transformer.transform(x, y) for x, y in polygon.exterior.coords]
            
            # Convert hole coordinates if any
            holes = []
            for interior in polygon.interiors:
                hole_coords = [transformer.transform(x, y) for x, y in interior.coords]
                holes.append(hole_coords)
            
            # Create new polygon with converted coordinates (lon, lat)
            converted_polygon = Polygon(exterior_coords, holes)
            converted_polygons.append(converted_polygon)
    
    return converted_polygons

def merge_overlapping_polygons(polygons):
    """Merge overlapping or touching polygons"""
    if not polygons:
        return []
    
    print(f"Starting with {len(polygons)} polygons")
    
    # Create a MultiPolygon and use unary_union to merge overlapping/touching polygons
    multi_polygon = MultiPolygon(polygons)
    merged = unary_union(multi_polygon)
    
    # Handle result - could be Polygon, MultiPolygon, or GeometryCollection
    if hasattr(merged, 'geoms'):
        # MultiPolygon or GeometryCollection
        result_polygons = [geom for geom in merged.geoms if geom.geom_type == 'Polygon']
    elif merged.geom_type == 'Polygon':
        # Single Polygon
        result_polygons = [merged]
    else:
        print(f"Unexpected geometry type after merge: {merged.geom_type}")
        result_polygons = polygons
    
    print(f"After merging: {len(result_polygons)} polygons")
    return result_polygons

def create_geodataframe(polygons):
    """Create GeoDataFrame from polygons"""
    if not polygons:
        return None
    
    # Create GeoDataFrame with WGS84 CRS
    gdf = gpd.GeoDataFrame(
        {'id': range(len(polygons))},
        geometry=polygons,
        crs="EPSG:4326"
    )
    
    # Add area calculation (in square degrees, for display purposes)
    gdf['area_deg2'] = gdf.geometry.area
    
    return gdf

def process_polygon_file(file_path):
    """Complete processing pipeline for polygon file"""
    print(f"Processing polygon file: {file_path}")
    
    # Step 1: Load Esri JSON
    esri_data = load_esri_json(file_path)
    print(f"Loaded {len(esri_data.get('features', []))} features from Esri JSON")
    
    # Step 2: Convert to Shapely polygons
    nztm_polygons = convert_esri_to_shapely_polygons(esri_data)
    print(f"Converted to {len(nztm_polygons)} Shapely polygons in NZTM")
    
    # Step 3: Convert coordinate system
    wgs84_polygons = convert_nztm_to_wgs84(nztm_polygons)
    print(f"Converted {len(wgs84_polygons)} polygons to WGS84")
    
    # Step 4: Merge overlapping polygons
    merged_polygons = merge_overlapping_polygons(wgs84_polygons)
    print(f"Merged to {len(merged_polygons)} polygons")
    
    # Step 5: Create GeoDataFrame
    gdf = create_geodataframe(merged_polygons)
    
    if gdf is not None:
        print(f"Created GeoDataFrame with {len(gdf)} polygons")
        print(f"Bounds: {gdf.total_bounds}")
        
        # Calculate centroid for display
        centroid = gdf.dissolve().centroid.iloc[0]
        print(f"Centroid: ({centroid.y:.6f}, {centroid.x:.6f})")
        
    return gdf

def save_as_geojson(gdf, output_path):
    """Save GeoDataFrame as GeoJSON"""
    if gdf is not None:
        gdf.to_file(output_path, driver='GeoJSON')
        print(f"Saved merged polygons to: {output_path}")
        return True
    return False

if __name__ == "__main__":
    # Process the new polygon data
    input_file = "attached_assets/myDrawing_1754734043555.json"
    output_file = "processed_clipping_polygons.geojson"
    
    # Process the polygons
    gdf = process_polygon_file(input_file)
    
    # Save result
    if gdf is not None:
        save_as_geojson(gdf, output_file)
        
        print("\n=== POLYGON PROCESSING SUMMARY ===")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Final polygon count: {len(gdf)}")
        print(f"Total area: {gdf['area_deg2'].sum():.8f} square degrees")
        print(f"Bounding box: {gdf.total_bounds}")
        
        # Print individual polygon info
        for idx, row in gdf.iterrows():
            print(f"Polygon {idx}: Area = {row['area_deg2']:.8f} sq deg")
    else:
        print("ERROR: Failed to process polygons")