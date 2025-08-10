#!/usr/bin/env python3
"""
Quick fix for polygon holes using the existing GeoJSON data.
Uses containment detection to identify and remove internal holes.
"""

import geopandas as gpd
from shapely.ops import unary_union
import pandas as pd

def fix_polygon_holes():
    """Process existing GeoJSON to remove internal holes using containment detection"""
    
    # Load existing GeoJSON file
    gdf = gpd.read_file('all_polygons_complete_latest.geojson')
    print(f"Loaded {len(gdf)} polygons from existing GeoJSON")
    
    # Find the largest polygon (main Canterbury Plains area)
    main_idx = gdf.geometry.area.idxmax()
    main_polygon = gdf.iloc[main_idx]
    main_geometry = main_polygon.geometry
    
    print(f"Main polygon area: {main_geometry.area:.8f} sq degrees")
    
    # Get all other polygons
    others = gdf.drop(main_idx)
    
    # Identify holes using containment detection
    holes = others[others.geometry.within(main_geometry)]
    external_polygons = others[~others.geometry.within(main_geometry)]
    
    print(f"Found {len(holes)} internal holes (bedrock exclusions)")
    print(f"Found {len(external_polygons)} external polygons (separate drainage areas)")
    
    # Create main polygon with holes subtracted
    if len(holes) > 0:
        holes_union = unary_union(holes.geometry.tolist())
        main_geometry_with_holes = main_geometry.difference(holes_union)
        print(f"Subtracted {len(holes)} holes from main polygon")
    else:
        main_geometry_with_holes = main_geometry
    
    # Create result polygons list
    result_data = []
    
    # Add main polygon with holes removed
    result_data.append({
        'polygon_type': 'main_with_holes_removed',
        'feature_id': 0,
        'area_sq_deg': main_geometry_with_holes.area,
        'geometry': main_geometry_with_holes
    })
    
    # Add external polygons (separate drainage areas)
    for i, (idx, row) in enumerate(external_polygons.iterrows(), 1):
        result_data.append({
            'polygon_type': 'external_drainage_area', 
            'feature_id': i,
            'area_sq_deg': row.geometry.area,
            'geometry': row.geometry
        })
    
    # Create final GeoDataFrame
    result_gdf = gpd.GeoDataFrame(result_data, crs='EPSG:4326')
    
    print(f"Final result: {len(result_gdf)} polygons")
    print(f"Total area: {result_gdf.geometry.area.sum():.8f} sq degrees")
    
    return result_gdf

if __name__ == "__main__":
    # Process polygons
    result = fix_polygon_holes()
    
    # Save processed result
    output_path = "comprehensive_polygons_processed.geojson"
    result.to_file(output_path, driver='GeoJSON')
    print(f"Saved processed polygons to {output_path}")
    
    # Also replace the original file for immediate use
    result.to_file("all_polygons_complete_latest.geojson", driver='GeoJSON')
    print(f"Updated original file with holes removed")