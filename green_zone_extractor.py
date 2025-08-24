"""
Green Zone Polygon Extractor
Creates boundary polygons around high-probability indicator kriging zones (≥0.7)
"""

import json
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd

def extract_green_zones_from_indicator_heatmaps(polygon_db):
    """
    Extract green zones (≥0.7 probability) from stored indicator kriging heatmaps
    and create a boundary polygon around them.
    
    Returns:
        dict: GeoJSON polygon representing the boundary of green zones
    """
    print("🔍 EXTRACTING GREEN ZONES: Loading indicator kriging heatmaps...")
    
    # Get all stored heatmaps
    all_heatmaps = polygon_db.get_all_stored_heatmaps()
    
    # Filter for indicator kriging heatmaps - use name pattern since they're clearly visible
    indicator_heatmaps = [
        hm for hm in all_heatmaps 
        if 'indicator_kriging' in hm.get('heatmap_name', '').lower()
    ]
    
    print(f"📊 Found {len(indicator_heatmaps)} indicator kriging heatmaps")
    
    # Debug: Show what we found
    if indicator_heatmaps:
        print("🔍 DETECTED INDICATOR HEATMAPS:")
        for i, hm in enumerate(indicator_heatmaps[:5]):  # Show first 5
            name = hm.get('heatmap_name', 'unknown')
            method = hm.get('interpolation_method', 'unknown')
            print(f"  {i+1}. {name} (method: {method})")
        if len(indicator_heatmaps) > 5:
            print(f"  ... and {len(indicator_heatmaps) - 5} more")
    else:
        print("🔍 DEBUG: No indicator heatmaps found. First 3 available heatmaps:")
        for i, hm in enumerate(all_heatmaps[:3]):
            name = hm.get('heatmap_name', 'unknown')
            method = hm.get('interpolation_method', 'unknown')
            print(f"  {i+1}. {name} (method: {method})")
    
    if not indicator_heatmaps:
        print("❌ No indicator kriging heatmaps found")
        return None
    
    # Collect all green zone features (≥0.7)
    green_features = []
    total_features = 0
    green_feature_count = 0
    
    for heatmap in indicator_heatmaps:
        geojson_data = heatmap.get('geojson_data')
        if not geojson_data or 'features' not in geojson_data:
            continue
            
        heatmap_name = heatmap.get('heatmap_name', 'unknown')
        features = geojson_data['features']
        total_features += len(features)
        
        heatmap_green_count = 0
        for feature in features:
            if 'properties' in feature:
                # Check for indicator value ≥ 0.7 (green zone)
                value = feature['properties'].get('value', 0)
                if value >= 0.7:
                    green_features.append(feature)
                    heatmap_green_count += 1
                    green_feature_count += 1
        
        print(f"  📍 {heatmap_name}: {heatmap_green_count}/{len(features)} green features")
    
    print(f"✅ EXTRACTED: {green_feature_count}/{total_features} total green zone features")
    
    if not green_features:
        print("❌ No green zone features found (≥0.7 threshold)")
        return None
    
    # Create polygons from green features
    green_polygons = []
    
    for feature in green_features:
        try:
            coords = feature['geometry']['coordinates']
            if feature['geometry']['type'] == 'Polygon':
                # Handle standard polygon
                polygon = Polygon(coords[0])  # First ring is exterior
                if polygon.is_valid:
                    green_polygons.append(polygon)
        except Exception as e:
            continue  # Skip invalid features
    
    print(f"🔷 Created {len(green_polygons)} valid polygons from green features")
    
    if not green_polygons:
        print("❌ No valid polygons created from green features")
        return None
    
    # Create unified boundary using union and convex hull
    try:
        # First, union all the polygons to merge overlapping areas
        unified_polygon = unary_union(green_polygons)
        
        # If result is MultiPolygon, use convex hull to create single boundary
        if isinstance(unified_polygon, MultiPolygon):
            boundary_polygon = unified_polygon.convex_hull
            print(f"🔗 Created convex hull boundary from {len(unified_polygon.geoms)} disconnected areas")
        else:
            # For single polygon, use convex hull to smooth the boundary
            boundary_polygon = unified_polygon.convex_hull
            print(f"🔗 Created convex hull boundary from unified polygon")
        
        # Convert to GeoJSON
        boundary_coords = list(boundary_polygon.exterior.coords)
        
        boundary_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [boundary_coords]
                },
                "properties": {
                    "name": "Green Zone Boundary",
                    "description": f"Boundary around {green_feature_count} high-probability indicator zones (≥0.7)",
                    "zone_type": "high_probability_boundary",
                    "threshold": 0.7,
                    "feature_count": green_feature_count
                }
            }]
        }
        
        print(f"✅ BOUNDARY CREATED: Polygon with {len(boundary_coords)} vertices")
        return boundary_geojson
        
    except Exception as e:
        print(f"❌ Error creating boundary polygon: {e}")
        return None

def store_green_zone_boundary(polygon_db, boundary_geojson):
    """
    Store the green zone boundary polygon in the database.
    
    Returns:
        int: ID of stored polygon, or None if failed
    """
    if not boundary_geojson:
        return None
        
    try:
        # Calculate center point for storage
        feature = boundary_geojson['features'][0]
        coords = feature['geometry']['coordinates'][0]
        
        # Calculate centroid
        lats = [coord[1] for coord in coords]
        lons = [coord[0] for coord in coords]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Store as a special polygon
        polygon_id = polygon_db.store_polygon(
            name="Green Zone Boundary",
            polygon_type="indicator_boundary",
            coordinates=coords,
            metadata={
                "description": "Boundary around high-probability indicator zones (≥0.7)",
                "threshold": 0.7,
                "feature_count": feature['properties']['feature_count'],
                "generated_at": str(np.datetime64('now'))
            }
        )
        
        print(f"💾 STORED: Green zone boundary as polygon ID {polygon_id}")
        return polygon_id
        
    except Exception as e:
        print(f"❌ Error storing boundary polygon: {e}")
        return None

def display_green_zone_boundary_on_map(folium_map, boundary_geojson):
    """
    Add the green zone boundary to a Folium map.
    """
    if not boundary_geojson or 'features' not in boundary_geojson:
        return False
        
    try:
        import folium
        
        feature = boundary_geojson['features'][0]
        coords = feature['geometry']['coordinates'][0]
        
        # Convert coordinates to lat/lon format for Folium
        folium_coords = [[coord[1], coord[0]] for coord in coords]
        
        # Add boundary polygon to map
        folium.Polygon(
            locations=folium_coords,
            color="#00AA00",  # Green border
            weight=3,
            opacity=0.8,
            fill=False,  # No fill, just boundary
            popup=f"Green Zone Boundary<br>Features: {feature['properties']['feature_count']}<br>Threshold: ≥{feature['properties']['threshold']}"
        ).add_to(folium_map)
        
        print(f"🗺️  DISPLAYED: Green zone boundary added to map")
        return True
        
    except Exception as e:
        print(f"❌ Error displaying boundary on map: {e}")
        return False