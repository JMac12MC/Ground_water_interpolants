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
    
    # Create detailed boundaries for separate green zones (no convex hull oversimplification)
    try:
        # First, union overlapping areas to get clean zones
        unified_geometry = unary_union(green_polygons)
        
        # Collect all separate green zone boundaries
        boundary_features = []
        zone_count = 0
        
        if isinstance(unified_geometry, MultiPolygon):
            # Multiple separate zones - create boundary for each
            for i, geom in enumerate(unified_geometry.geoms):
                if geom.area > 0:  # Only include valid geometries
                    # Use actual boundary, not convex hull for detailed shape
                    boundary_coords = list(geom.exterior.coords)
                    
                    boundary_features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [boundary_coords]
                        },
                        "properties": {
                            "name": f"Green Zone {i+1}",
                            "description": f"High-probability indicator zone (≥0.7)",
                            "zone_type": "high_probability_boundary",
                            "threshold": 0.7,
                            "zone_id": i+1
                        }
                    })
                    zone_count += 1
            
            print(f"🔗 Created {zone_count} separate detailed green zone boundaries")
            
        else:
            # Single unified zone
            boundary_coords = list(unified_geometry.exterior.coords)
            
            boundary_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon", 
                    "coordinates": [boundary_coords]
                },
                "properties": {
                    "name": "Green Zone 1",
                    "description": f"High-probability indicator zone (≥0.7)",
                    "zone_type": "high_probability_boundary", 
                    "threshold": 0.7,
                    "zone_id": 1
                }
            })
            zone_count = 1
            print(f"🔗 Created 1 detailed green zone boundary")
        
        # Create GeoJSON with multiple detailed boundaries
        boundary_geojson = {
            "type": "FeatureCollection",
            "features": boundary_features
        }
        
        print(f"✅ BOUNDARIES CREATED: {len(boundary_features)} detailed polygons for separate green zones")
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
        # Store as one comprehensive green zone boundary file (multiple features in one polygon)
        # Calculate overall center point for storage
        all_coords = []
        for feature in boundary_geojson['features']:
            coords = feature['geometry']['coordinates'][0]
            all_coords.extend(coords)
        
        # Store the entire boundary collection as one polygon
        polygon_id = polygon_db.store_polygon(
            name="Green Zone Boundaries",
            polygon_type="indicator_boundary",
            coordinates=json.dumps(boundary_geojson),  # Store the entire GeoJSON
            metadata={
                "description": f"Detailed boundaries around all high-probability indicator zones (≥0.7)",
                "threshold": 0.7,
                "total_zones": len(boundary_geojson['features']),
                "generated_at": str(np.datetime64('now'))
            }
        )
        
        print(f"💾 STORED: Green zone boundaries as single polygon file with ID {polygon_id}")
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
        
        # Add each separate green zone boundary to the map
        zones_added = 0
        
        for feature in boundary_geojson['features']:
            coords = feature['geometry']['coordinates'][0]
            zone_name = feature['properties']['name']
            
            # Convert coordinates to lat/lon format for Folium
            folium_coords = [[coord[1], coord[0]] for coord in coords]
            
            # Add boundary polygon with consistent green color
            folium.Polygon(
                locations=folium_coords,
                color="#00AA00",  # Single green color
                weight=3,
                opacity=0.8,
                fill=False,  # No fill, just boundary
                popup=f"{zone_name}<br>High-probability zone (≥{feature['properties']['threshold']})<br>Detailed boundary following actual green areas"
            ).add_to(folium_map)
            
            zones_added += 1
        
        print(f"🗺️  DISPLAYED: {zones_added} detailed green zone boundaries added to map")
        return True
        
    except Exception as e:
        print(f"❌ Error displaying boundary on map: {e}")
        return False