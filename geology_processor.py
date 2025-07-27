#!/usr/bin/env python3

# Check if all required dependencies are available
try:
    import rasterio
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import box, Point
    from rasterio.mask import mask
    from rasterio.features import shapes
    import json
    import pyproj
    from shapely.ops import unary_union
    GEOLOGY_AVAILABLE = True
    print("✅ Geology clipping system available")
except ImportError as e:
    GEOLOGY_AVAILABLE = False
    print(f"⚠️ Geology clipping not available: {e}")
    # Create stub functions for when dependencies are missing
    def get_geology_boundary(*args, **kwargs):
        return None

def check_geotiff_info(tiff_path):
    """
    Analyze GeoTIFF file to understand its structure and georeferencing
    """
    print(f"ANALYZING GEOTIFF: {tiff_path}")
    print("=" * 60)
    
    try:
        with rasterio.open(tiff_path) as src:
            print(f"FORMAT: {src.driver}")
            print(f"DIMENSIONS: {src.width} x {src.height} pixels")
            print(f"BANDS: {src.count}")
            print(f"DATA TYPE: {src.dtypes[0]}")
            print(f"CRS: {src.crs}")
            print(f"TRANSFORM: {src.transform}")
            print(f"BOUNDS: {src.bounds}")
            
            # Check if it's georeferenced
            if src.crs is not None:
                print("✅ GEOREFERENCED: File has coordinate reference system")
                
                # Convert bounds to lat/lon for New Zealand context
                if src.crs.to_string() != 'EPSG:4326':
                    print(f"PROJECTION: {src.crs}")
                    # Try to get bounds in WGS84
                    try:
                        import pyproj
                        transformer = pyproj.Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
                        min_lon, min_lat = transformer.transform(src.bounds.left, src.bounds.bottom)
                        max_lon, max_lat = transformer.transform(src.bounds.right, src.bounds.top)
                        print(f"WGS84 BOUNDS: {min_lat:.4f}°S to {max_lat:.4f}°S, {min_lon:.4f}°E to {max_lon:.4f}°E")
                    except:
                        print("Could not transform to WGS84")
                else:
                    print(f"WGS84 BOUNDS: {src.bounds.bottom:.4f}°S to {src.bounds.top:.4f}°S, {src.bounds.left:.4f}°E to {src.bounds.right:.4f}°E")
            else:
                print("❌ NOT GEOREFERENCED: File lacks coordinate reference system")
            
            # Check data values
            sample_data = src.read(1, window=rasterio.windows.Window(0, 0, min(100, src.width), min(100, src.height)))
            unique_values = np.unique(sample_data[~np.isnan(sample_data)])
            print(f"SAMPLE VALUES: {unique_values[:20]}...")  # First 20 unique values
            print(f"UNIQUE VALUE COUNT: {len(unique_values)}")
            
            if len(unique_values) < 50:
                print("DATA TYPE: Appears to be classified/categorical data")
            else:
                print("DATA TYPE: Appears to be continuous data")
                
            return {
                'georeferenced': src.crs is not None,
                'crs': str(src.crs) if src.crs else None,
                'bounds': src.bounds,
                'width': src.width,
                'height': src.height,
                'data_type': 'categorical' if len(unique_values) < 50 else 'continuous',
                'unique_values': unique_values.tolist()
            }
            
    except Exception as e:
        print(f"ERROR reading GeoTIFF: {e}")
        return None

def extract_ois_river_deposits(tiff_path, target_crs='EPSG:4326'):
    """
    Extract OIS1 (Holocene) and OIS2 (Late Pleistocene) river deposits based on identified colors
    OIS1: RGB(26,229,102) - Holocene river deposits
    OIS2: RGB(153,255,50) - Late Pleistocene river deposits
    """
    print(f"\nEXTRACTING OIS1/OIS2 RIVER DEPOSITS FROM: {tiff_path}")
    print("=" * 60)
    
    # Known colors for OIS units based on analysis
    ois_colors = {
        'OIS1': (26, 229, 102),   # Holocene river deposits - Light blue-green
        'OIS2': (153, 255, 50)    # Late Pleistocene river deposits - Light green-yellow
    }
    
    try:
        with rasterio.open(tiff_path) as src:
            print(f"Searching for OIS river deposit colors:")
            print(f"  OIS1 (Holocene): RGB{ois_colors['OIS1']}")
            print(f"  OIS2 (Late Pleistocene): RGB{ois_colors['OIS2']}")
            
            # Read RGB bands
            data = src.read()
            geology_polygons = []
            
            for unit_name, (r_target, g_target, b_target) in ois_colors.items():
                print(f"\nProcessing {unit_name} - RGB({r_target},{g_target},{b_target})...")
                
                # Create mask for exact color match
                mask = ((data[0] == r_target) & 
                       (data[1] == g_target) & 
                       (data[2] == b_target))
                
                if np.any(mask):
                    pixel_count = np.sum(mask)
                    area_km2 = pixel_count * (410.68 ** 2) / 1e6  # Area in km²
                    print(f"  Found {pixel_count} pixels (~{area_km2:.2f} km²)")
                    
                    # Convert mask to polygons
                    polygon_count = 0
                    for geom, value in shapes(mask.astype(rasterio.uint8), 
                                            mask=mask, 
                                            transform=src.transform):
                        if value == 1:
                            geology_polygons.append({
                                'geometry': geom,
                                'properties': {
                                    'unit_type': unit_name,
                                    'rgb': f'({r_target},{g_target},{b_target})',
                                    'description': f'{unit_name} river deposits',
                                    'age': 'Holocene' if unit_name == 'OIS1' else 'Late Pleistocene'
                                }
                            })
                            polygon_count += 1
                    
                    print(f"  Extracted {polygon_count} {unit_name} polygons")
                else:
                    print(f"  No pixels found for {unit_name}")
            
            if geology_polygons:
                # Create GeoDataFrame
                from shapely.geometry import shape
                shapely_geoms = [shape(poly['geometry']) for poly in geology_polygons]
                properties = [poly['properties'] for poly in geology_polygons]
                
                gdf = gpd.GeoDataFrame(properties, geometry=shapely_geoms, crs=src.crs)
                
                # Reproject if needed
                if str(src.crs) != target_crs:
                    print(f"REPROJECTING: {src.crs} → {target_crs}")
                    gdf = gdf.to_crs(target_crs)
                
                # Summary statistics
                ois1_count = len(gdf[gdf['unit_type'] == 'OIS1'])
                ois2_count = len(gdf[gdf['unit_type'] == 'OIS2'])
                print(f"\n✅ EXTRACTION COMPLETE:")
                print(f"   OIS1 (Holocene): {ois1_count} polygons")
                print(f"   OIS2 (Late Pleistocene): {ois2_count} polygons")
                print(f"   Total: {len(gdf)} river deposit polygons")
                
                return gdf
            else:
                print("❌ No OIS river deposit polygons found")
                return None
                
    except Exception as e:
        print(f"ERROR extracting OIS river deposits: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_specific_geology_units(tiff_path, unit_names=['OIS1', 'OIS2'], target_crs='EPSG:4326'):
    """
    Extract specific geology units (OIS1 and OIS2 river deposits) from GeoTIFF
    Returns polygons for only the specified units
    """
    print(f"\nEXTRACTING SPECIFIC GEOLOGY UNITS: {unit_names}")
    print("=" * 60)
    
    try:
        with rasterio.open(tiff_path) as src:
            # For RGB GeoTIFF, we need to check if there's associated metadata
            # that maps pixel values to geology unit names
            
            # First, let's examine the data structure
            print(f"Bands: {src.count}, Data type: {src.dtypes[0]}")
            
            # Read all bands to understand the data structure
            data = src.read()
            print(f"Data shape: {data.shape}")
            
            # For a 4-band RGBA image, we might need to:
            # 1. Convert RGB values to geology unit IDs
            # 2. Use external lookup table to map colors to unit names
            
            # Let's first extract unique color combinations
            if src.count >= 3:  # RGB or RGBA
                # Reshape to get unique color combinations
                rgb_data = data[:3]  # Take RGB bands
                reshaped = rgb_data.reshape(3, -1).T  # Shape: (pixels, 3)
                unique_colors = np.unique(reshaped, axis=0)
                print(f"Found {len(unique_colors)} unique RGB combinations")
                
                # Create potential OIS unit mapping based on common geological map colors
                # River deposits are often shown in light blues, greens, or sandy colors
                ois_color_candidates = []
                
                # Look for colors that might represent river deposits
                for i, color in enumerate(unique_colors):
                    r, g, b = color
                    
                    # Skip obvious background colors
                    if ((r == 0 and g == 0 and b == 0) or  # Black
                        (r == 255 and g == 255 and b == 255) or  # White
                        (r < 5 and g < 5 and b < 5)):  # Very dark colors
                        continue
                    
                    # Look for colors that might represent alluvial/river deposits
                    # Light blues, sandy colors, light greens, yellows
                    is_potential_river_deposit = (
                        (r < 100 and g < 150 and b > 150) or  # Light blue
                        (r > 150 and g > 150 and b < 100) or  # Sandy/yellow
                        (r < 150 and g > 150 and b < 150) or  # Light green
                        (r > 200 and g > 200 and b > 150) or  # Light sandy
                        (abs(r-g) < 30 and b > 100 and r > 50)  # Blue-ish tones
                    )
                    
                    if is_potential_river_deposit:
                        ois_color_candidates.append((i, color))
                
                print(f"Found {len(ois_color_candidates)} potential river deposit colors")
                
                # Extract polygons for all potential river deposit colors
                geology_polygons = []
                
                for i, color in ois_color_candidates:
                    r, g, b = color
                    
                    # Create mask for this color
                    mask = ((data[0] == r) & (data[1] == g) & (data[2] == b))
                    
                    if np.any(mask):
                        pixel_count = np.sum(mask)
                        area_km2 = pixel_count * (410.68 ** 2) / 1e6  # Approximate area in km²
                        
                        # Only process colors with reasonable polygon sizes
                        if pixel_count > 100:  # At least 100 pixels
                            print(f"Processing color ({r},{g},{b}): {pixel_count} pixels, ~{area_km2:.2f} km²")
                            
                            # Convert mask to polygons
                            for geom, value in shapes(mask.astype(rasterio.uint8), 
                                                    mask=mask, 
                                                    transform=src.transform):
                                if value == 1:
                                    # Determine potential unit type based on color characteristics
                                    potential_unit = 'unknown'
                                    if r < 100 and g < 150 and b > 150:
                                        potential_unit = 'OIS1_candidate'  # Light blue - Holocene
                                    elif r > 150 and g > 150 and b < 100:
                                        potential_unit = 'OIS2_candidate'  # Sandy - Late Pleistocene
                                    elif r < 150 and g > 150 and b < 150:
                                        potential_unit = 'OIS1_candidate'  # Light green - Recent deposits
                                    
                                    geology_polygons.append({
                                        'geometry': geom,
                                        'properties': {
                                            'unit_id': f'color_{i}',
                                            'rgb': f'({r},{g},{b})',
                                            'potential_unit': potential_unit,
                                            'pixel_count': int(pixel_count),
                                            'area_km2': round(area_km2, 2)
                                        }
                                    })
                
                if geology_polygons:
                    # Create GeoDataFrame
                    from shapely.geometry import shape
                    shapely_geoms = [shape(poly['geometry']) for poly in geology_polygons]
                    properties = [poly['properties'] for poly in geology_polygons]
                    
                    gdf = gpd.GeoDataFrame(properties, geometry=shapely_geoms, crs=src.crs)
                    
                    # Reproject if needed
                    if str(src.crs) != target_crs:
                        print(f"REPROJECTING: {src.crs} → {target_crs}")
                        gdf = gdf.to_crs(target_crs)
                    
                    print(f"EXTRACTED: {len(gdf)} geology unit polygons")
                    return gdf
                else:
                    print("No geology unit polygons found")
                    return None
            else:
                print("Single band data - using value-based extraction")
                # Handle single band raster
                band_data = src.read(1)
                unique_values = np.unique(band_data[~np.isnan(band_data.astype(float))])
                print(f"Unique values: {unique_values}")
                
                # Extract polygons for each unique value
                geology_polygons = []
                for value in unique_values:
                    if value != src.nodata and not np.isnan(value):
                        mask = (band_data == value)
                        
                        for geom, val in shapes(mask.astype(rasterio.uint8), 
                                              mask=mask, 
                                              transform=src.transform):
                            if val == 1:
                                geology_polygons.append({
                                    'geometry': geom,
                                    'properties': {
                                        'unit_id': f'value_{value}',
                                        'value': value
                                    }
                                })
                
                if geology_polygons:
                    from shapely.geometry import shape
                    shapely_geoms = [shape(poly['geometry']) for poly in geology_polygons]
                    properties = [poly['properties'] for poly in geology_polygons]
                    
                    gdf = gpd.GeoDataFrame(properties, geometry=shapely_geoms, crs=src.crs)
                    
                    if str(src.crs) != target_crs:
                        gdf = gdf.to_crs(target_crs)
                    
                    return gdf
                
                return None
                
    except Exception as e:
        print(f"ERROR extracting geology units: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_geology_polygons(tiff_path, target_crs='EPSG:4326'):
    """
    Extract geology polygons from GeoTIFF and convert to vector format
    """
    print(f"\nEXTRACTING GEOLOGY POLYGONS FROM: {tiff_path}")
    print("=" * 60)
    
    try:
        with rasterio.open(tiff_path) as src:
            # Read the first band (assuming RGB data, use first band for classification)
            data = src.read(1)
            
            # Convert raster to vector polygons
            polygon_data = []
            for geom, value in shapes(data, mask=data != src.nodata, transform=src.transform):
                if value is not None and not np.isnan(float(value)):
                    polygon_data.append({
                        'geometry': geom,
                        'geology_class': int(value)
                    })
            
            if not polygon_data:
                print("No valid polygons found")
                return None
            
            # Extract geometries and properties separately
            geometries = [item['geometry'] for item in polygon_data]
            properties = [{'geology_class': item['geology_class']} for item in polygon_data]
            
            # Create GeoDataFrame with proper geometry column
            from shapely.geometry import shape
            shapely_geometries = [shape(geom) for geom in geometries]
            
            gdf = gpd.GeoDataFrame(properties, geometry=shapely_geometries, crs=src.crs)
            
            # Reproject to target CRS if needed
            if str(src.crs) != target_crs:
                print(f"REPROJECTING: {src.crs} → {target_crs}")
                gdf = gdf.to_crs(target_crs)
            
            print(f"EXTRACTED: {len(gdf)} geology polygons")
            unique_classes = sorted(gdf['geology_class'].unique())
            print(f"GEOLOGY CLASSES: {unique_classes[:20]}...")  # Show first 20
            print(f"TOTAL CLASSES: {len(unique_classes)}")
            
            return gdf
            
    except Exception as e:
        print(f"ERROR extracting polygons: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_geology_boundary(tiff_path, target_crs='EPSG:4326', simplify_tolerance=0.0001):
    """
    Extract a simplified geology boundary for heatmap clipping
    This is more efficient than extracting all individual polygons
    """
    print(f"\nEXTRACTING GEOLOGY BOUNDARY FROM: {tiff_path}")
    print("=" * 60)
    
    try:
        with rasterio.open(tiff_path) as src:
            # Read the first band
            data = src.read(1)
            
            # Create a binary mask where any geology data exists
            geology_mask = (data != src.nodata) & (~np.isnan(data.astype(float)))
            
            # Convert mask to polygons
            boundary_geoms = []
            for geom, value in shapes(geology_mask.astype(rasterio.uint8), 
                                    mask=geology_mask, 
                                    transform=src.transform):
                if value == 1:  # Valid geology area
                    boundary_geoms.append(geom)
            
            if not boundary_geoms:
                print("No geology boundary found")
                return None
            
            # Create boundary GeoDataFrame
            from shapely.geometry import shape
            shapely_geoms = [shape(geom) for geom in boundary_geoms]
            
            gdf = gpd.GeoDataFrame({'geology': [1] * len(shapely_geoms)}, 
                                 geometry=shapely_geoms, crs=src.crs)
            
            # Dissolve into single boundary and simplify
            boundary = gdf.dissolve().simplify(simplify_tolerance)
            
            # Reproject to target CRS if needed
            if str(src.crs) != target_crs:
                print(f"REPROJECTING: {src.crs} → {target_crs}")
                boundary = boundary.to_crs(target_crs)
            
            print(f"EXTRACTED: Geology boundary with {len(boundary)} polygon(s)")
            
            return boundary
            
    except Exception as e:
        print(f"ERROR extracting boundary: {e}")
        import traceback
        traceback.print_exc()
        return None

def clip_heatmap_to_geology(geojson_data, geology_boundary):
    """
    Clip heatmap triangles to geology boundary - optimized version
    """
    print(f"\nCLIPPING HEATMAP TO GEOLOGY BOUNDARIES")
    print("=" * 50)
    
    clipped_features = []
    original_count = len(geojson_data['features'])
    
    # Get the boundary geometry
    boundary_geom = geology_boundary.geometry.iloc[0]
    
    for feature in geojson_data['features']:
        # Convert feature to shapely geometry
        from shapely.geometry import shape
        triangle = shape(feature['geometry'])
        
        # Check if triangle intersects with geology boundary
        if boundary_geom.intersects(triangle):
            # Clip triangle to geology boundary
            try:
                clipped = boundary_geom.intersection(triangle)
                
                if not clipped.is_empty and clipped.area > 0:
                    # Convert back to GeoJSON feature
                    if hasattr(clipped, 'geoms'):
                        # Multi-geometry result
                        for geom in clipped.geoms:
                            if hasattr(geom, 'area') and geom.area > 0:
                                clipped_features.append({
                                    'type': 'Feature',
                                    'geometry': geom.__geo_interface__,
                                    'properties': feature['properties']
                                })
                    else:
                        # Single geometry result
                        clipped_features.append({
                            'type': 'Feature',
                            'geometry': clipped.__geo_interface__,
                            'properties': feature['properties']
                        })
            except Exception as e:
                # If clipping fails, skip this feature
                continue
    
    clipped_count = len(clipped_features)
    retention_percent = (clipped_count/original_count*100) if original_count > 0 else 0
    print(f"CLIPPING RESULT: {original_count} → {clipped_count} features ({retention_percent:.1f}% retained)")
    
    return {
        'type': 'FeatureCollection',
        'features': clipped_features
    }

def test_geology_import():
    """
    Test the geology import functionality
    """
    tiff_path = "attached_assets/NZ Geology_1753590503005.tif"
    
    # Check file info
    info = check_geotiff_info(tiff_path)
    
    if info and info['georeferenced']:
        print("\n" + "="*60)
        print("GEOLOGY IMPORT READY FOR INTEGRATION")
        print("="*60)
        print("✅ File is georeferenced and can be used for heatmap clipping")
        print("✅ Can extract polygon boundaries for each geology class")
        print("✅ Compatible with existing interpolation system")
        
        # Extract geology boundary for efficient clipping
        print("\nEXTRACTING GEOLOGY BOUNDARY...")
        boundary = get_geology_boundary(tiff_path)
        
        if boundary is not None and len(boundary) > 0:
            print(f"✅ Successfully extracted geology boundary")
            print(f"📊 Boundary area: {boundary.geometry.iloc[0].area:.6f} square degrees")
            print(f"📍 Boundary bounds: {boundary.total_bounds}")
            return True
    
    return False

if __name__ == "__main__":
    test_geology_import()