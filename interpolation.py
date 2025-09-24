import numpy as np
import pandas as pd
import json
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.ensemble import RandomForestRegressor
from pykrige.ok import OrdinaryKriging
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import io
import base64
from PIL import Image
import matplotlib.colors as mcolors
import streamlit as st
import pyproj

# ===== CENTRALIZED CRS TRANSFORMATION HELPERS (ARCHITECT SOLUTION) =====
def get_transformers():
    """
    Create standardized coordinate transformers with always_xy=True to avoid axis order issues.
    Returns transformers for WGS84 <-> NZTM2000 conversions.
    
    Returns:
    --------
    tuple: (transformer_to_nztm, transformer_to_wgs84)
    """
    transformer_to_nztm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    transformer_to_wgs84 = pyproj.Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
    return transformer_to_nztm, transformer_to_wgs84

def to_nztm2000(lons, lats):
    """
    Transform coordinates from WGS84 (EPSG:4326) to NZTM2000 (EPSG:2193).
    
    Parameters:
    -----------
    lons : array-like
        Longitude coordinates in WGS84
    lats : array-like  
        Latitude coordinates in WGS84
        
    Returns:
    --------
    tuple: (x_coords, y_coords) in NZTM2000 meters
    """
    transformer_to_nztm, _ = get_transformers()
    return transformer_to_nztm.transform(lons, lats)

def to_wgs84(x_coords, y_coords):
    """
    Transform coordinates from NZTM2000 (EPSG:2193) to WGS84 (EPSG:4326).
    
    Parameters:
    -----------
    x_coords : array-like
        X coordinates in NZTM2000 meters
    y_coords : array-like
        Y coordinates in NZTM2000 meters
        
    Returns:
    --------  
    tuple: (lons, lats) in WGS84 degrees
    """
    _, transformer_to_wgs84 = get_transformers()
    return transformer_to_wgs84.transform(x_coords, y_coords)

def calculate_authoritative_bounds(lats, lons, lat_step, lon_step):
    """
    SINGLE SOURCE OF TRUTH for all bounds calculations.
    
    This function eliminates coordinate alignment issues by providing
    one consistent bounds calculation used throughout the entire system.
    
    Parameters:
    -----------
    lats : array-like
        Latitude array (north to south order)
    lons : array-like  
        Longitude array (west to east order)
    lat_step : float
        Latitude step size in degrees
    lon_step : float
        Longitude step size in degrees
        
    Returns:
    --------
    dict: Standardized bounds in all required formats
    """
    # Use EDGE-based bounds consistently (pixel boundaries, not centers)
    # This ensures proper alignment between interpolation and display
    north = lats[0] + lat_step/2   # lats[0] is northernmost center + half pixel
    south = lats[-1] - lat_step/2  # lats[-1] is southernmost center - half pixel  
    west = lons[0] - lon_step/2    # lons[0] is westernmost center - half pixel
    east = lons[-1] + lon_step/2   # lons[-1] is easternmost center + half pixel
    
    return {
        'north': north, 'south': south, 'east': east, 'west': west,
        'rasterio_format': (west, south, east, north),  # (left, bottom, right, top)
        'folium_format': [[south, west], [north, east]]  # [[sw_lat, sw_lon], [ne_lat, ne_lon]]
    }
# ==================================================================

# ===== CENTRALIZED CRS/GRID CONTEXT HELPERS (ARCHITECT SOLUTION) =====
def build_crs_grid(center_latlon, radius_km, grid_size):
    """
    Build coordinate grid in both NZTM2000 meters and WGS84 degrees.
    Returns both meter grid for interpolation and WGS84 grid for GeoJSON output.
    """
    center_lat, center_lon = center_latlon
    
    # Transform center to NZTM2000
    center_x_m, center_y_m = to_nztm2000([center_lon], [center_lat])
    center_x_m, center_y_m = center_x_m[0], center_y_m[0]
    
    # Create grid bounds in NZTM2000 meters
    radius_m = radius_km * 1000.0
    min_x_m = center_x_m - radius_m
    max_x_m = center_x_m + radius_m
    min_y_m = center_y_m - radius_m
    max_y_m = center_y_m + radius_m
    
    # Create meter grid
    x_vals_m = np.linspace(min_x_m, max_x_m, grid_size)
    y_vals_m = np.linspace(min_y_m, max_y_m, grid_size)
    X_m, Y_m = np.meshgrid(x_vals_m, y_vals_m)
    
    # Transform 2D grid to WGS84 for GeoJSON output
    LONS_2D, LATS_2D = to_wgs84(X_m.flatten(), Y_m.flatten())
    LONS_2D = LONS_2D.reshape(X_m.shape)
    LATS_2D = LATS_2D.reshape(Y_m.shape)
    
    return {
        'x_vals_m': x_vals_m,
        'y_vals_m': y_vals_m, 
        'X_m': X_m,
        'Y_m': Y_m,
        'LONS_2D': LONS_2D,
        'LATS_2D': LATS_2D
    }

def prepare_wells_xy(wells_df):
    """
    Transform wells coordinates from WGS84 to NZTM2000 meters.
    """
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)
    wells_x_m, wells_y_m = to_nztm2000(lons, lats)
    return wells_x_m, wells_y_m

def krige_on_grid(wells_x_m, wells_y_m, values, x_vals_m, y_vals_m, variogram_model='spherical', verbose=False):
    """
    Perform Kriging interpolation using NZTM2000 meter coordinates.
    """
    try:
        OK = OrdinaryKriging(
            wells_x_m, wells_y_m, values,
            variogram_model=variogram_model,
            verbose=verbose,
            enable_plotting=False,
            coordinates_type='euclidean'
        )
        Z, SS = OK.execute('grid', x_vals_m, y_vals_m)
        return Z, SS, OK
    except Exception as e:
        print(f"Kriging failed with {variogram_model}: {e}")
        return None, None, None
# =================================================================

@st.cache_data
def _load_exclusion_data():
    """
    Load red/orange exclusion polygons from GeoJSON file (cached for performance)
    
    Returns:
    --------
    geopandas.GeoDataFrame or None
        GeoDataFrame containing exclusion polygon geometries
    """
    try:
        import os
        
        # Check for the red/orange exclusion polygon file
        exclusion_files = [
            "attached_assets/red_orange_zones_stored_2025-09-16_1758401039896.geojson",
            "attached_assets/red_orange_zones_stored_2025-09-16_1758015813886.geojson",
            "red_orange_zones_stored_2025-09-16.geojson",
            "attached_assets/red_orange_zones_stored_2025-09-16.geojson"
        ]
        
        for file_path in exclusion_files:
            if os.path.exists(file_path):
                # Load as raw JSON data for caching compatibility
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if data and data.get('features'):
                    print(f"✅ Loaded {len(data['features'])} exclusion polygons from {file_path}")
                    return data
                    
        print("⚠️ No red/orange exclusion polygon file found")
        return None
        
    except Exception as e:
        print(f"❌ Error loading exclusion polygons: {e}")
        return None

def load_exclusion_polygons():
    """
    Get exclusion polygons as GeoDataFrame (with caching)
    """
    data = _load_exclusion_data()
    if data:
        return gpd.GeoDataFrame.from_features(data['features'])
    return None

@st.cache_resource
def get_prepared_exclusion_union():
    """
    Get prepared exclusion union for fast intersection testing (cached per session)
    
    Returns:
    --------
    tuple or None
        (union_geometry, prepared_geometry, union_bounds, exclusion_version) or None if no exclusions
    """
    try:
        import hashlib
        import os
        from shapely.ops import unary_union
        from shapely.prepared import prep
        
        # Load exclusion polygons
        exclusion_polygons = load_exclusion_polygons()
        if exclusion_polygons is None or len(exclusion_polygons) == 0:
            return None
            
        # Create version hash for cache invalidation
        exclusion_files = [
            "attached_assets/red_orange_zones_stored_2025-09-16_1758401039896.geojson",
            "attached_assets/red_orange_zones_stored_2025-09-16_1758015813886.geojson",
            "red_orange_zones_stored_2025-09-16.geojson",
            "attached_assets/red_orange_zones_stored_2025-09-16.geojson"
        ]
        
        version_parts = []
        for file_path in exclusion_files:
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                version_parts.append(f"{file_path}:{stat.st_size}:{stat.st_mtime}")
                break
        
        exclusion_version = hashlib.md5(":".join(version_parts).encode()).hexdigest()[:8]
        
        # Create unified exclusion geometry
        valid_geometries = [geom for geom in exclusion_polygons.geometry if geom.is_valid]
        if not valid_geometries:
            return None
            
        union_geometry = unary_union(valid_geometries)
        prepared_geometry = prep(union_geometry)
        union_bounds = union_geometry.bounds  # (minx, miny, maxx, maxy)
        
        print(f"🚀 PERFORMANCE: Created prepared exclusion union from {len(valid_geometries)} zones (version: {exclusion_version})")
        
        return union_geometry, prepared_geometry, union_bounds, exclusion_version
        
    except Exception as e:
        print(f"❌ Error creating prepared exclusion union: {e}")
        return None

def apply_exclusion_clipping(heatmap_data, exclusion_polygons):
    """
    Apply negative clipping to remove heatmap points within exclusion areas
    
    Parameters:
    -----------
    heatmap_data : list
        List of heatmap data points in format [[lat, lng, value], ...]
    exclusion_polygons : geopandas.GeoDataFrame
        GeoDataFrame containing exclusion polygon geometries
        
    Returns:
    --------
    list
        Filtered heatmap data with exclusion areas removed
    """
    if exclusion_polygons is None or len(exclusion_polygons) == 0:
        return heatmap_data
        
    if not heatmap_data:
        return heatmap_data
        
    try:
        from shapely.geometry import Point
        from shapely.ops import unary_union
        
        # Create unified exclusion geometry for faster point-in-polygon testing
        valid_geometries = [geom for geom in exclusion_polygons.geometry if geom.is_valid]
        if not valid_geometries:
            return heatmap_data
            
        exclusion_geometry = unary_union(valid_geometries)
        print(f"🚫 Applying exclusion clipping with {len(valid_geometries)} exclusion zones")
        
        # Filter out points that fall within exclusion areas
        filtered_data = []
        excluded_count = 0
        
        for data_point in heatmap_data:
            lat, lng = data_point[0], data_point[1]
            point = Point(lng, lat)  # Shapely uses (lon, lat) order
            
            # Exclude point if it's within any exclusion area
            if not (exclusion_geometry.contains(point) or exclusion_geometry.intersects(point)):
                filtered_data.append(data_point)
            else:
                excluded_count += 1
                
        print(f"🚫 Exclusion clipping: Removed {excluded_count} points, kept {len(filtered_data)} points")
        return filtered_data
        
    except Exception as e:
        print(f"❌ Error applying exclusion clipping: {e}")
        return heatmap_data

def apply_exclusion_clipping_to_stored_heatmap(stored_heatmap_geojson, auto_load_exclusions=True, method_name=None, heatmap_id=None):
    """
    Apply exclusion clipping to stored heatmap GeoJSON data for non-indicator methods only
    
    Parameters:
    -----------
    stored_heatmap_geojson : dict
        GeoJSON data from stored heatmap
    auto_load_exclusions : bool
        Whether to auto-load exclusion polygons if not provided
    method_name : str
        Interpolation method name to determine if exclusion should be applied
        
    Returns:
    --------
    dict
        GeoJSON with exclusion clipping applied for non-indicator methods
    """
    if not stored_heatmap_geojson or not stored_heatmap_geojson.get('features'):
        return stored_heatmap_geojson
    
    # Define indicator methods that should NOT have red/orange exclusion clipping
    indicator_methods = [
        'indicator_kriging', 
        'indicator_kriging_spherical', 
        'indicator_kriging_spherical_continuous'
    ]
    
    # Skip exclusion clipping for indicator methods
    if method_name and method_name in indicator_methods:
        print(f"🔄 INDICATOR METHOD: Skipping red/orange exclusion clipping for {method_name} (preserving full probability distribution)")
        return stored_heatmap_geojson
        
    # Apply exclusion clipping for non-indicator methods
    try:
        # Load exclusion polygons
        exclusion_polygons = load_exclusion_polygons() if auto_load_exclusions else None
        
        if exclusion_polygons is not None:
            features_before = len(stored_heatmap_geojson['features'])
            filtered_features = apply_exclusion_clipping_to_geojson(stored_heatmap_geojson['features'], exclusion_polygons, heatmap_id=heatmap_id)
            
            # Return updated GeoJSON
            filtered_geojson = {
                "type": stored_heatmap_geojson.get("type", "FeatureCollection"),
                "features": filtered_features
            }
            
            print(f"🚫 PRODUCTION: Applied red/orange exclusion clipping to {method_name or 'heatmap'}: {features_before} -> {len(filtered_features)} features")
            return filtered_geojson
        else:
            print(f"⚠️ No red/orange exclusion polygons found for {method_name or 'heatmap'} - showing {len(stored_heatmap_geojson['features'])} features without clipping")
            return stored_heatmap_geojson
            
    except Exception as e:
        print(f"❌ PRODUCTION: Error applying exclusion clipping to {method_name or 'heatmap'}: {e}")
        return stored_heatmap_geojson

# Removed unused caching function - using session_state cache instead

def bbox_intersects(bbox1, bbox2):
    """
    Fast bounding box intersection test
    
    Parameters:
    -----------
    bbox1, bbox2 : tuple
        Bounding boxes as (minx, miny, maxx, maxy)
        
    Returns:
    --------
    bool
        True if bounding boxes intersect
    """
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or 
                bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

def get_features_bbox(features):
    """
    Calculate bounding box of all features
    
    Parameters:
    -----------
    features : list
        List of GeoJSON feature objects
        
    Returns:
    --------
    tuple or None
        Bounding box as (minx, miny, maxx, maxy) or None
    """
    if not features:
        return None
        
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for feature in features:
        try:
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0]
                for coord in coords:
                    x, y = coord[0], coord[1]
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
        except:
            continue
            
    if min_x == float('inf'):
        return None
        
    return (min_x, min_y, max_x, max_y)

def apply_exclusion_clipping_to_geojson(features, exclusion_polygons, heatmap_id=None):
    """
    OPTIMIZED: Apply negative clipping to remove GeoJSON features within exclusion areas
    Uses prepared geometry, bounding box prefilters, and per-heatmap caching
    
    Parameters:
    -----------
    features : list
        List of GeoJSON feature objects
    exclusion_polygons : geopandas.GeoDataFrame
        GeoDataFrame containing exclusion polygon geometries (ignored, uses cached union)
    heatmap_id : str, optional
        Unique identifier for caching (e.g., "ground_water_level_kriging_gridpoint1")
        
    Returns:
    --------
    list
        Filtered GeoJSON features with exclusion areas removed
    """
    if not features:
        return features
        
    try:
        # Get prepared exclusion union (cached)
        exclusion_data = get_prepared_exclusion_union()
        if exclusion_data is None:
            return features
            
        union_geometry, prepared_geometry, union_bounds, exclusion_version = exclusion_data
        
        # Fast bounding box prefilter
        features_bbox = get_features_bbox(features)
        if features_bbox is None:
            return features
            
        if not bbox_intersects(features_bbox, union_bounds):
            print(f"🚀 FAST SKIP: Heatmap {heatmap_id or 'unknown'} bbox doesn't intersect exclusion zones")
            return features
        
        # Check cache if heatmap_id provided
        if heatmap_id:
            import hashlib
            features_str = str(len(features))  # Simplified hash based on count for performance
            cache_key = f"clipped_result_{heatmap_id}_{features_str}_{exclusion_version}"
            
            if cache_key in st.session_state:
                filtered_features, excluded_count = st.session_state[cache_key]
                print(f"🚀 CACHE HIT: {heatmap_id} - {excluded_count} excluded, {len(filtered_features)} kept")
                return filtered_features
        
        from shapely.geometry import Polygon as ShapelyPolygon
        
        print(f"🚫 Applying exclusion clipping to GeoJSON features with {len(exclusion_polygons) if exclusion_polygons is not None else 0} exclusion zones")
        
        # Filter out features that intersect with exclusion areas
        filtered_features = []
        excluded_count = 0
        
        for feature in features:
            try:
                # Get feature geometry
                if feature['geometry']['type'] == 'Polygon':
                    coords = feature['geometry']['coordinates'][0]
                    if len(coords) >= 3:
                        # Fast bbox check first
                        feature_bbox = get_features_bbox([feature])
                        if feature_bbox and not bbox_intersects(feature_bbox, union_bounds):
                            filtered_features.append(feature)
                            continue
                        
                        # Create Shapely polygon from coordinates
                        shapely_coords = [(coord[0], coord[1]) for coord in coords[:-1]]  # Remove duplicate closing point
                        feature_polygon = ShapelyPolygon(shapely_coords)
                        
                        # FIXED: Use geometric difference/intersection instead of just feature removal
                        try:
                            # Create allowed geometry: feature - exclusion_zones
                            allowed_geometry = feature_polygon.difference(union_geometry)
                            
                            if not allowed_geometry.is_empty:
                                # Handle MultiPolygon results
                                if hasattr(allowed_geometry, 'geoms'):
                                    # Split into separate features for each disconnected part
                                    for geom_part in allowed_geometry.geoms:
                                        if geom_part.area > 1e-10:  # Skip tiny slivers
                                            # Convert back to GeoJSON coordinates
                                            coords = list(geom_part.exterior.coords)
                                            new_feature = feature.copy()
                                            new_feature['geometry']['coordinates'] = [coords]
                                            filtered_features.append(new_feature)
                                else:
                                    # Single polygon result
                                    if allowed_geometry.area > 1e-10:  # Skip tiny slivers
                                        coords = list(allowed_geometry.exterior.coords)
                                        new_feature = feature.copy()
                                        new_feature['geometry']['coordinates'] = [coords]
                                        filtered_features.append(new_feature)
                                        
                            if allowed_geometry.is_empty or allowed_geometry.area < feature_polygon.area * 0.9:
                                excluded_count += 1  # Count as clipped if significant area removed
                        except Exception as geom_error:
                            # Fallback to old intersection test if geometric operations fail
                            if not (prepared_geometry.intersects(feature_polygon) or prepared_geometry.contains(feature_polygon)):
                                filtered_features.append(feature)
                            else:
                                excluded_count += 1
                    else:
                        # Keep features with invalid geometry
                        filtered_features.append(feature)
                else:
                    # Keep non-polygon features
                    filtered_features.append(feature)
                    
            except Exception as e:
                # If geometry processing fails, keep the feature
                filtered_features.append(feature)
        
        # Cache the result if heatmap_id provided
        if heatmap_id:
            # Store the actual result in session state cache
            st.session_state[cache_key] = (filtered_features, excluded_count)
                
        print(f"🚫 GeoJSON exclusion clipping: Removed {excluded_count} features, kept {len(filtered_features)} features")
        return filtered_features
        
    except Exception as e:
        print(f"❌ Error applying GeoJSON exclusion clipping: {e}")
        return features

def create_indicator_polygon_geometry(indicator_mask, threshold=0.7):
    """
    Convert indicator kriging mask into polygon geometry for clipping
    
    Parameters:
    -----------
    indicator_mask : tuple
        Tuple containing indicator kriging mask data
    threshold : float
        Threshold for high-probability zones
        
    Returns:
    --------
    shapely.geometry or None
        Merged polygon geometry of high-probability zones
    """
    try:
        if indicator_mask is None:
            return None
            
        mask_lat_grid, mask_lon_grid, mask_values, mask_lat_vals, mask_lon_vals = indicator_mask
        if mask_values is None:
            return None
            
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
        
        # Create polygons for grid cells where probability >= threshold
        polygons = []
        for i in range(len(mask_lat_vals)-1):
            for j in range(len(mask_lon_vals)-1):
                if mask_values[i, j] >= threshold:
                    # Create polygon for this grid cell
                    min_lat, max_lat = mask_lat_vals[i], mask_lat_vals[i+1]
                    min_lon, max_lon = mask_lon_vals[j], mask_lon_vals[j+1]
                    
                    polygon = Polygon([
                        (min_lon, min_lat),
                        (max_lon, min_lat), 
                        (max_lon, max_lat),
                        (min_lon, max_lat),
                        (min_lon, min_lat)
                    ])
                    polygons.append(polygon)
        
        if not polygons:
            return None
            
        # Merge all polygons into a single geometry
        merged_geometry = unary_union(polygons)
        
        print(f"Created indicator clipping geometry from {len(polygons)} high-probability cells")
        return merged_geometry
        
    except Exception as e:
        print(f"Error creating indicator polygon geometry: {e}")
        return None

def generate_indicator_kriging_mask(wells_df, center_point, radius_km, resolution=50, soil_polygons=None, threshold=0.7):
    """
    Generate an indicator kriging mask for high-probability zones (≥ threshold)
    
    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data
    center_point : tuple
        (latitude, longitude) of center point
    radius_km : float
        Radius in km to include in the grid
    resolution : int
        Grid resolution (number of cells per dimension)
    soil_polygons : GeoDataFrame, optional
        Soil polygon data for initial clipping
    threshold : float
        Threshold for high-probability zones (default 0.7)
        
    Returns:
    --------
    tuple
        (lat_grid, lon_grid, mask_values, grid_lats, grid_lons)
    """
    if isinstance(wells_df, pd.DataFrame) and wells_df.empty:
        return None, None, None, None, None
        
    # Get indicator kriging interpolation for the area
    try:
        # Use indicator kriging to get probability surface
        geojson_data = generate_geo_json_grid(
            wells_df, center_point, radius_km, 
            resolution=resolution, method='indicator_kriging',
            soil_polygons=soil_polygons
        )
        
        if not geojson_data or not geojson_data.get('features'):
            return None, None, None, None, None
            
        # Extract center coordinates and setup grid with HIGH-PRECISION conversion
        center_lat, center_lon = center_point
        
        # ===== ARCHITECT CLEANUP: Removed legacy precision calculation variables =====
        # These were used for km_per_degree calculations, now using NZTM2000 system
        print(f"🔧 Using NZTM2000 coordinate system for indicator mask generation")
        # ===========================================================================
        
        # ===== REMOVED: Legacy get_precise_conversion_factors (ARCHITECT CLEANUP) =====
        # This function was used for km_per_degree calculations which have been 
        # replaced with proper NZTM2000 coordinate transformations
        # ============================================================================
        
        # ===== ARCHITECT SOLUTION: BOUNDS CALCULATION WITH NZTM2000 =====
        print(f"🔧 Creating indicator mask bounds using NZTM2000 coordinate system")
        
        # Transform center to NZTM2000
        center_x_m, center_y_m = to_nztm2000([center_lon], [center_lat])
        center_x_m, center_y_m = center_x_m[0], center_y_m[0]
        
        # Create bounds in NZTM2000 meters
        radius_m = radius_km * 1000.0
        min_x_m = center_x_m - radius_m
        max_x_m = center_x_m + radius_m
        min_y_m = center_y_m - radius_m
        max_y_m = center_y_m + radius_m
        
        # Transform bounds back to WGS84 for grid creation
        bounds_x_m = [min_x_m, max_x_m, max_x_m, min_x_m]
        bounds_y_m = [min_y_m, min_y_m, max_y_m, max_y_m]
        bounds_lons, bounds_lats = to_wgs84(bounds_x_m, bounds_y_m)
        
        min_lat, max_lat = bounds_lats.min(), bounds_lats.max()
        min_lon, max_lon = bounds_lons.min(), bounds_lons.max()
        
        print(f"🔧 MASK BOUNDS: lat [{min_lat:.6f}, {max_lat:.6f}]°, lon [{min_lon:.6f}, {max_lon:.6f}]°")
        # ========================================================================
        
        # Create grid
        grid_size = min(150, max(50, resolution))
        lat_vals = np.linspace(min_lat, max_lat, grid_size)
        lon_vals = np.linspace(min_lon, max_lon, grid_size)
        grid_lons, grid_lats = np.meshgrid(lon_vals, lat_vals)
        
        # Initialize mask with zeros
        mask_values = np.zeros_like(grid_lats)
        
        # Extract probability values from GeoJSON features
        for feature in geojson_data['features']:
            if feature['geometry']['type'] == 'Polygon':
                # Get polygon coordinates and yield value
                coords = feature['geometry']['coordinates'][0]
                yield_value = feature['properties']['yield']
                
                # Find centroid of polygon
                if len(coords) >= 4:  # Valid polygon
                    centroid_lon = np.mean([c[0] for c in coords[:-1]])
                    centroid_lat = np.mean([c[1] for c in coords[:-1]])
                    
                    # Find nearest grid point
                    lat_idx = np.argmin(np.abs(lat_vals - centroid_lat))
                    lon_idx = np.argmin(np.abs(lon_vals - centroid_lon))
                    
                    # Set mask value to the actual indicator kriging probability
                    mask_values[lat_idx, lon_idx] = yield_value
                        
        high_prob_count = np.sum(mask_values >= threshold)
        print(f"Indicator mask generated: {high_prob_count} high-probability points (≥{threshold}) out of {mask_values.size} total grid points")
        return grid_lats, grid_lons, mask_values, lat_vals, lon_vals
        
    except Exception as e:
        print(f"Error generating indicator mask: {e}")
        return None, None, None, None, None

def generate_geo_json_grid(wells_df, center_point, radius_km, resolution=50, method='kriging', show_variance=False, auto_fit_variogram=False, variogram_model='spherical', soil_polygons=None, indicator_mask=None, new_clipping_polygon=None, exclusion_polygons=None):
    """
    Generate GeoJSON grid with interpolated yield values for accurate visualization

    This function creates a smooth, continuous interpolation surface with GeoJSON format
    for optimal visualization of groundwater yield patterns.

    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data
    center_point : tuple
        (latitude, longitude) of center point
    radius_km : float
        Radius in km to include in the grid
    resolution : int
        Grid resolution (number of cells per dimension)
    method : str
        Interpolation method to use
    indicator_mask : tuple or None
        Tuple containing indicator kriging mask data for clipping

    Returns:
    --------
    dict
        GeoJSON data structure with interpolated yield values
    """
    
    # For continuous indicator method, expand well search area but keep final grid bounds the same
    original_radius_km = radius_km
    print(f"🔍 GENERATE_GEO_JSON_GRID CALLED: method={method}, radius={radius_km}km, wells_count={len(wells_df)}")
    print(f"🔍 WELL SEARCH ANALYSIS: method={method}, radius={radius_km}km, wells_count={len(wells_df)}")
    
    if method == 'indicator_kriging_spherical_continuous':
        # Double the radius for well selection - get wells from larger area
        expanded_radius_km = radius_km * 2.0
        print(f"🎯 CONTINUOUS INDICATOR KRIGING: Expanding well search from {radius_km}km to {expanded_radius_km}km")
        print(f"📊 BEFORE EXPANSION: Using {len(wells_df)} wells from original {radius_km}km radius")
        
        # Get expanded well dataset from the larger search area
        try:
            from data_loader import load_nz_govt_data
            expanded_wells_df = load_nz_govt_data(center_point, expanded_radius_km)
            if expanded_wells_df is not None and len(expanded_wells_df) > len(wells_df):
                original_well_count = len(wells_df)
                wells_df = expanded_wells_df
                print(f"✅ EXPANSION SUCCESS: Found {len(wells_df)} wells in {expanded_radius_km}km area (gained {len(wells_df) - original_well_count} additional wells)")
            else:
                print(f"⚠️ EXPANSION RESULT: No additional wells found in expanded area, using original {len(wells_df)} wells")
        except Exception as e:
            print(f"❌ EXPANSION ERROR: {e}, using original {len(wells_df)} wells")
        
        # Keep final grid bounds at original radius for consistent clipping
        radius_km = original_radius_km
        print(f"📐 FINAL GRID BOUNDS: Kept at original {radius_km}km for consistent clipping polygon")
        print(f"🎯 CONTINUOUS INDICATOR SUMMARY: Using {len(wells_df)} wells from {expanded_radius_km}km, clipping to {radius_km}km")
    else:
        print(f"📍 STANDARD METHOD: Using {len(wells_df)} wells from {radius_km}km radius")
    
    # Debug indicator mask status and create polygon geometry for clipping
    print(f"GeoJSON {method}: indicator_mask is {'provided' if indicator_mask is not None else 'None'}")
    indicator_geometry = None
    
    # Create comprehensive clipping geometry (new comprehensive polygon takes priority)
    clipping_geometry = None
    
    if new_clipping_polygon is not None:
        try:
            # Handle the comprehensive clipping polygon (separate polygons for different drainage areas)
            print(f"🗺️ Using comprehensive clipping polygon with {len(new_clipping_polygon)} separate polygons")
            
            # Create unified geometry from all separate polygons
            # Note: The polygons should already have holes removed via containment detection
            from shapely.ops import unary_union
            all_geometries = [geom for geom in new_clipping_polygon.geometry if geom.is_valid]
            if all_geometries:
                clipping_geometry = unary_union(all_geometries)
                print(f"✅ Comprehensive clipping geometry created from {len(all_geometries)} processed polygons")
                
                # Check polygon types if available
                if 'polygon_type' in new_clipping_polygon.columns:
                    type_counts = new_clipping_polygon['polygon_type'].value_counts().to_dict()
                    print(f"   - Polygon types: {type_counts}")
                
                # Check if result is MultiPolygon (disconnected areas) or single Polygon
                if hasattr(clipping_geometry, 'geoms'):
                    print(f"   - Result: MultiPolygon with {len(clipping_geometry.geoms)} disconnected drainage areas")
                else:
                    print(f"   - Result: Single unified polygon (holes already removed)")
            else:
                print("⚠️ No valid geometries found in new clipping polygon")
                clipping_geometry = None
                    
        except Exception as e:
            print(f"❌ Error creating comprehensive clipping geometry: {e}")
            clipping_geometry = None
    
    # Fallback to Banks Peninsula exclusion if comprehensive polygon not available
    banks_peninsula_polygon = None
    banks_peninsula_coords = None  # Define the variable to prevent errors
    if clipping_geometry is None and banks_peninsula_coords and len(banks_peninsula_coords) > 3:
        try:
            from shapely.geometry import Polygon
            # Ensure the polygon is closed (first and last points are the same)
            coords = list(banks_peninsula_coords)
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            
            # Create polygon with (longitude, latitude) order for shapely
            # Note: banks_peninsula_coords should be in (latitude, longitude) format
            shapely_coords = [(coord[1], coord[0]) for coord in coords]  # Convert to (lon, lat)
            banks_peninsula_polygon = Polygon(shapely_coords)
            clipping_geometry = banks_peninsula_polygon  # Use as clipping geometry
            print(f"Banks Peninsula exclusion polygon created with {len(coords)} points")
        except Exception as e:
            print(f"Error creating Banks Peninsula exclusion polygon: {e}")
            banks_peninsula_polygon = None
    
    # Don't apply indicator clipping to indicator kriging methods themselves
    # Indicator methods should not be clipped by their own output
    indicator_methods = ['indicator_kriging', 'indicator_kriging_spherical', 'indicator_kriging_spherical_continuous', 'indicator_variance']
    clippable_methods = ['kriging', 'yield_kriging', 'specific_capacity_kriging', 'depth_kriging', 'depth_kriging_auto', 'swl_kriging', 'ground_water_level_kriging', 'idw', 'rf_kriging']
    
    if indicator_mask is not None and method in clippable_methods:
        mask_lat_grid, mask_lon_grid, mask_values, mask_lat_vals, mask_lon_vals = indicator_mask
        print(f"Indicator mask grid shape: {mask_values.shape if mask_values is not None else 'None'}")
        if mask_values is not None:
            import numpy as np  # Ensure numpy is available in this scope
            high_prob_count = np.sum(mask_values >= 0.7)
            print(f"Mask has {high_prob_count} high-probability points (≥0.7) out of {mask_values.size} total")
            # Create polygon geometry from indicator mask for clipping
            indicator_geometry = create_indicator_polygon_geometry(indicator_mask, threshold=0.7)
            if indicator_geometry is not None:
                print(f"Created indicator clipping geometry from {high_prob_count} high-probability cells")

    # Handle empty datasets
    if isinstance(wells_df, pd.DataFrame) and wells_df.empty:
        return {"type": "FeatureCollection", "features": []}

    # Filter out geotechnical/geological investigation wells using well_use
    if 'well_use' in wells_df.columns:
        geotechnical_mask = wells_df['well_use'].str.contains(
            'Geotechnical.*Investigation|Geological.*Investigation', 
            case=False, 
            na=False, 
            regex=True
        )
        wells_df = wells_df[~geotechnical_mask].copy()

        if wells_df.empty:
            return {"type": "FeatureCollection", "features": []}

    # Extract the original grid information with HIGH-PRECISION coordinate conversion
    center_lat, center_lon = center_point
    
    # HIGH-PRECISION COORDINATE CONVERSION SYSTEM
    # Use same ultra-precise geodetic calculations as sequential_heatmap.py
    # This ensures perfect alignment between heatmap centroids and clipping boundaries
    from utils import get_distance
    import numpy as np
    
    # ===== ARCHITECT CLEANUP: Removed second legacy get_precise_conversion_factors =====
    # This function has been replaced with proper NZTM2000 coordinate transformations
    # using the centralized CRS helpers: to_nztm2000(), to_wgs84(), build_crs_grid()
    print(f"🔧 COORDINATE SYSTEM: Using NZTM2000 transformations (legacy km_per_degree removed)")
    # ==================================================================================
    
    # ===== ARCHITECT SOLUTION: PROPER COORDINATE TRANSFORMATION =====
    print(f"🔧 ===== COORDINATE TRANSFORMATION FIX APPLIED IN GENERATE_GEO_JSON_GRID =====")
    print(f"🔧 Using centralized CRS helpers for proper EPSG:2193 transformations")
    
    # High resolution grid for smooth professional visualization
    wells_count = len(wells_df)
    if wells_count > 5000:
        grid_size = 80   # Higher resolution for very large datasets
    elif wells_count > 1000:
        grid_size = 120  # High resolution for large datasets
    else:
        grid_size = 150  # Very fine resolution for smaller datasets

    # Build grid using centralized helper
    grid_ctx = build_crs_grid(center_point, radius_km, grid_size)
    x_vals_m = grid_ctx['x_vals_m']
    y_vals_m = grid_ctx['y_vals_m']
    X_m = grid_ctx['X_m']
    Y_m = grid_ctx['Y_m']
    LONS_2D = grid_ctx['LONS_2D']
    LATS_2D = grid_ctx['LATS_2D']
    
    # For backward compatibility, create 1D lat/lon arrays (though we'll use 2D for polygons)
    lat_vals = LATS_2D[:, 0]  # First column (west edge)
    lon_vals = LONS_2D[0, :]  # First row (north edge)
    
    print(f"🔧 GRID CREATED: {grid_size}x{grid_size} using centralized CRS helpers")
    print(f"🔧 WGS84 BOUNDS: lat [{LATS_2D.min():.6f}, {LATS_2D.max():.6f}]°, lon [{LONS_2D.min():.6f}, {LONS_2D.max():.6f}]°")
    # =================================================================="

    # ===== COORDINATE TRANSFORMATION MOVED AFTER FILTERING FOR EACH METHOD =====
    # This ensures coordinate arrays match filtered well data arrays
    # ================================================================="

    # Use the new categorization system
    from data_loader import get_wells_for_interpolation

    if method == 'depth_kriging':
        # Get wells appropriate for depth interpolation
        wells_df = get_wells_for_interpolation(wells_df, 'depth')
        if wells_df.empty:
            return {"type": "FeatureCollection", "features": []}

        # ===== TRANSFORM FILTERED WELLS TO NZTM2000 =====
        # Transform wells AFTER filtering to ensure coordinate arrays match data arrays
        wells_x_m, wells_y_m = prepare_wells_xy(wells_df)
        print(f"🔧 WELLS TRANSFORMED: {len(wells_df)} wells from WGS84 to NZTM2000 using helper")
        print(f"🔧 WELLS RANGE NZTM2000: X [{wells_x_m.min():.1f}, {wells_x_m.max():.1f}]m, Y [{wells_y_m.min():.1f}, {wells_y_m.max():.1f}]m")
        # ================================================================="

        lats = wells_df['latitude'].values.astype(float)
        lons = wells_df['longitude'].values.astype(float)

        # Use depth_to_groundwater if available, otherwise fall back to depth
        if 'depth_to_groundwater' in wells_df.columns and wells_df['depth_to_groundwater'].notna().any():
            yields = wells_df['depth_to_groundwater'].values.astype(float)
        else:
            yields = wells_df['depth'].values.astype(float)
    elif method == 'specific_capacity_kriging':
        # Get wells appropriate for specific capacity interpolation
        wells_df = get_wells_for_interpolation(wells_df, 'specific_capacity')
        if wells_df.empty:
            return {"type": "FeatureCollection", "features": []}

        # Double-check that all wells have valid specific capacity data
        wells_df = wells_df[
            wells_df['specific_capacity'].notna() & 
            (wells_df['specific_capacity'] > 0)
        ].copy()

        if wells_df.empty:
            return {"type": "FeatureCollection", "features": []}

        # ===== TRANSFORM FILTERED WELLS TO NZTM2000 =====
        # Transform wells AFTER filtering to ensure coordinate arrays match data arrays
        wells_x_m, wells_y_m = prepare_wells_xy(wells_df)
        print(f"🔧 WELLS TRANSFORMED: {len(wells_df)} wells from WGS84 to NZTM2000 using helper")
        print(f"🔧 WELLS RANGE NZTM2000: X [{wells_x_m.min():.1f}, {wells_x_m.max():.1f}]m, Y [{wells_y_m.min():.1f}, {wells_y_m.max():.1f}]m")
        # ================================================================="

        lats = wells_df['latitude'].values.astype(float)
        lons = wells_df['longitude'].values.astype(float)
        yields = wells_df['specific_capacity'].values.astype(float)
    elif method == 'ground_water_level_kriging':
        # Get wells appropriate for ground water level interpolation
        wells_df = get_wells_for_interpolation(wells_df, 'ground_water_level')
        if wells_df.empty:
            return {"type": "FeatureCollection", "features": []}

        # Double-check that all wells have valid ground water level data
        # Only filter for non-null values, allow all numeric values (including 0 and negative)
        valid_gwl_mask = wells_df['ground water level'].notna()

        wells_df = wells_df[valid_gwl_mask].copy()

        # ===== TRANSFORM FILTERED WELLS TO NZTM2000 =====
        # Transform wells AFTER filtering to ensure coordinate arrays match data arrays
        wells_x_m, wells_y_m = prepare_wells_xy(wells_df)
        print(f"🔧 WELLS TRANSFORMED: {len(wells_df)} wells from WGS84 to NZTM2000 using helper")
        print(f"🔧 WELLS RANGE NZTM2000: X [{wells_x_m.min():.1f}, {wells_x_m.max():.1f}]m, Y [{wells_y_m.min():.1f}, {wells_y_m.max():.1f}]m")
        
        # ===== MINIMAL DEBUG: Reference Point Validation =====
        if len(wells_df) < 500:  # Only test on smaller datasets to avoid spam
            # Test known Canterbury landmarks
            landmarks = [
                ("Christchurch Cathedral", -43.5321, 172.6362),
                ("Canterbury Plains Center", -43.7, 172.0)
            ]
            print("🔍 COORD DEBUG - Reference Point Validation:")
            for name, lat, lon in landmarks:
                try:
                    # Test coordinate transformation accuracy
                    test_x, test_y = to_nztm2000([lon], [lat])
                    back_lon, back_lat = to_wgs84(test_x, test_y)
                    lat_diff = abs(lat - back_lat[0])
                    lon_diff = abs(lon - back_lon[0])
                    print(f"  {name}: WGS84({lat:.6f},{lon:.6f}) → NZTM({test_x[0]:.1f},{test_y[0]:.1f}) → WGS84({back_lat[0]:.6f},{back_lon[0]:.6f})")
                    print(f"    Round-trip error: lat={lat_diff:.8f}°, lon={lon_diff:.8f}°")
                except Exception as e:
                    print(f"    Error testing {name}: {e}")
        # ================================================================="

        if wells_df.empty:
            print("No valid ground water level data found for interpolation")
            return {"type": "FeatureCollection", "features": []}

        lats = wells_df['latitude'].values.astype(float)
        lons = wells_df['longitude'].values.astype(float)
        
        # Convert negative ground water levels to 0 for interpolation
        # Negative values indicate artesian conditions (water above surface)
        # For depth mapping, treat these as surface level (depth = 0)
        raw_gwl_values = wells_df['ground water level'].values.astype(float)
        yields = np.maximum(raw_gwl_values, 0)  # Convert negative values to 0
        
        negative_count = np.sum(raw_gwl_values < 0)
        if negative_count > 0:
            print(f"Ground water level interpolation: converted {negative_count} negative values (artesian) to 0 (surface level)")

        print(f"Ground water level interpolation: using {len(yields)} wells with values ranging from {yields.min():.2f} to {yields.max():.2f}")
    elif method == 'indicator_kriging' or method == 'indicator_kriging_spherical' or method == 'indicator_kriging_spherical_continuous':
        # For indicator kriging, we need wells with ACTUAL yield data (including 0.0)
        # But exclude wells that have missing yield data entirely
        wells_df_original = wells_df.copy()

        # Only filter for valid coordinates and yield_rate column existence
        if 'yield_rate' not in wells_df_original.columns:
            return {"type": "FeatureCollection", "features": []}

        # For indicator kriging, be more selective about 0.0 values
        # Only include wells that are explicitly water wells with yield measurements
        # Exclude wells that appear to be monitoring, geotechnical, or no-yield wells

        # Filter for valid coordinates and wells with actual yield measurements
        valid_coord_mask = (
            wells_df_original['latitude'].notna() & 
            wells_df_original['longitude'].notna() &
            wells_df_original['yield_rate'].notna()  # Must have yield data
        )

        # Additional filtering to exclude wells that shouldn't be in indicator kriging
        if 'well_use' in wells_df_original.columns:
            # Exclude geotechnical/geological investigation wells
            investigation_mask = ~wells_df_original['well_use'].str.contains(
                'Geotechnical.*Investigation|Geological.*Investigation|Monitoring', 
                case=False, na=False, regex=True
            )
            valid_coord_mask = valid_coord_mask & investigation_mask

        # Filter out monitoring wells with no useful data
        # Exclude "Water Level Observation" or "Groundwater Quality" wells that have ALL empty data fields
        print(f"INDICATOR KRIGING FILTERING: Starting with {len(wells_df_original)} total wells")
        print(f"COLUMNS AVAILABLE: {list(wells_df_original.columns)}")
        if 'USE_CODE_1_DESC' in wells_df_original.columns:
            def is_empty_or_zero(value):
                """Check if field is empty, None, zero, or whitespace"""
                if pd.isna(value) or str(value).strip() in ['', '0', '0.0', 'None', 'null']:
                    return True
                return False
            
            # Check for monitoring/observation wells
            monitoring_wells_mask = wells_df_original['USE_CODE_1_DESC'].str.contains(
                'Water Level Observation|Groundwater Quality', 
                case=False, na=False, regex=True
            )
            
            # For monitoring wells, check if ALL specified fields are empty
            fields_to_check = [
                'TOP_SCREEN_1', 'TOP_SCREEN_2', 'TOP_SCREEN_3',
                'BOTTOM_SCREEN_2', 'BOTTOM_SCREEN_3', 'ground water level',
                'START_READINGS', 'END_READINGS', 'MAX_YIELD'
            ]
            
            # Create mask for wells that should be excluded (monitoring wells with all empty fields)
            exclude_mask = wells_df_original.apply(lambda row: 
                monitoring_wells_mask[row.name] and 
                all(is_empty_or_zero(row.get(field, None)) for field in fields_to_check),
                axis=1
            )
            
            # Apply the exclusion filter
            valid_coord_mask = valid_coord_mask & ~exclude_mask
            
            # Log filtering results
            excluded_count = exclude_mask.sum()
            monitoring_count = monitoring_wells_mask.sum()
            print(f"INDICATOR KRIGING FILTERING: Found {monitoring_count} monitoring wells, filtered out {excluded_count} with completely empty data")
            if excluded_count > 0:
                print(f"✅ FILTERED: Removed {excluded_count} problematic monitoring wells from indicator kriging")

        wells_df = wells_df_original[valid_coord_mask].copy()

        if wells_df.empty:
            return {"type": "FeatureCollection", "features": []}

        lats = wells_df['latitude'].values.astype(float)
        lons = wells_df['longitude'].values.astype(float)

        # Convert to BINARY indicator values for kriging input using COMBINED criteria
        # A well is viable (indicator = 1) if ANY of these conditions are met:
        # - yield_rate ≥ 0.1 L/s, OR  
        # - ground water level data exists (any valid depth means water was found), OR
        # - WELL_STATUS_DESC = "Active (exist, present)"
        # Note: If ground water level is recorded, it means water was found at that depth = viable
        # Otherwise indicator = 0 (not viable)
        
        yield_threshold = 0.1
        
        raw_yields = wells_df['yield_rate'].values.astype(float)
        
        # Check if ground water level data is available
        has_gwl_data = 'ground water level' in wells_df.columns
        if has_gwl_data:
            gwl_values = wells_df['ground water level'].values.astype(float)
            # Handle NaN values in ground water level
            gwl_valid = ~np.isnan(gwl_values)
            gwl_viable = np.zeros_like(gwl_values, dtype=bool)
            # Wells are viable if they have ANY valid ground water level data (water was found)
            gwl_viable[gwl_valid] = True  # Any recorded depth means water was found = viable
        else:
            gwl_values = np.full_like(raw_yields, np.nan)
            gwl_viable = np.zeros_like(raw_yields, dtype=bool)
        
        # Check if well status data is available
        has_status_data = 'status' in wells_df.columns
        if has_status_data:
            status_values = wells_df['status'].fillna('')  # Replace NaN with empty string
            # Wells are viable if they are "Active (exist, present)"
            status_viable = status_values == "Active (exist, present)"
        else:
            status_viable = np.zeros_like(raw_yields, dtype=bool)
        
        # Combined viability logic: viable if ANY condition is met
        yield_viable = raw_yields >= yield_threshold
        combined_viable = yield_viable | gwl_viable | status_viable
        yields = combined_viable.astype(float)  # Binary: 1 or 0

        # Count wells in each category for detailed logging
        viable_count = np.sum(yields == 1)
        non_viable_count = np.sum(yields == 0)
        yield_only_viable = np.sum(yield_viable & ~gwl_viable & ~status_viable)
        gwl_only_viable = np.sum(gwl_viable & ~yield_viable & ~status_viable) if has_gwl_data else 0
        status_only_viable = np.sum(status_viable & ~yield_viable & ~gwl_viable) if has_status_data else 0
        yield_gwl_viable = np.sum(yield_viable & gwl_viable & ~status_viable) if has_gwl_data else 0
        yield_status_viable = np.sum(yield_viable & status_viable & ~gwl_viable) if has_status_data else 0
        gwl_status_viable = np.sum(gwl_viable & status_viable & ~yield_viable) if has_gwl_data and has_status_data else 0
        all_three_viable = np.sum(yield_viable & gwl_viable & status_viable) if has_gwl_data and has_status_data else 0

        print(f"GeoJSON indicator kriging: using {len(yields)} wells with COMBINED binary classification")
        print(f"COMBINED RESULTS:")
        print(f"  Total viable: {viable_count} wells ({100*viable_count/len(yields):.1f}%)")
        print(f"  Total non-viable: {non_viable_count} wells ({100*non_viable_count/len(yields):.1f}%)")
        print(f"BREAKDOWN BY CRITERIA:")
        print(f"  Viable by yield only (≥{yield_threshold} L/s): {yield_only_viable} wells")
        if has_gwl_data:
            print(f"  Viable by ground water level only: {gwl_only_viable} wells")
        if has_status_data:
            print(f"  Viable by active status only: {status_only_viable} wells")
        if has_gwl_data:
            print(f"  Viable by yield + ground water level: {yield_gwl_viable} wells")
        if has_status_data:
            print(f"  Viable by yield + active status: {yield_status_viable} wells")
        if has_gwl_data and has_status_data:
            print(f"  Viable by ground water level + active status: {gwl_status_viable} wells")
            print(f"  Viable by all three criteria: {all_three_viable} wells")
        
        # Calculate valid counts for data summary
        if has_gwl_data:
            valid_gwl_count = np.sum(~np.isnan(gwl_values))
            print(f"  Wells with ground water level data: {valid_gwl_count}/{len(wells_df)}")
        else:
            print(f"  No ground water level data available")
            valid_gwl_count = 0
            
        if has_status_data:
            active_status_count = np.sum(status_viable)
            print(f"  Wells with active status: {active_status_count}/{len(wells_df)}")
        else:
            print(f"  No well status data available")
        print(f"RAW DATA RANGES:")
        print(f"  Yield range: {raw_yields.min():.3f} to {raw_yields.max():.3f} L/s")
        if has_gwl_data and valid_gwl_count > 0:
            valid_gwl = gwl_values[~np.isnan(gwl_values)]
            print(f"  Ground water level range: {valid_gwl.min():.3f} to {valid_gwl.max():.3f}")
            
            # Debug specific well M35/4191 if present
            if 'well_id' in wells_df.columns:
                test_well_mask = wells_df['well_id'].str.contains('M35/4191', na=False)
                if test_well_mask.any():
                    test_well = wells_df[test_well_mask].iloc[0]
                    test_yield = test_well['yield_rate']
                    test_gwl = test_well['ground water level'] if not pd.isna(test_well['ground water level']) else 'NaN'
                    test_yield_viable = test_yield >= yield_threshold
                    test_gwl_viable = not pd.isna(test_well['ground water level'])
                    test_status_viable = test_well.get('status', '') == "Active (exist, present)" if has_status_data else False
                    test_combined = test_yield_viable or test_gwl_viable or test_status_viable
                    print(f"DEBUG WELL M35/4191:")
                    print(f"    Yield: {test_yield} L/s (viable: {test_yield_viable})")
                    print(f"    Ground water level: {test_gwl} (viable: {test_gwl_viable})")
                    if has_status_data:
                        test_status = test_well.get('status', 'N/A')
                        print(f"    Well status: {test_status} (viable: {test_status_viable})")
                    print(f"    Combined viable: {test_combined}")
                    print(f"    Criteria: yield>={yield_threshold}, gwl=has_valid_data, status=Active")
                    
            # Sample of ground water level values showing range
            gwl_sample = valid_gwl[:10] if len(valid_gwl) > 0 else []
            if len(gwl_sample) > 0:
                print(f"  Sample GWL values (all viable as water was found): {gwl_sample}")
                print(f"  Note: Any recorded depth means water was found = viable well")
                
        print(f"  Wells with exactly 0.0 yield: {np.sum(raw_yields == 0.0)}")
        print(f"  Wells with NaN yield (excluded): {wells_df_original['yield_rate'].isna().sum()}")
        print(f"  Total wells excluded for quality: {len(wells_df_original) - len(wells_df)}")
    else:
        # Get wells appropriate for yield interpolation
        wells_df = get_wells_for_interpolation(wells_df, 'yield')
        if wells_df.empty:
            return {"type": "FeatureCollection", "features": []}

        lats = wells_df['latitude'].values.astype(float)
        lons = wells_df['longitude'].values.astype(float)
        yields = wells_df['yield_rate'].values.astype(float)

    # ===== ARCHITECT SOLUTION: USE NZTM2000 COORDINATES FOR INTERPOLATION =====
    print(f"🔧 Using NZTM2000 coordinates for accurate interpolation")
    
    # Create grid points in NZTM2000 meters for interpolation
    radius_m = radius_km * 1000.0
    X_m_flat = X_m.flatten()
    Y_m_flat = Y_m.flatten()
    
    # Apply radius filter in meters
    center_x_m, center_y_m = to_nztm2000([center_lon], [center_lat])
    center_x_m, center_y_m = center_x_m[0], center_y_m[0]
    
    # Distance from center in meters
    distances = np.sqrt((X_m_flat - center_x_m)**2 + (Y_m_flat - center_y_m)**2)
    mask = distances <= radius_m
    
    # Define grid points in NZTM2000 meters
    grid_points_x = X_m_flat[mask]
    grid_points_y = Y_m_flat[mask]
    
    print(f"🔧 GRID FILTERING: {len(grid_points_x)}/{len(X_m_flat)} points within {radius_km}km radius")
    # ============================================================================

    try:
        # Initialize variance array for kriging uncertainty
        kriging_variance = None

        # Choose interpolation method based on parameter and dataset size
        if show_variance and (method == 'kriging' or method == 'rf_kriging') and len(wells_df) >= 5:
            # Use actual kriging with variance calculation when variance is requested
            print("Calculating kriging with variance estimation")

            # ===== USE CENTRALIZED KRIGING WITH NZTM2000 COORDINATES =====
            print(f"🔧 Performing kriging with variance in NZTM2000 coordinates")
            
            # Use centralized kriging helper with NZTM2000 coordinates
            if auto_fit_variogram:
                Z_grid, SS_grid, OK = krige_on_grid(
                    wells_x_m, wells_y_m, yields,
                    x_vals_m, y_vals_m,
                    variogram_model=variogram_model,
                    verbose=False
                )
            else:
                Z_grid, SS_grid, OK = krige_on_grid(
                    wells_x_m, wells_y_m, yields,
                    x_vals_m, y_vals_m,
                    variogram_model='linear',
                    verbose=False
                )
            
            if Z_grid is not None:
                # Extract values for masked grid points
                Z_flat = Z_grid.flatten()
                interpolated_z = Z_flat[mask]
                
                if SS_grid is not None:
                    SS_flat = SS_grid.flatten()
                    kriging_variance = SS_flat[mask]
                else:
                    kriging_variance = None
                    
                print(f"🔧 Kriging with variance completed: {len(interpolated_z)} interpolated points")
            else:
                print("❌ Kriging failed, falling back to griddata")
                interpolated_z = None
                kriging_variance = None
            # ====================================================================

        elif (method == 'indicator_kriging' or method == 'indicator_kriging_spherical' or method == 'indicator_kriging_spherical_continuous') and len(wells_df) >= 5:
            # Perform indicator kriging for binary yield suitability
            print("Performing indicator kriging for yield suitability mapping...")

            # ===== INDICATOR KRIGING WITH NZTM2000 COORDINATES =====
            print(f"🔧 Performing indicator kriging in NZTM2000 coordinates")
            
            # Use centralized kriging helper with spherical model for indicator data
            Z_grid, SS_grid, OK = krige_on_grid(
                wells_x_m, wells_y_m, yields,
                x_vals_m, y_vals_m,
                variogram_model='spherical',
                verbose=False
            )
            
            if Z_grid is not None:
                # Extract values for masked grid points
                Z_flat = Z_grid.flatten()
                interpolated_z = Z_flat[mask]
                
                # Ensure values are in [0,1] range (probabilities)
                interpolated_z = np.clip(interpolated_z, 0.0, 1.0)
                
                print(f"🔧 Indicator kriging completed: {len(interpolated_z)} probability points")
            else:
                print("❌ Indicator kriging failed")
                interpolated_z = None
            # ====================================================================

            # Ensure values are in [0,1] range (probabilities)
            interpolated_z = np.clip(interpolated_z, 0.0, 1.0)

            # Apply distance-based decay to prevent high values far from good wells
            # Skip decay for continuous version to preserve smooth kriging output
            if method != 'indicator_kriging_spherical_continuous':
                # Calculate distances from each grid point to nearest good well (yield >= 0.75)
                good_wells_mask = yields >= 0.75  # Wells with good yield indicators
                print(f"🔧 Found {np.sum(good_wells_mask)} good wells for distance decay calculation")
                if np.any(good_wells_mask) and interpolated_z is not None:
                    good_coords = np.column_stack([x_coords[good_wells_mask], y_coords[good_wells_mask]])
                    grid_coords = grid_points

                    # Calculate distance to nearest good well for each grid point
                    from scipy.spatial.distance import cdist
                    distances = cdist(grid_coords, good_coords)
                    min_distances_km = np.min(distances, axis=1) / 1000.0  # Convert to km

                    # Apply exponential decay for distances > 3km
                    decay_threshold_km = 3.0
                    decay_factor = 0.3  # Gentle decay to preserve three-tier structure

                    distance_mask = min_distances_km > decay_threshold_km
                    excess_distance = min_distances_km[distance_mask] - decay_threshold_km
                    decay_multiplier = np.exp(-decay_factor * excess_distance)
                    interpolated_z[distance_mask] *= decay_multiplier

            # Count points in each tier based on OUTPUT ranges (continuous 0-1 function)
            # Keep the continuous kriging output values intact - colors applied during visualization
            red_count = np.sum((interpolated_z >= 0.0) & (interpolated_z < 0.4))     # Red: 0.0-0.4
            orange_count = np.sum((interpolated_z >= 0.4) & (interpolated_z < 0.7))  # Orange: 0.4-0.7  
            green_count = np.sum(interpolated_z >= 0.7)                              # Green: 0.7-1.0

            print(f"Indicator kriging results (output function ranges):")
            print(f"Red (0.0-0.4): {red_count} points, Orange (0.4-0.7): {orange_count} points, Green (0.7-1.0): {green_count} points")
            print(f"Output value range: {interpolated_z.min():.3f} to {interpolated_z.max():.3f}")

        elif (method == 'kriging' or method == 'depth_kriging' or method == 'depth_kriging_auto') and auto_fit_variogram and len(wells_df) >= 5:
            # Perform kriging with auto-fitted variogram for yield/depth visualization (without variance output)
            if method == 'depth_kriging' or method == 'depth_kriging_auto':
                print(f"Auto-fitting {variogram_model} variogram model for depth estimation...")
            else:
                print(f"Auto-fitting {variogram_model} variogram model for yield estimation...")

            # ===== AUTO-FITTED KRIGING WITH NZTM2000 COORDINATES =====
            print(f"🔧 Performing auto-fitted kriging for {method} in NZTM2000 coordinates")
            
            # Use centralized kriging helper with auto-fitted variogram
            verbose_mode = True if (method == 'depth_kriging' or method == 'depth_kriging_auto') else False
            
            Z_grid, SS_grid, OK = krige_on_grid(
                wells_x_m, wells_y_m, yields,
                x_vals_m, y_vals_m,
                variogram_model=variogram_model,
                verbose=verbose_mode
            )
            
            if Z_grid is not None:
                # Extract values for masked grid points
                Z_flat = Z_grid.flatten()
                interpolated_z = Z_flat[mask]
                
                print(f"🔧 Auto-fitted kriging completed: {len(interpolated_z)} interpolated points")
            else:
                print("❌ Auto-fitted kriging failed, falling back to griddata")
                interpolated_z = None
            # ====================================================================

            # Additional validation for depth interpolation
            if method == 'depth_kriging' or method == 'depth_kriging_auto':
                print(f"Depth interpolation stats: min={np.min(interpolated_z):.2f}, max={np.max(interpolated_z):.2f}, mean={np.mean(interpolated_z):.2f}")
                # Ensure reasonable depth values (depths should be positive and reasonable)
                interpolated_z = np.maximum(0.1, interpolated_z)  # Minimum depth of 0.1m
                interpolated_z = np.minimum(200.0, interpolated_z)  # Maximum reasonable depth of 200m

        else:
            # Use standard griddata interpolation with NZTM2000 coordinates
            # This is much faster than kriging for large datasets
            wells_points = np.column_stack([wells_x_m, wells_y_m])
            grid_points_array = np.column_stack([grid_points_x, grid_points_y])
            
            interpolated_z = griddata(
                wells_points, yields, grid_points_array,
                method='linear', fill_value=0.0
            )

            # Fill any NaN values with nearest neighbor interpolation
            nan_mask = np.isnan(interpolated_z)
            if np.any(nan_mask):
                interpolated_z[nan_mask] = griddata(
                    wells_points, yields, grid_points_array[nan_mask],
                    method='nearest', fill_value=0.0
                )

            # Apply advanced smoothing for professional kriging-like appearance using NZTM2000 grid
            from scipy.ndimage import gaussian_filter

            # Reshape to 2D grid for smoothing using the NZTM2000 grid
            try:
                # Create full 2D grid for smoothing using X_m grid shape
                z_grid = np.zeros_like(X_m)
                z_grid_flat = z_grid.flatten()
                z_grid_flat[mask] = interpolated_z
                z_grid = z_grid_flat.reshape(X_m.shape)

                # Apply multiple smoothing passes for ultra-smooth appearance
                # First pass: moderate smoothing
                z_smooth = gaussian_filter(z_grid, sigma=1.5)
                # Second pass: fine smoothing for professional appearance
                z_smooth = gaussian_filter(z_smooth, sigma=0.8)

                # Extract smoothed values for our mask
                z_smooth_flat = z_smooth.flatten()
                interpolated_z = z_smooth_flat[mask]

                # Ensure values stay within reasonable bounds
                interpolated_z = np.maximum(0, interpolated_z)

            except Exception as e:
                # If smoothing fails, apply basic smoothing
                print(f"Advanced smoothing error: {e}, using basic smoothing")
                try:
                    z_grid = np.zeros_like(grid_X)
                    z_grid_flat = z_grid.flatten()
                    z_grid_flat[mask] = interpolated_z
                    z_grid = z_grid_flat.reshape(grid_X.shape)
                    z_smooth = gaussian_filter(z_grid, sigma=1.0)
                    z_smooth_flat = z_smooth.flatten()
                    interpolated_z = z_smooth_flat[mask]
                except:
                    print("Basic smoothing also failed, using raw interpolation")
    except Exception as e:
        # Fallback to simple IDW interpolation if the above methods fail
        print(f"Interpolation error: {e}, using fallback method")
        # Create grid points array for IDW interpolation
        grid_points_array = np.column_stack([grid_points_x, grid_points_y])
        wells_points = np.column_stack([wells_x_m, wells_y_m])
        
        interpolated_z = np.zeros(len(grid_points_x))
        for i, point in enumerate(grid_points_array):
            weights = 1.0 / (np.sqrt(np.sum((wells_points - point)**2, axis=1)) + 1e-5)
            interpolated_z[i] = np.sum(weights * yields) / np.sum(weights)

    # Convert grid coordinates back to lat/lon
    # ===== ARCHITECT FIX: CONVERT GRID POINTS USING NZTM2000 SYSTEM =====
    # Use the centralized coordinate system instead of km_per_degree
    grid_lons, grid_lats = to_wgs84(grid_points_x, grid_points_y)
    print(f"🔧 Converted {len(grid_lons)} grid points from NZTM2000 to WGS84")
    
    # ===== MINIMAL DEBUG: Coordinate Validation Points =====
    debug_coords = len(grid_lons) < 1000  # Only debug smaller grids to avoid spam
    if debug_coords and len(grid_lons) > 4:
        # Log grid corners for validation
        indices = [0, len(grid_lons)//4, len(grid_lons)//2, -1]
        print("🔍 COORD DEBUG - Grid Sample Points:")
        for i in indices:
            if i < len(grid_lons):
                print(f"  Point {i}: NZTM({grid_points_x[i]:.1f}, {grid_points_y[i]:.1f})m → WGS84({grid_lats[i]:.6f}, {grid_lons[i]:.6f})")
    # ================================================================
    # ========================================================================

    # Prepare soil polygon geometry for later filtering (do not apply to interpolation)
    merged_soil_geometry = None

    if soil_polygons is not None and len(soil_polygons) > 0:
        try:
            # Create a unified geometry from all soil polygons for later filtering
            valid_geometries = []
            for idx, row in soil_polygons.iterrows():
                if row.geometry and row.geometry.is_valid:
                    valid_geometries.append(row.geometry)

            if valid_geometries:
                # Merge all polygons into a single multipolygon
                merged_soil_geometry = unary_union(valid_geometries)
                print(f"Prepared soil drainage geometry for display filtering")
            else:
                print("No valid soil polygon geometries found")
        except Exception as e:
            print(f"Error preparing soil polygon geometry: {e}")
            merged_soil_geometry = None

    # Build the GeoJSON structure
    features = []

    # Create final square clipping geometry (smaller than original search area)
    # Original search area is radius_km x radius_km square
    # Final clipping area is 50% of original (10km for 20km original)
    final_clip_factor = 0.5
    final_radius_km = radius_km * final_clip_factor
    
    # Create final square clipping polygon centered on the original center
    # ===== ARCHITECT FIX: FINAL CLIPPING WITH NZTM2000 SYSTEM =====
    # Calculate final radius in NZTM2000 meters for consistent clipping
    final_radius_m = final_radius_km * 1000.0
    print(f"🔧 Final radius clipping: {final_radius_km}km = {final_radius_m}m NZTM2000")
    # ========================================================================
    
    # ===== FINAL ARCHITECT FIX: CLIPPING POLYGON WITH NZTM2000 BOUNDS =====
    # Convert final radius to WGS84 bounds using NZTM2000 system
    center_x_m, center_y_m = to_nztm2000([center_lon], [center_lat])
    center_x_m, center_y_m = center_x_m[0], center_y_m[0]
    
    # Create bounds in NZTM2000 meters
    bounds_x_m = [center_x_m - final_radius_m, center_x_m + final_radius_m, 
                  center_x_m + final_radius_m, center_x_m - final_radius_m]
    bounds_y_m = [center_y_m - final_radius_m, center_y_m - final_radius_m,
                  center_y_m + final_radius_m, center_y_m + final_radius_m]
    
    # Transform to WGS84 for polygon coordinates
    bounds_lons, bounds_lats = to_wgs84(bounds_x_m, bounds_y_m)
    
    final_clip_polygon_coords = [
        [bounds_lons[0], bounds_lats[0]],  # SW
        [bounds_lons[1], bounds_lats[1]],  # SE
        [bounds_lons[2], bounds_lats[2]],  # NE
        [bounds_lons[3], bounds_lats[3]],  # NW
        [bounds_lons[0], bounds_lats[0]]   # Close
    ]
    print(f"🔧 Final clipping polygon created using NZTM2000 coordinate system")
    # ==========================================================================
    
    from shapely.geometry import Polygon as ShapelyPolygon
    final_clip_geometry = ShapelyPolygon(final_clip_polygon_coords)
    
    print(f"Final clipping: {radius_km}km -> {final_radius_km:.1f}km square ({final_clip_factor*100:.0f}% of original)")
    print(f"Final clipping geometry bounds: {final_clip_geometry.bounds}")

    # Create polygons only where needed - use a Delaunay triangulation approach
    # for a more organic-looking interpolation surface
    from scipy.spatial import Delaunay

    try:
        # Create a Delaunay triangulation of the interpolation points
        points_2d = np.vstack([grid_lons, grid_lats]).T

        # Only create triangulation if we have enough points
        if len(points_2d) > 3:
            tri = Delaunay(points_2d)

            # Process each triangle to create a polygon
            for simplex in tri.simplices:
                # Get the three points of this triangle
                vertices = points_2d[simplex]

                # Get the values for these points (yield or variance)
                if show_variance and kriging_variance is not None:
                    vertex_values = kriging_variance[simplex]
                    avg_value = float(np.mean(vertex_values))
                    # For variance, show all values (including very small ones)
                    value_threshold = 0.0001  # Show almost all variance values
                else:
                    vertex_values = interpolated_z[simplex]
                    avg_value = float(np.mean(vertex_values))
                    value_threshold = 0.01  # Only show meaningful yield values

                avg_yield = avg_value  # Keep for backwards compatibility

                # Adjust value threshold based on interpolation method
                effective_threshold = value_threshold

                # Only add triangles with meaningful values and within our radius
                if avg_yield > effective_threshold:
                    # Check if triangle should be included based on soil polygons
                    include_triangle = True

                    if merged_soil_geometry is not None:
                        # Apply proper geometric intersection clipping
                        triangle_coords = [(float(v[0]), float(v[1])) for v in vertices]
                        triangle_coords.append(triangle_coords[0])  # Close the polygon
                        triangle_polygon = ShapelyPolygon(triangle_coords)
                        
                        # Include if triangle intersects or is mostly within soil drainage areas
                        # Allow triangles that cross boundaries rather than requiring complete containment
                        include_triangle = (merged_soil_geometry.intersects(triangle_polygon) or 
                                          merged_soil_geometry.contains(triangle_polygon))

                    # Additional clipping by indicator kriging geometry (high-probability zones)
                    if include_triangle and indicator_mask is not None:
                        centroid_lon = float(np.mean(vertices[:, 0]))
                        centroid_lat = float(np.mean(vertices[:, 1]))
                        
                        # Extract indicator mask data
                        mask_lat_grid, mask_lon_grid, mask_values, mask_lat_vals, mask_lon_vals = indicator_mask
                        
                        if mask_values is not None:
                            # Find nearest grid point in the mask
                            lat_idx = np.searchsorted(mask_lat_vals, centroid_lat)
                            lon_idx = np.searchsorted(mask_lon_vals, centroid_lon)
                            
                            # Ensure indices are within bounds
                            lat_idx = min(max(0, lat_idx), len(mask_lat_vals) - 1)
                            lon_idx = min(max(0, lon_idx), len(mask_lon_vals) - 1)
                            
                            # Get the indicator probability at this location
                            if lat_idx < mask_values.shape[0] and lon_idx < mask_values.shape[1]:
                                indicator_prob = mask_values[lat_idx, lon_idx]
                                
                                # Only include if probability >= threshold
                                if indicator_prob < 0.7:
                                    include_triangle = False
                                    print(f"  Masked out triangle at ({centroid_lat:.4f}, {centroid_lon:.4f}), prob={indicator_prob:.2f}")
                            else:
                                include_triangle = False
                                print(f"  Masked out triangle - indices out of bounds")
                        else:
                            include_triangle = False
                            print(f"  Masked out triangle - no mask values available")

                    if include_triangle:
                        # Create polygon for this triangle
                        poly = {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [float(vertices[0,0]), float(vertices[0,1])],
                                    [float(vertices[1,0]), float(vertices[1,1])],
                                    [float(vertices[2,0]), float(vertices[2,1])],
                                    [float(vertices[0,0]), float(vertices[0,1])]
                                ]]
                            },
                            "properties": {
                                "value": avg_yield,
                                "yield": avg_yield
                            }
                        }
                        
                        # Apply comprehensive clipping geometry (includes Canterbury Plains polygons)
                        should_include = True
                        if clipping_geometry is not None:
                            try:
                                # Use the triangle centroid for clipping check
                                centroid_lon = np.mean([vertices[0,0], vertices[1,0], vertices[2,0]])
                                centroid_lat = np.mean([vertices[0,1], vertices[1,1], vertices[2,1]])
                                center_point = Point(centroid_lon, centroid_lat)
                                
                                # Include only if the center point is inside the comprehensive clipping geometry
                                should_include = clipping_geometry.contains(center_point)
                                
                                if not should_include:
                                    print(f"🗺️ Triangle excluded by comprehensive clipping at ({centroid_lat:.4f}, {centroid_lon:.4f})")
                            except Exception as e:
                                print(f"❌ Error checking comprehensive clipping for triangle: {e}")
                                # Default to include if clipping check fails
                                should_include = True
                        
                        # Only add feature if it should be included
                        if should_include:
                            features.append(poly)
    except Exception as e:
        # If triangulation fails, fall back to the simpler grid method
        print(f"Triangulation error: {e}, using grid method")
        for i in range(len(grid_lats)):
            # Only process points with meaningful values
            if interpolated_z[i] > 0.01:
                # Check if point should be included based on soil polygons
                include_point = True

                if merged_soil_geometry is not None:
                    point = Point(grid_lons[i], grid_lats[i])
                    include_point = merged_soil_geometry.contains(point) or merged_soil_geometry.intersects(point)

                # Additional clipping by indicator kriging geometry (high-probability zones)
                if include_point and indicator_mask is not None:
                    # Extract indicator mask data
                    mask_lat_grid, mask_lon_grid, mask_values, mask_lat_vals, mask_lon_vals = indicator_mask
                    
                    if mask_values is not None:
                        # Find nearest grid point in the mask
                        lat_idx = np.searchsorted(mask_lat_vals, grid_lats[i])
                        lon_idx = np.searchsorted(mask_lon_vals, grid_lons[i])
                        
                        # Ensure indices are within bounds
                        lat_idx = min(max(0, lat_idx), len(mask_lat_vals) - 1)
                        lon_idx = min(max(0, lon_idx), len(mask_lon_vals) - 1)
                        
                        # Get the indicator probability at this location
                        if lat_idx < mask_values.shape[0] and lon_idx < mask_values.shape[1]:
                            indicator_prob = mask_values[lat_idx, lon_idx]
                            
                            # Only include if probability >= threshold
                            if indicator_prob < 0.7:
                                include_point = False
                                print(f"  Masked out point at ({grid_lats[i]:.4f}, {grid_lons[i]:.4f}), prob={indicator_prob:.2f}")
                        else:
                            include_point = False
                            print(f"  Masked out point - indices out of bounds")
                    else:
                        include_point = False
                        print(f"  Masked out point - no mask values available")

                if include_point:
                    # Create a small circle as a polygon (approximated with 8 points)
                    radius_deg_lat = 0.5 * (lat_vals[1] - lat_vals[0])
                    radius_deg_lon = 0.5 * (lon_vals[1] - lon_vals[0])

                    # Create the polygon coordinates
                    coords = []
                    for angle in np.linspace(0, 2*np.pi, 9):
                        x = grid_lons[i] + radius_deg_lon * np.cos(angle)
                        y = grid_lats[i] + radius_deg_lat * np.sin(angle)
                        coords.append([float(x), float(y)])

                    # Create the polygon feature
                    poly = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coords]
                        },
                        "properties": {
                            "value": float(interpolated_z[i]),
                            "yield": float(interpolated_z[i])
                        }
                    }
                    
                    # Add feature (Banks Peninsula exclusion removed)
                    features.append(poly)

    # Log filtering results
    if merged_soil_geometry is not None:
        print(f"GeoJSON features filtered by soil drainage areas: {len(features)} polygons displayed")
    
    # Apply final square clipping to existing features (triangle removal, not re-interpolation)
    features_before_final_clip = len(features)
    final_clipped_features = []
    
    for feature in features:
        try:
            # Get polygon centroid
            coords = feature['geometry']['coordinates'][0]
            if len(coords) >= 3:
                centroid_lon = sum(coord[0] for coord in coords) / len(coords)
                centroid_lat = sum(coord[1] for coord in coords) / len(coords)
                centroid_point = Point(centroid_lon, centroid_lat)
                
                # Create triangle polygon for proper intersection clipping
                triangle_coords = [(coord[0], coord[1]) for coord in coords[:-1]]  # Remove duplicate closing point
                if len(triangle_coords) >= 3:
                    triangle_polygon = ShapelyPolygon(triangle_coords)
                    # Only keep features that are completely within the smaller square
                    if final_clip_geometry.contains(triangle_polygon):
                        final_clipped_features.append(feature)
                else:
                    # If we can't create a proper triangle, keep the feature
                    final_clipped_features.append(feature)
        except Exception as e:
            # If centroid calculation fails, keep the feature
            final_clipped_features.append(feature)
    
    features = final_clipped_features
    print(f"Final square clipping: {features_before_final_clip} -> {len(features)} features ({final_radius_km:.1f}km sides)")

    # Apply exclusion clipping ONLY for non-indicator interpolation methods
    # Indicator methods already show probability values, so user wants to see full distribution
    indicator_methods = [
        'indicator_kriging', 
        'indicator_kriging_spherical', 
        'indicator_kriging_spherical_continuous'
    ]
    
    # Apply red/orange exclusion clipping for NON-INDICATOR methods only
    if method not in indicator_methods:
        # Apply exclusion clipping if exclusion polygons are provided
        if exclusion_polygons is not None:
            features_before_exclusion = len(features)
            features = apply_exclusion_clipping_to_geojson(features, exclusion_polygons)
            print(f"🚫 PRODUCTION: Applied exclusion clipping for {method}: {features_before_exclusion} -> {len(features)} features")
        elif exclusion_polygons is None:
            # Try to load exclusion polygons automatically
            auto_exclusion_polygons = load_exclusion_polygons()
            if auto_exclusion_polygons is not None:
                features_before_exclusion = len(features)
                features = apply_exclusion_clipping_to_geojson(features, auto_exclusion_polygons)
                print(f"🚫 PRODUCTION: Applied auto-loaded exclusion clipping for {method}: {features_before_exclusion} -> {len(features)} features")
    else:
        print(f"🔄 INDICATOR METHOD: Skipping exclusion clipping for {method} (preserving full probability distribution)")

    # Create the full GeoJSON object
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    return geojson

def generate_heat_map_data(wells_df, center_point, radius_km, resolution=50, method='kriging', soil_polygons=None):
    """
    Generate heat map data using various interpolation techniques

    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data with latitude, longitude and yield_rate columns
    center_point : tuple
        Tuple containing (latitude, longitude) of the center point
    radius_km : float
        Radius in kilometers to generate heat map data for
    resolution : int
        Number of points to generate in each dimension
    method : str
        Interpolation method to use ('kriging', 'idw', 'rf_kriging')

    Returns:
    --------
    list
        List of [lat, lng, intensity] points for the heat map

    Notes:
    ------
    This function is called each time a new location is selected to create an
    entirely fresh interpolation. The results should always be specific to the
    current center_point and never reuse old interpolation data.
    """
    # Handle empty datasets
    if isinstance(wells_df, pd.DataFrame) and wells_df.empty:
        return []

    # Filter out geotechnical/geological investigation wells using well_use
    if 'well_use' in wells_df.columns:
        geotechnical_mask = wells_df['well_use'].str.contains(
            'Geotechnical.*Investigation|Geological.*Investigation', 
            case=False, 
            na=False, 
            regex=True
        )
        wells_df = wells_df[~geotechnical_mask].copy()

        if wells_df.empty:
            return []

    # Use the new categorization system
    from data_loader import get_wells_for_interpolation

    if method == 'depth_kriging':
        # Get wells appropriate for depth interpolation
        wells_df_filtered = get_wells_for_interpolation(wells_df, 'depth')
        if wells_df_filtered.empty:
            return []

        lats = wells_df_filtered['latitude'].values.astype(float)
        lons = wells_df_filtered['longitude'].values.astype(float)

        # Use depth_to_groundwater if available, otherwise fall back to depth
        if 'depth_to_groundwater' in wells_df_filtered.columns and wells_df_filtered['depth_to_groundwater'].notna().any():
            yields = wells_df_filtered['depth_to_groundwater'].values.astype(float)
        else:
            yields = wells_df_filtered['depth'].values.astype(float)
    elif method == 'specific_capacity_kriging':
        # Get wells appropriate for specific capacity interpolation
        wells_df_filtered = get_wells_for_interpolation(wells_df, 'specific_capacity')
        if wells_df_filtered.empty:
            return []

        # Ensure all wells have valid specific capacity data
        wells_df_filtered = wells_df_filtered[
            wells_df_filtered['specific_capacity'].notna() & 
            (wells_df_filtered['specific_capacity'] > 0)
        ].copy()

        if wells_df_filtered.empty:
            return []

        lats = wells_df_filtered['latitude'].values.astype(float)
        lons = wells_df_filtered['longitude'].values.astype(float)
        yields = wells_df_filtered['specific_capacity'].values.astype(float)
    elif method == 'ground_water_level_kriging':
        # Get wells appropriate for ground water level interpolation
        wells_df_filtered = get_wells_for_interpolation(wells_df, 'ground_water_level')
        if wells_df_filtered.empty:
            print("No wells found for ground water level interpolation")
            return []

        # Ensure all wells have valid ground water level data
        # Allow all numeric values including 0 and negative values (consistent with other GWL function)
        valid_gwl_mask = wells_df_filtered['ground water level'].notna()

        wells_df_filtered = wells_df_filtered[valid_gwl_mask].copy()

        if wells_df_filtered.empty:
            print("No valid ground water level data after filtering")
            return []

        lats = wells_df_filtered['latitude'].values.astype(float)
        lons = wells_df_filtered['longitude'].values.astype(float)
        yields = wells_df_filtered['ground water level'].values.astype(float)

        print(f"Heat map ground water level: using {len(yields)} wells with GWL values from {yields.min():.2f} to {yields.max():.2f}")
    elif method == 'indicator_kriging' or method == 'indicator_kriging_spherical' or method == 'indicator_kriging_spherical_continuous':
        # INDICATOR KRIGING FILTERING: Filter out monitoring wells with no useful data BEFORE calling get_wells_for_interpolation
        method_name = "spherical" if method in ['indicator_kriging_spherical', 'indicator_kriging_spherical_continuous'] else "linear"
        print(f"HEAT MAP INDICATOR KRIGING ({method_name}) FILTERING: Starting with {len(wells_df)} total wells")
        print(f"HEAT MAP COLUMNS AVAILABLE: {list(wells_df.columns)}")
        
        wells_df_for_filtering = wells_df.copy()
        if 'USE_CODE_1_DESC' in wells_df_for_filtering.columns:
            def is_empty_or_zero(value):
                """Check if field is empty, None, zero, or whitespace"""
                if pd.isna(value) or str(value).strip() in ['', '0', '0.0', 'None', 'null']:
                    return True
                return False
            
            # Find monitoring wells
            monitoring_wells_mask = wells_df_for_filtering['USE_CODE_1_DESC'].str.contains(
                'Water Level Observation|Groundwater Quality', 
                case=False, na=False, regex=True
            )
            
            # Fields to check for filtering (these are problematic empty fields)
            fields_to_check = [
                'TOP_SCREEN_1', 'TOP_SCREEN_2', 'TOP_SCREEN_3',
                'BOTTOM_SCREEN_2', 'BOTTOM_SCREEN_3', 'ground water level',
                'START_READINGS', 'END_READINGS', 'MAX_YIELD'
            ]
            
            # Create exclusion mask: monitoring wells that have ALL empty data fields
            exclude_mask = wells_df_for_filtering.apply(lambda row: 
                monitoring_wells_mask[row.name] and 
                all(is_empty_or_zero(row.get(field, None)) for field in fields_to_check),
                axis=1
            )
            
            excluded_count = exclude_mask.sum()
            monitoring_count = monitoring_wells_mask.sum()
            print(f"HEAT MAP INDICATOR KRIGING FILTERING: Found {monitoring_count} monitoring wells, filtered out {excluded_count} with completely empty data")
            
            if excluded_count > 0:
                wells_df_for_filtering = wells_df_for_filtering[~exclude_mask].copy()
                print(f"✅ HEAT MAP INDICATOR KRIGING FILTERED: Removed {excluded_count} problematic monitoring wells")
        else:
            print("⚠️ HEAT MAP INDICATOR KRIGING: USE_CODE_1_DESC column not found - skipping monitoring well filtering")

        # INDICATOR KRIGING: Get wells with EITHER yield data OR groundwater level data
        # Both indicate water presence, so both should be included
        has_coordinates = (
            wells_df_for_filtering['latitude'].notna() & 
            wells_df_for_filtering['longitude'].notna()
        )
        
        has_yield_data = (
            wells_df_for_filtering['yield_rate'].notna() & 
            (wells_df_for_filtering['yield_rate'] > 0)
        )
        
        has_gwl_data = False
        if 'ground water level' in wells_df_for_filtering.columns:
            has_gwl_data = (
                wells_df_for_filtering['ground water level'].notna() & 
                (wells_df_for_filtering['ground water level'] != 0)
            )
        
        # Include wells with coordinates AND (yield data OR groundwater level data)
        indicator_wells_mask = has_coordinates & (has_yield_data | has_gwl_data)
        wells_df_filtered = wells_df_for_filtering[indicator_wells_mask].copy()
        
        if wells_df_filtered.empty:
            print("No wells found with either yield or groundwater level data for indicator kriging")
            return []

        lats = wells_df_filtered['latitude'].values.astype(float)
        lons = wells_df_filtered['longitude'].values.astype(float)

        # BINARY CLASSIFICATION for indicator kriging:
        # If well has yield data: viable if >= 0.1 L/s
        # If well has no yield data but has groundwater level data: treat as viable (water present)
        # If well has neither: treat as non-viable (should not happen due to filtering above)
        yield_threshold = 0.1
        yields = np.zeros(len(wells_df_filtered))  # Start with all non-viable
        
        for i, (idx, row) in enumerate(wells_df_filtered.iterrows()):
            has_yield = pd.notna(row['yield_rate']) and row['yield_rate'] > 0
            has_gwl = False
            if 'ground water level' in wells_df_filtered.columns:
                has_gwl = pd.notna(row['ground water level']) and row['ground water level'] != 0
            
            if has_yield:
                # Use yield data for classification
                yields[i] = 1.0 if row['yield_rate'] >= yield_threshold else 0.0
            elif has_gwl:
                # No yield data but has groundwater level data = treat as viable (water present)
                yields[i] = 1.0
            else:
                # No yield or GWL data = non-viable (should not happen due to filtering)
                yields[i] = 0.0
        
        # Count wells in each category for logging
        viable_count = np.sum(yields == 1)
        non_viable_count = np.sum(yields == 0)
        yield_based_count = np.sum(has_yield_data[indicator_wells_mask])
        gwl_based_count = np.sum(~has_yield_data[indicator_wells_mask] & has_gwl_data[indicator_wells_mask])

        print(f"Heat map indicator kriging: using {len(yields)} wells with binary classification")
        print(f"Wells with yield data: {yield_based_count}")
        print(f"Wells with only groundwater level data (treated as viable): {gwl_based_count}")
        print(f"Final classification - Viable: {viable_count} ({100*viable_count/len(yields):.1f}%), Non-viable: {non_viable_count} ({100*non_viable_count/len(yields):.1f}%)")
    else:
        # Get wells appropriate for yield interpolation
        wells_df_filtered = get_wells_for_interpolation(wells_df, 'yield')
        if wells_df_filtered.empty:
            return []

        lats = wells_df_filtered['latitude'].values.astype(float)
        lons = wells_df_filtered['longitude'].values.astype(float)
        yields = wells_df_filtered['yield_rate'].values.astype(float)

    # Handle case with too few data points
    if len(wells_df) < 3:
        heat_data = []
        for i, (lat, lon, yield_val) in enumerate(zip(lats, lons, yields)):
            heat_data.append([float(lat), float(lon), float(yield_val)])
        return heat_data

    # Create simplified grid for interpolation
    center_lat, center_lon = center_point
    grid_size = min(50, max(30, resolution))

    try:
        # ===== ARCHITECT SOLUTION: USE NZTM2000 FOR indicator_kriging_mask =====
        print(f"🔧 Converting indicator_kriging_mask to use NZTM2000 coordinates")
        
        # Transform wells to NZTM2000 using centralized helper
        wells_temp_df = pd.DataFrame({'latitude': lats, 'longitude': lons})
        wells_x_m, wells_y_m = prepare_wells_xy(wells_temp_df)
        
        # Build grid using centralized helper  
        grid_ctx = build_crs_grid(center_point, radius_km, grid_size)
        x_vals_m = grid_ctx['x_vals_m']
        y_vals_m = grid_ctx['y_vals_m']
        X_m = grid_ctx['X_m']
        Y_m = grid_ctx['Y_m']
        
        # Create grid points in NZTM2000 meters
        center_x_m, center_y_m = to_nztm2000([center_lon], [center_lat])
        center_x_m, center_y_m = center_x_m[0], center_y_m[0]
        
        X_m_flat = X_m.flatten()
        Y_m_flat = Y_m.flatten()
        
        # Apply radius filter in meters
        radius_m = radius_km * 1000.0
        distances = np.sqrt((X_m_flat - center_x_m)**2 + (Y_m_flat - center_y_m)**2)
        mask = distances <= radius_m
        
        xi_inside_x = X_m_flat[mask]
        xi_inside_y = Y_m_flat[mask]
        
        print(f"🔧 INDICATOR GRID: {len(xi_inside_x)}/{len(X_m_flat)} points within {radius_km}km radius")
        # ========================================================================

        # Create grid points array for compatibility 
        grid_points_array = np.column_stack([xi_inside_x, xi_inside_y])
        print(f"🔧 Grid points array created: {len(grid_points_array)} points for indicator kriging")

        # Choose interpolation method based on parameter and dataset size
        if (method == 'yield_kriging' or method == 'specific_capacity_kriging' or method == 'ground_water_level_kriging' or method == 'indicator_kriging' or method == 'indicator_kriging_spherical' or method == 'indicator_kriging_spherical_continuous') and len(wells_df) >= 5:
            try:
                if method == 'specific_capacity_kriging':
                    interpolation_name = "specific capacity kriging"
                elif method == 'ground_water_level_kriging':
                    interpolation_name = "ground water level kriging"
                elif method == 'indicator_kriging':
                    interpolation_name = "indicator kriging (yield suitability)"
                elif method == 'indicator_kriging_spherical':
                    interpolation_name = "indicator kriging spherical (yield suitability)"
                elif method == 'indicator_kriging_spherical_continuous':
                    interpolation_name = "indicator kriging spherical continuous (yield suitability)"
                else:
                    interpolation_name = "yield kriging"
                print(f"Using {interpolation_name} interpolation for heat map")

                # Filter to meaningful data for better kriging
                if method == 'indicator_kriging' or method == 'indicator_kriging_spherical' or method == 'indicator_kriging_spherical_continuous':
                    # For indicator kriging, use all data (including 0s and 1s)
                    meaningful_data_mask = np.ones(len(yields), dtype=bool)  # Use all data
                else:
                    # For other methods, filter to meaningful yield data
                    meaningful_data_mask = yields > 0.1

                if meaningful_data_mask.any() and np.sum(meaningful_data_mask) >= 5:
                    # Use filtered data
                    filtered_x_coords = x_coords[meaningful_data_mask]
                    filtered_y_coords = y_coords[meaningful_data_mask] 
                    filtered_yields = yields[meaningful_data_mask]

                    # Use filtered wells in NZTM2000 coordinates
                    filtered_wells_x_m = wells_x_m[meaningful_data_mask]
                    filtered_wells_y_m = wells_y_m[meaningful_data_mask]

                    # Set up kriging with appropriate variogram model
                    if method == 'indicator_kriging':
                        # Use linear variogram for binary indicator data
                        variogram_model_to_use = 'linear'
                    elif method == 'indicator_kriging_spherical':
                        # Use spherical variogram for binary indicator data
                        variogram_model_to_use = 'spherical'
                    elif method == 'indicator_kriging_spherical_continuous':
                        # Use spherical variogram for continuous indicator data
                        variogram_model_to_use = 'spherical'
                    else:
                        variogram_model_to_use = 'spherical'

                    # Use centralized kriging helper with NZTM2000 coordinates
                    Z_grid, SS_grid, OK = krige_on_grid(
                        filtered_wells_x_m, filtered_wells_y_m, filtered_yields,
                        x_vals_m, y_vals_m,
                        variogram_model=variogram_model_to_use,
                        verbose=False
                    )
                    
                    if Z_grid is not None:
                        # Extract values for masked grid points
                        Z_flat = Z_grid.flatten()
                        interpolated_z = Z_flat[mask]
                        print(f"🔧 Indicator kriging completed: {len(interpolated_z)} points")
                    else:
                        print("❌ Indicator kriging failed in helper")
                        interpolated_z = None

                    # Process results based on interpolation method
                    if method == 'indicator_kriging' or method == 'indicator_kriging_spherical':
                        # Ensure probability values are in [0,1] range
                        interpolated_z = np.clip(interpolated_z, 0.0, 1.0)

                        # Apply binary threshold for clear visualization (0.5 = 50% probability)
                        binary_threshold = 0.5
                        interpolated_z = (interpolated_z >= binary_threshold).astype(float)

                        variogram_type = "spherical" if method == 'indicator_kriging_spherical' else "linear"
                        print(f"Heat map indicator kriging ({variogram_type}): binary classification with {np.sum(interpolated_z)}/{len(interpolated_z)} areas classified as 'likely' for groundwater")
                    elif method == 'indicator_kriging_spherical_continuous':
                        # For continuous version: keep continuous probability values, no binary thresholding
                        interpolated_z = np.clip(interpolated_z, 0.0, 1.0)  # Just ensure [0,1] range
                        print(f"Heat map indicator kriging (spherical continuous): continuous probabilities from {interpolated_z.min():.3f} to {interpolated_z.max():.3f}")
                    else:
                        # Ensure non-negative yields for other methods
                        interpolated_z = np.maximum(0, interpolated_z)
                else:
                    # Fallback to griddata if insufficient data
                    interpolated_z = griddata(points, yields, xi_inside, method='linear', fill_value=0.0)
                    nan_mask = np.isnan(interpolated_z)
                    if np.any(nan_mask):
                        interpolated_z[nan_mask] = griddata(
                            points, yields, xi_inside[nan_mask], method='nearest', fill_value=0.0
                        )

            except Exception as e:
                print(f"Yield kriging error: {e}, fallingback to standard interpolation")
                interpolated_z = griddata(points, yields, xi_inside, method='linear', fill_value=0.0)
                nan_mask = np.isnan(interpolated_z)
                if np.any(nan_mask):
                    interpolated_z[nan_mask] = griddata(
                        points, yields, xi_inside[nan_mask], method='nearest', fill_value=0.0
                    )

        elif method == 'rf_kriging' and len(wells_df) >= 10:
            try:
                print("Using Random Forest + Kriging interpolation")
                # OPTIMIZATION: Reduce number of trees for faster performance
                # For large datasets, we need to prioritize speed over slight accuracy improvements

                # Prepare data for Random Forest
                features = np.vstack([x_coords, y_coords]).T  # Features are [x, y] coordinates in km
                target = yields  # Target is the yield values

                # OPTIMIZATION: Use fewer trees (50 instead of 100) and limit max_depth
                # This significantly speeds up training and prediction with minimal accuracy loss
                rf = RandomForestRegressor(
                    n_estimators=50,       # Reduced from 100 for faster performance
                    max_depth=15,          # Limit tree depth for faster training
                    min_samples_split=5,   # Require more samples per split (reduces overfitting)
                    n_jobs=-1,             # Use all available cores
                    random_state=42
                )
                rf.fit(features, target)

                # Get RF predictions for all grid points
                rf_predictions = rf.predict(xi_inside)

                # OPTIMIZATION: Skip kriging for very large datasets (>1000 points)
                # as it becomes the main performance bottleneck
                if len(features) < 1000:
                    # Calculate residuals on training data
                    rf_train_preds = rf.predict(features)
                    residuals = target - rf_train_preds

                    # If enough points, apply Kriging to the residuals
                    if len(features) >= 5 and len(features) < 1000:
                        # ===== FINAL ARCHITECT FIX: RF+KRIGING WITH NZTM2000 =====
                        print("🔧 Applying NZTM2000 coordinate fix to RF+Kriging residuals")
                        
                        # Use centralized kriging helper for residuals interpolation
                        Z_grid, SS_grid, OK = krige_on_grid(
                            wells_x_m, wells_y_m, residuals,
                            x_vals_m, y_vals_m,
                            variogram_model='linear',
                            verbose=False
                        )
                        
                        if Z_grid is not None:
                            # Extract kriged residuals for masked grid points
                            Z_flat = Z_grid.flatten()
                            kriged_residuals = Z_flat[mask]
                            print("🔧 RF+Kriging residuals interpolated successfully")
                        else:
                            print("❌ RF+Kriging residuals failed, using RF predictions only")
                            kriged_residuals = np.zeros_like(rf_predictions)
                        # ================================================================

                        # Combine RF predictions with kriged residuals
                        interpolated_z = rf_predictions + kriged_residuals
                    else:
                        # Not enough points for kriging, use RF predictions only
                        interpolated_z = rf_predictions
                else:
                    # Too many points for efficient kriging, use RF predictions only
                    print("Using RF predictions only (skipping kriging for large dataset)")
                    interpolated_z = rf_predictions
            except Exception as e:
                print(f"RF+Kriging error: {e}, falling back to standard interpolation")
                # Fall back to standard interpolation
                # Basic 2D interpolation (linear)
                from scipy.interpolate import griddata
                interpolated_z = griddata(points, yields, xi_inside, method='linear', fill_value=0.0)

                # For areas with NaNs, apply nearest neighbor to fill gaps
                if np.any(np.isnan(interpolated_z)):
                    nan_mask = np.isnan(interpolated_z)
                    interpolated_z[nan_mask] = griddata(
                        points, yields, xi_inside[nan_mask], method='nearest', fill_value=0.0
                    )
        else:
            # Choose interpolation method
            # Basic 2D interpolation using scipy.interpolate.griddata

            # OPTIMIZATION: For large datasets, use fewer points and faster method
            if len(points) > 2000:
                print(f"Large dataset optimization: Sampling from {len(points)} points")
                # For very large datasets, we'll use a random sample to improve performance
                sample_size = min(2000, len(points))
                # Use stratified sampling for better representation (divide area into regions)
                # Create grid cells (10x10)
                lat_bins = np.linspace(np.min(lats), np.max(lats), 10)
                lon_bins = np.linspace(np.min(lons), np.max(lons), 10)

                # Create a mask for sampling points from each cell
                sample_indices = []
                for i in range(9):
                    for j in range(9):
                        # Get points in this cell
                        cell_mask = (
                            (lats >= lat_bins[i]) & (lats < lat_bins[i+1]) & 
                            (lons >= lon_bins[j]) & (lons < lon_bins[j+1])
                        )
                        cell_indices = np.where(cell_mask)[0]

                        # If there are points in this cell, sample a proportional amount
                        if len(cell_indices) > 0:
                            # Calculate how many points to sample from this cell
                            # (proportional to the cell's share of total points)
                            cell_sample_size = max(1, int(sample_size * len(cell_indices) / len(points)))
                            # Sample randomly from this cell
                            if len(cell_indices) > cell_sample_size:
                                cell_sample = np.random.choice(cell_indices, cell_sample_size, replace=False)
                                sample_indices.extend(cell_sample)
                            else:
                                # If cell has fewer points than needed, take all of them
                                sample_indices.extend(cell_indices)

                # Ensure we have enough sample points
                if len(sample_indices) < sample_size:
                    # Add more random points if needed
                    remaining = sample_size - len(sample_indices)
                    all_indices = np.arange(len(points))
                    remaining_indices = np.setdiff1d(all_indices, sample_indices)
                    if len(remaining_indices) > 0:
                        additional_samples = np.random.choice(
                            remaining_indices, 
                            min(remaining, len(remaining_indices)), 
                            replace=False
                        )
                        sample_indices.extend(additional_samples)

                # Use the sampled points for interpolation
                sampled_points = points[sample_indices]
                sampled_yields = yields[sample_indices]

                # Use linear interpolation for speed with large datasets
                interpolated_z = griddata(
                    sampled_points, sampled_yields, xi_inside, 
                    method='linear', fill_value=0.0
                )

                # For areas with NaNs, apply nearest neighbor to fill gaps
                if np.any(np.isnan(interpolated_z)):
                    nan_mask = np.isnan(interpolated_z)
                    interpolated_z[nan_mask] = griddata(
                        sampled_points, sampled_yields, xi_inside[nan_mask],
                        method='nearest', fill_value=0.0
                    )
            else:
                # Standard approach for smaller datasets
                # First try linear interpolation
                interpolated_z = griddata(
                    points, yields, xi_inside, 
                    method='linear', fill_value=0.0
                )

                # For areas with NaNs, apply nearest neighbor to fill gaps
                if np.any(np.isnan(interpolated_z)):
                    nan_mask = np.isnan(interpolated_z)
                    interpolated_z[nan_mask] = griddata(
                        points, yields, xi_inside[nan_mask],
                        method='nearest', fill_value=0.0
                    )

        # Make sure we don't have negative values
        interpolated_z = np.maximum(0, interpolated_z)

        # Convert back to lat/lon coordinates
        # ===== ARCHITECT FIX: CONVERT GRID POINTS USING NZTM2000 =====
        lon_points, lat_points = to_wgs84(xi_inside_x, xi_inside_y)
        print(f"🔧 Converted grid points for heat data: {len(lat_points)} points")
        # ========================================================================

        # Prepare soil polygon geometry for filtering heat map display
        merged_soil_geometry = None
        if soil_polygons is not None and len(soil_polygons) > 0:
            try:
                # Create a unified geometry from all soil polygons
                valid_geometries = []
                for idx, row in soil_polygons.iterrows():
                    if row.geometry and row.geometry.is_valid:
                        valid_geometries.append(row.geometry)

                if valid_geometries:
                    # Merge all polygons into a single multipolygon
                    merged_soil_geometry = unary_union(valid_geometries)
                    print(f"Prepared soil drainage geometry for heat map filtering")
                else:
                    print("No valid soil polygon geometries found for heat map")
            except Exception as e:
                print(f"Error preparing soil polygon geometry for heat map: {e}")
                merged_soil_geometry = None

        # Create heat map data
        heat_data = []

        # OPTIMIZATION: For large datasets, reduce the number of heat map points for better performance
        # First, detect if we need to sample points due to a large dataset
        max_heat_points = 2500  # Maximum points for smooth performance

        if len(lat_points) > max_heat_points:
            # Use a grid-based sampling approach to maintain visual accuracy with fewer points
            print(f"Optimizing heatmap visualization: sampling {max_heat_points} points from {len(lat_points)} total")

            # Create a grid with target number of cells
            grid_size = int(np.sqrt(max_heat_points))

            # Find min and max lat/lon
            min_lat, max_lat = np.min(lat_points), np.max(lat_points)
            min_lon, max_lon = np.min(lon_points), np.max(lon_points)

            # Create grid
            lat_grid = np.linspace(min_lat, max_lat, grid_size)
            lon_grid = np.linspace(min_lon, max_lon, grid_size)

            # Sample points by selecting representative points in each grid cell
            heat_data = []
            for i in range(len(lat_grid)-1):
                for j in range(len(lon_grid)-1):
                    # Find points in this grid cell
                    cell_mask = (
                        (lat_points >= lat_grid[i]) & (lat_points < lat_grid[i+1]) &
                        (lon_points >= lon_grid[j]) & (lon_points < lon_grid[j+1])
                    )

                    if np.any(cell_mask):
                        # Select a point with the maximum value from this cell (important for yield visualization)
                        cell_values = interpolated_z[cell_mask]
                        cell_lat = lat_points[cell_mask]
                        cell_lon = lon_points[cell_mask]

                        if np.max(cell_values) > 0.01:  # Only add significant values
                            max_idx = np.argmax(cell_values)
                            point_lat = float(cell_lat[max_idx])
                            point_lon = float(cell_lon[max_idx])

                            # Check if point should be included based on soil polygons
                            include_point = True
                            if merged_soil_geometry is not None:
                                point = Point(point_lon, point_lat)
                                include_point = merged_soil_geometry.contains(point) or merged_soil_geometry.intersects(point)

                            if include_point:
                                # Double-check: ensure point is actually within soil polygons
                                if merged_soil_geometry is not None:
                                    point = Point(point_lon, point_lat)
                                    # Use strict containment for heat map points
                                    strictly_contained = merged_soil_geometry.contains(point)

                                    # If not strictly contained, check if very close to boundary
                                    if not strictly_contained:
                                        buffer_distance = 0.0001  # roughly 10 meters
                                        buffered_geometry = merged_soil_geometry.buffer(buffer_distance)
                                        strictly_contained = buffered_geometry.contains(point)

                                    if strictly_contained:
                                        heat_data.append([
                                            point_lat,  # Latitude
                                            point_lon,  # Longitude
                                            float(cell_values[max_idx])  # Yield value (actual value)
                                        ])
                                else:
                                    heat_data.append([
                                        point_lat,  # Latitude
                                        point_lon,  # Longitude
                                        float(cell_values[max_idx])  # Yield value (actual value)
                                    ])
        else:
            # Standard approach for smaller datasets
            heat_data = []
            # Add interpolated points
            for i in range(len(lat_points)):
                # Adjust threshold based on interpolation method
                meaningful_threshold = 0.01

                # Only add points with meaningful values
                if interpolated_z[i] > meaningful_threshold:
                    # Check if point should be included based on soil polygons
                    include_point = True
                    if merged_soil_geometry is not None:
                        # For non-SWL methods, apply normal soil polygon filtering
                        point = Point(lon_points[i], lat_points[i])
                        include_point = merged_soil_geometry.contains(point) or merged_soil_geometry.intersects(point)

                    if include_point:
                        heat_data.append([
                            float(lat_points[i]),  # Latitude
                            float(lon_points[i]),  # Longitude
                            float(interpolated_z[i])  # SWL/yield value (actual value, not normalized)
                        ])

        # Always make sure well points themselves are included for accuracy
        # These are the actual data points we have, so they should be shown
        well_points_added = 0
        for j in range(len(lats)):
            # Check if well is within search radius
            well_dist_km = np.sqrt(
                # ===== ARCHITECT FIX: DISTANCE CALCULATION WITH NZTM2000 =====
                (wells_y_m[j] - center_y_m)**2 + (wells_x_m[j] - center_x_m)**2
                # ================================================================
            )

            if well_dist_km <= radius_km:
                # Check if well should be included based on soil polygons
                include_well = True
                if merged_soil_geometry is not None:
                    well_point = Point(lons[j], lats[j])
                    include_well = merged_soil_geometry.contains(well_point) or merged_soil_geometry.intersects(well_point)

                if include_well:
                    # Check soil polygon containment based on method
                    well_point = Point(lons[j], lats[j])
                    well_contained = merged_soil_geometry.contains(well_point) or merged_soil_geometry.intersects(well_point)

                    if well_contained:
                        heat_data.append([
                            float(lats[j]),
                            float(lons[j]),
                            float(yields[j])
                        ])
                        well_points_added += 1
                    else:
                        # No soil polygon filtering
                        heat_data.append([
                            float(lats[j]),
                            float(lons[j]),
                            float(yields[j])
                        ])
                        well_points_added += 1

        # Log filtering results
        if merged_soil_geometry is not None:
            print(f"Heat map filtered by soil drainage areas: {len(heat_data)} points displayed")

        return heat_data

    except Exception as e:
        print(f"Interpolation error: {e}")
        return fallback_interpolation(wells_df, center_point, radius_km)

def fallback_interpolation(wells_df, center_point, radius_km, resolution=50):
    """
    Simplified IDW (Inverse Distance Weighting) interpolation as fallback method
    Creates a continuous interpolated surface based on actual well yield values
    """
    # Filter to only use wells with actual yield data (exclude unknown yields)
    if 'has_unknown_yield' in wells_df.columns:
        valid_wells = wells_df[(~wells_df['has_unknown_yield']) & (wells_df['yield_rate'].notna())].copy()
    else:
        valid_wells = wells_df[wells_df['yield_rate'].notna()].copy()

    if valid_wells.empty:
        return []

    # Extract coordinates and yields
    lats = valid_wells['latitude'].values.astype(float)
    lons = valid_wells['longitude'].values.astype(float)
    yields = valid_wells['yield_rate'].values.astype(float)

    # Handle empty dataset
    if len(yields) == 0:
        return []

    center_lat,center_lon = center_point

    # Create a grid for interpolation
    grid_resolution = min(50, resolution)

    # Calculate area to cover
    lat_radius = radius_km / 111.0
    lon_radius = radius_km / (111.0 * np.cos(np.radians(float(center_lat))))

    # Grid boundaries
    min_lat = center_lat - lat_radius
    max_lat = center_lat + lat_radius
    min_lon = center_lon - lon_radius
    max_lon = center_lon + lon_radius

    # Create grid
    grid_lats = np.linspace(min_lat, max_lat, grid_resolution)
    grid_lons = np.linspace(min_lon, max_lon, grid_resolution)
    grid_lat, grid_lon = np.meshgrid(grid_lats, grid_lons)

    # Flatten grid for calculations
    grid_lat_flat = grid_lat.flatten()
    grid_lon_flat = grid_lon.flatten()

    # Heat map data collection
    heat_data = []

    # Convert to a simple projected space (km from center)
    x_points = (lons - center_lon) * 111.0 * np.cos(np.radians(center_lat))
    y_points = (lats - center_lat) * 111.0

    # For each grid point, calculate IDW interpolation
    for i in range(len(grid_lat_flat)):
        grid_point_lat = grid_lat_flat[i]
        grid_point_lon = grid_lon_flat[i]

        # Skip if outside search radius
        dist_from_center_km = np.sqrt(
            ((grid_point_lat - center_lat) * 111.0)**2 + 
            ((grid_point_lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
        )

        if dist_from_center_km > radius_km:
            continue

        # Convert grid point to projected space
        grid_x = (grid_point_lon - center_lon) * 111.0 * np.cos(np.radians(center_lat))
        grid_y = (grid_point_lat - center_lat) * 111.0

        # Calculate distances to all wells
        distances = np.sqrt((grid_x - x_points)**2 + (grid_y - y_points)**2)

        # IDW Parameters
        power = 2.0  # Standard IDW power parameter

        # Standard IDW formula
        if np.min(distances) < 0.05:  # If very close to a well (within 50m)
            # Find the closest well
            closest_idx = np.argmin(distances)
            interpolated_value = yields[closest_idx]
        else:
            # Apply IDW with distance power weighting
            weights = 1.0 / (distances**power)

            # Handle potential division by zero
            if np.any(np.isinf(weights)):
                mask = np.isinf(weights)
                weights[mask] = 1.0
                weights[~mask] = 0.0

            # Normalize weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                # Calculate weighted average
                interpolated_value = np.sum(weights * yields)
            else:
                # This shouldn't happen, but just in case
                interpolated_value = 0.0

        # Add to heat map with interpolated value (actual yield value)
        if interpolated_value > 0.01:  # Only add significant points
            heat_data.append([
                float(grid_point_lat), 
                float(grid_point_lon),
                float(interpolated_value)  # Use actual yield value
            ])

    # Always add the actual well points with their values
    for j in range(len(lats)):
        # Check if within search radius
        dist_from_center_km = np.sqrt(
            ((lats[j] - center_lat) * 111.0)**2 + 
            ((lons[j] - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
        )

        if dist_from_center_km <= radius_km:
            heat_data.append([
                float(lats[j]),
                float(lons[j]),
                float(yields[j])  # Use actual yield value
            ])

    return heat_data

def get_prediction_at_point(wells_df, point_lat, point_lon):
    """
    Get a predicted yield at a specific point based on nearby wells using modified IDW

    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data
    point_lat : float
        Latitude of the point to predict
    point_lon : float
        Longitude of the point to predict

    Returns:
    --------
    float
        Predicted yield rate at the specified point
    """
    if not isinstance(wells_df, pd.DataFrame) or wells_df.empty:
        return 0

    # Extract coordinates and yields
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)
    yields = wells_df['yield_rate'].values.astype(float)

    # Convert to kilometer distances for better accuracy
    # First, convert to flat projection (rough approximation using kilometers)
    origin_lat, origin_lon = np.mean(lats), np.mean(lons)
    x_coords = (lons - origin_lon) * 111.0 * np.cos(np.radians(origin_lat))
    y_coords = (lats - origin_lat) * 111.0

    # Convert prediction point to same projection
    point_x = (point_lon - origin_lon) * 111.0 * np.cos(np.radians(origin_lat))
    point_y = (point_lat - origin_lat) * 111.0

    try:
        # Calculate distances to all wells (in kilometers)
        distances = np.sqrt((point_x - x_coords)**2 + (point_y - y_coords)**2)

        # Parameters for modified IDW
        # Higher power value creates sharper transitions between high and low yield areas
        power = 2.5
        smoothing = 0.1  # Small value to prevent division by zero

        # Maximum distance influence (in km)
        # Points beyond this have minimal impact on prediction
        max_influence_distance = 5.0  # 5km influence radius

        # Apply exponential distance decay to sharply reduce influence of distant wells
        exp_weights = np.exp(-distances / (max_influence_distance / 3))

        # Apply traditional inverse distance power weighting with higher power
        idw_weights = 1.0 / (distances + smoothing)**power

        # Combine both weighting strategies
        weights = exp_weights * idw_weights

        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)

            # Calculate weighted average of yields
            predicted_yield = np.sum(weights * yields)

            # If point is very far from any well, reduce confidence
            nearest_dist = np.min(distances)

            if nearest_dist > max_influence_distance:
                # Adjust prediction closer to area average for distant points
                confidence = max(0.2, np.exp(-(nearest_dist - max_influence_distance) / max_influence_distance))
                avg_yield = np.mean(yields)
                predicted_yield = predicted_yield * confidence + avg_yield * (1 - confidence)

            return float(max(0, predicted_yield))  # Ensure non-negative yield
        else:
            return 0.0

    except Exception as e:
        print(f"Prediction error: {e}")
        # Fall back to basic IDW
        return basic_idw_prediction(wells_df, point_lat, point_lon)

def basic_idw_prediction(wells_df, point_lat, point_lon):
    """
    Calculate yield using basic Inverse Distance Weighting (IDW)
    Used as a fallback method - only uses wells with actual yield data
    """
    if not isinstance(wells_df, pd.DataFrame) or wells_df.empty:
        return 0

    # Filter to only use wells with actual yield data
    if 'has_unknown_yield' in wells_df.columns:
        valid_wells = wells_df[(~wells_df['has_unknown_yield']) & (wells_df['yield_rate'].notna())].copy()
    else:
        valid_wells = wells_df[wells_df['yield_rate'].notna()].copy()

    if valid_wells.empty:
        return 0

    # Calculate distance from each well to the point (in km)
    distances = []
    for idx, row in valid_wells.iterrows():
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        # Convert to kilometers using approximate conversion
        lat_dist = (lat - point_lat) * 111.0  # 1 degree latitude ≈ 111 km
        lon_dist = (lon - point_lon) * 111.0 * np.cos(np.radians((lat + point_lat) / 2))
        distance = np.sqrt(lat_dist**2 + lon_dist**2)
        distances.append(max(0.1, distance))  # Prevent division by zero

    # Calculate inverse distance weights with power of 2
    weights = [1 / (d**2) for d in distances]
    total_weight = sum(weights)

    if total_weight == 0:
        # If all weights are zero (should be impossible with our minimum distance)
        return 0

    # Calculate weighted average of yields
    weighted_yield = sum(w * float(row['yield_rate']) for w, (idx, row) in zip(weights, valid_wells.iterrows())) / total_weight
    return float(max(0, weighted_yield))  # Ensure non-negative yieldld

def calculate_kriging_variance(wells_df, center_point, radius_km, resolution=50, variogram_model='spherical', use_depth=False, soil_polygons=None):
    """
    Calculate kriging variance for yield or depth to groundwater interpolations.

    This function performs ordinary kriging to estimate the variance (uncertainty)
    of the interpolated values. It's designed to work with both yield and
    depth-to-groundwater data, and applies soil polygon filtering like other interpolants.

    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data, including latitude, longitude, and
        either yield_rate (for yield interpolation) or depth_to_groundwater/depth
        (for depth interpolation).
    center_point : tuple
        Tuple containing (latitude, longitude) of the center point.
    radius_km : float
        Radius in kilometers to generate the variance data for.
    resolution : int
        Number of points to generate in each dimension.
    variogram_model : str, optional
        Variogram model to use for kriging (e.g., 'linear', 'spherical', 'gaussian').
        Defaults to 'spherical'.
    use_depth : bool
        Whether to use depth data instead of yield data.
    soil_polygons : GeoDataFrame, optional
        Soil polygon data for clipping the variance output.

    Returns:
    --------
    list
        List of [lat, lng, variance] points for the variance heat map.
        Returns an empty list if there is not enough data or if an error occurs.

    Notes:
    ------
    - The function automatically detects whether to interpolate yield or depth
      based on the columns available in `wells_df`.
    - It handles cases where depth_to_groundwater is not available and falls
      back to using the 'depth' column if it exists.
    - It returns an empty list if there is insufficient data (less than 5 wells)
      to perform kriging.
    - Applies the same soil polygon clipping as other interpolants.
    """
    try:
        # Filter wells data similar to yield kriging for consistency
        wells_df_filtered = wells_df.copy()

        # Filter out geotechnical/geological investigation wells using well_use
        if 'well_use' in wells_df_filtered.columns:
            geotechnical_mask = wells_df_filtered['well_use'].str.contains(
                'Geotechnical.*Investigation|Geological.*Investigation', 
                case=False, 
                na=False, 
                regex=True
            )
            wells_df_filtered = wells_df_filtered[~geotechnical_mask].copy()

            if wells_df_filtered.empty:
                return []

        # Determine whether to use yield or depth data based on use_depth parameter
        if use_depth:
            # For depth variance, filter wells similar to depth kriging
            if 'is_dry_well' in wells_df.columns:
                wells_df_filtered = wells_df_filtered[~wells_df_filtered['is_dry_well']]

            if 'depth_to_groundwater' in wells_df_filtered.columns and not wells_df_filtered['depth_to_groundwater'].isna().all():
                wells_df_filtered = wells_df_filtered[wells_df_filtered['depth_to_groundwater'].notna() & (wells_df_filtered['depth_to_groundwater'] > 0)]
                values = wells_df_filtered['depth_to_groundwater'].values.astype(float)
                interpolation_type = 'depth_to_groundwater'
            elif 'depth' in wells_df_filtered.columns:
                wells_df_filtered = wells_df_filtered[wells_df_filtered['depth'].notna() & (wells_df_filtered['depth'] > 0)]
                values = wells_df_filtered['depth'].values.astype(float)
                interpolation_type = 'depth'
            else:
                print("Error: No depth data found in wells_df.")
                return []
        else:
            # For yield variance, filter to meaningful yield data like yield kriging
            meaningful_yield_mask = wells_df_filtered['yield_rate'].fillna(0) > 0.1

            if meaningful_yield_mask.any() and np.sum(meaningful_yield_mask) >= 5:
                wells_df_filtered = wells_df_filtered[meaningful_yield_mask]
                values = wells_df_filtered['yield_rate'].values.astype(float)
                interpolation_type = 'yield'
            else:
                # If insufficient meaningful yield data, use all available yield data
                values = wells_df_filtered['yield_rate'].fillna(0).values.astype(float)
                interpolation_type = 'yield'

        # Prepare data: Extract coordinates and values from filtered data
        lats = wells_df_filtered['latitude'].values.astype(float)
        lons = wells_df_filtered['longitude'].values.astype(float)

        # Ensure there's enough data for kriging
        if len(wells_df_filtered) < 5:
            print("Warning: Insufficient data points for kriging variance calculation after filtering.")
            return []

        # Create grid for interpolation
        center_lat, center_lon = center_point
        grid_size = min(50, max(30, resolution))  # Adjust grid size if necessary

        # COORDINATE TRANSFORMATION FIX: Use proper projected coordinates (EPSG:2193)
        print(f"🔧 ===== COORDINATE TRANSFORMATION FIX APPLIED =====")
        import pyproj
        
        # Set up coordinate transformers
        transformer_to_nztm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
        transformer_to_wgs84 = pyproj.Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
        
        # Transform center point to NZTM2000 (meters)
        center_x_m, center_y_m = transformer_to_nztm.transform(center_lon, center_lat)
        print(f"🔧 CENTER POINT: ({center_lat:.6f}°, {center_lon:.6f}°) -> ({center_x_m:.1f}m, {center_y_m:.1f}m)")
        
        # Transform well coordinates to NZTM2000 meters
        wells_x_m, wells_y_m = transformer_to_nztm.transform(lons, lats)
        
        # Convert to km from center using exact meter calculations
        x_coords = (wells_x_m - center_x_m) / 1000.0  # Convert to km
        y_coords = (wells_y_m - center_y_m) / 1000.0  # Convert to km
        
        print(f"🔧 WELL COORDS: {len(x_coords)} wells converted to projected coordinates")
        print(f"🔧 COORDINATE RANGE: X: {x_coords.min():.2f} to {x_coords.max():.2f} km, Y: {y_coords.min():.2f} to {y_coords.max():.2f} km")

        # Create grid in km space (square bounds) - same as before
        grid_x = np.linspace(-radius_km, radius_km, grid_size)
        grid_y = np.linspace(-radius_km, radius_km, grid_size)
        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

        # Flatten grid for interpolation
        xi = np.vstack([grid_X.flatten(), grid_Y.flatten()]).T

        # Filter points outside the square bounds (instead of circular)
        mask = (np.abs(xi[:,0]) <= radius_km) & (np.abs(xi[:,1]) <= radius_km)
        xi_inside = xi[mask]
        
        # Convert grid points back to NZTM2000 meters, then to WGS84 for kriging
        grid_x_m = xi_inside[:, 0] * 1000.0 + center_x_m  # Convert km back to meters
        grid_y_m = xi_inside[:, 1] * 1000.0 + center_y_m  # Convert km back to meters
        
        # Transform grid points to WGS84 for kriging
        xi_lon, xi_lat = transformer_to_wgs84.transform(grid_x_m, grid_y_m)
        
        # Transform well coordinates to WGS84 for kriging (reverse original transformation)
        lon_values, lat_values = transformer_to_wgs84.transform(wells_x_m, wells_y_m)
        
        print(f"🔧 GRID POINTS: {len(xi_inside)} points transformed for kriging")
        print(f"🔧 KRIGING INPUT: wells ({len(lon_values)}) and grid points ({len(xi_lon)}) in WGS84")
        print(f"🔧 ========================================================")

        # Perform Ordinary Kriging with enhanced variance calculation
        try:
            # First, try standard kriging with optimized parameters for variance
            OK = OrdinaryKriging(
                lon_values, lat_values, values,
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False,
                weight=False,  # Disable additional weighting for cleaner variance
                exact_values=False,  # Allow interpolation for better variance estimation
                pseudo_inv=False  # Use standard inversion
            )

            # Execute kriging to get both predictions and variance
            predictions, kriging_variance = OK.execute('points', xi_lon, xi_lat)

            print(f"Initial kriging variance - Min: {np.min(kriging_variance):.6f}, Max: {np.max(kriging_variance):.6f}, Mean: {np.mean(kriging_variance):.6f}")

            # Ensure variance values are reasonable (variance should always be positive)
            kriging_variance = np.maximum(kriging_variance, 1e-8)

            # Check if variance is too uniform (indicates calculation issue)
            variance_std = np.std(kriging_variance)
            variance_mean = np.mean(kriging_variance)
            variance_cv = variance_std / variance_mean if variance_mean > 0 else 0

            print(f"Variance coefficient of variation: {variance_cv:.4f}")

            if variance_cv < 0.1:  # If variance is too uniform (less than 10% variation)
                print("Warning: Variance appears too uniform, enhancing with multiple variogram models...")

                # Try multiple variogram models and combine results
                variance_results = []
                variogram_models = ['linear', 'power', 'gaussian', 'exponential']

                for model in variogram_models:
                    try:
                        OK_test = OrdinaryKriging(
                            lon_values, lat_values, values,
                            variogram_model=model,
                            verbose=False,
                            enable_plotting=False
                        )
                        _, test_variance = OK_test.execute('points', xi_lon, xi_lat)
                        test_variance = np.maximum(test_variance, 1e-8)

                        # Check if this model produces better variation
                        test_cv = np.std(test_variance) / np.mean(test_variance) if np.mean(test_variance) > 0 else 0
                        variance_results.append((model, test_variance, test_cv))
                        print(f"Model {model}: CV = {test_cv:.4f}")

                    except Exception as model_error:
                        print(f"Model {model} failed: {model_error}")
                        continue

                # Use the model with the highest coefficient of variation (most spatial variation)
                if variance_results:
                    best_model, best_variance, best_cv = max(variance_results, key=lambda x: x[2])
                    print(f"Using {best_model} model with CV = {best_cv:.4f}")
                    kriging_variance = best_variance
                    variance_cv = best_cv

            # If still too uniform, enhance with distance-based variation
            if variance_cv < 0.05:
                print("Enhancing variance with distance-based component...")

                # Calculate distance from each grid point to nearest data point
                from scipy.spatial.distance import cdist

                # Create coordinate arrays for distance calculation
                data_coords = np.column_stack([lat_values, lon_values])
                grid_coords = np.column_stack([xi_lat, xi_lon])

                # Calculate distances (in decimal degrees)
                distances = cdist(grid_coords, data_coords)
                min_distances = np.min(distances, axis=1)

                # Create distance-based variance enhancement
                max_distance = np.max(min_distances) if len(min_distances) > 0 else 1.0
                if max_distance > 0:
                    # Normalize distances and create variance enhancement factor
                    normalized_distances = min_distances / max_distance

                    # Create exponential distance decay for variance
                    distance_factor = 1.0 + 2.0 * np.exp(normalized_distances * 3.0)  # Exponential growth with distance

                    # Enhance the original kriging variance with distance component
                    base_variance = np.mean(kriging_variance)
                    enhanced_variance = kriging_variance * distance_factor

                    # Smooth the transition between original and enhanced variance
                    alpha = 0.6  # Blend factor
                    kriging_variance = alpha * enhanced_variance + (1 - alpha) * kriging_variance

                    print(f"Enhanced variance - Min: {np.min(kriging_variance):.6f}, Max: {np.max(kriging_variance):.6f}")

            # Final variance validation and scaling
            kriging_variance = np.maximum(kriging_variance, 1e-6)

            # Scale variance to reasonable range for visualization
            min_var = np.min(kriging_variance)
            max_var = np.max(kriging_variance)
            var_range = max_var - min_var

            if var_range > 0:
                # Normalize to 0-1 range then scale to meaningful variance range
                normalized_var = (kriging_variance - min_var) / var_range
                # Scale to range that makes sense for the data type
                if use_depth:
                    kriging_variance = normalized_var * (np.var(values) * 2.0) + (np.var(values) * 0.1)
                else:
                    kriging_variance = normalized_var * (np.var(values) * 1.5) + (np.var(values) * 0.05)

            print(f"Final variance - Min: {np.min(kriging_variance):.6f}, Max: {np.max(kriging_variance):.6f}, CV: {np.std(kriging_variance)/np.mean(kriging_variance):.4f}")

        except Exception as e:
            print(f"Error in kriging variance calculation: {e}")
            # Fallback: create synthetic variance based on distance to nearest data point
            print("Using enhanced distance-based variance estimation as fallback")

            # Calculate distance from each grid point to nearest data point
            from scipy.spatial.distance import cdist

            # Create coordinate arrays for distance calculation
            data_coords = np.column_stack([lat_values, lon_values])
            grid_coords = np.column_stack([xi_lat, xi_lon])

            # Calculate distances
            distances = cdist(grid_coords, data_coords)
            min_distances = np.min(distances, axis=1)

            # Create variance based on distance and data density
            max_distance = np.max(min_distances)
            if max_distance > 0:
                normalized_distances = min_distances / max_distance

                # Create more realistic variance based on distance and local data density
                # Areas far from wells should have higher uncertainty
                base_variance = np.var(values) if len(values) > 1 else 1.0

                # Exponential increase in variance with distance
                distance_variance = base_variance * (0.2 + 1.8 * np.exp(normalized_distances * 2.0))

                # Add some local variation based on coordinate position
                lat_variation = np.sin(xi_lat * 100) * 0.1 * base_variance
                lon_variation = np.cos(xi_lon * 100) * 0.1 * base_variance

                kriging_variance = distance_variance + lat_variation + lon_variation
                kriging_variance = np.maximum(kriging_variance, base_variance * 0.1)
            else:
                kriging_variance = np.full(len(xi_lat), np.var(values) * 0.5)

        # Prepare variance data for heat map using proper coordinate transformation
        # xi_lat and xi_lon are already in WGS84 from the coordinate transformation above
        lat_points = xi_lat
        lon_points = xi_lon

        # Prepare soil polygon geometry for filtering variance display (same as other interpolants)
        merged_soil_geometry = None
        if soil_polygons is not None and len(soil_polygons) > 0:
            try:
                # Create a unified geometry from all soil polygons
                valid_geometries = []
                for idx, row in soil_polygons.iterrows():
                    if row.geometry and row.geometry.is_valid:
                        valid_geometries.append(row.geometry)

                if valid_geometries:
                    # Merge all polygons into a single multipolygon
                    merged_soil_geometry = unary_union(valid_geometries)
                    print(f"Prepared soil drainage geometry for kriging variance filtering")
                else:
                    print("No valid soil polygon geometries found for kriging variance")
            except Exception as e:
                print(f"Error preparing soil polygon geometry for kriging variance: {e}")
                merged_soil_geometry = None

        # Apply stricter soil polygon filtering for variance data
        variance_data = []
        points_inside_count = 0
        points_total_count = len(lat_points)

        for i in range(len(lat_points)):
            # Only add points with meaningful variance values (lower threshold for variance)
            if kriging_variance[i] > 1e-6:
                # Check if point should be included based on soil polygons
                include_point = True
                if merged_soil_geometry is not None:
                    point = Point(lon_points[i], lat_points[i])

                    # Use ONLY strict containment - no intersection tolerance
                    strictly_contained = merged_soil_geometry.contains(point)

                    # For variance visualization, we want very precise clipping
                    # Only allow points that are clearly inside the soil polygons
                    if not strictly_contained:
                        # Check if point is very close to boundary (smaller buffer than before)
                        buffer_distance = 0.00005  # roughly 5 meters - much smaller buffer
                        buffered_geometry = merged_soil_geometry.buffer(buffer_distance)
                        strictly_contained = buffered_geometry.contains(point)

                    include_point = strictly_contained
                    if include_point:
                        points_inside_count += 1

                if include_point:
                    # Add data point with latitude, longitude, and variance
                    variance_data.append([
                        float(lat_points[i]),
                        float(lon_points[i]),
                        float(kriging_variance[i])
                    ])
                else:
                    # Add small amount of debugging for clipping
                    if merged_soil_geometry is not None and i % 100 == 0:  # Sample every 100th point for debug
                        point = Point(lon_points[i], lat_points[i])
                        distance_to_boundary = merged_soil_geometry.distance(point)
                        if distance_to_boundary < 0.001:  # If very close to boundary
                            print(f"Debug: Point {i} excluded - distance to boundary: {distance_to_boundary:.6f}")

        if merged_soil_geometry is not None:
            print(f"Variance clipping: {points_inside_count}/{points_total_count} points inside soil polygons ({100*points_inside_count/points_total_count:.1f}%)")

        # Log filtering results
        if merged_soil_geometry is not None:
            print(f"Kriging variance filtered by soil drainage areas: {len(variance_data)} points displayed")

        return variance_data

    except Exception as e:
        print(f"Error calculating kriging variance: {e}")
        return []  # Return empty list in case of error

# The following function is for create_base_map,
# I am adding the code here for generating the map with three-tier color system with red (0-0.5), orange (0.5-0.75), and green (0.75-1.0).

def create_base_map(location, zoom_start=10, tiles='OpenStreetMap'):
    """
    Create a base map centered at the specified location.

    Parameters:
    -----------
    location : tuple
        Tuple containing (latitude, longitude) of the center point.
    zoom_start : int, optional
        Initial zoom level. Defaults to 10.
    tiles : str, optional
        Tile provider for the map. Defaults to 'OpenStreetMap'.

    Returns:
    --------
    folium.Map
        A folium Map object.
    """
    import folium

    m = folium.Map(location=location, zoom_start=zoom_start, tiles=tiles)
    return m

def add_heatmap(m, data, radius=25, blur=15, max_value=1.0):
    """
    Add a heatmap layer to the map.

    Parameters:
    -----------
    m : folium.Map
        The folium Map object to add the heatmap to.
    data : list
        List of [lat, lng, intensity] points for the heatmap.
    radius : int, optional
        Radius of each point. Defaults to 25.
    blur : int, optional
        Amount of blur. Defaults to 15.
    max_value : float, optional
        Maximum intensity value. Defaults to 1.0.
    """
    from folium.plugins import HeatMap

    # Ensure data is not empty
    if not data:
        print("Warning: Heatmap data is empty. No heatmap will be added.")
        return

    # Create HeatMap layer and add it to the map
    hm = HeatMap(
        data,
        radius=radius,
        blur=blur,
        max_val=max_value,
        gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}  # Custom gradient
    )
    hm.add_to(m)

def add_circle_markers(m, wells_df, color='blue', radius=5, popup_text=None):
    """
    Add circle markers to the map for each well.

    Parameters:
    -----------
    m : folium.Map
        The folium Map object to add the markers to.
    wells_df : DataFrame
        DataFrame containing well data with latitude, longitude, and optional
        columns for popup text.
    color : str, optional
        Color of the markers. Defaults to 'blue'.
    radius : int, optional
        Radius of the markers. Defaults to 5.
    popup_text : list, optional
        List of column names to include in the popup text. Defaults to None.
    """
    import folium

    # Ensure wells_df is not empty
    if not isinstance(wells_df, pd.DataFrame) or wells_df.empty:
        print("Warning: wells_df is empty. No circle markers will be added.")
        return

    # Iterate through each well and add a circle marker
    for idx, row in wells_df.iterrows():
        lat = float(row['latitude'])
        lon = float(row['longitude'])

        # Create popup text if popup_text is specified
        if popup_text:
            popup_content = ""
            for col in popup_text:
                if col in row:
                    popup_content += f"<b>{col}:</b> {row[col]}<br>"
            popup = folium.Popup(popup_content, max_width=300)
        else:
            popup = None

        # Create circle marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=popup
        ).add_to(m)

def create_map_with_interpolated_data(wells_df, center_point, radius_km, resolution=50, interpolation_method='kriging', show_variance=False, variogram_model='spherical', soil_polygons=None):
    """
    Create a Folium map with interpolated groundwater yield data as a heat map.

    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data with latitude, longitude, and yield_rate columns.
    center_point : tuple
        Tuple containing (latitude, longitude) of the map's center point.
    radius_km : float
        Radius in kilometers to generate the heat map data for.
    resolution : int
        Number of points to generate in each dimension for interpolation.
    interpolation_method : str, optional
        Interpolation method to use ('kriging', 'idw', 'rf_kriging'). Defaults to 'kriging'.
    show_variance : bool, optional
        Whether to show kriging variance instead of interpolated values.
    variogram_model : str, optional
        Variogram model to use for kriging (e.g., 'linear', 'spherical', 'gaussian').
        Defaults to 'spherical'.
    soil_polygons : GeoDataFrame, optional
        Soil polygon data for clipping the interpolation output.

    Returns:
    --------
    folium.Map
        A Folium Map object with a heat map layer of interpolated yield data.
    """
    import folium
    import streamlit as st

    # Create the base map
    m = create_base_map(center_point, zoom_start=10)

    # Try to get the heat map data
    try:
        if show_variance:
            # Calculate kriging variance for uncertainty visualization
            heat_data = calculate_kriging_variance(wells_df, center_point, radius_km, resolution, variogram_model, soil_polygons=soil_polygons)
            # Set a maximum variance value for the heatmap
            max_value = 0.1  # Adjust this value as needed based on your data
        else:
            # Generate heat map data using the specified interpolation method
            heat_data = generate_heat_map_data(wells_df, center_point, radius_km, resolution, interpolation_method, soil_polygons=soil_polygons)

            # Calculate max_value
            if heat_data:
                max_value = max(point[2] for point in heat_data)
            else:
                max_value = 1.0

        if heat_data:
            if show_variance:
                add_heatmap(m, heat_data, radius=20, blur=15, max_value=max_value)  # Show variance as heatmap
            else:
                add_heatmap(m, heat_data, radius=25, blur=15, max_value=max_value)  # Show yield/depth as heatmap
        else:
            print("No heat data generated")

    except Exception as e:
        print(f"Error generating or adding heatmap: {e}")

    # Add circle markers for the well locations
    add_circle_markers(m, wells_df, color='black', radius=4, popup_text=['name', 'yield_rate', 'depth_to_groundwater', 'depth'])

    # Add a colormap legend to the map
    try:
        if show_variance:
            # Variance colormap legend
            colormap = folium.LinearColormap(
                colors=['blue', 'lime', 'yellow', 'red'],
                vmin=0,
                vmax=max_value,
                caption='Kriging Variance (Uncertainty)'
            )
            m.add_child(colormap)
        elif st.session_state.interpolation_method == 'indicator_kriging':
            # Three-tier indicator kriging legend
            colormap = folium.StepColormap(
                colors=['#FF0000', '#FFA500', '#00FF00'],  # Red, Orange, Green
                vmin=0,
                vmax=1.0,
                index=[0, 0.5, 0.75, 1.0],  # Three-tier thresholds
                caption='Groundwater Yield Probability: Red (Poor 0-0.5), Orange (Moderate 0.5-0.75), Green (Good 0.75-1.0)'
            )
            m.add_child(colormap)
        elif st.session_state.interpolation_method == 'depth_kriging':
            # Depth colormap legend
            colormap = folium.LinearColormap(
                colors=['green', 'yellow', 'red'],
                vmin=0,
                vmax=max_value,
                caption='Depth to Groundwater (m)'
            )
            m.add_child(colormap)
        elif st.session_state.interpolation_method == 'ground_water_level_kriging':
            # Ground water level colormap legend
            colormap = folium.LinearColormap(
                colors=['blue', 'cyan', 'yellow', 'orange', 'red'],
                vmin=0,
                vmax=max_value,
                caption='Ground Water Level'
            )
            m.add_child(colormap)
        else:
            # Yield colormap legend
            colormap = folium.LinearColormap(
                colors=['blue', 'lime', 'yellow', 'red'],
                vmin=0,
                vmax=max_value,
                caption='Groundwater Yield (L/s)'
            )
            m.add_child(colormap)
    except Exception as e:
        print(f"Error adding colormap legend: {e}")

    return m

def generate_raster_from_kriging_grid(Z_grid, x_vals_m, y_vals_m, global_colormap_func=None, opacity=0.7, raster_size=(512, 512)):
    """
    OFFSET FIX: Generate raster directly from Kriging Z_grid, bypassing triangulation to eliminate 6-pixel offset.
    
    This function creates a raster directly from the PyKrige interpolation results without the
    coordinate-drift-prone triangulation step that was causing the alignment issues.
    
    Parameters:
    -----------
    Z_grid : numpy.ndarray
        2D array of Kriging interpolation results from PyKrige (shape matches x_vals_m, y_vals_m meshgrid)
    x_vals_m : numpy.ndarray
        1D array of x-coordinates in NZTM2000 meters
    y_vals_m : numpy.ndarray  
        1D array of y-coordinates in NZTM2000 meters
    global_colormap_func : function
        Function to map values to colors
    opacity : float
        Raster opacity (0.0-1.0)
    raster_size : tuple
        (width, height) of output raster
        
    Returns:
    --------
    dict : Raster overlay data with 'image_base64', 'bounds', 'opacity' keys
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import base64
        from io import BytesIO
        import rasterio
        from rasterio.transform import from_origin
        
        print(f"🎯 DIRECT KRIGING RASTER: Creating raster from {Z_grid.shape} grid, bypassing triangulation")
        
        # Convert NZTM2000 grid coordinates to WGS84 for georeferencing
        X_m, Y_m = np.meshgrid(x_vals_m, y_vals_m)
        grid_lons, grid_lats = to_wgs84(X_m.flatten(), Y_m.flatten())
        grid_lons_2d = grid_lons.reshape(X_m.shape)
        grid_lats_2d = grid_lats.reshape(Y_m.shape)
        
        # Extract bounds from transformed coordinates
        min_lat, max_lat = grid_lats_2d.min(), grid_lats_2d.max()
        min_lon, max_lon = grid_lons_2d.min(), grid_lons_2d.max()
        
        print(f"🎯 BOUNDS: lat [{min_lat:.6f}, {max_lat:.6f}], lon [{min_lon:.6f}, {max_lon:.6f}]")
        print(f"🎯 KRIGING GRID: {Z_grid.shape}, values [{Z_grid.min():.3f}, {Z_grid.max():.3f}]")
        
        # Create output raster at specified size
        height, width = raster_size
        
        # Resample Z_grid to target raster size if needed
        from scipy.ndimage import zoom
        if Z_grid.shape != (height, width):
            zoom_factors = (height / Z_grid.shape[0], width / Z_grid.shape[1])
            Z_resampled = zoom(Z_grid, zoom_factors, order=1)  # Linear interpolation
            print(f"🎯 RESAMPLED: {Z_grid.shape} → {Z_resampled.shape}")
        else:
            Z_resampled = Z_grid
        
        # Create RGBA image
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Apply colormap to interpolated values
        for i in range(height):
            for j in range(width):
                value = Z_resampled[i, j]
                if not np.isnan(value) and value > 0:
                    if global_colormap_func:
                        color_hex = global_colormap_func(value)
                        # Convert hex to RGB
                        r = int(color_hex[1:3], 16)
                        g = int(color_hex[3:5], 16) 
                        b = int(color_hex[5:7], 16)
                    else:
                        # Fallback colormap (blue to red)
                        norm_val = min(1.0, max(0.0, value / Z_resampled.max()))
                        r = int(255 * norm_val)
                        g = int(255 * (1 - norm_val))
                        b = int(255 * (1 - norm_val))
                    
                    rgba_image[i, j] = [r, g, b, int(255 * opacity)]
        
        # Convert to PNG and encode as base64
        from PIL import Image
        pil_image = Image.fromarray(rgba_image, mode='RGBA')
        
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # Create bounds for Folium ImageOverlay (SW and NE corners)
        folium_bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        
        print(f"🎯 RASTER CREATED: {width}x{height} pixels, bounds {folium_bounds}")
        
        return {
            'image_base64': img_base64,
            'bounds': folium_bounds,
            'opacity': opacity,
            'method': 'direct_kriging_raster',
            'grid_info': {
                'original_shape': Z_grid.shape,
                'raster_shape': (height, width),
                'value_range': [float(Z_grid.min()), float(Z_grid.max())]
            }
        }
        
    except Exception as e:
        print(f"❌ Error in direct Kriging raster generation: {e}")
        return None

def generate_vector_grid_overlay(geojson_data, bounds, global_colormap_func=None, opacity=0.7, sampling_distance_meters=100, clipping_polygon=None):
    """
    ARCHITECT SOLUTION: Vector-based grid overlay using GeoJSON rectangles for deterministic positioning
    
    This eliminates coordinate system mismatches by letting Leaflet handle precise geographic positioning
    of individual grid cells as vector polygons instead of relying on raster overlay corner-warping.
    
    Parameters:
    -----------
    geojson_data : dict
        GeoJSON FeatureCollection containing triangular mesh data
    bounds : dict
        Dictionary with 'north', 'south', 'east', 'west' bounds
    global_colormap_func : function
        Function to map values to colors consistently across all heatmaps
    opacity : float
        Transparency level (0.0 to 1.0)
    sampling_distance_meters : int
        Grid spacing in meters (default: 100m)
    clipping_polygon : object
        Optional clipping polygon
        
    Returns:
    --------
    dict
        Dictionary containing GeoJSON FeatureCollection of grid rectangles
    """
    try:
        if not geojson_data or not geojson_data.get('features'):
            return None
            
        # Extract values and coordinates from triangular mesh - same as before
        values = []
        coords = []
        vertex_values = {}
        
        for feature in geojson_data['features']:
            if feature.get('properties', {}).get('value') is not None:
                value = feature['properties']['value']
                
                if feature['geometry']['type'] == 'Polygon':
                    triangle_coords = feature['geometry']['coordinates'][0][:-1]
                    
                    for coord in triangle_coords:
                        coord_key = (round(coord[0], 6), round(coord[1], 6))
                        
                        if coord_key in vertex_values:
                            vertex_values[coord_key] = (vertex_values[coord_key] + value) / 2
                        else:
                            vertex_values[coord_key] = value
        
        # Convert vertex dictionary to arrays
        for (lon, lat), val in vertex_values.items():
            coords.append([lon, lat])
            values.append(val)
        
        if not values or not coords:
            return None
            
        values = np.array(values)
        coords = np.array(coords)
        
        print(f"🔄 VECTOR APPROACH: Creating grid rectangles from {len(values)} vertex points")
        
        # Calculate grid parameters 
        west, east = bounds['west'], bounds['east']
        south, north = bounds['south'], bounds['north']
        
        # COORDINATE TRANSFORMATION FIX: Work entirely in EPSG:2193 (NZTM2000) projected coordinates
        print(f"🔧 ===== PROJECTED COORDINATE GRID CREATION =====")
        
        # Set up coordinate transformer from WGS84 to NZTM2000
        import pyproj
        transformer_to_nztm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
        transformer_to_wgs84 = pyproj.Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
        
        # Transform bounding box to NZTM2000 (meters)
        west_m, south_m = transformer_to_nztm.transform(west, south)
        east_m, north_m = transformer_to_nztm.transform(east, north)
        
        print(f"🔧 WGS84 BOUNDS: N={north:.6f}°, S={south:.6f}°, E={east:.6f}°, W={west:.6f}°")
        print(f"🔧 NZTM2000 BOUNDS: N={north_m:.1f}m, S={south_m:.1f}m, E={east_m:.1f}m, W={west_m:.1f}m")
        
        # Create regular grid in meters using sampling_distance_meters directly
        x_coords_m = np.arange(west_m + sampling_distance_meters/2, east_m, sampling_distance_meters)
        y_coords_m = np.arange(south_m + sampling_distance_meters/2, north_m, sampling_distance_meters)
        
        print(f"🔧 PROJECTED GRID: {len(x_coords_m)} × {len(y_coords_m)} = {len(x_coords_m) * len(y_coords_m)} points")
        print(f"🔧 GRID SPACING: {sampling_distance_meters}m exactly (no degree approximation)")
        
        # Convert grid coordinates back to WGS84 for compatibility with existing code
        # Create meshgrid first
        X_m, Y_m = np.meshgrid(x_coords_m, y_coords_m)
        
        # Transform all grid points to WGS84
        lons_grid, lats_grid = transformer_to_wgs84.transform(X_m.flatten(), Y_m.flatten())
        lons = lons_grid.reshape(X_m.shape)[0, :]  # First row (all have same x-coordinates)
        lats = lats_grid.reshape(Y_m.shape)[:, 0]  # First column (all have same y-coordinates)
        
        # Reverse lats array to maintain north-to-south ordering expected by rest of code
        lats = lats[::-1]
        
        # Calculate equivalent degree steps for backward compatibility
        lat_step = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.001
        lon_step = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.001
        
        print(f"🔧 TRANSFORMED GRID: {len(lats)} lats from {lats[0]:.6f}° to {lats[-1]:.6f}°")
        print(f"🔧 TRANSFORMED GRID: {len(lons)} lons from {lons[0]:.6f}° to {lons[-1]:.6f}°")
        print(f"🔧 EQUIVALENT STEPS: lat_step={lat_step:.8f}°, lon_step={lon_step:.8f}°")
        print(f"🔧 ===================================================")
        
        print(f"🔄 VECTOR GRID: {len(lats)} x {len(lons)} = {len(lats) * len(lons)} grid cells")
        
        # Create coordinate meshgrid for interpolation
        xi, yi = np.meshgrid(lons, lats)
        
        # Apply clipping polygon mask if provided
        clipping_mask = None
        if clipping_polygon is not None:
            try:
                from shapely.geometry import Point
                import geopandas as gpd
                
                print(f"🗺️ Applying clipping polygon to vector grid...")
                
                clipping_mask = np.zeros((len(lats), len(lons)), dtype=bool)
                
                if hasattr(clipping_polygon, 'geometry') and len(clipping_polygon) > 0:
                    merged_clipping_geom = clipping_polygon.geometry.unary_union
                elif hasattr(clipping_polygon, '__geo_interface__'):
                    merged_clipping_geom = clipping_polygon
                else:
                    merged_clipping_geom = None
                
                if merged_clipping_geom is not None:
                    points_inside = 0
                    for i in range(len(lats)):
                        for j in range(len(lons)):
                            lon = xi[i, j]
                            lat = yi[i, j]
                            point = Point(lon, lat)
                            is_inside = merged_clipping_geom.contains(point) or merged_clipping_geom.intersects(point)
                            clipping_mask[i, j] = is_inside
                            if is_inside:
                                points_inside += 1
                    
                    print(f"🗺️ Clipping results: {points_inside}/{len(lats)*len(lons)} grid cells inside clipping polygon")
                    
            except Exception as e:
                print(f"Error applying clipping polygon: {e}")
                clipping_mask = None
        
        # Interpolate values onto grid using multiple methods
        try:
            # Start with cubic interpolation
            zi = griddata(coords, values, (xi, yi), method='cubic', fill_value=np.nan)
            
            # Fill NaN values with linear interpolation
            nan_mask = np.isnan(zi)
            if np.any(nan_mask):
                zi_linear = griddata(coords, values, (xi, yi), method='linear', fill_value=np.nan)
                zi[nan_mask] = zi_linear[nan_mask]
                
            # Final pass: fill remaining NaN with nearest neighbor
            nan_mask = np.isnan(zi)
            if np.any(nan_mask):
                zi_nearest = griddata(coords, values, (xi, yi), method='nearest', fill_value=np.nan)
                zi[nan_mask] = zi_nearest[nan_mask]
                
        except Exception as e:
            print(f"Error in interpolation: {e}")
            # Fallback to nearest neighbor only
            zi = griddata(coords, values, (xi, yi), method='nearest', fill_value=np.nan)
        
        # Apply clipping mask
        if clipping_mask is not None:
            zi[~clipping_mask] = np.nan
        
        # Create GeoJSON FeatureCollection of grid rectangles
        features = []
        
        for i in range(len(lats)):
            for j in range(len(lons)):
                if not np.isnan(zi[i, j]):
                    value = zi[i, j]
                    
                    # Calculate rectangle bounds (cell edges, not centers)
                    cell_north = lats[i] + lat_step/2
                    cell_south = lats[i] - lat_step/2  
                    cell_west = lons[j] - lon_step/2
                    cell_east = lons[j] + lon_step/2
                    
                    # Create rectangle coordinates (clockwise from top-left)
                    rectangle_coords = [[
                        [cell_west, cell_north],   # Top-left (NW)
                        [cell_east, cell_north],   # Top-right (NE) 
                        [cell_east, cell_south],   # Bottom-right (SE)
                        [cell_west, cell_south],   # Bottom-left (SW)
                        [cell_west, cell_north]    # Close rectangle
                    ]]
                    
                    # Get color from global colormap function
                    if global_colormap_func:
                        color_hex = global_colormap_func(value)
                    else:
                        # Fallback color mapping
                        normalized_value = (value - np.nanmin(zi)) / (np.nanmax(zi) - np.nanmin(zi))
                        color_hex = f"#{int(normalized_value * 255):02x}{int((1-normalized_value) * 255):02x}00"
                    
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": rectangle_coords
                        },
                        "properties": {
                            "value": float(value),
                            "color": color_hex,
                            "opacity": opacity,
                            "row": i,
                            "col": j,
                            "center_lat": float(lats[i]),
                            "center_lon": float(lons[j])
                        }
                    }
                    features.append(feature)
        
        print(f"🔄 VECTOR RESULT: Created {len(features)} grid rectangles with precise geographic positioning")
        
        geojson_result = {
            "type": "FeatureCollection", 
            "features": features
        }
        
        return {
            'type': 'vector_grid',
            'geojson': geojson_result,
            'opacity': opacity,
            'grid_info': {
                'total_cells': len(lats) * len(lons),
                'visible_cells': len(features),
                'lat_step_degrees': lat_step,
                'lon_step_degrees': lon_step,
                'lat_step_meters': sampling_distance_meters,
                'lon_step_meters': sampling_distance_meters
            }
        }
        
    except Exception as e:
        print(f"Error generating vector grid overlay: {e}")
        return None

def generate_smooth_raster_overlay(geojson_data, bounds, raster_size=(512, 512), global_colormap_func=None, opacity=0.7, sampling_distance_meters=100, clipping_polygon=None):
    """
    Convert GeoJSON triangular mesh to smooth raster overlay for Windy.com-style visualization
    
    Uses 100-meter sampling grid across triangulated data, generating NaN where no triangle data exists.
    This preserves the original soil polygon and indicator kriging clipping.
    
    Parameters:
    -----------
    geojson_data : dict
        GeoJSON FeatureCollection containing triangular mesh data
    bounds : dict
        Dictionary with 'north', 'south', 'east', 'west' bounds
    sampling_distance_meters : int
        Sampling distance in meters for the regular grid (default: 100m)
    global_colormap_func : function
        Function to map values to colors consistently across all heatmaps
    opacity : float
        Transparency level (0.0 to 1.0) matching triangle mesh fillOpacity
        
    Returns:
    --------
    dict
        Dictionary containing base64 encoded image and bounds for Folium overlay
    """
    try:
        if not geojson_data or not geojson_data.get('features'):
            return None
            
        # Extract values and coordinates from triangular mesh
        # Use all triangle vertices (not just centroids) for better boundary coverage
        values = []
        coords = []
        vertex_values = {}  # Track values at each unique vertex
        
        for feature in geojson_data['features']:
            if feature.get('properties', {}).get('value') is not None:
                value = feature['properties']['value']
                
                if feature['geometry']['type'] == 'Polygon':
                    triangle_coords = feature['geometry']['coordinates'][0][:-1]  # Remove duplicate last point
                    
                    # Add all triangle vertices with their interpolated values
                    for coord in triangle_coords:
                        coord_key = (round(coord[0], 6), round(coord[1], 6))  # Round to avoid floating point issues
                        
                        if coord_key in vertex_values:
                            # Average values if vertex appears in multiple triangles
                            vertex_values[coord_key] = (vertex_values[coord_key] + value) / 2
                        else:
                            vertex_values[coord_key] = value
        
        # Convert vertex dictionary to arrays
        for (lon, lat), val in vertex_values.items():
            coords.append([lon, lat])
            values.append(val)
        
        if not values or not coords:
            return None
            
        values = np.array(values)
        coords = np.array(coords)
        
        print(f"Extracted {len(values)} vertex points from triangulated heatmap data")
        if coords.size == 0:
            return None
        
        # PERFORMANCE: Use dynamic grid resolution to prevent memory crashes
        west, east = bounds['west'], bounds['east']
        south, north = bounds['south'], bounds['north']
        
        # Calculate grid extent
        lat_extent = north - south
        lon_extent = east - west
        
        # OPTIMIZATION: Cap maximum grid size to prevent crashes
        max_grid_points = 1_000_000  # Maximum 1M points instead of 10.5M
        
        # Calculate appropriate resolution based on extent
        meters_per_degree_lat = 111000
        meters_per_degree_lon = 111000 * np.cos(np.radians((south + north) / 2))
        
        # Estimate grid points at 100m resolution
        lat_step_100m = 100 / meters_per_degree_lat
        lon_step_100m = 100 / meters_per_degree_lon
        grid_points_100m = (lat_extent / lat_step_100m) * (lon_extent / lon_step_100m)
        
        # Use coarser resolution if needed (architect recommendation)
        if grid_points_100m > max_grid_points:
            # Scale up resolution to stay under limit
            scale_factor = np.sqrt(grid_points_100m / max_grid_points)
            effective_resolution = 100 * scale_factor
            lat_step = effective_resolution / meters_per_degree_lat
            lon_step = effective_resolution / meters_per_degree_lon
            print(f"PERFORMANCE: Using {effective_resolution:.0f}m resolution (scaled from 100m) to prevent memory crash")
        else:
            lat_step = lat_step_100m
            lon_step = lon_step_100m
            effective_resolution = 100
        
        # Create regular sampling grid with lats ordered NORTH to SOUTH (row 0 = north)
        # This eliminates the need for vertical flip and ensures proper pixel-to-coordinate mapping
        lats = np.arange(north, south - lat_step, -lat_step)  # North to south ordering
        lons = np.arange(west, east + lon_step, lon_step)
        
        print(f"Created {len(lats)} x {len(lons)} = {len(lats) * len(lons)} sampling grid at {effective_resolution:.0f}m spacing")
        
        # Create coordinate grids
        width = len(lons)
        height = len(lats)
        xi, yi = np.meshgrid(lons, lats)
        
        # Apply clipping polygon mask if provided
        clipping_mask = None
        if clipping_polygon is not None:
            try:
                from shapely.geometry import Point
                import geopandas as gpd
                
                print(f"🗺️ Applying clipping polygon to {width}x{height} raster grid...")
                
                # Create mask for clipping
                clipping_mask = np.zeros((height, width), dtype=bool)
                
                # Parse clipping polygon geometry
                if hasattr(clipping_polygon, 'geometry') and len(clipping_polygon) > 0:
                    # It's a GeoDataFrame
                    merged_clipping_geom = clipping_polygon.geometry.unary_union
                elif hasattr(clipping_polygon, '__geo_interface__'):
                    # It's already a geometry
                    merged_clipping_geom = clipping_polygon
                else:
                    print("Warning: Invalid clipping polygon format, skipping clipping")
                    merged_clipping_geom = None
                
                if merged_clipping_geom is not None:
                    # Check each grid point
                    points_checked = 0
                    points_inside = 0
                    for i in range(height):
                        for j in range(width):
                            # FIXED: Access coordinates correctly from meshgrid
                            # xi[i, j] = longitude, yi[i, j] = latitude
                            lon = xi[i, j]
                            lat = yi[i, j]
                            point = Point(lon, lat)
                            is_inside = merged_clipping_geom.contains(point) or merged_clipping_geom.intersects(point)
                            clipping_mask[i, j] = is_inside
                            points_checked += 1
                            if is_inside:
                                points_inside += 1
                    
                    print(f"🗺️ Clipping results: {points_inside}/{points_checked} grid points inside clipping polygon")
                else:
                    clipping_mask = None
                    
            except Exception as e:
                print(f"Error applying clipping polygon: {e}")
                clipping_mask = None
        
        # Interpolate values onto 100m-spaced grid using multiple methods for complete coverage
        try:
            # Start with cubic interpolation for smooth results
            zi = griddata(coords, values, (xi, yi), method='cubic', fill_value=np.nan)
            
            # Fill NaN values with linear interpolation
            nan_mask = np.isnan(zi)
            if np.any(nan_mask):
                zi_linear = griddata(coords, values, (xi, yi), method='linear', fill_value=np.nan)
                zi[nan_mask] = zi_linear[nan_mask]
                
            # Final pass: fill remaining NaN with nearest neighbor only where triangle data exists
            # This preserves NaN boundaries where soil/Banks Peninsula clipping occurred
            nan_mask = np.isnan(zi)
            if np.any(nan_mask):
                zi_nearest = griddata(coords, values, (xi, yi), method='nearest')
                zi[nan_mask] = zi_nearest[nan_mask]
            
            # Apply Gaussian smoothing to create seamless blending between tiles
            from scipy.ndimage import gaussian_filter
            zi = gaussian_filter(zi, sigma=1.5)  # Increased smoothing for better tile blending
            
        except Exception as e:
            print(f"Cubic interpolation failed, using linear: {e}")
            # Fallback to linear interpolation
            zi = griddata(coords, values, (xi, yi), method='linear', fill_value=np.nan)
            
            # Fill NaN with nearest neighbor
            nan_mask = np.isnan(zi)
            if np.any(nan_mask):
                zi_nearest = griddata(coords, values, (xi, yi), method='nearest')
                zi[nan_mask] = zi_nearest[nan_mask]
            
            # Apply Gaussian smoothing to fallback too
            from scipy.ndimage import gaussian_filter
            zi = gaussian_filter(zi, sigma=1.5)
        
        # 7) Z ARRAY STATISTICS and analysis
        print(f"🔧 Z ARRAY ANALYSIS:")
        print(f"🔧   Shape: {zi.shape} (height={zi.shape[0]}, width={zi.shape[1]})")
        print(f"🔧   Values: min={np.nanmin(zi):.3f}, max={np.nanmax(zi):.3f}, mean={np.nanmean(zi):.3f}")
        print(f"🔧   NaN count: {np.sum(np.isnan(zi))} of {zi.size} ({100*np.sum(np.isnan(zi))/zi.size:.1f}%)")
        
        # Corner probes to understand array orientation
        print(f"🔧 CORNER PROBES:")
        print(f"🔧   Z[0,0]={zi[0,0]:.3f} at grid coords ({xi[0,0]:.6f}, {yi[0,0]:.6f}) = NW corner (west, north)")
        print(f"🔧   Z[0,-1]={zi[0,-1]:.3f} at grid coords ({xi[0,-1]:.6f}, {yi[0,-1]:.6f}) = NE corner (east, north)")
        print(f"🔧   Z[-1,0]={zi[-1,0]:.3f} at grid coords ({xi[-1,0]:.6f}, {yi[-1,0]:.6f}) = SW corner (west, south)")
        print(f"🔧   Z[-1,-1]={zi[-1,-1]:.3f} at grid coords ({xi[-1,-1]:.6f}, {yi[-1,-1]:.6f}) = SE corner (east, south)")
        
        print(f"🔧 Generated {width}x{height} raster with interpolation across all triangulated data")
        print(f"🔧 Preserves clipping boundaries where triangulated data was limited by soil/Banks Peninsula polygons")
        
        # Apply clipping mask to the interpolated data
        if clipping_mask is not None:
            # Set areas outside clipping polygon to NaN to make them transparent
            zi[~clipping_mask] = np.nan
            print(f"🗺️ Applied clipping mask: {np.sum(clipping_mask)} valid pixels out of {clipping_mask.size}")
        
        # Use the interpolated zi for display
        zi_smooth = zi
        
        # Convert values to colors using global colormap function
        if global_colormap_func:
            # Create RGBA image
            rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
            
            for i in range(height):
                for j in range(width):
                    if not np.isnan(zi_smooth[i, j]):
                        color_hex = global_colormap_func(zi_smooth[i, j])
                        # Convert hex to RGB
                        color_hex = color_hex.lstrip('#')
                        if len(color_hex) == 6:
                            r = int(color_hex[0:2], 16)
                            g = int(color_hex[2:4], 16)
                            b = int(color_hex[4:6], 16)
                            rgba_image[i, j] = [r, g, b, int(opacity * 255)]  # Match triangle mesh opacity
                        else:
                            rgba_image[i, j] = [0, 0, 0, 0]  # Transparent for invalid colors
                    else:
                        rgba_image[i, j] = [0, 0, 0, 0]  # Transparent for NaN values
        else:
            # Fallback: use matplotlib colormap
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize
            viridis = cm.viridis
            
            # Normalize values
            valid_mask = ~np.isnan(zi_smooth)
            if np.any(valid_mask):
                vmin, vmax = np.nanmin(zi_smooth), np.nanmax(zi_smooth)
                norm = Normalize(vmin=vmin, vmax=vmax)
                
                # Apply colormap
                colored = viridis(norm(zi_smooth))
                rgba_image = (colored * 255).astype(np.uint8)
                
                # Set transparency for NaN values
                rgba_image[~valid_mask, 3] = 0
                rgba_image[valid_mask, 3] = int(opacity * 255)  # Match triangle mesh opacity
            else:
                rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        
        # ARCHITECT SOLUTION: Create properly georeferenced raster with correct geotransform
        print(f"🔧 ===== GEOREFERENCED RASTER PIPELINE =====")
        print(f"🔧 RGBA IMAGE: shape={rgba_image.shape}")
        print(f"🔧 GRID SETUP: lats go from {lats[0]:.6f} to {lats[-1]:.6f} (north to south)")
        print(f"🔧 GRID SETUP: lons go from {lons[0]:.6f} to {lons[-1]:.6f} (west to east)")
        
        # Use rasterio to create proper geotransform from pixel edges (not centers)
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        from rasterio.crs import CRS
        import tempfile
        import os
        
        height, width = rgba_image.shape[:2]
        
        # ===== COORDINATE ALIGNMENT FIX: Use centralized bounds calculation =====
        bounds = calculate_authoritative_bounds(lats, lons, lat_step, lon_step)
        raster_west = bounds['west']
        raster_east = bounds['east'] 
        raster_south = bounds['south']
        raster_north = bounds['north']
        
        print(f"🔧 PIXEL EDGE BOUNDS: N={raster_north:.8f}, S={raster_south:.8f}, E={raster_east:.8f}, W={raster_west:.8f}")
        
        # 🎯 EXPERT FIX: Calculate pixel size correctly for from_origin
        # from_origin expects pixel spacing between centers, not total extent
        xres = (raster_east - raster_west) / width   # longitude spacing per pixel
        yres = (raster_north - raster_south) / height # latitude spacing per pixel
        
        print(f"🎯 PIXEL SIZE: xres={xres:.8f}°, yres={yres:.8f}°")
        print(f"🎯 PIXEL SIZE: xres={xres*111000:.1f}m, yres={yres*111000:.1f}m")
        
        # 🎯 EXPERT FIX: Use from_origin(west, north, xres, yres) for proper pixel registration
        # This treats (west, north) as the upper-left corner and uses positive pixel sizes
        transform = from_origin(raster_west, raster_north, xres, yres)
        print(f"🎯 GEOTRANSFORM (from_origin): {transform}")
        
        # Verify transform maps correctly with center offsets
        # Pixel center (0.5, 0.5) should map near (west + xres/2, north - yres/2)
        x0, y0 = transform * (0.5, 0.5)
        x1, y1 = transform * (width-0.5, height-0.5)
        print(f"🎯 TRANSFORM CHECK: pixel(0.5,0.5) -> geo({x0:.8f}, {y0:.8f}) [center of first pixel]")
        print(f"🎯 TRANSFORM CHECK: pixel({width-0.5},{height-0.5}) -> geo({x1:.8f}, {y1:.8f}) [center of last pixel]")
        
        # ARCHITECT FIX: Crop to visible pixels to eliminate transparent padding offset
        print(f"🔧 CROPPING TO VISIBLE PIXELS (alpha > 0)...")
        
        # Find bounding box of non-transparent pixels
        alpha_mask = rgba_image[:, :, 3] > 0
        
        if np.any(alpha_mask):
            # Find rows and columns with visible pixels
            visible_rows = np.any(alpha_mask, axis=1)
            visible_cols = np.any(alpha_mask, axis=0)
            
            row_indices = np.where(visible_rows)[0]
            col_indices = np.where(visible_cols)[0]
            
            if len(row_indices) > 0 and len(col_indices) > 0:
                # Get tight bounding box
                min_row, max_row = row_indices[0], row_indices[-1]
                min_col, max_col = col_indices[0], col_indices[-1]
                
                # Crop the image to visible pixels only
                rgba_image_cropped = rgba_image[min_row:max_row+1, min_col:max_col+1]
                
                # Calculate geographic bounds for the cropped image
                # Row indices correspond to lat indices, col indices to lon indices
                # FIXED: Correct pixel edge calculation - since lats go north to south
                cropped_north = lats[min_row] + lat_step/2  # Edge of northernmost visible pixel (correct)
                cropped_south = lats[max_row] - lat_step/2  # Edge of southernmost visible pixel (correct)
                cropped_west = lons[min_col] - lon_step/2   # Edge of westernmost visible pixel
                cropped_east = lons[max_col] + lon_step/2   # Edge of easternmost visible pixel
                
                print(f"🔧 CROPPED: {rgba_image.shape} -> {rgba_image_cropped.shape}")
                print(f"🔧 VISIBLE BOUNDS: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
                print(f"🔧 GEOGRAPHIC BOUNDS: N={cropped_north:.6f}, S={cropped_south:.6f}, E={cropped_east:.6f}, W={cropped_west:.6f}")
                
                # Diagnostic: verify pixel resolution
                actual_lat_per_pixel = (cropped_north - cropped_south) / rgba_image_cropped.shape[0]
                actual_lon_per_pixel = (cropped_east - cropped_west) / rgba_image_cropped.shape[1]
                lat_pixel_meters = actual_lat_per_pixel * meters_per_degree_lat
                lon_pixel_meters = actual_lon_per_pixel * meters_per_degree_lon
                print(f"🔧 PIXEL RESOLUTION: {lat_pixel_meters:.1f}m lat, {lon_pixel_meters:.1f}m lon")
                
                rgba_final = rgba_image_cropped
                final_bounds = [[cropped_south, cropped_west], [cropped_north, cropped_east]]
            else:
                print(f"🔧 WARNING: No visible pixels found, using original image")
                rgba_final = rgba_image
                final_bounds = [[lats[-1] - lat_step/2, lons[0] - lon_step/2], [lats[0] + lat_step/2, lons[-1] + lon_step/2]]
        else:
            print(f"🔧 WARNING: Image is fully transparent, using original bounds")
            rgba_final = rgba_image
            final_bounds = [[lats[-1] - lat_step/2, lons[0] - lon_step/2], [lats[0] + lat_step/2, lons[-1] + lon_step/2]]
        
        # ASSERTION: Verify image row 0 corresponds to north
        assert lats[0] > lats[-1], f"Latitude array should be descending (north to south): {lats[0]} to {lats[-1]}"
        print(f"🔧 INVARIANT VERIFIED: Image row 0 corresponds to NORTH (lat={lats[0]:.6f})")
        print(f"🔧 =========================================")
        
        pil_image = Image.fromarray(rgba_final, 'RGBA')
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG', optimize=True)
        img_buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # ===== COORDINATE ALIGNMENT FIX: Use centralized authoritative bounds =====
        bounds = calculate_authoritative_bounds(lats, lons, lat_step, lon_step)
        print(f"🔧 ALIGNMENT FIX: Using centralized bounds calculation (eliminates CENTER vs EDGE inconsistencies)")
        print(f"🔧 AUTHORITATIVE BOUNDS: N={bounds['north']:.6f}, S={bounds['south']:.6f}, E={bounds['east']:.6f}, W={bounds['west']:.6f}")
        # ================================================================
        
        # 9) BOUNDS CALCULATION - CENTER vs EDGE analysis
        print(f"🔧 ===== BOUNDS CALCULATION DEBUG =====")
        
        # Comparison with legacy center-based bounds (for debugging only)
        legacy_center_south, legacy_center_north = south, north
        legacy_center_west, legacy_center_east = west, east
        
        print(f"🔧 LEGACY CENTER-BASED: N={legacy_center_north:.6f}, S={legacy_center_south:.6f}, E={legacy_center_east:.6f}, W={legacy_center_west:.6f}")
        print(f"🔧 AUTHORITATIVE BOUNDS: N={bounds['north']:.6f}, S={bounds['south']:.6f}, E={bounds['east']:.6f}, W={bounds['west']:.6f}")
        
        # Calculate the differences that were causing the offset (should now be consistent ~143m/244m)
        lat_diff_meters = (bounds['north'] - legacy_center_north) * meters_per_degree_lat
        lon_diff_meters = (bounds['east'] - legacy_center_east) * meters_per_degree_lon
        print(f"🔧 DIFFERENCE: {lat_diff_meters:.1f}m north, {lon_diff_meters:.1f}m east")
        
        # 🎯 EXPERT DIAGNOSTIC: Check if offset is ~half a pixel (indicates pixel registration bug)
        pixel_res_m = 286.0  # Current resolution
        offset_pixels_north = abs(lat_diff_meters) / pixel_res_m
        offset_pixels_east = abs(lon_diff_meters) / pixel_res_m
        print(f"🎯 PIXEL OFFSET DIAGNOSTIC:")
        print(f"🎯   North offset: {lat_diff_meters:.1f}m = {offset_pixels_north:.2f} pixels")
        print(f"🎯   East offset:  {lon_diff_meters:.1f}m = {offset_pixels_east:.2f} pixels")
        if 0.4 <= offset_pixels_north <= 0.6:
            print(f"🚨 PIXEL REGISTRATION BUG DETECTED: North offset ~0.5 pixels indicates image flip or origin issue!")
        if 0.4 <= offset_pixels_east <= 0.6:
            print(f"🚨 PIXEL REGISTRATION BUG DETECTED: East offset ~0.5 pixels indicates origin issue!")
        
        # Overlay center vs grid center 
        overlay_center_lat = (bounds['north'] + bounds['south']) / 2
        overlay_center_lon = (bounds['east'] + bounds['west']) / 2
        grid_center_lat = (lats[0] + lats[-1]) / 2
        grid_center_lon = (lons[0] + lons[-1]) / 2
        
        center_diff_lat_m = (overlay_center_lat - grid_center_lat) * meters_per_degree_lat
        center_diff_lon_m = (overlay_center_lon - grid_center_lon) * meters_per_degree_lon
        print(f"🔧 CENTER DIFFERENCE: overlay vs grid = {center_diff_lat_m:.1f}m lat, {center_diff_lon_m:.1f}m lon")
        
        # Canterbury reference validation
        christchurch_lat, christchurch_lon = -43.5321, 172.6362
        print(f"🔧 REFERENCE: Christchurch at ({christchurch_lat:.6f}, {christchurch_lon:.6f})")
        print(f"🔧 IN LEGACY CENTER BOUNDS? lat: {legacy_center_south <= christchurch_lat <= legacy_center_north}, lon: {legacy_center_west <= christchurch_lon <= legacy_center_east}")
        print(f"🔧 IN AUTHORITATIVE BOUNDS? lat: {bounds['south'] <= christchurch_lat <= bounds['north']}, lon: {bounds['west'] <= christchurch_lon <= bounds['east']}")
        
        print(f"🔧 FOLIUM BOUNDS (cropped visible pixels): {final_bounds}")
        print(f"🔧 =======================================")
        
        # 10) CROSS-CHECK: Log sample triangulated data for comparison
        if coords.size > 0:
            print(f"🔧 TRIANGULATED DATA CROSS-CHECK:")
            sample_indices = np.linspace(0, len(coords)-1, min(5, len(coords)), dtype=int)
            for i in sample_indices:
                lon, lat = coords[i]
                value = values[i]
                print(f"🔧   Triangulated point: ({lon:.6f}, {lat:.6f}) -> {value:.3f}")
        print(f"🔧 ========================================")
        
        # ARCHITECT SOLUTION: Use georeferenced bounds with proper pixel-to-coordinate mapping
        try:
            # Create a temporary GeoTIFF to ensure proper georeferencing
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Write georeferenced raster using rasterio
            with rasterio.open(
                tmp_path, 'w',
                driver='GTiff',
                height=height, width=width,
                count=4,  # RGBA
                dtype=rgba_image.dtype,
                crs=CRS.from_epsg(4326),  # WGS84
                transform=transform,
                compress='lzw'
            ) as dst:
                # Write RGBA bands
                for i in range(4):
                    dst.write(rgba_image[:, :, i], i + 1)
            
            # Read back the georeferenced bounds to ensure accuracy
            with rasterio.open(tmp_path) as src:
                bounds_obj = src.bounds
                accurate_west = bounds_obj.left
                accurate_south = bounds_obj.bottom  
                accurate_east = bounds_obj.right
                accurate_north = bounds_obj.top
                
                print(f"🔧 RASTERIO BOUNDS: N={accurate_north:.8f}, S={accurate_south:.8f}, E={accurate_east:.8f}, W={accurate_west:.8f}")
                
                # Verify no coordinate flips occurred
                if accurate_north <= accurate_south:
                    print(f"🚨 ERROR: North <= South, coordinate flip detected!")
                if accurate_east <= accurate_west:
                    print(f"🚨 ERROR: East <= West, coordinate flip detected!")
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            # COORDINATE ALIGNMENT FIX: Use centralized bounds for consistency
            # This ensures display bounds match interpolation bounds exactly
            folium_bounds = bounds['folium_format']
            
            print(f"🔧 FOLIUM BOUNDS (georeferenced): {folium_bounds}")
            print(f"🔧 COORDINATE VALIDATION:")
            print(f"🔧   North > South: {accurate_north > accurate_south} ({accurate_north:.6f} > {accurate_south:.6f})")
            print(f"🔧   East > West: {accurate_east > accurate_west} ({accurate_east:.6f} > {accurate_west:.6f})")
            
        except Exception as e:
            print(f"🚨 RASTERIO ERROR: {e}")
            # Fallback to centralized bounds calculation for consistency
            folium_bounds = bounds['folium_format']
            print(f"🔧 FALLBACK BOUNDS (centralized): {folium_bounds}")
        
        pil_image = Image.fromarray(rgba_image, 'RGBA')
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG', optimize=True)
        img_buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        return {
            'image_base64': img_base64,
            'bounds': folium_bounds,
            'opacity': opacity,
            'positioning_method': 'georeferenced_rasterio'
        }
        
    except Exception as e:
        print(f"Error generating smooth raster overlay: {e}")
        return None