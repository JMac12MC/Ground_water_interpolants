"""
Covariate Processing Module for Regression Kriging and Quantile Regression Forests

This module generates environmental covariates from spatial data sources:
- River distance (horizontal)
- Soil/rock type (one-hot encoded)
- Artificial zero-DTW points along rivers
- Coordinates (Easting/Northing)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')


def generate_artificial_zeros(river_centerlines, num_points='adaptive', buffer_meters=50):
    """
    Generate artificial zero-DTW (depth-to-water) points along river buffers.
    
    Based on the attached code examples but with adaptive point generation
    based on total river length rather than fixed count.
    
    Parameters:
    -----------
    river_centerlines : GeoDataFrame
        River centerline features in NZTM2000 (EPSG:2193)
    num_points : int or 'adaptive'
        Number of points to generate. If 'adaptive', generates points based on
        river length (1 point per 500m of river)
    buffer_meters : float
        Buffer distance around rivers in meters (default 50m)
    
    Returns:
    --------
    GeoDataFrame
        Points with DTW=0.0 sampled inside river buffers
    """
    try:
        # Ensure CRS is NZTM2000
        if river_centerlines.crs != 'EPSG:2193':
            river_centerlines = river_centerlines.to_crs('EPSG:2193')
        
        # Buffer rivers
        river_buffers = river_centerlines.copy()
        river_buffers['geometry'] = river_buffers.geometry.buffer(buffer_meters)
        
        # Calculate adaptive number of points if requested
        if num_points == 'adaptive':
            total_length_m = river_centerlines.geometry.length.sum()
            num_points = int(total_length_m / 500.0)  # 1 point per 500m
            num_points = max(500, min(num_points, 5000))  # Clamp between 500-5000
            print(f"🌊 ADAPTIVE ZEROS: Generating {num_points} points for {total_length_m/1000:.1f}km of rivers")
        
        # Generate random points inside buffers
        zeros = []
        np.random.seed(42)
        
        attempts = 0
        max_attempts = num_points * 10
        
        while len(zeros) < num_points and attempts < max_attempts:
            # Random buffer selection weighted by area
            buffer_idx = np.random.randint(0, len(river_buffers))
            geom = river_buffers.iloc[buffer_idx].geometry
            
            if geom.is_empty:
                attempts += 1
                continue
            
            # Generate random point in bounding box
            bounds = geom.bounds
            x = np.random.uniform(bounds[0], bounds[2])
            y = np.random.uniform(bounds[1], bounds[3])
            pt = Point(x, y)
            
            # Check if point is inside buffer
            if geom.contains(pt):
                zeros.append(pt)
            
            attempts += 1
        
        # Create GeoDataFrame
        zeros_gdf = gpd.GeoDataFrame(
            {'DTW': [0.0] * len(zeros), 'source': ['artificial_river'] * len(zeros)},
            geometry=zeros,
            crs='EPSG:2193'
        )
        
        print(f"✅ Generated {len(zeros)} artificial zero-DTW points in {attempts} attempts")
        return zeros_gdf
        
    except Exception as e:
        print(f"❌ Error generating artificial zeros: {e}")
        return None


def compute_river_distance(points_gdf, river_centerlines):
    """
    Compute horizontal distance from points to nearest river.
    
    Parameters:
    -----------
    points_gdf : GeoDataFrame
        Points where distance should be computed (NZTM2000)
    river_centerlines : GeoDataFrame
        River centerline features (NZTM2000)
    
    Returns:
    --------
    np.ndarray
        Distance in meters to nearest river for each point
    """
    try:
        # Ensure same CRS
        if points_gdf.crs != 'EPSG:2193':
            points_gdf = points_gdf.to_crs('EPSG:2193')
        if river_centerlines.crs != 'EPSG:2193':
            river_centerlines = river_centerlines.to_crs('EPSG:2193')
        
        # Merge all rivers into single geometry for faster distance calc
        rivers_union = unary_union(river_centerlines.geometry)
        
        # Compute distances
        distances = points_gdf.geometry.distance(rivers_union).values
        
        print(f"📏 River distances: min={distances.min():.1f}m, max={distances.max():.1f}m, mean={distances.mean():.1f}m")
        return distances
        
    except Exception as e:
        print(f"❌ Error computing river distance: {e}")
        return np.zeros(len(points_gdf))


def extract_soil_rock_types(points_gdf, soil_rock_polygons, type_column='rock_type'):
    """
    Extract soil/rock type at each point location via spatial join.
    
    Parameters:
    -----------
    points_gdf : GeoDataFrame
        Points where soil/rock type should be extracted
    soil_rock_polygons : GeoDataFrame
        Polygon features with soil/rock classification
    type_column : str
        Column name containing rock/soil type labels
    
    Returns:
    --------
    pd.DataFrame
        One-hot encoded soil/rock types
    """
    try:
        # Ensure same CRS
        if points_gdf.crs != soil_rock_polygons.crs:
            soil_rock_polygons = soil_rock_polygons.to_crs(points_gdf.crs)
        
        # Auto-detect type column if the default doesn't exist
        if type_column not in soil_rock_polygons.columns:
            # Common column names for rock/soil type classification
            possible_columns = ['rock_type', 'ROCK_TYPE', 'geology', 'GEOLOGY', 
                              'soil_type', 'SOIL_TYPE', 'lithology', 'LITHOLOGY',
                              'type', 'TYPE', 'class', 'CLASS', 'name', 'NAME']
            
            for col in possible_columns:
                if col in soil_rock_polygons.columns:
                    type_column = col
                    print(f"🔍 Auto-detected soil/rock type column: '{type_column}'")
                    break
            else:
                # If no suitable column found, use first non-geometry column
                non_geom_cols = [c for c in soil_rock_polygons.columns if c != 'geometry']
                if non_geom_cols:
                    type_column = non_geom_cols[0]
                    print(f"⚠️ Using first available column '{type_column}' for rock/soil types")
                else:
                    print(f"❌ No suitable type column found in soil/rock polygons")
                    return pd.DataFrame({'rock_unknown': np.ones(len(points_gdf))})
        
        # Spatial join to get rock type at each point
        joined = gpd.sjoin(points_gdf, soil_rock_polygons[[type_column, 'geometry']], 
                          how='left', predicate='within')
        
        # Get rock types (fill NaN with 'unknown')
        rock_types = joined[type_column].fillna('unknown')
        
        # One-hot encode
        one_hot = pd.get_dummies(rock_types, prefix='rock')
        
        print(f"🪨 Soil/Rock types found: {rock_types.nunique()} unique types")
        print(f"   Categories: {', '.join(rock_types.value_counts().head(5).index.tolist())}")
        
        return one_hot
        
    except Exception as e:
        print(f"❌ Error extracting soil/rock types: {e}")
        # Return dummy encoding
        return pd.DataFrame({'rock_unknown': np.ones(len(points_gdf))})


def build_covariate_matrix(wells_gdf, river_centerlines=None, soil_rock_polygons=None, 
                           include_artificial_zeros=True):
    """
    Build complete feature matrix X for ML methods (RK/QRF).
    
    Combines wells + artificial zeros, then extracts all covariates.
    
    Parameters:
    -----------
    wells_gdf : GeoDataFrame
        Well locations with DTW values (NZTM2000)
    river_centerlines : GeoDataFrame or None
        River features for distance covariate
    soil_rock_polygons : GeoDataFrame or None
        Soil/rock polygons for geology covariate
    include_artificial_zeros : bool
        Whether to generate and include artificial zero-DTW points
    
    Returns:
    --------
    tuple: (X, y, training_points_gdf)
        X: Feature matrix (n_samples, n_features)
        y: Target values (DTW in meters)
        training_points_gdf: Combined wells + zeros geodataframe
    """
    try:
        # Ensure wells are in NZTM2000
        if wells_gdf.crs != 'EPSG:2193':
            wells_gdf = wells_gdf.to_crs('EPSG:2193')
        
        # Generate artificial zeros if rivers available
        training_points = wells_gdf.copy()
        
        if include_artificial_zeros and river_centerlines is not None:
            zeros = generate_artificial_zeros(river_centerlines, num_points='adaptive')
            if zeros is not None and len(zeros) > 0:
                # Ensure zeros have same columns as wells
                zeros = zeros.to_crs('EPSG:2193')
                training_points = pd.concat([wells_gdf, zeros], ignore_index=True)
                print(f"📊 Training data: {len(wells_gdf)} wells + {len(zeros)} zeros = {len(training_points)} total")
        
        # Extract coordinates (always available)
        coords = training_points.geometry.apply(lambda pt: (pt.x, pt.y))
        easting = np.array([c[0] for c in coords])
        northing = np.array([c[1] for c in coords])
        
        X_list = []
        feature_names = []
        
        # Add coordinates
        X_list.append(easting.reshape(-1, 1))
        X_list.append(northing.reshape(-1, 1))
        feature_names.extend(['easting', 'northing'])
        
        # Add river distance if available
        if river_centerlines is not None:
            dist_river = compute_river_distance(training_points, river_centerlines)
            X_list.append(dist_river.reshape(-1, 1))
            feature_names.append('dist_river')
        
        # Add soil/rock types if available
        if soil_rock_polygons is not None:
            rock_onehot = extract_soil_rock_types(training_points, soil_rock_polygons)
            X_list.append(rock_onehot.values)
            feature_names.extend(rock_onehot.columns.tolist())
        
        # Combine all features
        X = np.hstack(X_list)
        y = training_points['DTW'].values
        
        print(f"✅ Feature matrix built: {X.shape} (samples × features)")
        print(f"   Features: {', '.join(feature_names)}")
        
        return X, y, training_points
        
    except Exception as e:
        print(f"❌ Error building covariate matrix: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def build_prediction_grid_covariates(grid_points_gdf, river_centerlines=None, 
                                     soil_rock_polygons=None, training_rock_types=None):
    """
    Build feature matrix for prediction grid (same covariates as training).
    
    Parameters:
    -----------
    grid_points_gdf : GeoDataFrame
        Grid points where predictions are needed (NZTM2000)
    river_centerlines : GeoDataFrame or None
        River features
    soil_rock_polygons : GeoDataFrame or None
        Soil/rock polygons
    training_rock_types : list or None
        Rock type columns from training to ensure alignment
    
    Returns:
    --------
    np.ndarray
        Feature matrix for grid (n_grid_points, n_features)
    """
    try:
        if grid_points_gdf.crs != 'EPSG:2193':
            grid_points_gdf = grid_points_gdf.to_crs('EPSG:2193')
        
        # Extract coordinates
        coords = grid_points_gdf.geometry.apply(lambda pt: (pt.x, pt.y))
        easting = np.array([c[0] for c in coords])
        northing = np.array([c[1] for c in coords])
        
        X_list = []
        
        # Add coordinates
        X_list.append(easting.reshape(-1, 1))
        X_list.append(northing.reshape(-1, 1))
        
        # Add river distance
        if river_centerlines is not None:
            dist_river = compute_river_distance(grid_points_gdf, river_centerlines)
            X_list.append(dist_river.reshape(-1, 1))
        
        # Add soil/rock types
        if soil_rock_polygons is not None:
            rock_onehot = extract_soil_rock_types(grid_points_gdf, soil_rock_polygons)
            
            # Ensure same columns as training data
            if training_rock_types is not None:
                rock_onehot = rock_onehot.reindex(columns=training_rock_types, fill_value=0)
            
            X_list.append(rock_onehot.values)
        
        X_grid = np.hstack(X_list)
        
        print(f"✅ Prediction grid features: {X_grid.shape}")
        return X_grid
        
    except Exception as e:
        print(f"❌ Error building grid covariates: {e}")
        return None
