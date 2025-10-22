# Fix JSON parsing issues in database operations
import os
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text, Table, Column, Integer, String, Float, DateTime, MetaData, inspect
from sqlalchemy.dialects.postgresql import JSON
import geopandas as gpd
from shapely import wkt
import json
from datetime import datetime

class PolygonDatabase:
    def __init__(self):
        """Initialize database connection using environment variables"""
        self.database_url = os.environ.get('DATABASE_URL')
        if not self.database_url:
            print("Warning: DATABASE_URL not found, using default SQLite")
            self.database_url = 'sqlite:///groundwater.db'

        try:
            self.engine = create_engine(self.database_url)
            self.metadata = MetaData()
            self._create_tables()
            print(f"Database initialized successfully: {self.database_url}")
        except Exception as e:
            print(f"Database initialization failed: {e}")
            raise

        # PostgreSQL connection for heatmap data
        self.pg_engine = None
        try:
            heatmap_db_url = os.environ.get('HEATMAP_DATABASE_URL')
            if heatmap_db_url:
                self.pg_engine = create_engine(heatmap_db_url)
        except Exception as e:
            print(f"PostgreSQL connection not available: {e}")

    def get_heatmap_data(self, heatmap_type='yield', bounds=None):
        """Retrieve pre-computed heatmap data from database

        Args:
            heatmap_type: 'yield' or 'depth'
            bounds: dict with 'north', 'south', 'east', 'west' to filter by map bounds
        """
        if not self.pg_engine:
            return None

        table_name = f"{heatmap_type}_heatmap"
        value_field = f"{heatmap_type}_value"

        try:
            query = f"SELECT latitude, longitude, {value_field} as value FROM {table_name}"

            # Add spatial filtering if bounds provided
            if bounds:
                where_clauses = [
                    f"latitude >= {bounds['south']}",
                    f"latitude <= {bounds['north']}",
                    f"longitude >= {bounds['west']}",
                    f"longitude <= {bounds['east']}"
                ]
                query += " WHERE " + " AND ".join(where_clauses)

            # Limit results for performance
            query += " LIMIT 10000"

            df = pd.read_sql(query, self.pg_engine)
            return df.to_dict('records')

        except Exception as e:
            print(f"Error retrieving {heatmap_type} heatmap: {e}")
            return None

    def get_heatmap_bounds(self, heatmap_type='yield'):
        """Get the spatial bounds of the heatmap data"""
        if not self.pg_engine:
            return None

        table_name = f"{heatmap_type}_heatmap"

        try:
            query = f"""
                SELECT 
                    MIN(latitude) as min_lat,
                    MAX(latitude) as max_lat,
                    MIN(longitude) as min_lon,
                    MAX(longitude) as max_lon
                FROM {table_name}
            """

            result = pd.read_sql(query, self.pg_engine)
            if len(result) > 0:
                return {
                    'south': result.iloc[0]['min_lat'],
                    'north': result.iloc[0]['max_lat'],
                    'west': result.iloc[0]['min_lon'],
                    'east': result.iloc[0]['max_lon']
                }
        except Exception as e:
            print(f"Error getting heatmap bounds: {e}")
            return None

    def _create_tables(self):
        """Create the merged_polygons and stored_heatmaps tables if they don't exist"""
        try:
            # Check if tables exist
            inspector = inspect(self.engine)
            table_names = inspector.get_table_names()
            
            with self.engine.connect() as conn:
                # Create merged_polygons table
                if 'merged_polygons' not in table_names:
                    conn.execute(text("""
                        CREATE TABLE merged_polygons (
                            id SERIAL PRIMARY KEY,
                            polygon_name VARCHAR(255) NOT NULL,
                            geometry_wkt TEXT NOT NULL,
                            properties JSON,
                            area_km2 FLOAT,
                            centroid_lat FLOAT,
                            centroid_lng FLOAT,
                            well_count INTEGER DEFAULT 0,
                            avg_yield FLOAT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
                    print("Created merged_polygons table")
                
                # Create stored_heatmaps table
                if 'stored_heatmaps' not in table_names:
                    conn.execute(text("""
                        CREATE TABLE stored_heatmaps (
                            id SERIAL PRIMARY KEY,
                            heatmap_name VARCHAR(255) NOT NULL,
                            center_lat FLOAT NOT NULL,
                            center_lon FLOAT NOT NULL,
                            radius_km FLOAT NOT NULL,
                            interpolation_method VARCHAR(100) NOT NULL,
                            heatmap_data JSON NOT NULL,
                            geojson_data JSON,
                            well_count INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            colormap_metadata TEXT,
                            indicator_range FLOAT,
                            indicator_sill FLOAT,
                            indicator_nugget FLOAT,
                            indicator_auto_fit BOOLEAN DEFAULT false
                        )
                    """))
                    print("Created stored_heatmaps table")
                
                # Create saved_rasters table for storing generated raster visualizations (file-based storage)
                if 'saved_rasters' not in table_names:
                    conn.execute(text("""
                        CREATE TABLE saved_rasters (
                            id SERIAL PRIMARY KEY,
                            name VARCHAR(255) NOT NULL UNIQUE,
                            file_path VARCHAR(512) NOT NULL,
                            bounds_json TEXT NOT NULL,
                            opacity FLOAT DEFAULT 0.7,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
                    print("Created saved_rasters table")
                
                conn.commit()
        except Exception as e:
            print(f"Error creating tables: {e}")

    def store_merged_polygon(self, polygon_name, geometry, properties=None, well_data=None):
        """
        Store a merged polygon with its associated data

        Parameters:
        -----------
        polygon_name : str
            Name or identifier for the polygon
        geometry : shapely.geometry
            The polygon geometry
        properties : dict, optional
            Additional properties to store as JSON
        well_data : DataFrame, optional
            Well data associated with this polygon

        Returns:
        --------
        int
            The ID of the stored polygon
        """
        try:
            # Calculate area and centroid
            area_km2 = geometry.area * 111000 * 111000 / 1000000  # Rough conversion to km²
            centroid = geometry.centroid
            centroid_lat = centroid.y
            centroid_lng = centroid.x

            # Calculate well statistics if well_data provided
            well_count = 0
            avg_yield = None
            if well_data is not None and not well_data.empty:
                well_count = len(well_data)
                if 'yield_rate' in well_data.columns:
                    avg_yield = well_data['yield_rate'].mean()

            # Store the polygon
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    INSERT INTO merged_polygons 
                    (polygon_name, geometry_wkt, properties, area_km2, centroid_lat, centroid_lng, 
                     well_count, avg_yield, created_at, updated_at)
                    VALUES (:polygon_name, :geometry_wkt, :properties, :area_km2, :centroid_lat, 
                            :centroid_lng, :well_count, :avg_yield, :created_at, :updated_at)
                    RETURNING id
                """), {
                    'polygon_name': polygon_name,
                    'geometry_wkt': geometry.wkt,
                    'properties': json.dumps(properties) if properties else None,
                    'area_km2': area_km2,
                    'centroid_lat': centroid_lat,
                    'centroid_lng': centroid_lng,
                    'well_count': well_count,
                    'avg_yield': avg_yield,
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                })
                conn.commit()
                row = result.fetchone()
                if row:
                    polygon_id = row[0]
                    return polygon_id
                return None

        except Exception as e:
            print(f"Error storing polygon: {e}")
            return None

    def get_polygon_by_name(self, polygon_name):
        """
        Retrieve a polygon by its name

        Parameters:
        -----------
        polygon_name : str
            Name of the polygon to retrieve

        Returns:
        --------
        dict or None
            Polygon data including geometry and properties
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT * FROM merged_polygons 
                    WHERE polygon_name = :polygon_name
                    ORDER BY created_at DESC
                    LIMIT 1
                """), {'polygon_name': polygon_name})

                row = result.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'polygon_name': row[1],
                        'geometry_wkt': row[2],
                        'properties': row[3] if isinstance(row[3], dict) else (json.loads(row[3]) if row[3] else {}),
                        'area_km2': row[4],
                        'centroid_lat': row[5],
                        'centroid_lng': row[6],
                        'well_count': row[7],
                        'avg_yield': row[8],
                        'created_at': row[9],
                        'updated_at': row[10]
                    }
                return None

        except Exception as e:
            print(f"Error retrieving polygon: {e}")
            return None

    def get_all_polygons(self):
        """
        Retrieve all stored polygons

        Returns:
        --------
        list
            List of polygon dictionaries
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT * FROM merged_polygons 
                    ORDER BY created_at DESC
                """))

                polygons = []
                for row in result:
                    polygons.append({
                        'id': row[0],
                        'polygon_name': row[1],
                        'geometry_wkt': row[2],
                        'properties': row[3] if isinstance(row[3], dict) else (json.loads(row[3]) if row[3] else {}),
                        'area_km2': row[4],
                        'centroid_lat': row[5],
                        'centroid_lng': row[6],
                        'well_count': row[7],
                        'avg_yield': row[8],
                        'created_at': row[9],
                        'updated_at': row[10]
                    })
                return polygons

        except Exception as e:
            print(f"Error retrieving polygons: {e}")
            return []

    def update_polygon_well_data(self, polygon_id, well_data):
        """
        Update well statistics for a stored polygon

        Parameters:
        -----------
        polygon_id : int
            ID of the polygon to update
        well_data : DataFrame
            Updated well data
        """
        try:
            well_count = len(well_data) if well_data is not None and not well_data.empty else 0
            avg_yield = None
            if well_data is not None and not well_data.empty and 'yield_rate' in well_data.columns:
                avg_yield = well_data['yield_rate'].mean()

            with self.engine.connect() as conn:
                conn.execute(text("""
                    UPDATE merged_polygons 
                    SET well_count = :well_count, avg_yield = :avg_yield, updated_at = :updated_at
                    WHERE id = :polygon_id
                """), {
                    'well_count': well_count,
                    'avg_yield': avg_yield,
                    'updated_at': datetime.now(),
                    'polygon_id': polygon_id
                })
                conn.commit()

        except Exception as e:
            print(f"Error updating polygon well data: {e}")

    def delete_polygon(self, polygon_id):
        """
        Delete a polygon by its ID

        Parameters:
        -----------
        polygon_id : int
            ID of the polygon to delete

        Returns:
        --------
        bool
            True if deletion was successful
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    DELETE FROM merged_polygons WHERE id = :polygon_id
                """), {'polygon_id': polygon_id})
                conn.commit()
                return result.rowcount > 0

        except Exception as e:
            print(f"Error deleting polygon: {e}")
            return False

    def get_polygon_statistics(self):
        """
        Get summary statistics about stored polygons

        Returns:
        --------
        dict
            Statistics about the stored polygons
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_polygons,
                        AVG(area_km2) as avg_area_km2,
                        SUM(well_count) as total_wells,
                        AVG(avg_yield) as overall_avg_yield
                    FROM merged_polygons
                """))

                row = result.fetchone()
                if row:
                    return {
                        'total_polygons': row[0],
                        'avg_area_km2': row[1],
                        'total_wells': row[2],
                        'overall_avg_yield': row[3]
                    }
                return {
                    'total_polygons': 0,
                    'avg_area_km2': 0,
                    'total_wells': 0,
                    'overall_avg_yield': 0
                }

        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {
                'total_polygons': 0,
                'avg_area_km2': 0,
                'total_wells': 0,
                'overall_avg_yield': 0
            }

    def store_heatmap(self, heatmap_name, center_lat, center_lon, radius_km, interpolation_method, heatmap_data, geojson_data=None, well_count=0, colormap_metadata=None, indicator_range=None, indicator_sill=None, indicator_nugget=None, indicator_auto_fit=False):
        """
        Store a heatmap in the database for persistent display

        Parameters:
        -----------
        heatmap_name : str
            Name or identifier for the heatmap
        center_lat : float
            Center latitude of the heatmap
        center_lon : float
            Center longitude of the heatmap
        radius_km : float
            Search radius in km
        interpolation_method : str
            Method used for interpolation
        heatmap_data : list
            Heat map data as list of [lat, lon, value] points
        geojson_data : dict, optional
            GeoJSON data for more detailed visualization
        well_count : int
            Number of wells used in the interpolation
        colormap_metadata : dict, optional
            Metadata for consistent colormap application including global min/max values
        indicator_range : float, optional
            Variogram range parameter used for indicator kriging
        indicator_sill : float, optional
            Variogram sill parameter used for indicator kriging
        indicator_nugget : float, optional
            Variogram nugget parameter used for indicator kriging
        indicator_auto_fit : bool, optional
            Whether auto-fit variogram was used (default: False)

        Returns:
        --------
        int
            The ID of the stored heatmap
        """
        import time
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to store heatmap: {heatmap_name} (attempt {attempt + 1}/{max_retries})")
                print(f"GeoJSON data present: {bool(geojson_data)}")
                if geojson_data:
                    print(f"GeoJSON features count: {len(geojson_data.get('features', []))}")
                
                # Ensure fresh engine connection for each attempt
                if not self.database_url:
                    print("Error: No database URL available")
                    return None
                
                # Create fresh connection for each retry attempt
                self.engine = create_engine(self.database_url)
                
                with self.engine.connect() as conn:
                    # Check for existing heatmap with same name AND location to prevent TRUE duplicates
                    # Ensure coordinates are Python float types, not numpy types
                    center_lat_float = float(center_lat)
                    center_lon_float = float(center_lon)
                    
                    existing_check = conn.execute(text("""
                        SELECT id FROM stored_heatmaps 
                        WHERE heatmap_name = :heatmap_name 
                        AND ABS(center_lat - :center_lat) < 0.001 
                        AND ABS(center_lon - :center_lon) < 0.001
                    """), {
                        'heatmap_name': heatmap_name,
                        'center_lat': center_lat_float,
                        'center_lon': center_lon_float
                    })
                    
                    existing_row = existing_check.fetchone()
                    if existing_row:
                        existing_id = existing_row[0]
                        print(f"⚠️ DUPLICATE DETECTED: '{heatmap_name}' already exists with ID {existing_id}, returning existing ID")
                        return -existing_id  # Return negative ID to signal duplicate
                    
                    # Insert new heatmap if no duplicate found - include colormap metadata for consistency
                    colormap_json = json.dumps(colormap_metadata) if colormap_metadata else None
                    print(f"💾 STORING COLORMAP METADATA: {colormap_metadata}")
                    print(f"💾 STORING VARIOGRAM PARAMS: Range={indicator_range}, Sill={indicator_sill}, Nugget={indicator_nugget}, Auto-fit={indicator_auto_fit}")
                    
                    result = conn.execute(text("""
                        INSERT INTO stored_heatmaps (
                            heatmap_name, center_lat, center_lon, radius_km, 
                            interpolation_method, heatmap_data, geojson_data, well_count, colormap_metadata,
                            indicator_range, indicator_sill, indicator_nugget, indicator_auto_fit
                        ) VALUES (
                            :heatmap_name, :center_lat, :center_lon, :radius_km,
                            :interpolation_method, :heatmap_data, :geojson_data, :well_count, :colormap_metadata,
                            :indicator_range, :indicator_sill, :indicator_nugget, :indicator_auto_fit
                        ) RETURNING id
                    """), {
                        'heatmap_name': heatmap_name,
                        'center_lat': center_lat_float,
                        'center_lon': center_lon_float,
                        'radius_km': float(radius_km),
                        'interpolation_method': interpolation_method,
                        'heatmap_data': json.dumps(heatmap_data),
                        'geojson_data': json.dumps(geojson_data) if geojson_data else None,
                        'well_count': int(well_count),
                        'colormap_metadata': colormap_json,
                        'indicator_range': float(indicator_range) if indicator_range is not None else None,
                        'indicator_sill': float(indicator_sill) if indicator_sill is not None else None,
                        'indicator_nugget': float(indicator_nugget) if indicator_nugget is not None else None,
                        'indicator_auto_fit': bool(indicator_auto_fit)
                    })
                    conn.commit()
                    row = result.fetchone()
                    if row:
                        heatmap_id = row[0]
                        print(f"Successfully stored NEW heatmap '{heatmap_name}' with ID {heatmap_id}")
                        if geojson_data:
                            print(f"Stored GeoJSON with {len(geojson_data.get('features', []))} triangular features")
                        return heatmap_id
                    else:
                        print("Failed to get heatmap ID after insert")
                        return None

            except Exception as e:
                print(f"Error storing heatmap on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"Failed to store heatmap after {max_retries} attempts")
                    return None

    def get_grid_point_locations(self):
        """
        Extract grid point numbers and their coordinates from stored heatmaps
        
        Returns:
        --------
        list
            List of dicts with 'grid_point_num', 'lat', 'lon', 'heatmap_id', 'heatmap_name'
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT id, heatmap_name, center_lat, center_lon
                    FROM stored_heatmaps
                    WHERE heatmap_name LIKE '%gridpoint%'
                    ORDER BY id
                """))
                
                grid_points = []
                for row in result:
                    heatmap_id, name, lat, lon = row
                    # Extract grid point number from name like "indicator_kriging_spherical_continuous_gridpoint116_..."
                    if 'gridpoint' in name:
                        try:
                            parts = name.split('gridpoint')[1].split('_')
                            grid_num = int(parts[0])
                            grid_points.append({
                                'grid_point_num': grid_num,
                                'lat': lat,
                                'lon': lon,
                                'heatmap_id': heatmap_id,
                                'heatmap_name': name
                            })
                        except:
                            pass
                
                return grid_points
                
        except Exception as e:
            print(f"Error getting grid point locations: {e}")
            return []

    def get_all_stored_heatmaps(self):
        """
        Retrieve all stored heatmaps from the database

        Returns:
        --------
        list
            List of heatmap dictionaries
        """
        print("📊 FETCH OPERATION START: Retrieving all stored heatmaps from database")
        
        try:
            # Recreate engine connection if needed
            if not hasattr(self, 'engine') or self.engine is None:
                print("🔄 Creating new database engine connection")
                self.engine = create_engine(self.database_url)
            
            with self.engine.connect() as conn:
                print("🔍 QUERYING DATABASE: Executing SELECT query for all heatmaps")
                result = conn.execute(text("""
                    SELECT id, heatmap_name, center_lat, center_lon, radius_km,
                           interpolation_method, heatmap_data, geojson_data, well_count, created_at, colormap_metadata,
                           indicator_range, indicator_sill, indicator_nugget, indicator_auto_fit
                    FROM stored_heatmaps
                    ORDER BY created_at DESC
                """))
                
                heatmaps = []
                raw_rows = result.fetchall()
                print(f"📋 RAW QUERY RESULT: Found {len(raw_rows)} rows in database")
                
                for i, row in enumerate(raw_rows):
                    print(f"📄 PROCESSING ROW {i+1}: ID {row[0]} - {row[1]}")
                    
                    # Handle JSON data that might already be parsed
                    heatmap_data = row[6]
                    if isinstance(heatmap_data, str):
                        try:
                            heatmap_data = json.loads(heatmap_data)
                        except json.JSONDecodeError:
                            print(f"⚠️  JSON decode error for heatmap_data in ID {row[0]}")
                            heatmap_data = []
                    elif not isinstance(heatmap_data, list):
                        heatmap_data = []
                    
                    geojson_data = row[7]
                    if isinstance(geojson_data, str):
                        try:
                            geojson_data = json.loads(geojson_data)
                        except json.JSONDecodeError:
                            print(f"⚠️  JSON decode error for geojson_data in ID {row[0]}")
                            geojson_data = None
                    
                    # Validate GeoJSON structure
                    if geojson_data and isinstance(geojson_data, dict):
                        if 'features' not in geojson_data:
                            print(f"⚠️  GeoJSON missing 'features' for ID {row[0]}")
                            geojson_data = None
                        elif not isinstance(geojson_data['features'], list):
                            print(f"⚠️  GeoJSON 'features' is not a list for ID {row[0]}")
                            geojson_data = None
                        elif len(geojson_data['features']) == 0:
                            print(f"⚠️  GeoJSON has 0 features for ID {row[0]}")
                        else:
                            print(f"  ✅ Valid GeoJSON with {len(geojson_data['features'])} features for ID {row[0]}")
                    elif geojson_data is not None:
                        print(f"⚠️  Invalid GeoJSON type {type(geojson_data)} for ID {row[0]}")
                        geojson_data = None
                    
                    # Parse colormap metadata if available
                    colormap_metadata = None
                    try:
                        if len(row) > 10 and row[10]:  # colormap_metadata column
                            if isinstance(row[10], str):
                                colormap_metadata = json.loads(row[10])
                            else:
                                colormap_metadata = row[10]
                            print(f"  🎨 LOADED COLORMAP METADATA: {colormap_metadata}")
                    except (json.JSONDecodeError, IndexError) as e:
                        print(f"  ⚠️ WARNING: Could not parse colormap_metadata for {row[1]}: {e}")
                    
                    # Extract variogram parameters (indices 11, 12, 13, 14)
                    indicator_range = row[11] if len(row) > 11 else None
                    indicator_sill = row[12] if len(row) > 12 else None
                    indicator_nugget = row[13] if len(row) > 13 else None
                    indicator_auto_fit = row[14] if len(row) > 14 else False
                    
                    heatmap = {
                        'id': row[0],
                        'heatmap_name': row[1],
                        'center_lat': row[2],
                        'center_lon': row[3],
                        'radius_km': row[4],
                        'interpolation_method': row[5],
                        'heatmap_data': heatmap_data,
                        'geojson_data': geojson_data,
                        'well_count': row[8],
                        'created_at': row[9],
                        'colormap_metadata': colormap_metadata,
                        'indicator_range': indicator_range,
                        'indicator_sill': indicator_sill,
                        'indicator_nugget': indicator_nugget,
                        'indicator_auto_fit': indicator_auto_fit
                    }
                    heatmaps.append(heatmap)
                    print(f"✅ PROCESSED: Heatmap ID {row[0]} added to result list")
                
                print(f"📊 FETCH RESULT: Successfully retrieved {len(heatmaps)} stored heatmaps")
                print(f"📋 HEATMAP IDS: {[h['id'] for h in heatmaps]}")
                return heatmaps

        except Exception as e:
            print(f"❌ FETCH ERROR: Error retrieving stored heatmaps: {e}")
            import traceback
            print(f"📍 STACK TRACE: {traceback.format_exc()}")
            return []

    def delete_all_stored_heatmaps(self):
        """
        Delete all stored heatmaps from the database

        Returns:
        --------
        int
            Number of heatmaps deleted
        """
        try:
            # Recreate connection to ensure fresh database state
            if not hasattr(self, 'engine') or self.engine is None:
                self.engine = create_engine(self.database_url)
            
            with self.engine.connect() as conn:
                # Use TRUNCATE for complete clear
                result = conn.execute(text("TRUNCATE TABLE stored_heatmaps"))
                conn.commit()
                print("Truncated stored_heatmaps table - all heatmaps deleted")
                return 999  # Return high number to indicate complete clear

        except Exception as e:
            print(f"Error deleting stored heatmaps: {e}")
            # Try alternative DELETE approach
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text("DELETE FROM stored_heatmaps"))
                    conn.commit()
                    deleted_count = result.rowcount
                    print(f"Deleted {deleted_count} stored heatmaps using DELETE")
                    return deleted_count
            except Exception as e2:
                print(f"Error with DELETE approach: {e2}")
                return 0

    def delete_stored_heatmap(self, heatmap_id):
        """
        Delete a specific stored heatmap by ID

        Parameters:
        -----------
        heatmap_id : int
            ID of the heatmap to delete

        Returns:
        --------
        bool
            True if deletion was successful
        """
        print(f"🗑️ DELETE OPERATION START: Attempting to delete heatmap ID {heatmap_id}")
        
        try:
            # Ensure engine connection is available
            if not hasattr(self, 'engine') or self.engine is None:
                if not self.database_url:
                    print("❌ ERROR: No database URL available for deletion")
                    return False
                print("🔄 Recreating database engine connection")
                self.engine = create_engine(self.database_url)
            
            with self.engine.connect() as conn:
                print(f"📋 PRE-DELETE CHECK: Verifying heatmap {heatmap_id} exists")
                
                # Get all heatmaps before deletion for logging
                pre_delete_result = conn.execute(text("SELECT id, heatmap_name FROM stored_heatmaps ORDER BY id"))
                pre_delete_heatmaps = pre_delete_result.fetchall()
                print(f"📊 PRE-DELETE STATE: Found {len(pre_delete_heatmaps)} total heatmaps in database:")
                for hm in pre_delete_heatmaps:
                    print(f"   - ID {hm[0]}: {hm[1]}")
                
                # First verify the target heatmap exists
                check_result = conn.execute(text("""
                    SELECT id, heatmap_name FROM stored_heatmaps WHERE id = :heatmap_id
                """), {'heatmap_id': heatmap_id})
                
                target_heatmap = check_result.fetchone()
                if not target_heatmap:
                    print(f"❌ TARGET NOT FOUND: Heatmap with ID {heatmap_id} not found in database")
                    return False
                
                print(f"✅ TARGET FOUND: Heatmap ID {heatmap_id} ('{target_heatmap[1]}') exists and will be deleted")
                
                # Perform the deletion
                print(f"🗑️ EXECUTING DELETE: Removing heatmap ID {heatmap_id} from database")
                result = conn.execute(text("""
                    DELETE FROM stored_heatmaps WHERE id = :heatmap_id
                """), {'heatmap_id': heatmap_id})
                conn.commit()
                
                deleted_count = result.rowcount
                print(f"📊 DELETE RESULT: {deleted_count} row(s) deleted for heatmap ID {heatmap_id}")
                
                # Verify deletion by checking post-delete state
                post_delete_result = conn.execute(text("SELECT id, heatmap_name FROM stored_heatmaps ORDER BY id"))
                post_delete_heatmaps = post_delete_result.fetchall()
                print(f"📊 POST-DELETE STATE: {len(post_delete_heatmaps)} heatmaps remaining in database:")
                for hm in post_delete_heatmaps:
                    print(f"   - ID {hm[0]}: {hm[1]}")
                
                # Double-check that the target is actually gone
                verify_result = conn.execute(text("""
                    SELECT id FROM stored_heatmaps WHERE id = :heatmap_id
                """), {'heatmap_id': heatmap_id})
                still_exists = verify_result.fetchone()
                
                if still_exists:
                    print(f"❌ DELETE FAILED: Heatmap ID {heatmap_id} still exists after deletion!")
                    return False
                else:
                    print(f"✅ DELETE CONFIRMED: Heatmap ID {heatmap_id} successfully removed from database")
                    return deleted_count > 0

        except Exception as e:
            print(f"❌ DELETE ERROR: Exception during deletion of heatmap ID {heatmap_id}: {e}")
            # Try to reconnect and retry once
            try:
                print("🔄 RETRY: Attempting to reconnect and retry deletion")
                self.engine = create_engine(self.database_url)
                with self.engine.connect() as conn:
                    result = conn.execute(text("""
                        DELETE FROM stored_heatmaps WHERE id = :heatmap_id
                    """), {'heatmap_id': heatmap_id})
                    conn.commit()
                    deleted_count = result.rowcount
                    print(f"✅ RETRY SUCCESS: {deleted_count} row(s) deleted on retry")
                    return deleted_count > 0
            except Exception as retry_error:
                print(f"❌ RETRY FAILED: {retry_error}")
                return False

    def update_stored_heatmap_geojson(self, heatmap_id, geojson_data):
        """
        Update the GeoJSON data for a specific stored heatmap
        
        Parameters:
        -----------
        heatmap_id : int
            ID of the heatmap to update
        geojson_data : str
            JSON string of the updated GeoJSON data
            
        Returns:
        --------
        bool
            True if update was successful
        """
        import time
        try:
            # Ensure engine connection is available with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if not hasattr(self, 'engine') or self.engine is None:
                        if not self.database_url:
                            return False
                        self.engine = create_engine(self.database_url)
                    
                    with self.engine.connect() as conn:
                        result = conn.execute(text("""
                            UPDATE stored_heatmaps 
                            SET geojson_data = :geojson_data
                            WHERE id = :heatmap_id
                        """), {
                            'geojson_data': geojson_data,
                            'heatmap_id': heatmap_id
                        })
                        conn.commit()
                        
                        return result.rowcount > 0
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Retry {attempt + 1}: Database connection failed, retrying...")
                        time.sleep(1)
                        # Force engine recreation
                        self.engine = None
                        continue
                    else:
                        raise e
                        
        except Exception as e:
            print(f"Error updating heatmap {heatmap_id}: {e}")
            return False

    def store_polygon(self, name, polygon_type, coordinates, metadata=None):
        """
        Store a polygon in the database for persistent display
        
        Parameters:
        -----------
        name : str
            Name or identifier for the polygon
        polygon_type : str
            Type of polygon (e.g., 'indicator_boundary', 'green_zone', etc.)
        coordinates : list
            List of coordinate pairs [(lon, lat), ...]
        metadata : dict, optional
            Additional metadata for the polygon
            
        Returns:
        --------
        int
            The ID of the stored polygon, or None if failed
        """
        try:
            if not hasattr(self, 'engine') or self.engine is None:
                if not self.database_url:
                    return None
                self.engine = create_engine(self.database_url)
            
            with self.engine.connect() as conn:
                # Check if polygons table exists, if not create it
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS stored_polygons (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        polygon_type TEXT NOT NULL,
                        coordinates TEXT NOT NULL,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Insert the polygon
                result = conn.execute(text("""
                    INSERT INTO stored_polygons (name, polygon_type, coordinates, metadata)
                    VALUES (:name, :polygon_type, :coordinates, :metadata)
                    RETURNING id
                """), {
                    'name': name,
                    'polygon_type': polygon_type,
                    'coordinates': json.dumps(coordinates),
                    'metadata': json.dumps(metadata) if metadata else None
                })
                
                conn.commit()
                polygon_id = result.fetchone()[0]
                
                print(f"✅ Stored polygon '{name}' with ID {polygon_id}")
                return polygon_id
                
        except Exception as e:
            print(f"Error storing polygon: {e}")
            return None
    
    def save_raster(self, name, raster_image_base64, bounds, opacity=0.7):
        """
        Save a generated raster visualization to disk and database
        
        Parameters:
        -----------
        name : str
            Unique name for the saved raster
        raster_image_base64 : str
            Base64-encoded PNG image of the raster
        bounds : list
            Bounds [[south, west], [north, east]] for the raster overlay
        opacity : float
            Opacity level (0.0 to 1.0)
        
        Returns:
        --------
        int or None
            The ID of the saved raster, or None if failed
        """
        try:
            import base64
            import os
            from datetime import datetime
            
            # Decode the base64 image
            img_data = base64.b64decode(raster_image_base64)
            
            # Create saved_rasters directory if it doesn't exist
            os.makedirs('saved_rasters', exist_ok=True)
            
            # Generate unique filename using timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
            filename = f"{safe_name}_{timestamp}.png"
            file_path = os.path.join('saved_rasters', filename)
            
            # Save to disk
            with open(file_path, 'wb') as f:
                f.write(img_data)
            
            file_size_mb = len(img_data) / (1024 * 1024)
            print(f"💾 Saved raster to disk: {file_path} ({file_size_mb:.2f} MB)")
            
            # Save metadata to database
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    INSERT INTO saved_rasters (name, file_path, bounds_json, opacity)
                    VALUES (:name, :file_path, :bounds_json, :opacity)
                    RETURNING id
                """), {
                    'name': name,
                    'file_path': file_path,
                    'bounds_json': json.dumps(bounds),
                    'opacity': opacity
                })
                
                conn.commit()
                raster_id = result.fetchone()[0]
                print(f"✅ Saved raster '{name}' with ID {raster_id}")
                return raster_id
                
        except Exception as e:
            print(f"❌ Error saving raster: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_all_saved_rasters(self):
        """
        Retrieve all saved rasters sorted by creation date (newest first)
        
        Returns:
        --------
        list of dict
            List of saved rasters with id, name, and created_at
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT id, name, created_at
                    FROM saved_rasters
                    ORDER BY created_at DESC
                """))
                
                rasters = []
                for row in result:
                    rasters.append({
                        'id': row[0],
                        'name': row[1],
                        'created_at': row[2]
                    })
                
                return rasters
                
        except Exception as e:
            print(f"Error retrieving saved rasters: {e}")
            return []
    
    def load_raster(self, raster_id):
        """
        Load a saved raster from disk by ID
        
        Parameters:
        -----------
        raster_id : int
            ID of the raster to load
        
        Returns:
        --------
        dict or None
            Dictionary with raster_image_base64, bounds, opacity, name
        """
        try:
            import base64
            import os
            
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT name, file_path, bounds_json, opacity
                    FROM saved_rasters
                    WHERE id = :raster_id
                """), {'raster_id': raster_id})
                
                row = result.fetchone()
                if row:
                    name = row[0]
                    file_path = row[1]
                    bounds = json.loads(row[2])
                    opacity = row[3]
                    
                    # Load image from disk
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            img_data = f.read()
                        raster_image_base64 = base64.b64encode(img_data).decode('utf-8')
                        
                        print(f"📂 Loaded raster from disk: {file_path}")
                        return {
                            'name': name,
                            'raster_image_base64': raster_image_base64,
                            'bounds': bounds,
                            'opacity': opacity
                        }
                    else:
                        print(f"❌ Raster file not found: {file_path}")
                        return None
                return None
                
        except Exception as e:
            print(f"Error loading raster: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def delete_raster(self, raster_id):
        """
        Delete a saved raster file and database entry by ID
        
        Parameters:
        -----------
        raster_id : int
            ID of the raster to delete
        
        Returns:
        --------
        bool
            True if deleted successfully, False otherwise
        """
        try:
            import os
            
            # Get file path before deleting from database
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT file_path FROM saved_rasters WHERE id = :raster_id
                """), {'raster_id': raster_id})
                row = result.fetchone()
                
                if row:
                    file_path = row[0]
                    
                    # Delete from database first
                    conn.execute(text("""
                        DELETE FROM saved_rasters WHERE id = :raster_id
                    """), {'raster_id': raster_id})
                    conn.commit()
                    
                    # Delete file from disk
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"🗑️ Deleted raster file: {file_path}")
                    
                    print(f"✅ Deleted raster ID {raster_id}")
                    return True
                else:
                    print(f"Raster ID {raster_id} not found")
                    return False
                
        except Exception as e:
            print(f"Error deleting raster: {e}")
            import traceback
            traceback.print_exc()
            return False