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
            raise ValueError("DATABASE_URL environment variable not found")

        self.engine = create_engine(self.database_url)
        self.metadata = MetaData()
        self._create_tables()

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
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
                    print("Created stored_heatmaps table")
                
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

    def store_heatmap(self, heatmap_name, center_lat, center_lon, radius_km, interpolation_method, heatmap_data, geojson_data=None, well_count=0):
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

        Returns:
        --------
        int
            The ID of the stored heatmap
        """
        try:
            # Recreate engine connection if needed
            if not hasattr(self, 'engine') or self.engine is None:
                self.engine = create_engine(self.database_url)
            
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    INSERT INTO stored_heatmaps (
                        heatmap_name, center_lat, center_lon, radius_km, 
                        interpolation_method, heatmap_data, geojson_data, well_count
                    ) VALUES (
                        :heatmap_name, :center_lat, :center_lon, :radius_km,
                        :interpolation_method, :heatmap_data, :geojson_data, :well_count
                    ) RETURNING id
                """), {
                    'heatmap_name': heatmap_name,
                    'center_lat': center_lat,
                    'center_lon': center_lon,
                    'radius_km': radius_km,
                    'interpolation_method': interpolation_method,
                    'heatmap_data': json.dumps(heatmap_data),
                    'geojson_data': json.dumps(geojson_data) if geojson_data else None,
                    'well_count': well_count
                })
                conn.commit()
                row = result.fetchone()
                if row:
                    heatmap_id = row[0]
                    print(f"Successfully stored heatmap '{heatmap_name}' with ID {heatmap_id}")
                    return heatmap_id
                else:
                    print("Failed to get heatmap ID after insert")
                    return None

        except Exception as e:
            print(f"Error storing heatmap: {e}")
            return None

    def get_all_stored_heatmaps(self):
        """
        Retrieve all stored heatmaps from the database

        Returns:
        --------
        list
            List of heatmap dictionaries
        """
        try:
            # Recreate engine connection if needed
            if not hasattr(self, 'engine') or self.engine is None:
                self.engine = create_engine(self.database_url)
            
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT id, heatmap_name, center_lat, center_lon, radius_km,
                           interpolation_method, heatmap_data, geojson_data, well_count, created_at
                    FROM stored_heatmaps
                    ORDER BY created_at DESC
                """))
                
                heatmaps = []
                for row in result:
                    # Handle JSON data that might already be parsed
                    heatmap_data = row[6]
                    if isinstance(heatmap_data, str):
                        try:
                            heatmap_data = json.loads(heatmap_data)
                        except json.JSONDecodeError:
                            heatmap_data = []
                    elif not isinstance(heatmap_data, list):
                        heatmap_data = []
                    
                    geojson_data = row[7]
                    if isinstance(geojson_data, str):
                        try:
                            geojson_data = json.loads(geojson_data)
                        except json.JSONDecodeError:
                            geojson_data = None
                    
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
                        'created_at': row[9]
                    }
                    heatmaps.append(heatmap)
                
                print(f"Successfully retrieved {len(heatmaps)} stored heatmaps")
                return heatmaps

        except Exception as e:
            print(f"Error retrieving stored heatmaps: {e}")
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
            with self.engine.connect() as conn:
                result = conn.execute(text("DELETE FROM stored_heatmaps"))
                conn.commit()
                deleted_count = result.rowcount
                print(f"Deleted {deleted_count} stored heatmaps")
                return deleted_count

        except Exception as e:
            print(f"Error deleting stored heatmaps: {e}")
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
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    DELETE FROM stored_heatmaps WHERE id = :heatmap_id
                """), {'heatmap_id': heatmap_id})
                conn.commit()
                return result.rowcount > 0

        except Exception as e:
            print(f"Error deleting heatmap: {e}")
            return False