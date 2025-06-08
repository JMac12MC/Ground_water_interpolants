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
    
    def _create_tables(self):
        """Create the merged_polygons table if it doesn't exist"""
        try:
            # Check if table exists
            inspector = inspect(self.engine)
            if 'merged_polygons' not in inspector.get_table_names():
                # Create the table
                with self.engine.connect() as conn:
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
                    conn.commit()
                    print("Created merged_polygons table successfully")
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
                        'properties': json.loads(row[3]) if row[3] else {},
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
                        'properties': json.loads(row[3]) if row[3] else {},
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