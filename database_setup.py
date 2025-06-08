
import os
import psycopg2
from psycopg2.extras import execute_values
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import mapping
import json
import streamlit as st

def create_database_tables():
    """Create the soil_polygons table in PostgreSQL"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        st.error("DATABASE_URL environment variable not set. Please create a PostgreSQL database in Replit.")
        return False
    
    try:
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        # Create table for merged soil polygons
        cur.execute("""
            CREATE TABLE IF NOT EXISTS merged_soil_polygons (
                id SERIAL PRIMARY KEY,
                drainage_type VARCHAR(255),
                geometry_geojson TEXT,
                area_sqkm FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_drainage_type ON merged_soil_polygons(drainage_type);
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Error creating database tables: {e}")
        return False

def load_and_merge_soil_polygons():
    """Load all soil polygons, merge adjacent ones by drainage type, and store in database"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        st.error("DATABASE_URL environment variable not set.")
        return False
    
    try:
        # Set GDAL config to restore corrupted .shx files
        os.environ['SHAPE_RESTORE_SHX'] = 'YES'
        
        # Load the complete shapefile
        st.info("Loading all soil polygons from shapefile...")
        soil_gdf = gpd.read_file("attached_assets/s-map-soil-drainage-aug-2024_1749379069732.shp")
        
        # Convert to WGS84 if needed
        if soil_gdf.crs and soil_gdf.crs.to_string() != 'EPSG:4326':
            soil_gdf = soil_gdf.to_crs('EPSG:4326')
        elif not soil_gdf.crs:
            soil_gdf.crs = 'EPSG:4326'
        
        st.success(f"Loaded {len(soil_gdf)} soil polygons")
        
        # Group by drainage type and merge adjacent polygons
        st.info("Merging adjacent polygons by drainage type...")
        
        drainage_groups = {}
        if 'DRAINAGE' in soil_gdf.columns:
            for drainage_type in soil_gdf['DRAINAGE'].unique():
                if drainage_type and str(drainage_type) != 'nan':
                    group_polygons = soil_gdf[soil_gdf['DRAINAGE'] == drainage_type]
                    
                    # Merge all polygons of the same drainage type
                    merged_geometry = unary_union(group_polygons.geometry.tolist())
                    
                    # Calculate area in square kilometers
                    area_sqkm = merged_geometry.area * 111.32 * 111.32  # Rough conversion from degrees to km²
                    
                    drainage_groups[drainage_type] = {
                        'geometry': merged_geometry,
                        'area_sqkm': area_sqkm
                    }
        else:
            # If no DRAINAGE column, treat all as one group
            merged_geometry = unary_union(soil_gdf.geometry.tolist())
            area_sqkm = merged_geometry.area * 111.32 * 111.32
            drainage_groups['Unknown'] = {
                'geometry': merged_geometry,
                'area_sqkm': area_sqkm
            }
        
        st.success(f"Merged into {len(drainage_groups)} drainage type groups")
        
        # Store in database
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        # Clear existing data
        cur.execute("DELETE FROM merged_soil_polygons")
        
        # Insert merged polygons
        for drainage_type, data in drainage_groups.items():
            geometry_geojson = json.dumps(mapping(data['geometry']))
            
            cur.execute("""
                INSERT INTO merged_soil_polygons (drainage_type, geometry_geojson, area_sqkm)
                VALUES (%s, %s, %s)
            """, (drainage_type, geometry_geojson, data['area_sqkm']))
        
        conn.commit()
        cur.close()
        conn.close()
        
        st.success(f"Successfully stored {len(drainage_groups)} merged soil polygon groups in database")
        return True
        
    except Exception as e:
        st.error(f"Error processing soil polygons: {e}")
        return False

def load_soil_polygons_from_database():
    """Load merged soil polygons from database"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        return None
    
    try:
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT drainage_type, geometry_geojson, area_sqkm 
            FROM merged_soil_polygons 
            ORDER BY area_sqkm DESC
        """)
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        if results:
            # Convert back to GeoDataFrame
            data = []
            for drainage_type, geom_json, area_sqkm in results:
                geom_dict = json.loads(geom_json)
                data.append({
                    'DRAINAGE': drainage_type,
                    'area_sqkm': area_sqkm,
                    'geometry': geom_dict
                })
            
            gdf = gpd.GeoDataFrame.from_features(data, crs='EPSG:4326')
            return gdf
        
        return None
        
    except Exception as e:
        st.error(f"Error loading soil polygons from database: {e}")
        return None

if __name__ == "__main__":
    # Setup database and process polygons
    if create_database_tables():
        load_and_merge_soil_polygons()
