
import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
from sklearn.model_selection import train_test_split
import psycopg2
from sqlalchemy import create_engine, text
import geopandas as gpd
from shapely.geometry import Point, Polygon
import json
import os
from data_loader import load_nz_govt_data
from database import PolygonDatabase
import time
import gc

class HeatmapPreprocessor:
    def __init__(self, db_connection_string="postgresql://localhost:5432/groundwater"):
        self.engine = create_engine(db_connection_string)
        self.polygon_db = PolygonDatabase()
        
    def create_heatmap_tables(self):
        """Create tables for storing pre-computed heatmaps"""
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS yield_heatmap (
                    id SERIAL PRIMARY KEY,
                    latitude FLOAT NOT NULL,
                    longitude FLOAT NOT NULL,
                    yield_value FLOAT NOT NULL,
                    zone_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(latitude, longitude)
                );
                
                CREATE TABLE IF NOT EXISTS depth_heatmap (
                    id SERIAL PRIMARY KEY,
                    latitude FLOAT NOT NULL,
                    longitude FLOAT NOT NULL,
                    depth_value FLOAT NOT NULL,
                    zone_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(latitude, longitude)
                );
                
                CREATE INDEX IF NOT EXISTS idx_yield_coords ON yield_heatmap(latitude, longitude);
                CREATE INDEX IF NOT EXISTS idx_depth_coords ON depth_heatmap(latitude, longitude);
                CREATE INDEX IF NOT EXISTS idx_yield_zone ON yield_heatmap(zone_id);
                CREATE INDEX IF NOT EXISTS idx_depth_zone ON depth_heatmap(zone_id);
            """))
            conn.commit()
    
    def divide_into_zones(self, wells_data, grid_size=8):
        """Divide spatial extent into zones for processing"""
        min_lat, max_lat = wells_data['latitude'].min(), wells_data['latitude'].max()
        min_lon, max_lon = wells_data['longitude'].min(), wells_data['longitude'].max()
        
        lat_bins = np.linspace(min_lat, max_lat, grid_size + 1)
        lon_bins = np.linspace(min_lon, max_lon, grid_size + 1)
        
        zones = []
        zone_id = 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                zone = {
                    'id': zone_id,
                    'lat_min': lat_bins[i],
                    'lat_max': lat_bins[i + 1],
                    'lon_min': lon_bins[j],
                    'lon_max': lon_bins[j + 1]
                }
                zones.append(zone)
                zone_id += 1
        
        return zones
    
    def get_zone_wells(self, wells_data, zone, buffer=0.1):
        """Get wells within a zone plus buffer to avoid edge artifacts"""
        lat_min = zone['lat_min'] - buffer
        lat_max = zone['lat_max'] + buffer
        lon_min = zone['lon_min'] - buffer
        lon_max = zone['lon_max'] + buffer
        
        zone_wells = wells_data[
            (wells_data['latitude'] >= lat_min) &
            (wells_data['latitude'] <= lat_max) &
            (wells_data['longitude'] >= lon_min) &
            (wells_data['longitude'] <= lon_max)
        ].copy()
        
        return zone_wells
    
    def interpolate_zone(self, zone_wells, zone, target_field='yield_rate', grid_resolution=50):
        """Perform kriging interpolation for a single zone"""
        try:
            # Clean data
            clean_wells = zone_wells.dropna(subset=[target_field, 'latitude', 'longitude'])
            
            if len(clean_wells) < 10:
                print(f"Zone {zone['id']}: Insufficient data ({len(clean_wells)} wells)")
                return None
            
            # Prepare data
            X = clean_wells[['longitude', 'latitude']].values
            y = clean_wells[target_field].values
            
            # Create interpolation grid
            lat_grid = np.linspace(zone['lat_min'], zone['lat_max'], grid_resolution)
            lon_grid = np.linspace(zone['lon_min'], zone['lon_max'], grid_resolution)
            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
            
            # Perform kriging
            try:
                ok = OrdinaryKriging(
                    X[:, 0], X[:, 1], y,
                    variogram_model='spherical',
                    verbose=False,
                    enable_plotting=False
                )
                
                z, ss = ok.execute('grid', lon_grid, lat_grid)
                
                # Convert to point data
                points = []
                for i in range(len(lat_grid)):
                    for j in range(len(lon_grid)):
                        if not np.isnan(z[i, j]):
                            points.append({
                                'latitude': lat_mesh[i, j],
                                'longitude': lon_mesh[i, j],
                                'value': float(z[i, j]),
                                'zone_id': zone['id']
                            })
                
                print(f"Zone {zone['id']}: Generated {len(points)} interpolated points")
                return points
                
            except Exception as e:
                print(f"Zone {zone['id']}: Kriging failed, using linear fallback: {e}")
                # Fallback to simple interpolation
                from scipy.interpolate import griddata
                
                points_interp = griddata(
                    X, y, 
                    (lon_mesh, lat_mesh), 
                    method='linear',
                    fill_value=np.nan
                )
                
                points = []
                for i in range(len(lat_grid)):
                    for j in range(len(lon_grid)):
                        if not np.isnan(points_interp[i, j]):
                            points.append({
                                'latitude': lat_mesh[i, j],
                                'longitude': lon_mesh[i, j],
                                'value': float(points_interp[i, j]),
                                'zone_id': zone['id']
                            })
                
                return points
                
        except Exception as e:
            print(f"Zone {zone['id']}: Complete failure: {e}")
            return None
    
    def clip_by_soil_polygons(self, interpolated_points):
        """Clip interpolated points by soil drainage polygons"""
        if not self.polygon_db.soil_polygons:
            print("No soil polygons available for clipping")
            return interpolated_points
        
        clipped_points = []
        
        for point in interpolated_points:
            point_geom = Point(point['longitude'], point['latitude'])
            
            # Check if point is within any suitable soil polygon
            for _, polygon_row in self.polygon_db.soil_polygons.iterrows():
                if polygon_row.geometry.contains(point_geom):
                    # Check if soil type is suitable (well drained)
                    if polygon_row.get('drainage', '').lower() in ['well', 'moderately well', 'rapid']:
                        clipped_points.append(point)
                        break
        
        print(f"Clipped from {len(interpolated_points)} to {len(clipped_points)} points")
        return clipped_points
    
    def store_heatmap_data(self, heatmap_points, table_name, value_field):
        """Store interpolated heatmap data in database"""
        if not heatmap_points:
            return
        
        # Clear existing data
        with self.engine.connect() as conn:
            conn.execute(text(f"DELETE FROM {table_name}"))
            conn.commit()
        
        # Prepare data for bulk insert
        df = pd.DataFrame(heatmap_points)
        df = df.rename(columns={'value': value_field})
        
        # Bulk insert
        df.to_sql(table_name, self.engine, if_exists='append', index=False)
        print(f"Stored {len(heatmap_points)} points in {table_name}")
    
    def precompute_yield_heatmap(self, wells_data):
        """Pre-compute yield heatmap for all data"""
        print("Starting yield heatmap pre-computation...")
        
        zones = self.divide_into_zones(wells_data)
        all_points = []
        
        for zone in zones:
            print(f"Processing yield zone {zone['id']}/{len(zones)-1}")
            zone_wells = self.get_zone_wells(wells_data, zone)
            
            if len(zone_wells) > 0:
                zone_points = self.interpolate_zone(zone_wells, zone, 'yield_rate')
                if zone_points:
                    all_points.extend(zone_points)
            
            # Memory cleanup
            gc.collect()
        
        # Clip by soil polygons
        clipped_points = self.clip_by_soil_polygons(all_points)
        
        # Store in database
        self.store_heatmap_data(clipped_points, 'yield_heatmap', 'yield_value')
        
        print(f"Yield heatmap pre-computation complete: {len(clipped_points)} points")
    
    def precompute_depth_heatmap(self, wells_data):
        """Pre-compute depth heatmap for all data"""
        print("Starting depth heatmap pre-computation...")
        
        zones = self.divide_into_zones(wells_data)
        all_points = []
        
        for zone in zones:
            print(f"Processing depth zone {zone['id']}/{len(zones)-1}")
            zone_wells = self.get_zone_wells(wells_data, zone)
            
            if len(zone_wells) > 0:
                zone_points = self.interpolate_zone(zone_wells, zone, 'depth')
                if zone_points:
                    all_points.extend(zone_points)
            
            # Memory cleanup
            gc.collect()
        
        # Clip by soil polygons
        clipped_points = self.clip_by_soil_polygons(all_points)
        
        # Store in database
        self.store_heatmap_data(clipped_points, 'depth_heatmap', 'depth_value')
        
        print(f"Depth heatmap pre-computation complete: {len(clipped_points)} points")
    
    def run_full_preprocessing(self):
        """Run complete heatmap preprocessing pipeline"""
        print("Starting full heatmap preprocessing...")
        
        # Load all well data
        print("Loading well data...")
        wells_data = load_nz_govt_data()
        
        if wells_data is None or len(wells_data) == 0:
            print("No well data available!")
            return
        
        print(f"Loaded {len(wells_data)} wells")
        
        # Create database tables
        self.create_heatmap_tables()
        
        # Pre-compute both heatmaps
        self.precompute_yield_heatmap(wells_data)
        self.precompute_depth_heatmap(wells_data)
        
        print("Full preprocessing complete!")

if __name__ == "__main__":
    preprocessor = HeatmapPreprocessor()
    preprocessor.run_full_preprocessing()
