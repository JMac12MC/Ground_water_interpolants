
import os
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, Polygon
import pandas as pd
from database import PolygonDatabase

def merge_adjacent_polygons(soil_gdf, buffer_distance=0.0001):
    """
    Merge adjacent soil polygons using spatial operations
    
    Parameters:
    -----------
    soil_gdf : GeoDataFrame
        The soil polygons to merge
    buffer_distance : float
        Small buffer to help merge polygons that are very close
    
    Returns:
    --------
    GeoDataFrame
        Merged polygons
    """
    print(f"Starting with {len(soil_gdf)} individual polygons")
    
    # Convert to a projected CRS to avoid buffer warnings
    if soil_gdf.crs and soil_gdf.crs.to_string() == 'EPSG:4326':
        # Use New Zealand Transverse Mercator 2000 for accurate area calculations
        soil_gdf = soil_gdf.to_crs('EPSG:2193')
        print("Converted to NZTM2000 projected coordinate system")
    
    # Ensure valid geometries
    soil_gdf = soil_gdf[soil_gdf.geometry.is_valid].copy()
    print(f"After removing invalid geometries: {len(soil_gdf)} polygons")
    
    # Apply small buffer to help merge touching polygons (in meters now)
    if buffer_distance > 0:
        # Use 10 meters instead of degrees
        soil_gdf['geometry'] = soil_gdf.geometry.buffer(10)
    
    # Group by drainage type if available
    if 'DRAINAGE' in soil_gdf.columns:
        grouped_polygons = []
        drainage_types = soil_gdf['DRAINAGE'].unique()
        
        for drainage_type in drainage_types:
            if pd.isna(drainage_type):
                continue
                
            print(f"Processing drainage type: {drainage_type}")
            group_polygons = soil_gdf[soil_gdf['DRAINAGE'] == drainage_type]
            
            if len(group_polygons) > 0:
                # Merge all polygons of the same drainage type
                merged_geometry = unary_union(group_polygons.geometry.tolist())
                
                # Handle both single polygons and multipolygons
                if isinstance(merged_geometry, (Polygon, MultiPolygon)):
                    # Create a new row for the merged polygon
                    merged_row = {
                        'geometry': merged_geometry,
                        'DRAINAGE': drainage_type,
                        'merged_count': len(group_polygons),
                        'total_area_km2': merged_geometry.area / 1000000  # Convert m² to km²
                    }
                    grouped_polygons.append(merged_row)
                    print(f"  Merged {len(group_polygons)} polygons into 1 for {drainage_type}")
        
        # Create new GeoDataFrame with merged polygons
        if grouped_polygons:
            merged_gdf = gpd.GeoDataFrame(grouped_polygons, crs=soil_gdf.crs)
        else:
            merged_gdf = gpd.GeoDataFrame(columns=['geometry', 'DRAINAGE', 'merged_count', 'total_area_km2'], crs=soil_gdf.crs)
    else:
        # If no drainage column, create smaller chunks to avoid memory issues
        print("No DRAINAGE column found, processing in chunks")
        chunk_size = 1000
        chunks = []
        
        for i in range(0, len(soil_gdf), chunk_size):
            chunk = soil_gdf.iloc[i:i+chunk_size]
            chunk_geometry = unary_union(chunk.geometry.tolist())
            chunks.append(chunk_geometry)
            print(f"  Processed chunk {i//chunk_size + 1}/{(len(soil_gdf)-1)//chunk_size + 1}")
        
        # Merge all chunks
        print("Merging all chunks...")
        merged_geometry = unary_union(chunks)
        
        merged_gdf = gpd.GeoDataFrame([{
            'geometry': merged_geometry,
            'DRAINAGE': 'All_Merged',
            'merged_count': len(soil_gdf),
            'total_area_km2': merged_geometry.area / 1000000  # Convert m² to km²
        }], crs=soil_gdf.crs)
    
    # Remove buffer if it was applied
    if buffer_distance > 0:
        merged_gdf['geometry'] = merged_gdf.geometry.buffer(-10)  # Remove 10m buffer
        # Ensure geometries are still valid after removing buffer
        merged_gdf = merged_gdf[merged_gdf.geometry.is_valid]
    
    # Convert back to WGS84 for storage
    merged_gdf = merged_gdf.to_crs('EPSG:4326')
    print("Converted back to WGS84 for storage")
    
    print(f"Final result: {len(merged_gdf)} merged polygons")
    return merged_gdf

def process_and_store_soil_polygons():
    """
    Main function to process soil polygons and store them in the database
    """
    try:
        # Initialize database connection
        print("Connecting to database...")
        polygon_db = PolygonDatabase()
        
        # Check if we already have processed polygons
        existing_polygons = polygon_db.get_all_polygons()
        if existing_polygons:
            print(f"Found {len(existing_polygons)} existing polygons in database")
            response = input("Do you want to reprocess and replace existing polygons? (y/n): ")
            if response.lower() != 'y':
                print("Skipping processing. Existing polygons will be used.")
                return
            else:
                print("Clearing existing polygons...")
                for polygon in existing_polygons:
                    polygon_db.delete_polygon(polygon['id'])
        
        # Load soil polygons from shapefile
        print("Loading soil polygons from shapefile...")
        
        # Set GDAL config to restore corrupted .shx files
        os.environ['SHAPE_RESTORE_SHX'] = 'YES'
        
        # Try to load the newer shapefile first, fall back to older one
        try:
            soil_gdf = gpd.read_file("attached_assets/s-map-soil-drainage-aug-2024_1749427998471.shp")
            print("Loaded newer shapefile")
        except:
            try:
                soil_gdf = gpd.read_file("attached_assets/s-map-soil-drainage-aug-2024_1749379069732.shp")
                print("Loaded older shapefile")
            except Exception as e:
                print(f"Failed to load shapefile: {e}")
                return False
        print(f"Loaded {len(soil_gdf)} soil polygons from shapefile")
        
        # Convert to WGS84 if needed
        if soil_gdf.crs and soil_gdf.crs.to_string() != 'EPSG:4326':
            print("Converting to WGS84 coordinate system...")
            soil_gdf = soil_gdf.to_crs('EPSG:4326')
        elif not soil_gdf.crs:
            soil_gdf.crs = 'EPSG:4326'
        
        # Display basic info about the data
        print(f"Columns available: {list(soil_gdf.columns)}")
        if 'DRAINAGE' in soil_gdf.columns:
            drainage_types = soil_gdf['DRAINAGE'].value_counts()
            print(f"Drainage types found: {len(drainage_types)}")
            print(drainage_types.head(10))
        
        # Merge adjacent polygons
        print("\nMerging adjacent polygons...")
        merged_gdf = merge_adjacent_polygons(soil_gdf)
        
        # Store merged polygons in database
        print("\nStoring merged polygons in database...")
        stored_count = 0
        
        for idx, row in merged_gdf.iterrows():
            try:
                # Create polygon name
                drainage_name = row.get('DRAINAGE', 'Unknown')
                polygon_name = f"Soil_Drainage_{drainage_name}_{idx}"
                
                # Create properties dictionary
                properties = {
                    'drainage_type': drainage_name,
                    'merged_count': int(row.get('merged_count', 1)),
                    'total_area_km2': float(row.get('total_area_km2', 0)),
                    'processing_date': pd.Timestamp.now().isoformat(),
                    'source': 'S-Map Soil Drainage Shapefile'
                }
                
                # Check geometry size and simplify if needed
                geometry = row.geometry
                if hasattr(geometry, 'wkt') and len(geometry.wkt) > 1000000:  # If WKT is too large
                    print(f"Simplifying large geometry for {polygon_name}")
                    # Simplify geometry to reduce size
                    geometry = geometry.simplify(0.001, preserve_topology=True)
                
                # Store in database with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Reconnect to database if needed
                        if attempt > 0:
                            polygon_db = PolygonDatabase()
                            
                        polygon_id = polygon_db.store_merged_polygon(
                            polygon_name=polygon_name,
                            geometry=geometry,
                            properties=properties
                        )
                        
                        if polygon_id:
                            stored_count += 1
                            print(f"Stored polygon {stored_count}: {polygon_name} (ID: {polygon_id})")
                            break
                        else:
                            print(f"Failed to store polygon: {polygon_name}")
                            
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"Attempt {attempt + 1} failed for {polygon_name}, retrying...")
                            import time
                            time.sleep(2)  # Wait 2 seconds before retry
                        else:
                            print(f"Failed to store polygon after {max_retries} attempts: {polygon_name} - {e}")
                            
            except Exception as e:
                print(f"Error processing polygon {idx}: {e}")
                continue
        
        print(f"\n✅ Successfully processed and stored {stored_count} merged soil polygons in database!")
        
        # Display summary statistics
        try:
            stats = polygon_db.get_polygon_statistics()
            print(f"\nDatabase Summary:")
            print(f"- Total polygons: {stats.get('total_polygons', 0)}")
            if stats.get('avg_area_km2') is not None:
                print(f"- Average area: {stats['avg_area_km2']:.2f} km²")
            else:
                print("- Average area: No data available")
        except Exception as e:
            print(f"Error getting database statistics: {e}")
        
        return stored_count > 0
        
    except Exception as e:
        print(f"❌ Error processing soil polygons: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🌱 Soil Polygon Processing and Merging Tool")
    print("=" * 50)
    
    # Check if database is available
    if not os.environ.get('DATABASE_URL'):
        print("❌ DATABASE_URL environment variable not found!")
        print("Please create a PostgreSQL database in Replit first.")
        exit(1)
    
    # Run the processing
    success = process_and_store_soil_polygons()
    
    if success:
        print("\n🎉 Processing complete! The merged polygons are now stored in your database.")
        print("Your Streamlit app will now load these merged polygons much faster.")
    else:
        print("\n❌ Processing failed. Check the error messages above.")
