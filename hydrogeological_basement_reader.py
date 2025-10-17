"""
Hydrogeological Basement TIFF Reader
Extracts exclusion zones (1-3) from depth-to-hydrogeological-basement map
"""

import rasterio
import numpy as np
from rasterio import features
from shapely.geometry import shape
from shapely.ops import unary_union
import geopandas as gpd

def load_hydrogeological_exclusions(tiff_path='attached_assets/depth-to-hydrogeological-basement-map-2019_1760681867545.tif'):
    """
    Extract zones 1-3 from hydrogeological basement TIFF as exclusion polygons.
    
    Zones 1-3: Areas without groundwater (exclude from heatmaps)
    Zones 4-7: Areas with groundwater (keep in heatmaps)
    
    Parameters:
    -----------
    tiff_path : str
        Path to the hydrogeological basement TIFF file
        
    Returns:
    --------
    geopandas.GeoDataFrame or None
        GeoDataFrame containing exclusion polygons in WGS84, with merged boundaries
    """
    try:
        print(f"🗺️ Loading hydrogeological basement exclusions from TIFF...")
        
        with rasterio.open(tiff_path) as src:
            print(f"   TIFF dimensions: {src.width} x {src.height}")
            print(f"   CRS: {src.crs}")
            print(f"   Bounds: {src.bounds}")
            
            # Read raster data
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            
            # Get data statistics (excluding NoData value -9999)
            valid_data = data[(data != -9999) & (~np.isnan(data))]
            print(f"   Value range: {valid_data.min():.2f} to {valid_data.max():.2f}")
            print(f"   Mean depth: {valid_data.mean():.2f}")
            
            # Classify depth values into 7 zones
            # Zones 1-3: Shallow/no groundwater (EXCLUDE from heatmaps)
            # Zones 4-7: Deep groundwater basement (KEEP in heatmaps)
            
            # Create zone classification based on depth
            zones = np.zeros_like(data)
            zones[data == -9999] = 0  # NoData
            
            # Count valid pixels for reporting
            valid_pixels = np.sum(data != -9999)
            
            # Bin depths into 7 zones
            # Zone 1: < 0 (negative/invalid) - EXCLUDE
            # Zone 2: 0-2m (very shallow) - EXCLUDE
            # Zone 3: 2-3.5m (shallow) - EXCLUDE
            # Zone 4: 3.5-5m (moderate) - KEEP
            # Zone 5: 5-6m (deep) - KEEP
            # Zone 6: 6-7m (very deep) - KEEP
            # Zone 7: >= 7m (extremely deep) - KEEP
            
            zones[(data < 0) & (data != -9999)] = 1  # Exclude NoData
            zones[(data >= 0) & (data < 2)] = 2
            zones[(data >= 2) & (data < 3.5)] = 3
            zones[(data >= 3.5) & (data < 5)] = 4
            zones[(data >= 5) & (data < 6)] = 5
            zones[(data >= 6) & (data < 7)] = 6
            zones[data >= 7] = 7
            
            # Report zone distribution
            for z in range(1, 8):
                count = np.sum(zones == z)
                if count > 0:
                    pct = 100 * count / valid_pixels
                    print(f"   Zone {z}: {count:,} pixels ({pct:.1f}%)")
            
            # Create exclusion mask for zones 1-3
            exclusion_mask = np.isin(zones, [1, 2, 3])
            
            # Count exclusion pixels
            exclusion_count = np.sum(exclusion_mask)
            valid_pixels = np.sum(data != -9999)
            print(f"   Exclusion zones (1-3): {exclusion_count:,} pixels ({100*exclusion_count/valid_pixels:.1f}% of valid data)")
            
            # Convert exclusion zones to polygons using rasterio features
            # This automatically merges adjacent pixels with same value
            exclusion_shapes = []
            
            for geom, value in features.shapes(
                exclusion_mask.astype(np.uint8),
                mask=exclusion_mask,
                transform=transform
            ):
                if value == 1:  # Only process exclusion areas
                    exclusion_shapes.append(shape(geom))
            
            if not exclusion_shapes:
                print("   ⚠️ No exclusion zones found")
                return None
            
            print(f"   Extracted {len(exclusion_shapes)} polygon(s) from zones 1-3")
            
            # Merge adjacent polygons automatically
            merged_exclusions = unary_union(exclusion_shapes)
            print(f"   Merged into unified exclusion geometry")
            
            # Create GeoDataFrame with proper CRS
            gdf = gpd.GeoDataFrame(
                {'geometry': [merged_exclusions], 'zone_type': ['hydrogeological_basement_exclusion']},
                crs=crs
            )
            
            # Convert to WGS84 for compatibility with existing system
            if crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
                print(f"   Reprojected from {crs} to WGS84")
            
            # Get polygon count after union
            if merged_exclusions.geom_type == 'MultiPolygon':
                poly_count = len(merged_exclusions.geoms)
            else:
                poly_count = 1
            
            print(f"✅ Loaded hydrogeological basement exclusions: {poly_count} merged polygon(s)")
            
            return gdf
            
    except FileNotFoundError:
        print(f"❌ TIFF file not found: {tiff_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading hydrogeological basement exclusions: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_combined_exclusion_union():
    """
    Get unified exclusion geometry combining red/orange zones AND hydrogeological basement zones.
    
    Returns:
    --------
    tuple or None
        (union_geometry, prepared_geometry, bounds, version) or None if error
    """
    try:
        from interpolation import get_prepared_exclusion_union
        from shapely.prepared import prep
        
        # Get existing red/orange exclusions
        red_orange_data = get_prepared_exclusion_union()
        
        # Get hydrogeological basement exclusions
        hydro_exclusions = load_hydrogeological_exclusions()
        
        geometries_to_union = []
        
        # Add red/orange exclusions if available
        if red_orange_data is not None:
            red_orange_union, _, _, _ = red_orange_data
            geometries_to_union.append(red_orange_union)
            print("🔗 Including red/orange exclusions in combined union")
        
        # Add hydrogeological basement exclusions if available
        if hydro_exclusions is not None and len(hydro_exclusions) > 0:
            hydro_geom = hydro_exclusions.geometry.iloc[0]
            geometries_to_union.append(hydro_geom)
            print("🔗 Including hydrogeological basement exclusions in combined union")
        
        if not geometries_to_union:
            print("⚠️ No exclusion zones available")
            return None
        
        # Combine all exclusions
        combined_union = unary_union(geometries_to_union)
        prepared_union = prep(combined_union)
        bounds = combined_union.bounds
        
        print(f"✅ Combined exclusion union created from {len(geometries_to_union)} layer(s)")
        
        return combined_union, prepared_union, bounds, "combined_with_hydrogeological"
        
    except Exception as e:
        print(f"❌ Error creating combined exclusion union: {e}")
        return None


if __name__ == "__main__":
    # Test the reader
    exclusions = load_hydrogeological_exclusions()
    if exclusions is not None:
        print(f"\n📊 Test Results:")
        print(f"   Exclusion GeoDataFrame shape: {exclusions.shape}")
        print(f"   CRS: {exclusions.crs}")
        print(f"   Geometry type: {exclusions.geometry.iloc[0].geom_type}")
