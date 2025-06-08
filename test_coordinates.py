
"""
Test coordinate transformation and geological data fetching
"""
import pyproj
from geology_service import GeologyService

def test_coordinate_transformation():
    """Test NZTM to WGS84 coordinate transformation"""
    
    # Known Canterbury locations in NZTM (EPSG:2193)
    test_points_nztm = [
        (1576237, 5183688),  # Christchurch area
        (1522359, 5204387),  # Canterbury Plains
        (1580000, 5200000),  # East Canterbury
    ]
    
    # Create transformer
    transformer = pyproj.Transformer.from_crs(
        "EPSG:2193",  # NZTM2000
        "EPSG:4326",  # WGS84
        always_xy=True
    )
    
    print("="*60)
    print("🧭 COORDINATE TRANSFORMATION TEST")
    print("="*60)
    
    for i, (nztm_x, nztm_y) in enumerate(test_points_nztm):
        # Transform to WGS84
        lon, lat = transformer.transform(nztm_x, nztm_y)
        
        print(f"Test Point {i+1}:")
        print(f"  NZTM (EPSG:2193): X={nztm_x}, Y={nztm_y}")
        print(f"  WGS84 (EPSG:4326): Lat={lat:.6f}, Lon={lon:.6f}")
        
        # Validate coordinates are in NZ bounds
        if -48 <= lat <= -34 and 166 <= lon <= 179:
            print(f"  ✅ Coordinates within New Zealand bounds")
        else:
            print(f"  ❌ Coordinates outside New Zealand bounds!")
        
        print()

def test_geological_data_fetch():
    """Test geological data fetching for known locations"""
    
    geology_service = GeologyService()
    
    # Test locations in WGS84
    test_locations = [
        (-43.5320, 172.6306, "Christchurch"),
        (-43.4, 172.0, "Canterbury Plains"),
        (-43.6, 172.7, "Banks Peninsula"),
    ]
    
    print("="*60)
    print("🪨 GEOLOGICAL DATA FETCH TEST")
    print("="*60)
    
    for lat, lon, location_name in test_locations:
        print(f"Testing {location_name}: {lat:.4f}, {lon:.4f}")
        
        try:
            polygons = geology_service.get_sedimentary_polygons(lat, lon, 10)
            
            if polygons and 'features' in polygons:
                print(f"  ✅ Found {len(polygons['features'])} geological polygons")
                
                if len(polygons['features']) > 0:
                    # Show first polygon properties
                    first_feature = polygons['features'][0]
                    props = first_feature.get('properties', {})
                    print(f"  📊 Sample properties: {list(props.keys())[:5]}")
                    
                    geom = first_feature.get('geometry', {})
                    print(f"  📐 Geometry type: {geom.get('type', 'Unknown')}")
            else:
                print(f"  ❌ No geological data found")
                
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
        
        print()

if __name__ == "__main__":
    test_coordinate_transformation()
    test_geological_data_fetch()
