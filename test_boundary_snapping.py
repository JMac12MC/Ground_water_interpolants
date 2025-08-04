#!/usr/bin/env python3
"""
Automated End-to-End Boundary Snapping Test
==========================================

This test verifies that boundary snapping is actually working in the 1×2 grid system.
It programmatically generates heatmaps and tracks vertex coordinate changes.

Test Methodology:
1. Generate original heatmap first, capture its boundary vertices
2. Generate east heatmap with boundary snapping enabled
3. Verify east heatmap's west boundary matches original's east boundary
4. Check final GeoJSON output for seamless triangulation
5. Report detailed findings with coordinate differences
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json

# Add current directory to path for imports
sys.path.append('.')

# Import project modules
from data_loader import load_sample_data
from interpolation import generate_geo_json_grid
from database import get_database_connection, setup_database
from utils import get_distance

class BoundarySnappingTester:
    """Automated test suite for boundary snapping verification"""
    
    def __init__(self):
        self.test_results = {}
        self.original_boundaries = None
        self.east_boundaries = None
        self.snap_log = []
        
    def setup_test_environment(self):
        """Initialize test environment with sample data"""
        print("🔧 SETTING UP TEST ENVIRONMENT")
        print("=" * 50)
        
        # Load wells data
        try:
            wells_data = load_sample_data()
            print(f"✓ Loaded {len(wells_data)} wells for testing")
            self.wells_data = wells_data
        except Exception as e:
            print(f"✗ Failed to load wells data: {e}")
            return False
            
        # Setup database
        try:
            setup_database()
            print("✓ Database initialized")
        except Exception as e:
            print(f"✗ Database setup failed: {e}")
            return False
            
        return True
        
    def extract_boundary_coordinates(self, geojson_data: Dict) -> Dict[str, float]:
        """Extract boundary coordinates from GeoJSON features"""
        if not geojson_data or 'features' not in geojson_data:
            return {}
            
        features = geojson_data['features']
        if not features:
            return {}
            
        # Extract all coordinates from all triangular features
        all_coords = []
        for feature in features:
            if feature.get('geometry', {}).get('type') == 'Polygon':
                coords = feature['geometry']['coordinates'][0]  # Get outer ring
                all_coords.extend(coords)
                
        if not all_coords:
            return {}
            
        # Calculate boundary extents
        lons = [coord[0] for coord in all_coords]
        lats = [coord[1] for coord in all_coords]
        
        boundaries = {
            'north': max(lats),
            'south': min(lats), 
            'east': max(lons),
            'west': min(lons)
        }
        
        print(f"  📐 Extracted boundaries: N={boundaries['north']:.8f}, S={boundaries['south']:.8f}")
        print(f"                          E={boundaries['east']:.8f}, W={boundaries['west']:.8f}")
        
        return boundaries
        
    def generate_test_heatmap(self, location_name: str, center_point: List[float], 
                            adjacent_boundaries: Optional[Dict] = None) -> Dict:
        """Generate a single heatmap for testing with detailed boundary tracking"""
        print(f"\n🗺️ GENERATING {location_name.upper()} HEATMAP")
        print("-" * 40)
        print(f"  Center: {center_point[0]:.6f}, {center_point[1]:.6f}")
        
        if adjacent_boundaries:
            print(f"  Adjacent boundaries to snap to: {list(adjacent_boundaries.keys())}")
            for direction, coord in adjacent_boundaries.items():
                print(f"    {direction}: {coord:.8f}")
        else:
            print(f"  No adjacent boundaries (first heatmap)")
            
        # Filter wells for this location
        search_radius = 12.5  # km
        wells_df = self.wells_data.copy()
        wells_df['within_square'] = wells_df.apply(
            lambda row: self.is_within_square(
                row['latitude'], 
                row['longitude'],
                center_point[0],
                center_point[1],
                search_radius
            ), 
            axis=1
        )
        
        filtered_wells = wells_df[wells_df['within_square']]
        print(f"  Wells found: {len(filtered_wells)}")
        
        if len(filtered_wells) == 0:
            print(f"  ✗ No wells found for {location_name}")
            return {}
            
        # Generate heatmap with boundary tracking
        try:
            # Track vertices before snapping (if applicable)
            pre_snap_boundaries = None
            if adjacent_boundaries:
                # Calculate expected boundaries before snapping
                lat, lon = center_point
                radius_km = search_radius
                km_per_deg_lat = 111.0
                km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
                
                pre_snap_boundaries = {
                    'north': lat + (radius_km / km_per_deg_lat),
                    'south': lat - (radius_km / km_per_deg_lat),
                    'east': lon + (radius_km / km_per_deg_lon),
                    'west': lon - (radius_km / km_per_deg_lon)
                }
                print(f"  📐 Pre-snap boundaries: N={pre_snap_boundaries['north']:.8f}, S={pre_snap_boundaries['south']:.8f}")
                print(f"                          E={pre_snap_boundaries['east']:.8f}, W={pre_snap_boundaries['west']:.8f}")
            
            geojson_data = generate_geo_json_grid(
                filtered_wells.copy(),
                center_point,
                search_radius,
                resolution=50,  # Lower resolution for faster testing
                method='indicator_kriging',
                show_variance=False,
                auto_fit_variogram=True,
                variogram_model='spherical',
                soil_polygons=None,  # Skip soil clipping for testing
                indicator_mask=None,
                banks_peninsula_coords=None,  # Skip Banks Peninsula clipping
                adjacent_boundaries=adjacent_boundaries,
                boundary_vertices=None
            )
            
            if not geojson_data or 'features' not in geojson_data:
                print(f"  ✗ Failed to generate GeoJSON for {location_name}")
                return {}
                
            print(f"  ✓ Generated {len(geojson_data['features'])} triangular features")
            
            # Extract final boundaries
            final_boundaries = self.extract_boundary_coordinates(geojson_data)
            
            # Analyze boundary snapping if applicable
            if adjacent_boundaries and pre_snap_boundaries:
                self.analyze_boundary_snapping(location_name, pre_snap_boundaries, 
                                             final_boundaries, adjacent_boundaries)
            
            return {
                'geojson': geojson_data,
                'boundaries': final_boundaries,
                'pre_snap_boundaries': pre_snap_boundaries,
                'wells_count': len(filtered_wells),
                'features_count': len(geojson_data['features'])
            }
            
        except Exception as e:
            print(f"  ✗ Error generating {location_name} heatmap: {e}")
            import traceback
            traceback.print_exc()
            return {}
            
    def analyze_boundary_snapping(self, location_name: str, pre_snap: Dict, 
                                final: Dict, adjacent: Dict):
        """Analyze whether boundary snapping actually occurred"""
        print(f"\n🔍 BOUNDARY SNAPPING ANALYSIS FOR {location_name.upper()}")
        print("-" * 40)
        
        snapping_detected = False
        coordinate_changes = {}
        
        for direction, target_coord in adjacent.items():
            if direction in pre_snap and direction in final:
                pre_coord = pre_snap[direction]
                final_coord = final[direction]
                
                # Calculate distances
                distance_to_target = abs(final_coord - target_coord)
                distance_moved = abs(final_coord - pre_coord)
                
                print(f"  {direction.upper()} BOUNDARY:")
                print(f"    Pre-snap:  {pre_coord:.8f}")
                print(f"    Target:    {target_coord:.8f}")
                print(f"    Final:     {final_coord:.8f}")
                print(f"    Moved:     {distance_moved:.8f} degrees ({distance_moved * 111000:.2f} meters)")
                print(f"    To target: {distance_to_target:.8f} degrees ({distance_to_target * 111000:.2f} meters)")
                
                # Check if snapping occurred
                if distance_moved > 0.0001:  # Significant movement (>10m)
                    snapping_detected = True
                    print(f"    ✓ SNAPPING DETECTED: Boundary moved significantly")
                    
                    # Check accuracy of snapping
                    if distance_to_target < 0.0001:  # Within 10m of target
                        print(f"    ✓ SNAPPING ACCURATE: Within 10m of target")
                    else:
                        print(f"    ⚠️ SNAPPING INACCURATE: {distance_to_target * 111000:.2f}m from target")
                else:
                    print(f"    ✗ NO SNAPPING: Boundary did not move significantly")
                    
                coordinate_changes[direction] = {
                    'pre_snap': pre_coord,
                    'target': target_coord,
                    'final': final_coord,
                    'moved': distance_moved,
                    'to_target': distance_to_target,
                    'snapped': distance_moved > 0.0001,
                    'accurate': distance_to_target < 0.0001
                }
                
        # Store analysis results
        self.snap_log.append({
            'location': location_name,
            'snapping_detected': snapping_detected,
            'coordinate_changes': coordinate_changes
        })
        
        if snapping_detected:
            print(f"  🎯 VERDICT: Boundary snapping is WORKING for {location_name}")
        else:
            print(f"  ❌ VERDICT: Boundary snapping NOT DETECTED for {location_name}")
            
    def test_1x2_grid_boundary_snapping(self):
        """Test boundary snapping in 1×2 grid configuration"""
        print(f"\n🧪 TESTING 1×2 GRID BOUNDARY SNAPPING")
        print("=" * 50)
        
        # Test center point (Canterbury region)
        center_lat, center_lon = -43.5321, 171.7622
        print(f"Test center: {center_lat:.6f}, {center_lon:.6f}")
        
        # Calculate east offset (19.82km spacing)
        east_offset_km = 19.82
        km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat))
        east_offset_degrees = east_offset_km / km_per_deg_lon
        
        # Define 1×2 grid locations
        locations = [
            ('original', [center_lat, center_lon]),
            ('east', [center_lat, center_lon + east_offset_degrees])
        ]
        
        print(f"East offset: {east_offset_degrees:.8f} degrees ({east_offset_km:.2f}km)")
        print(f"East heatmap center: {locations[1][1][0]:.6f}, {locations[1][1][1]:.6f}")
        
        # Generate original heatmap first
        print(f"\n📍 STEP 1: GENERATE ORIGINAL HEATMAP")
        original_result = self.generate_test_heatmap('original', locations[0][1])
        
        if not original_result:
            print("❌ FAILED: Could not generate original heatmap")
            return False
            
        self.original_boundaries = original_result['boundaries']
        
        # Generate east heatmap with boundary snapping
        print(f"\n📍 STEP 2: GENERATE EAST HEATMAP WITH BOUNDARY SNAPPING")
        adjacent_boundaries = {
            'west': self.original_boundaries['east']
        }
        
        east_result = self.generate_test_heatmap('east', locations[1][1], adjacent_boundaries)
        
        if not east_result:
            print("❌ FAILED: Could not generate east heatmap")
            return False
            
        self.east_boundaries = east_result['boundaries']
        
        # Final verification
        return self.verify_final_alignment()
        
    def verify_final_alignment(self):
        """Verify final boundary alignment between heatmaps"""
        print(f"\n🎯 FINAL BOUNDARY ALIGNMENT VERIFICATION")
        print("=" * 50)
        
        if not self.original_boundaries or not self.east_boundaries:
            print("❌ FAILED: Missing boundary data")
            return False
            
        # Check east boundary of original vs west boundary of east
        original_east = self.original_boundaries['east']
        east_west = self.east_boundaries['west']
        
        boundary_gap = abs(original_east - east_west)
        gap_meters = boundary_gap * 111000  # Convert to meters
        
        print(f"Original heatmap east boundary:  {original_east:.8f}")
        print(f"East heatmap west boundary:      {east_west:.8f}")
        print(f"Gap between boundaries:          {boundary_gap:.8f} degrees")
        print(f"Gap in meters:                   {gap_meters:.2f} meters")
        
        # Success criteria: gap < 10 meters
        if gap_meters < 10.0:
            print(f"✅ SUCCESS: Boundaries aligned within {gap_meters:.2f}m (< 10m threshold)")
            return True
        else:
            print(f"❌ FAILED: Gap of {gap_meters:.2f}m exceeds 10m threshold")
            return False
            
    def is_within_square(self, well_lat: float, well_lon: float, 
                        center_lat: float, center_lon: float, radius_km: float) -> bool:
        """Check if well is within square search area"""
        # Convert radius to degrees
        lat_offset = radius_km / 111.0
        lon_offset = radius_km / (111.0 * np.cos(np.radians(center_lat)))
        
        return (abs(well_lat - center_lat) <= lat_offset and 
                abs(well_lon - center_lon) <= lon_offset)
                
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print(f"\n📊 BOUNDARY SNAPPING TEST REPORT")
        print("=" * 50)
        
        if not self.snap_log:
            print("❌ No snapping data collected")
            return
            
        for entry in self.snap_log:
            location = entry['location']
            detected = entry['snapping_detected']
            changes = entry['coordinate_changes']
            
            print(f"\n{location.upper()} HEATMAP:")
            print(f"  Snapping detected: {'✅ YES' if detected else '❌ NO'}")
            
            for direction, data in changes.items():
                moved = data['moved'] * 111000  # meters
                accuracy = data['to_target'] * 111000  # meters
                print(f"  {direction} boundary: moved {moved:.1f}m, accuracy {accuracy:.1f}m")
                
    def run_full_test(self):
        """Run complete boundary snapping test suite"""
        print("🚀 STARTING AUTOMATED BOUNDARY SNAPPING TEST")
        print("=" * 60)
        
        # Setup
        if not self.setup_test_environment():
            print("❌ Test environment setup failed")
            return False
            
        # Run 1×2 grid test
        success = self.test_1x2_grid_boundary_snapping()
        
        # Generate report
        self.generate_test_report()
        
        # Final verdict
        print(f"\n🏁 FINAL TEST VERDICT")
        print("=" * 30)
        if success:
            print("✅ BOUNDARY SNAPPING TEST PASSED")
            print("   Heatmaps are properly aligned with seamless boundaries")
        else:
            print("❌ BOUNDARY SNAPPING TEST FAILED") 
            print("   Boundary alignment issues detected")
            
        return success

def main():
    """Main test execution"""
    tester = BoundarySnappingTester()
    return tester.run_full_test()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)