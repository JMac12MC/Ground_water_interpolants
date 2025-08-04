import pandas as pd
import numpy as np
import math
import csv
import io

def get_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points in kilometers
    """
    R = 6371  # Earth radius in kilometers
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def calculate_yield_score(yield_rate, max_yield=None):
    """
    Convert yield rate to a score between 0 and 1
    """
    if max_yield is None:
        # If no max provided, use a reasonable maximum (adjust as needed)
        max_yield = 100
    
    # Normalize the yield rate
    score = min(1.0, max(0.0, yield_rate / max_yield))
    return score

def is_within_square(lat, lon, center_lat, center_lon, radius_km):
    """
    Check if a point is within a square area defined by radius from center
    For radius_km = 10, creates a 20km x 20km square
    Uses high-precision conversion factors for consistency with heatmap system
    """
    # HIGH-PRECISION COORDINATE CONVERSION - Same as interpolation.py system
    TOLERANCE_KM = 0.0001  # 10cm tolerance
    MAX_ITERATIONS = 50    # Reduced iterations for performance in checking function
    ADAPTIVE_STEP_SIZE = 0.000001
    
    def get_precise_conversion_factors(reference_lat, reference_lon):
        """Calculate ultra-precise km-to-degree conversion factors"""
        test_distance = 1.0  # 1km test distance
        
        # Ultra-precise latitude conversion
        lat_offset_initial = test_distance / 111.0
        best_lat_factor = 111.0
        best_lat_error = float('inf')
        
        for i in range(MAX_ITERATIONS):
            test_lat = reference_lat + lat_offset_initial
            actual_distance = get_distance(reference_lat, reference_lon, test_lat, reference_lon)
            error = abs(actual_distance - test_distance)
            
            current_factor = test_distance / lat_offset_initial
            if error < best_lat_error:
                best_lat_factor = current_factor
                best_lat_error = error
            
            if error < TOLERANCE_KM:
                break
                
            # Simplified adaptive refinement for performance
            if error > 0.001:
                adjustment_factor = test_distance / actual_distance  
                lat_offset_initial *= adjustment_factor
            else:
                step_size = max(ADAPTIVE_STEP_SIZE, error / 10.0)
                if actual_distance > test_distance:
                    lat_offset_initial -= step_size
                else:
                    lat_offset_initial += step_size
        
        # Ultra-precise longitude conversion
        lon_offset_initial = test_distance / (111.0 * abs(np.cos(np.radians(reference_lat))))
        best_lon_factor = 111.0 * abs(np.cos(np.radians(reference_lat)))
        best_lon_error = float('inf')
        
        for i in range(MAX_ITERATIONS):
            test_lon = reference_lon + lon_offset_initial
            actual_distance = get_distance(reference_lat, reference_lon, reference_lat, test_lon)
            error = abs(actual_distance - test_distance)
            
            current_factor = test_distance / lon_offset_initial
            if error < best_lon_error:
                best_lon_factor = current_factor
                best_lon_error = error
            
            if error < TOLERANCE_KM:
                break
                
            # Simplified adaptive refinement for performance
            if error > 0.001:
                adjustment_factor = test_distance / actual_distance
                lon_offset_initial *= adjustment_factor
            else:
                step_size = max(ADAPTIVE_STEP_SIZE, error / 10.0)
                if actual_distance > test_distance:
                    lon_offset_initial -= step_size
                else:
                    lon_offset_initial += step_size
        
        return best_lat_factor, best_lon_factor
    
    # Get precise conversion factors for this location
    km_per_degree_lat, km_per_degree_lon = get_precise_conversion_factors(center_lat, center_lon)
    
    # Convert radius to degrees using high-precision factors
    lat_radius_deg = radius_km / km_per_degree_lat
    lon_radius_deg = radius_km / km_per_degree_lon
    
    # Check if point is within square bounds
    lat_within = abs(lat - center_lat) <= lat_radius_deg
    lon_within = abs(lon - center_lon) <= lon_radius_deg
    
    return lat_within and lon_within

def download_as_csv(dataframe):
    """
    Convert a DataFrame to a CSV string for download
    """
    # Create a string buffer
    buffer = io.StringIO()
    
    # Write the DataFrame to the buffer as a CSV
    dataframe.to_csv(buffer, index=False)
    
    # Get the value of the buffer as a string
    csv_string = buffer.getvalue()
    
    return csv_string
