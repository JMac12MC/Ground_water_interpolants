import numpy as np
import math
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

def is_within_square(lat, lon, center_lat, center_lon, radius_km):
    """
    Check if a point is within a square area defined by radius from center
    For radius_km = 10, creates a 20km x 20km square
    """
    # Convert radius to degrees (approximate)
    lat_radius_deg = radius_km / 111.0  # ~111km per degree latitude
    lon_radius_deg = radius_km / (111.0 * np.cos(np.radians(center_lat)))  # adjust for longitude
    
    # Check if point is within square bounds
    lat_within = abs(lat - center_lat) <= lat_radius_deg
    lon_within = abs(lon - center_lon) <= lon_radius_deg
    
    return lat_within and lon_within

def get_red_orange_polygon_for_download(polygon_db):
    """
    Retrieve stored red/orange zone polygons from database for download
    
    Returns:
        tuple: (success: bool, data: str/None, message: str)
    """
    try:
        from green_zone_extractor import get_stored_red_orange_polygon
        
        # Retrieve the stored polygon data
        polygon_data = get_stored_red_orange_polygon(polygon_db)
        
        if polygon_data:
            import json
            
            # Handle both dict and string returns from get_stored_red_orange_polygon
            if isinstance(polygon_data, str):
                # Data is already a JSON string, validate it
                try:
                    # Parse to validate it's valid JSON
                    parsed_data = json.loads(polygon_data)
                    # Re-format with proper indentation
                    geojson_string = json.dumps(parsed_data, indent=2)
                except json.JSONDecodeError:
                    return False, None, "Invalid JSON data in stored polygon"
            else:
                # Data is a dict/object, convert to JSON string
                geojson_string = json.dumps(polygon_data, indent=2)
            
            return True, geojson_string, "Red/orange polygon data retrieved successfully"
        else:
            return False, None, "No red/orange polygon data found. Generate indicator kriging heatmaps first, then extract boundaries."
            
    except Exception as e:
        return False, None, f"Error retrieving polygon data: {str(e)}"
