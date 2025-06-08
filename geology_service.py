
import requests
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import json
import streamlit as st

class GeologyService:
    """
    Service to fetch geological data and determine if areas are suitable for groundwater
    """
    
    def __init__(self):
        # GNS Science QMAP WMS service URL - updated to new source
        self.wms_base_url = "https://services1.arcgisonline.co.nz/arcgis/rest/services/GNS/qmapgeology/MapServer"
        
        # Sedimentary unit codes that are suitable for groundwater (to be expanded based on actual data)
        self.sedimentary_codes = [
            'Q1a',  # Recent alluvium
            'Q1f',  # Recent flood deposits  
            'Q1s',  # Recent swamp deposits
            'Q2a',  # Late Pleistocene alluvium
            'Q2f',  # Late Pleistocene fan deposits
            'Q2g',  # Late Pleistocene glacial outwash
            'Q3a',  # Mid Pleistocene alluvium
            'Q3g',  # Mid Pleistocene glacial deposits
            'Q4a',  # Early Pleistocene alluvium
            'Ts',   # Tertiary sediments
            'Ms',   # Mesozoic sediments
            'Pz',   # Paleozoic sediments
            # Add more codes as needed based on actual QMAP data
        ]
        
        # Cache for geological queries
        self.geology_cache = {}
    
    def get_geology_at_point(self, lat, lon):
        """
        Get geological unit at a specific point using WMS GetFeatureInfo
        """
        cache_key = f"{lat:.6f},{lon:.6f}"
        if cache_key in self.geology_cache:
            return self.geology_cache[cache_key]
        
        try:
            # GetFeatureInfo request URL - try different layer configurations
            url = f"{self.wms_base_url}/identify"
            
            params = {
                'f': 'json',
                'geometry': f"{lon},{lat}",
                'geometryType': 'esriGeometryPoint',
                'sr': '4326',  # WGS84
                'layers': 'visible:0',  # Try visible layers
                'tolerance': 3,  # Increase tolerance
                'mapExtent': f"{lon-0.01},{lat-0.01},{lon+0.01},{lat+0.01}",
                'imageDisplay': '400,400,96',
                'returnGeometry': 'false'
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and len(data['results']) > 0:
                    # Extract geological unit from the first result
                    attributes = data['results'][0].get('attributes', {})
                    
                    # Try multiple attribute names
                    unit_code = (attributes.get('UNIT_CODE') or 
                                attributes.get('ROCK_UNIT') or 
                                attributes.get('GEOLOGY') or 
                                attributes.get('UNIT') or
                                attributes.get('FORMATION') or
                                'Unknown')
                    
                    # Debug print for first few queries
                    if len(self.geology_cache) < 5:
                        print(f"Geology at {lat:.4f}, {lon:.4f}: {unit_code} (attributes: {list(attributes.keys())})")
                    
                    self.geology_cache[cache_key] = unit_code
                    return unit_code
                else:
                    # Try alternative layer specification
                    params['layers'] = 'all'
                    response2 = requests.get(url, params=params, timeout=15)
                    if response2.status_code == 200:
                        data2 = response2.json()
                        if 'results' in data2 and len(data2['results']) > 0:
                            attributes = data2['results'][0].get('attributes', {})
                            unit_code = (attributes.get('UNIT_CODE') or 
                                        attributes.get('ROCK_UNIT') or 
                                        'Unknown')
                            self.geology_cache[cache_key] = unit_code
                            return unit_code
            
            # If API call fails, assume hard rock to be conservative with masking
            print(f"No geology data found for {lat:.4f}, {lon:.4f}, assuming hard rock")
            self.geology_cache[cache_key] = 'HARD_ROCK'
            return 'HARD_ROCK'
            
        except Exception as e:
            print(f"Error fetching geology data for {lat}, {lon}: {e}")
            # Assume hard rock on error to be conservative
            self.geology_cache[cache_key] = 'HARD_ROCK'
            return 'HARD_ROCK'
    
    def is_sedimentary(self, unit_code):
        """
        Check if a geological unit code represents sedimentary rock suitable for groundwater
        """
        if unit_code == 'Unknown':
            # If we can't determine geology, default to restricting (conservative for masking)
            return False
        
        # Convert to uppercase for consistent comparison
        unit_code = str(unit_code).upper()
        
        # Explicitly identify hard rock formations that should be masked
        hard_rock_patterns = [
            'K',    # Igneous intrusions
            'G',    # Granite
            'A',    # Andesite
            'B',    # Basalt
            'R',    # Rhyolite
            'D',    # Diorite
            'V',    # Volcanic rocks
            'I',    # Igneous
            'M',    # Metamorphic
            'S',    # Schist
            'GN',   # Gneiss
            'SL',   # Slate
        ]
        
        # Check if it's a hard rock formation
        for pattern in hard_rock_patterns:
            if unit_code.startswith(pattern):
                return False
        
        # Sedimentary patterns suitable for groundwater (expanded for nationwide NZ)
        sedimentary_patterns = [
            'Q1',   # Recent alluvium, gravels
            'Q2',   # Late Pleistocene alluvium
            'Q3',   # Mid Pleistocene alluvium
            'Q4',   # Early Pleistocene alluvium
            'QF',   # Fan deposits
            'QG',   # Glacial outwash
            'QS',   # Swamp deposits
            'QL',   # Lake deposits
            'QM',   # Marine deposits
            'QA',   # Alluvial deposits
            'QC',   # Colluvial deposits
            'QE',   # Estuarine deposits
            'T',    # Tertiary sediments
            'N',    # Neogene sediments
            'P',    # Paleogene sediments
            'C',    # Cretaceous sediments (some)
            'J',    # Jurassic sediments (some)
        ]
        
        # Check if unit code starts with any sedimentary patterns
        for pattern in sedimentary_patterns:
            if unit_code.startswith(pattern):
                return True
        
        # Also check explicit sedimentary codes
        return unit_code in self.sedimentary_codes
    
    def get_sedimentary_mask(self, center_lat, center_lon, radius_km, resolution=50):
        """
        Create a boolean mask for sedimentary areas within a given radius
        """
        # Create a grid of points to check
        km_per_degree_lat = 111.0
        km_per_degree_lon = 111.0 * np.cos(np.radians(center_lat))
        
        lat_radius = radius_km / km_per_degree_lat
        lon_radius = radius_km / km_per_degree_lon
        
        # Create grid
        lats = np.linspace(center_lat - lat_radius, center_lat + lat_radius, resolution)
        lons = np.linspace(center_lon - lon_radius, center_lon + lon_radius, resolution)
        
        # Create mask array
        mask = np.zeros((resolution, resolution), dtype=bool)
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                # Check if point is within radius
                dist_km = self._haversine_distance(center_lat, center_lon, lat, lon)
                if dist_km <= radius_km:
                    # Get geology at this point
                    unit_code = self.get_geology_at_point(lat, lon)
                    mask[i, j] = self.is_sedimentary(unit_code)
        
        return mask, lats, lons
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate haversine distance between two points in km
        """
        R = 6371  # Earth radius in km
        
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def clip_interpolation_by_geology(self, geojson_data):
        """
        Clip interpolation results by removing features in hard rock areas
        This is applied AFTER interpolation is generated
        """
        if not geojson_data or 'features' not in geojson_data:
            return geojson_data
        
        clipped_features = []
        
        for feature in geojson_data['features']:
            # Get centroid of the polygon/triangle
            coords = feature['geometry']['coordinates'][0]
            
            # Calculate centroid lat/lon
            lats = [coord[1] for coord in coords[:-1]]  # Exclude last duplicate point
            lons = [coord[0] for coord in coords[:-1]]
            centroid_lat = sum(lats) / len(lats)
            centroid_lon = sum(lons) / len(lons)
            
            # Check geology at centroid
            unit_code = self.get_geology_at_point(centroid_lat, centroid_lon)
            
            # Only keep features in sedimentary areas
            if self.is_sedimentary(unit_code):
                clipped_features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": clipped_features
        }

    def filter_wells_by_geology(self, wells_df):
        """
        Filter wells to only include those in sedimentary areas
        """
        if wells_df.empty:
            return wells_df
        
        # Add geology information to wells
        wells_df = wells_df.copy()
        wells_df['geology_unit'] = wells_df.apply(
            lambda row: self.get_geology_at_point(row['latitude'], row['longitude']),
            axis=1
        )
        
        # Filter to sedimentary areas only
        wells_df['is_sedimentary'] = wells_df['geology_unit'].apply(self.is_sedimentary)
        
        return wells_df[wells_df['is_sedimentary']].copy()
