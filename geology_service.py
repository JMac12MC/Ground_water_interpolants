
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
            
            # Very short timeout to prevent stalling
            response = requests.get(url, params=params, timeout=2)
            
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
                                'SEDIMENTARY')  # Default to sedimentary to avoid blocking
                    
                    # Debug print for first few queries only
                    if len(self.geology_cache) < 3:
                        print(f"Geology at {lat:.4f}, {lon:.4f}: {unit_code}")
                    
                    self.geology_cache[cache_key] = unit_code
                    return unit_code
            
            # If API call fails, default to sedimentary to avoid blocking interpolation
            self.geology_cache[cache_key] = 'SEDIMENTARY'
            return 'SEDIMENTARY'
            
        except Exception as e:
            # On any error, default to sedimentary to avoid blocking
            if len(self.geology_cache) < 3:
                print(f"Geology API error for {lat:.4f}, {lon:.4f}: {e}, defaulting to sedimentary")
            self.geology_cache[cache_key] = 'SEDIMENTARY'
            return 'SEDIMENTARY'
    
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
    
    def get_sedimentary_polygons(self, center_lat, center_lon, radius_km):
        """
        Fetch sedimentary geological polygons from GNS Science QMAP service
        Returns GeoJSON of sedimentary areas only
        """
        try:
            # Calculate bounding box for the search area
            km_per_degree_lat = 111.0
            km_per_degree_lon = 111.0 * np.cos(np.radians(center_lat))
            
            lat_radius = radius_km / km_per_degree_lat
            lon_radius = radius_km / km_per_degree_lon
            
            min_lat = center_lat - lat_radius
            max_lat = center_lat + lat_radius
            min_lon = center_lon - lon_radius
            max_lon = center_lon + lon_radius
            
            # Query QMAP for geological polygons in the area
            url = f"{self.wms_base_url}/query"
            
            params = {
                'f': 'json',
                'where': '1=1',  # Get all features
                'outFields': '*',
                'returnGeometry': 'true',
                'spatialRel': 'esriSpatialRelIntersects',
                'geometryType': 'esriGeometryEnvelope',
                'geometry': f"{min_lon},{min_lat},{max_lon},{max_lat}",
                'inSR': '4326',
                'outSR': '4326'
            }
            
            print(f"Fetching geological polygons for area...")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'features' in data and len(data['features']) > 0:
                    # Filter for sedimentary features only
                    sedimentary_features = []
                    
                    for feature in data['features']:
                        try:
                            attributes = feature.get('attributes', {})
                            
                            # Get unit code from various possible field names
                            unit_code = (attributes.get('UNIT_CODE') or 
                                        attributes.get('ROCK_UNIT') or 
                                        attributes.get('GEOLOGY') or 
                                        attributes.get('UNIT') or
                                        attributes.get('FORMATION') or
                                        'Unknown')
                            
                            # Only keep sedimentary polygons
                            if self.is_sedimentary(unit_code):
                                sedimentary_features.append(feature)
                                
                        except Exception as e:
                            print(f"Error processing geological feature: {e}")
                            continue
                    
                    print(f"Found {len(sedimentary_features)} sedimentary polygons out of {len(data['features'])} total geological features")
                    
                    return {
                        "type": "FeatureCollection",
                        "features": sedimentary_features
                    }
                else:
                    print("No geological features found in the area")
                    return None
            else:
                print(f"Failed to fetch geological data: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching geological polygons: {e}")
            return None

    def clip_interpolation_by_polygons(self, interpolation_geojson, geological_polygons):
        """
        Clip interpolation results using actual geological polygon boundaries
        This uses geometric intersection to properly mask the interpolated surface
        """
        if not interpolation_geojson or not geological_polygons:
            print("No interpolation data or geological polygons available for clipping")
            return interpolation_geojson
        
        if 'features' not in interpolation_geojson or len(interpolation_geojson['features']) == 0:
            return interpolation_geojson
            
        if 'features' not in geological_polygons or len(geological_polygons['features']) == 0:
            print("No sedimentary polygons found - returning unclipped interpolation")
            return interpolation_geojson
        
        try:
            from shapely.geometry import Polygon, MultiPolygon, Point
            from shapely.ops import unary_union
            import json
            
            print(f"Clipping {len(interpolation_geojson['features'])} interpolation features using {len(geological_polygons['features'])} sedimentary polygons...")
            
            # Convert geological polygons to Shapely geometries and union them
            sedimentary_polygons = []
            
            for feature in geological_polygons['features']:
                try:
                    geom = feature['geometry']
                    if geom['type'] == 'Polygon':
                        coords = geom['coordinates']
                        if len(coords) > 0 and len(coords[0]) >= 4:  # Valid polygon
                            poly = Polygon(coords[0])
                            if poly.is_valid:
                                sedimentary_polygons.append(poly)
                    elif geom['type'] == 'MultiPolygon':
                        for poly_coords in geom['coordinates']:
                            if len(poly_coords) > 0 and len(poly_coords[0]) >= 4:
                                poly = Polygon(poly_coords[0])
                                if poly.is_valid:
                                    sedimentary_polygons.append(poly)
                except Exception as e:
                    print(f"Error processing geological polygon: {e}")
                    continue
            
            if not sedimentary_polygons:
                print("No valid sedimentary polygons found")
                return interpolation_geojson
            
            # Union all sedimentary polygons into a single mask
            print("Creating sedimentary mask...")
            try:
                sedimentary_mask = unary_union(sedimentary_polygons)
            except Exception as e:
                print(f"Error creating sedimentary mask: {e}")
                # Fallback: use individual polygons
                sedimentary_mask = sedimentary_polygons
            
            # Clip interpolation features
            clipped_features = []
            
            for i, feature in enumerate(interpolation_geojson['features']):
                try:
                    # Convert interpolation feature to Shapely geometry
                    geom = feature['geometry']
                    if geom['type'] == 'Polygon':
                        coords = geom['coordinates']
                        if len(coords) > 0 and len(coords[0]) >= 4:
                            interp_poly = Polygon(coords[0])
                            
                            if interp_poly.is_valid:
                                # Check intersection with sedimentary mask
                                if isinstance(sedimentary_mask, (Polygon, MultiPolygon)):
                                    if sedimentary_mask.intersects(interp_poly):
                                        # Calculate intersection
                                        intersection = sedimentary_mask.intersection(interp_poly)
                                        
                                        if not intersection.is_empty and intersection.area > 0:
                                            # Convert back to GeoJSON feature
                                            if intersection.geom_type == 'Polygon':
                                                new_coords = [list(intersection.exterior.coords)]
                                                clipped_feature = {
                                                    "type": "Feature",
                                                    "geometry": {
                                                        "type": "Polygon",
                                                        "coordinates": new_coords
                                                    },
                                                    "properties": feature['properties'].copy()
                                                }
                                                clipped_features.append(clipped_feature)
                                            elif intersection.geom_type == 'MultiPolygon':
                                                # Handle MultiPolygon results
                                                for poly in intersection.geoms:
                                                    if poly.area > 0:
                                                        new_coords = [list(poly.exterior.coords)]
                                                        clipped_feature = {
                                                            "type": "Feature",
                                                            "geometry": {
                                                                "type": "Polygon",
                                                                "coordinates": new_coords
                                                            },
                                                            "properties": feature['properties'].copy()
                                                        }
                                                        clipped_features.append(clipped_feature)
                                else:
                                    # Fallback: check against individual polygons
                                    for sed_poly in sedimentary_polygons:
                                        if sed_poly.intersects(interp_poly):
                                            clipped_features.append(feature)
                                            break
                    
                    # Progress indication
                    if i > 0 and i % 100 == 0:
                        print(f"Processed {i}/{len(interpolation_geojson['features'])} interpolation features...")
                        
                except Exception as e:
                    print(f"Error processing interpolation feature {i}: {e}")
                    continue
            
            print(f"Geological clipping complete: {len(clipped_features)} features retained from {len(interpolation_geojson['features'])} original features")
            
            return {
                "type": "FeatureCollection",
                "features": clipped_features
            }
            
        except ImportError:
            print("Shapely not available - falling back to simple point-based clipping")
            return self.clip_interpolation_by_geology(interpolation_geojson)
        except Exception as e:
            print(f"Error in polygon-based clipping: {e}")
            print("Falling back to point-based clipping")
            return self.clip_interpolation_by_geology(interpolation_geojson)

    def clip_interpolation_by_geology(self, geojson_data):
        """
        Fallback method: Clip interpolation results by removing features in hard rock areas
        This is applied AFTER interpolation is generated
        """
        if not geojson_data or 'features' not in geojson_data:
            return geojson_data
        
        # Try polygon-based clipping first
        center_coords = self._estimate_center_from_features(geojson_data['features'])
        if center_coords:
            center_lat, center_lon = center_coords
            radius_km = self._estimate_radius_from_features(geojson_data['features'], center_lat, center_lon)
            
            sedimentary_polygons = self.get_sedimentary_polygons(center_lat, center_lon, radius_km)
            if sedimentary_polygons:
                return self.clip_interpolation_by_polygons(geojson_data, sedimentary_polygons)
        
        # Fallback to original point-based method
        print("Using fallback point-based geological clipping...")
        
        clipped_features = []
        features = geojson_data.get('features', [])
        
        # Limit features for performance
        max_features = min(500, len(features))
        
        for i, feature in enumerate(features[:max_features]):
            try:
                coords = feature['geometry']['coordinates'][0]
                lats = [coord[1] for coord in coords[:-1]]
                lons = [coord[0] for coord in coords[:-1]]
                centroid_lat = sum(lats) / len(lats)
                centroid_lon = sum(lons) / len(lons)
                
                # Check geology at centroid
                unit_code = self.get_geology_at_point(centroid_lat, centroid_lon)
                
                if self.is_sedimentary(unit_code):
                    clipped_features.append(feature)
                    
            except Exception as e:
                # On error, include the feature to avoid data loss
                clipped_features.append(feature)
        
        print(f"Point-based geological clipping complete: {len(clipped_features)} features retained")
        
        return {
            "type": "FeatureCollection",
            "features": clipped_features
        }

    def _estimate_center_from_features(self, features):
        """Estimate center coordinates from GeoJSON features"""
        try:
            all_lats = []
            all_lons = []
            
            for feature in features[:10]:  # Sample first 10 features
                coords = feature['geometry']['coordinates'][0]
                lats = [coord[1] for coord in coords[:-1]]
                lons = [coord[0] for coord in coords[:-1]]
                all_lats.extend(lats)
                all_lons.extend(lons)
            
            if all_lats and all_lons:
                return sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)
        except:
            pass
        return None

    def _estimate_radius_from_features(self, features, center_lat, center_lon):
        """Estimate radius from GeoJSON features"""
        try:
            max_dist = 0
            for feature in features[:10]:
                coords = feature['geometry']['coordinates'][0]
                for coord in coords[:-1]:
                    lon, lat = coord
                    dist = self._haversine_distance(center_lat, center_lon, lat, lon)
                    max_dist = max(max_dist, dist)
            return max_dist + 5  # Add 5km buffer
        except:
            return 25  # Default 25km radius

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
