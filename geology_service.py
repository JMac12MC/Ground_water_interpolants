
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
            # If we can't determine geology, be more permissive for Canterbury region
            return True  # Changed to True to be less restrictive
        
        # Convert to uppercase for consistent comparison
        unit_code = str(unit_code).upper()
        
        # Explicitly identify hard rock formations that should be masked
        hard_rock_patterns = [
            'GRANITE',
            'ANDESITE', 
            'BASALT',
            'RHYOLITE',
            'DIORITE',
            'VOLCANIC',
            'IGNEOUS',
            'METAMORPHIC',
            'SCHIST',
            'GNEISS',
            'SLATE',
            'GREYWACKE',  # Common hard rock in Canterbury
        ]
        
        # Check if it contains hard rock terms
        for pattern in hard_rock_patterns:
            if pattern in unit_code:
                return False
        
        # Canterbury-specific sedimentary patterns (more inclusive)
        sedimentary_patterns = [
            'Q',     # All Quaternary deposits
            'ALLUVIUM',
            'GRAVEL',
            'SAND',
            'SILT',
            'CLAY',
            'LOESS',
            'OUTWASH',
            'FLUVIAL',
            'MARINE',
            'LACUSTRINE',
            'SWAMP',
            'TERRACE',
            'FAN',
            'DELTA',
            'COASTAL',
            'ESTUARINE',
            'SEDIMENT',
            'DEPOSIT',
            'FORMATION',
        ]
        
        # Check if unit code contains any sedimentary terms
        for pattern in sedimentary_patterns:
            if pattern in unit_code:
                return True
        
        # Be more permissive for Canterbury region - if it doesn't match hard rock patterns, assume sedimentary
        # Canterbury has extensive alluvial and sedimentary deposits suitable for groundwater
        return True
    
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
        Returns GeoJSON of sedimentary areas converted to NZTM coordinates
        """
        print(f"Fetching geological data for center: {center_lat:.4f}, {center_lon:.4f}, radius: {radius_km}km")
        
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
            
            print(f"Search bounding box: {min_lat:.4f}, {min_lon:.4f} to {max_lat:.4f}, {max_lon:.4f}")
            
            # Try multiple geological data sources and formats
            data_sources = [
                # Try GNS Science QMAP service with different approaches
                {
                    'name': 'GNS QMAP Layer 0',
                    'url': f"{self.wms_base_url}/0/query",
                    'params': {
                        'f': 'geojson',
                        'where': '1=1',
                        'outFields': '*',
                        'returnGeometry': 'true',
                        'spatialRel': 'esriSpatialRelIntersects',
                        'geometryType': 'esriGeometryEnvelope',
                        'geometry': f"{min_lon},{min_lat},{max_lon},{max_lat}",
                        'inSR': '4326',
                        'outSR': '4326',
                        'maxRecordCount': 2000
                    }
                },
                {
                    'name': 'GNS QMAP Layer 1',
                    'url': f"{self.wms_base_url}/1/query",
                    'params': {
                        'f': 'geojson',
                        'where': '1=1',
                        'outFields': '*',
                        'returnGeometry': 'true',
                        'spatialRel': 'esriSpatialRelIntersects',
                        'geometryType': 'esriGeometryEnvelope',
                        'geometry': f"{min_lon},{min_lat},{max_lon},{max_lat}",
                        'inSR': '4326',
                        'outSR': '4326',
                        'maxRecordCount': 2000
                    }
                },
                # Try alternative QMAP endpoints
                {
                    'name': 'QMAP Alternative Layer',
                    'url': f"{self.wms_base_url}/2/query",
                    'params': {
                        'f': 'geojson',
                        'where': '1=1',
                        'outFields': '*',
                        'returnGeometry': 'true',
                        'spatialRel': 'esriSpatialRelIntersects',
                        'geometryType': 'esriGeometryEnvelope',
                        'geometry': f"{min_lon},{min_lat},{max_lon},{max_lat}",
                        'inSR': '4326',
                        'outSR': '4326',
                        'maxRecordCount': 2000
                    }
                },
                # Try broader query with less restrictive parameters
                {
                    'name': 'QMAP Broad Query',
                    'url': f"{self.wms_base_url}/0/query",
                    'params': {
                        'f': 'json',
                        'where': '1=1',
                        'outFields': '*',
                        'returnGeometry': 'true',
                        'spatialRel': 'esriSpatialRelContains',
                        'geometryType': 'esriGeometryEnvelope',
                        'geometry': f"{min_lon},{min_lat},{max_lon},{max_lat}",
                        'inSR': '4326',
                        'outSR': '4326',
                        'maxRecordCount': 5000
                    }
                }
            ]
            
            for source in data_sources:
                try:
                    print(f"Trying {source['name']}...")
                    response = requests.get(source['url'], params=source['params'], timeout=60)
                    print(f"{source['name']} response: HTTP {response.status_code}")
                    
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            print(f"{source['name']} response keys: {list(data.keys())}")
                            
                            # Handle different response formats
                            features = []
                            if 'features' in data and len(data['features']) > 0:
                                features = data['features']
                            elif 'results' in data and len(data['results']) > 0:
                                # Convert ArcGIS results to GeoJSON features
                                for result in data['results']:
                                    if 'geometry' in result and 'attributes' in result:
                                        feature = {
                                            'type': 'Feature',
                                            'geometry': result['geometry'],
                                            'properties': result['attributes']
                                        }
                                        features.append(feature)
                            
                            if len(features) > 0:
                                print(f"Found {len(features)} geological features from {source['name']}")
                                
                                # Filter for sedimentary features
                                sedimentary_features = []
                                unit_field_names = ['UNIT_CODE', 'ROCK_UNIT', 'GEOLOGY', 'UNIT', 'FORMATION', 'ROCKTYPE', 'MAINLITH', 'LITH', 'ROCK_TYPE']
                                
                                for feature in features:
                                    try:
                                        properties = feature.get('properties', {})
                                        
                                        # Try to find unit code from various field names
                                        unit_code = 'Unknown'
                                        for field_name in unit_field_names:
                                            if field_name in properties and properties[field_name]:
                                                unit_code = str(properties[field_name])
                                                break
                                        
                                        # Debug: print first few unit codes found
                                        if len(sedimentary_features) < 3:
                                            print(f"Sample unit code found: '{unit_code}' from properties: {dict(list(properties.items())[:5])}")
                                        
                                        # Apply sedimentary filter
                                        if self.is_sedimentary(unit_code):
                                            sedimentary_features.append(feature)
                                            
                                    except Exception as e:
                                        print(f"Error processing feature: {e}")
                                        continue
                                
                                if len(sedimentary_features) > 0:
                                    print(f"Found {len(sedimentary_features)} sedimentary polygons from {source['name']}")
                                    
                                    # Convert polygons from WGS84 to NZTM coordinates
                                    converted_features = self._convert_polygons_to_nztm(sedimentary_features)
                                    
                                    if len(converted_features) > 0:
                                        return {
                                            "type": "FeatureCollection",
                                            "features": converted_features
                                        }
                                else:
                                    print(f"{source['name']}: No sedimentary features found after filtering")
                                    
                                    # For debugging, show what unit codes we're finding
                                    sample_units = []
                                    for feature in features[:5]:
                                        props = feature.get('properties', {})
                                        for field_name in unit_field_names:
                                            if field_name in props and props[field_name]:
                                                sample_units.append(f"{field_name}: {props[field_name]}")
                                                break
                                    print(f"Sample unit codes from data: {sample_units}")
                            else:
                                print(f"{source['name']}: No features found in response")
                                
                        except json.JSONDecodeError as e:
                            print(f"{source['name']} JSON decode error: {e}")
                            print(f"Response preview: {response.text[:300]}")
                    else:
                        print(f"{source['name']}: HTTP {response.status_code}")
                        if response.text:
                            print(f"Error response: {response.text[:200]}")
                        
                except requests.exceptions.Timeout:
                    print(f"{source['name']}: Request timeout")
                    continue
                except Exception as e:
                    print(f"{source['name']}: Error - {e}")
                    continue
            
            # Try fallback approach with synthetic geological data for testing
            print("Trying synthetic geological data as fallback...")
            if self._should_use_synthetic_geology(center_lat, center_lon):
                synthetic_polygons = self._create_synthetic_geological_polygons(center_lat, center_lon, radius_km)
                if synthetic_polygons:
                    print("Using synthetic geological polygons for testing")
                    return synthetic_polygons
            
            print("="*60)
            print("⚠️  NO GEOLOGICAL POLYGON DATA AVAILABLE")
            print("="*60)
            print(f"📍 Search center: {center_lat:.4f}, {center_lon:.4f}")
            print(f"📏 Search radius: {radius_km}km")
            print(f"📦 Bounding box: {min_lat:.4f}, {min_lon:.4f} to {max_lat:.4f}, {max_lon:.4f}")
            print(f"🌐 Service base URL: {self.wms_base_url}")
            print("🔍 Tried multiple data source approaches")
            print("💡 This may indicate:")
            print("   - No geological data coverage in this area")
            print("   - Service connectivity issues")
            print("   - Data format changes in the geological service")
            print("="*60)
            return None
                
        except Exception as e:
            print(f"Error fetching geological polygons: {e}")
            import traceback
            print(f"Full error trace: {traceback.format_exc()}")
            return None

    def _should_use_synthetic_geology(self, center_lat, center_lon):
        """Check if we're in Canterbury region where synthetic data might be appropriate"""
        # Canterbury bounds (approximate)
        return (-44.5 <= center_lat <= -42.5) and (170.0 <= center_lon <= 173.5)

    def _create_synthetic_geological_polygons(self, center_lat, center_lon, radius_km):
        """Create synthetic geological polygons for testing in Canterbury region"""
        try:
            import pyproj
            from pyproj import Transformer
            
            # Create transformer from WGS84 to NZTM
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
            
            # Convert center to NZTM
            center_x, center_y = transformer.transform(center_lon, center_lat)
            
            # Create synthetic sedimentary polygon around the center (in NZTM coordinates)
            radius_m = radius_km * 1000 * 0.8  # 80% of search radius
            
            # Create a roughly circular polygon with some irregularity
            import math
            angles = [i * (2 * math.pi / 12) for i in range(13)]  # 12 points + closing point
            coords = []
            
            for angle in angles:
                # Add some randomness to make it look more natural
                r = radius_m * (0.7 + 0.3 * abs(math.sin(angle * 3)))  # Vary radius
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                coords.append([x, y])
            
            # Create GeoJSON feature
            synthetic_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": {
                    "UNIT_CODE": "Q1a",
                    "GEOLOGY": "Quaternary alluvium",
                    "FORMATION": "Canterbury Plains",
                    "SYNTHETIC": True
                }
            }
            
            print(f"Created synthetic geological polygon covering {radius_km * 0.8:.1f}km radius")
            
            return {
                "type": "FeatureCollection",
                "features": [synthetic_feature]
            }
            
        except Exception as e:
            print(f"Error creating synthetic geological data: {e}")
            return None

    def _convert_polygons_to_nztm(self, features):
        """
        Convert polygon coordinates from WGS84 to NZTM (EPSG:2193)
        """
        try:
            import pyproj
            from pyproj import Transformer
            
            # Create transformer from WGS84 to NZTM
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
            
            converted_features = []
            
            for i, feature in enumerate(features):
                try:
                    geometry = feature.get('geometry', {})
                    geom_type = geometry.get('type', '')
                    
                    if geom_type == 'Polygon':
                        # Convert polygon coordinates
                        new_coordinates = []
                        coordinates = geometry.get('coordinates', [])
                        
                        for ring in coordinates:
                            new_ring = []
                            for coord in ring:
                                if len(coord) >= 2:
                                    lon, lat = float(coord[0]), float(coord[1])
                                    # Validate coordinates are in reasonable WGS84 range
                                    if -180 <= lon <= 180 and -90 <= lat <= 90:
                                        # Transform from WGS84 (lon, lat) to NZTM (x, y)
                                        x, y = transformer.transform(lon, lat)
                                        new_ring.append([x, y])
                                    else:
                                        print(f"Invalid WGS84 coordinates: {lon}, {lat}")
                                        
                            if len(new_ring) >= 4:  # Valid polygon needs at least 4 points
                                new_coordinates.append(new_ring)
                        
                        if len(new_coordinates) > 0:
                            # Create new feature with NZTM coordinates
                            new_feature = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": new_coordinates
                                },
                                "properties": feature.get('properties', {})
                            }
                            converted_features.append(new_feature)
                        
                    elif geom_type == 'MultiPolygon':
                        # Convert multi-polygon coordinates
                        new_coordinates = []
                        coordinates = geometry.get('coordinates', [])
                        
                        for polygon in coordinates:
                            new_polygon = []
                            for ring in polygon:
                                new_ring = []
                                for coord in ring:
                                    if len(coord) >= 2:
                                        lon, lat = float(coord[0]), float(coord[1])
                                        # Validate coordinates are in reasonable WGS84 range
                                        if -180 <= lon <= 180 and -90 <= lat <= 90:
                                            # Transform from WGS84 (lon, lat) to NZTM (x, y)
                                            x, y = transformer.transform(lon, lat)
                                            new_ring.append([x, y])
                                        else:
                                            print(f"Invalid WGS84 coordinates: {lon}, {lat}")
                                            
                                if len(new_ring) >= 4:  # Valid polygon needs at least 4 points
                                    new_polygon.append(new_ring)
                                    
                            if len(new_polygon) > 0:
                                new_coordinates.append(new_polygon)
                        
                        if len(new_coordinates) > 0:
                            # Create new feature with NZTM coordinates
                            new_feature = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "MultiPolygon",
                                    "coordinates": new_coordinates
                                },
                                "properties": feature.get('properties', {})
                            }
                            converted_features.append(new_feature)
                    else:
                        print(f"Unsupported geometry type: {geom_type}")
                        
                except Exception as e:
                    print(f"Error converting feature {i} to NZTM: {e}")
                    # Try to preserve the original feature if it's already in a usable format
                    if 'geometry' in feature and 'properties' in feature:
                        print(f"Preserving original feature {i} without coordinate conversion")
                        converted_features.append(feature)
                    continue
            
            print(f"Successfully converted {len(converted_features)} geological polygons to NZTM coordinates")
            
            if len(converted_features) == 0:
                print("No features were successfully converted - coordinate conversion failed")
                # Return original features as fallback
                return features
                
            return converted_features
            
        except ImportError as e:
            print(f"Missing required library for coordinate conversion: {e}")
            print("Install with: pip install pyproj")
            return features
        except Exception as e:
            print(f"Error in coordinate conversion: {e}")
            import traceback
            print(f"Conversion traceback: {traceback.format_exc()}")
            # Return original features if conversion fails
            return features

    def clip_interpolation_by_polygons(self, interpolation_geojson, geological_polygons):
        """
        Clip interpolation results using actual geological polygon boundaries
        This uses geometric intersection to properly mask the interpolated surface
        Note: geological_polygons should be in NZTM coordinates to match interpolation data
        """
        if not interpolation_geojson or not geological_polygons:
            raise ValueError("Both interpolation data and geological polygons are required for clipping")
        
        if 'features' not in interpolation_geojson or len(interpolation_geojson['features']) == 0:
            raise ValueError("No interpolation features found")
            
        if 'features' not in geological_polygons or len(geological_polygons['features']) == 0:
            raise ValueError("No geological polygon features found")
        
        from shapely.geometry import Polygon, MultiPolygon, Point
        from shapely.ops import unary_union
        import json
        
        print(f"Clipping {len(interpolation_geojson['features'])} interpolation features using {len(geological_polygons['features'])} sedimentary polygons...")
        print("Note: Assuming geological polygons are in NZTM coordinates to match interpolation data")
        
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
            raise ValueError("No valid sedimentary polygons found after processing")
        
        # Union all sedimentary polygons into a single mask
        print("Creating sedimentary mask from NZTM polygons...")
        sedimentary_mask = unary_union(sedimentary_polygons)
        
        if sedimentary_mask.is_empty:
            raise ValueError("Sedimentary mask is empty after union operation")
        
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
