import pandas as pd
import numpy as np
import streamlit as st

def load_sample_data():
    """
    Load sample well data for demonstration purposes
    Returns a pandas DataFrame with well information
    """
    try:
        # Try to load from file first
        df = pd.read_csv("sample_data/wells_sample.csv")
        return df
    except:
        # Generate sample data if file not found
        # This mimics real well data from New Zealand
        
        # Create a base location (Christchurch, NZ area)
        base_lat = -43.5320
        base_lon = 172.6306
        
        # Create random wells around this location
        num_wells = 100
        np.random.seed(42)  # For reproducibility
        
        # Generate random coordinates within ~50km
        lats = np.random.uniform(base_lat - 0.5, base_lat + 0.5, num_wells)
        lons = np.random.uniform(base_lon - 0.5, base_lon + 0.5, num_wells)
        
        # Generate well IDs
        well_ids = [f"W-{i+1000}" for i in range(num_wells)]
        
        # Generate depths - realistic for NZ area (10-300m)
        depths = np.random.uniform(10, 300, num_wells)
        
        # Generate yield rates - L/s (0.1 to 50 L/s)
        # Use a distribution that creates more low-yield wells than high-yield ones
        base_yields = np.random.exponential(scale=5, size=num_wells)
        yield_rates = np.clip(base_yields, 0.1, 50)
        
        # Create some spatial correlation - wells close together should have similar yields
        # Create 5 "high yield zones"
        high_yield_centers = []
        for _ in range(5):
            center_lat = np.random.uniform(base_lat - 0.4, base_lat + 0.4)
            center_lon = np.random.uniform(base_lon - 0.4, base_lon + 0.4)
            high_yield_centers.append((center_lat, center_lon))
        
        # Adjust yields based on proximity to high yield zones
        for i in range(num_wells):
            for center_lat, center_lon in high_yield_centers:
                distance = np.sqrt((lats[i] - center_lat)**2 + (lons[i] - center_lon)**2)
                # If well is close to a high yield zone, increase its yield
                if distance < 0.1:  # Within about 10km
                    boost_factor = (0.1 - distance) * 10  # Closer wells get bigger boost
                    yield_rates[i] += boost_factor * 20  # Boost by up to 20 L/s
        
        # Cap yield rates at a realistic maximum
        yield_rates = np.clip(yield_rates, 0.1, 50)
        
        # Generate well status
        statuses = np.random.choice(
            ["Active", "Inactive", "Monitoring", "Abandoned"], 
            size=num_wells,
            p=[0.7, 0.1, 0.15, 0.05]  # Probabilities for each status
        )
        
        # Create the DataFrame
        df = pd.DataFrame({
            'well_id': well_ids,
            'latitude': lats,
            'longitude': lons,
            'depth': depths,
            'yield_rate': yield_rates,
            'status': statuses
        })
        
        # Save to file for future use
        try:
            df.to_csv("sample_data/wells_sample.csv", index=False)
        except:
            pass
        
        return df

def load_custom_data(uploaded_file):
    """
    Load custom data from an uploaded CSV file
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        The uploaded CSV file
    
    Returns:
    --------
    DataFrame
        Pandas DataFrame with the well data
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if required columns exist
        required_columns = ['latitude', 'longitude', 'yield_rate']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.error("Your CSV must have columns for latitude, longitude, and yield_rate at minimum.")
            return None
        
        # Add missing optional columns with default values if they don't exist
        if 'well_id' not in df.columns:
            df['well_id'] = [f"C-{i+1}" for i in range(len(df))]
        
        if 'depth' not in df.columns:
            df['depth'] = np.nan
            
        if 'status' not in df.columns:
            df['status'] = "Unknown"
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def load_api_data(api_url, api_key=None):
    """
    Load data from an external API (placeholder for future implementation)
    
    Parameters:
    -----------
    api_url : str
        URL of the API endpoint
    api_key : str, optional
        API key for authentication
    
    Returns:
    --------
    DataFrame
        Pandas DataFrame with the well data
    """
    # This is a placeholder for future implementation
    st.warning("API data loading is not yet implemented.")
    return load_sample_data()
