#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# filename: flow_map_app.py

# --- Imports ---
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.img_tiles import Stamen
import numpy as np
import streamlit as st

# --- Data Paths ---
csv_path = "untrade_wide2.csv"
shapefile_path = "ne_110m_admin_0_countries.shp"

# --- Load and Clean Data ---
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df = df.rename(columns={
    'Origin': 'origin',
    'Destination': 'destination',
    'Origin_ISO': 'origin_iso',
    'Destination_ISO': 'destination_iso',
    'Transport_Mode': 'mode'
})

world = gpd.read_file(shapefile_path)
centroids = world.to_crs("+proj=robin")
centroids['coords'] = centroids['geometry'].centroid.to_crs("EPSG:4326")
country_coords = centroids.set_index('ISO_A3_EH')['coords'].to_dict()

# --- Helper Functions ---
def get_coords(iso_code):
    return country_coords.get(iso_code)

def scaled_width(value, min_v, max_v):
    norm = (np.log10(value + 1) - np.log10(min_v + 1)) / (np.log10(max_v + 1) - np.log10(min_v + 1) + 1e-6)
    return float(0.5 + norm * 5)

def plot_flows(view_type, country, top_n, mode, variable):
    if mode != "All modes":
        filtered_df = df[df['mode'] == mode]
    else:
        filtered_df = df.copy()

    filtered = filtered_df[filtered_df[view_type] == country].sort_values(by=variable, ascending=False).head(top_n)

    fig = plt.figure(figsize=(14, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())
    ax.add_image(Stamen('terrain'), 4)

    for _, row in filtered.iterrows():
        orig = get_coords(row['origin_iso'])
        dest = get_coords(row['destination_iso'])
        if orig and dest:
            lw = scaled_width(row[variable], filtered[variable].min(), filtered[variable].max())
            ax.plot([orig.x, dest.x], [orig.y, dest.y],
                    transform=ccrs.Geodetic(), linewidth=lw, color='crimson', alpha=0.7)
            ax.plot(dest.x, dest.y, 'o', markersize=lw + 2, color='crimson', alpha=0.9, transform=ccrs.PlateCarree())
    return fig, filtered

# --- Streamlit Interface ---
st.set_page_config(layout="wide")
st.title("🌐 Trade Transport Flow Map")

country = st.selectbox("Select Country", sorted(set(df['origin']).union(df['destination'])))
view_type = st.radio("View as", ["origin", "destination"])
top_n = st.selectbox("Top N Flows", [5, 10, 15, 20])
mode = st.selectbox("Transport Mode", ["All modes"] + df['mode'].dropna().unique().tolist())
variable = st.selectbox("Variable", [
    'Transport_expenditure_US_Value',
    'FOB_value_US_Value',
    'Perunit_freight_rate_USkg_Value',
    'Transport_work_in_tonkm_Value',
    'Transport_work_in_1000_km_Value',
    'Transport_cost_intensity_in_US_per_tonkm_Value',
    'Transport_cost_intensity_in_US_per_1000_km_Value',
    'Tons_Value',
    'Advalorem_freight_rate_Value',
    'Unit_value_USkg_Value'
])

if st.button("Generate Map"):
    fig, summary_df = plot_flows(view_type, country, top_n, mode, variable)
    st.pyplot(fig)
    st.markdown("### 📊 Summary Table")
    summary_df['Flow'] = summary_df['origin'] + " → " + summary_df['destination']
    display_df = summary_df[['Flow', variable]].copy()
    display_df = display_df.rename(columns={variable: "Value"})
    st.dataframe(display_df.reset_index(drop=True))

