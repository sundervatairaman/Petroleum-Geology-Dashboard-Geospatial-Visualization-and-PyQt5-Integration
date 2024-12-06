import geopandas as gpd
import pandas as pd  # Import pandas
import matplotlib.pyplot as plt

# List of shapefiles to read
shapefiles = ['geo_sval.shp', 'geo1ec.shp', 'geo3al.shp', 'geo4_2l.shp', 
              'geo7_2ag.shp', 'geo8apg.shp', 'geo3bl.shp', 'geo3cl.shp']

# Read all shapefiles into separate GeoDataFrames
gdfs = [gpd.read_file(shp) for shp in shapefiles]

# Choose a common CRS, e.g., WGS 84 (EPSG:4326)
common_crs = 'EPSG:4326'

# Transform each GeoDataFrame to the common CRS
for i in range(len(gdfs)):
    if gdfs[i].crs is not None:  # Check if the CRS is defined
        gdfs[i] = gdfs[i].to_crs(common_crs)  # Transform to common CRS

# Concatenate all GeoDataFrames into one
combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

# Plot the combined GeoDataFrame
fig, ax = plt.subplots(figsize=(10, 10))  # Optional: Set figure size
combined_gdf.plot(ax=ax, color='blue', edgecolor='black')  # Customize colors

# Add title and labels (optional)
plt.title('Combined GeoDataFrame Plot')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Show the plot
plt.show()
