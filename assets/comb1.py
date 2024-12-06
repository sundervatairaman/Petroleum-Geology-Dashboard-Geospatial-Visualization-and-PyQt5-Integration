import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

# Function to ensure polygons are closed
def ensure_closed_polygon(geometry):
    if isinstance(geometry, Polygon):
        coords = list(geometry.exterior.coords)
        if coords[0] != coords[-1]:  # Check if the first and last points are the same
            coords.append(coords[0])  # Close the polygon by adding the first point to the end
        return Polygon(coords, [interior.coords for interior in geometry.interiors])
    return geometry

# List of shapefiles to read
shapefiles = ['geo_sval.shp', 'geo1ec.shp', 'geo3al.shp', 'geo4_2l.shp', 
              'geo7_2ag.shp', 'geo8apg.shp', 'geo3bl.shp', 'geo3cl.shp','geo8bg.shp','geo2cg.shp','geo6bg.shp','geo6ag.shp','Geologic_units.shp']# Read all shapefiles into separate GeoDataFrames
gdfs = []
for shp in shapefiles:
    try:
        gdf = gpd.read_file(shp)
        # Apply the ensure_closed_polygon function to correct non-closed polygons
        gdf['geometry'] = gdf['geometry'].apply(ensure_closed_polygon)
        gdfs.append(gdf)
    except Exception as e:
        print(f"Error reading {shp}: {e}")

# Choose a common CRS, e.g., WGS 84 (EPSG:4326)
common_crs = 'EPSG:4326'

# Transform each GeoDataFrame to the common CRS
for i in range(len(gdfs)):
    if gdfs[i].crs is not None:  # Check if the CRS is defined
        gdfs[i] = gdfs[i].to_crs(common_crs)  # Transform to common CRS

# Concatenate all GeoDataFrames into one
combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

# Define the output shapefile path
output_shapefile = 'combined_geodata.shp'

# Export the combined GeoDataFrame to a new shapefile
combined_gdf.to_file(output_shapefile, driver='ESRI Shapefile')

print(f"Combined shapefile created: {output_shapefile}")
