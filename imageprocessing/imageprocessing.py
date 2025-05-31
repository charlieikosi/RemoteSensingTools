import planetary_computer as pc
import pystac_client
#from pystac_client import Client
import rioxarray
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
import shapely
from shapely.geometry import box, mapping
from shapely.wkt import loads
import numpy as np
import time
from datetime import datetime, timedelta
import dask.array as da
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from IPython.display import Image
import rich.table
import odc.stac
import io
from rasterio.warp import transform_bounds
from pyproj import Transformer

import rioxarray
#from PIL import Image

from rasterio.plot import show
import json
import folium
import string
import stackstac

from shapely.geometry import box
import string
from shapely.affinity import scale

from matplotlib.colors import ListedColormap

def initialize_client():
    '''Function that initilizes the planetary client'''
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

def load_shapefile(filepath):
    '''Function that takes a filepath as string to load a shapefile'''
    return gpd.read_file(filepath)

def convert_geojson(geopandas_df):
    '''Function that takes a geopandas dataframe and converts to a geojson. Dataframe must have crs EPSG:4326'''
    # Reproject
    if geopandas_df.crs != "EPSG:4326":
        geopandas_df = geopandas_df.to_crs("EPSG:4326")
        # Convert to GeoJSON
        geojson_data = json.loads(geopandas_df.to_json())
        # Extract the geometry part
        geojson_dict = geojson_data["features"][0]["geometry"]
        return geojson_dict

def reverse_coordinates(geojson_polygon):
    """
    Reverse coordinates in a GeoJSON Polygon from (lon, lat) to (lat, lon).

    Parameters:
        geojson_polygon (dict): A GeoJSON Polygon dictionary.

    Returns:
        list: Reversed coordinates list suitable for Folium.
    """
    return [[lat, lon] for lon, lat in geojson_polygon['coordinates'][0]]

def sat_feature_geometry(sat_collection, sat_feature_num):
    '''Function that takes the satellite collection and satellite feature number as integer input to return the 
    geometry of the feature.
    '''
    feature_geometry = reverse_coordinates(sat_collection[sat_feature_num].geometry)    
    return feature_geometry

def view_roi(geopandas_df,label):
    """
    Function that takes in a geopandas dataframe and plots an interactive map,
    displaying the polygon ID (index) on hover.
    (Integrated into the original structure)
    """
    # --- 1. CRS Handling (Original) ---
    # Work on a copy to avoid modifying the original DataFrame outside the function,
    # especially since we're adding a temporary column.
    df_display = geopandas_df.copy()

    if df_display.crs != "EPSG:4326":
        df_display = df_display.to_crs("EPSG:4326")

    # --- 2. Add ID Column for Tooltip (Addition) ---
    # Create a temporary column holding the index value as a string.
    # We do this *before* converting to GeoJSON.
    # Using '__id' as a column name to minimize potential conflicts.
    id_col_name = label
    df_display[id_col_name] = df_display.values.astype(str)

    # --- 3. Convert to GeoJSON (Original - now includes ID column) ---
    geojson = df_display.to_json()

    # --- 4. Create Folium Map (Original - with empty check) ---
    # Check if the dataframe is empty to avoid errors accessing iloc[0]
    if not df_display.empty:
        centroid = df_display.geometry.centroid.iloc[0]  # Get first centroid for centering
        map_location = [centroid.y, centroid.x]
        zoom = 12
    else:
        # Provide default values if the GeoDataFrame is empty
        map_location = [0, 0]
        zoom = 2
        print("Warning: Input GeoDataFrame is empty. Centering map at [0,0].")

    m = folium.Map(location=map_location, zoom_start=zoom)

    # --- 5. Define Tooltip (Addition) ---
    tooltip = folium.features.GeoJsonTooltip(
        fields=[id_col_name],  # The column name we added
        aliases=['ID:'],       # The label text shown before the ID value
        sticky=True,          # Tooltip follows the mouse cursor
        style=("background-color: white; color: black; font-family: sans-serif; font-size: 12px; padding: 5px;") # Optional styling
    )

    # --- 6. Add GeoJSON Layer (Original - now with Tooltip) ---
    folium.GeoJson(
        geojson,
        name="Shapefile Layer",
        tooltip=tooltip, # Pass the created tooltip object here
        style_function=lambda x: { # Optional: Basic styling for visibility
            'fillColor': '#add8e6', # Light blue fill
            'color': 'black',      # Black border
            'weight': 1,
            'fillOpacity': 0.6
        },
        highlight_function=lambda x: { # Optional: Styling on hover
            'fillColor': '#ffeda0', # Yellow fill on hover
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.8
        }
    ).add_to(m)

    # --- 7. Add Layer Control (Optional but recommended) ---
    folium.LayerControl().add_to(m)
    return m

def sat_img_collection(date_start, date_end, geojson, sensor_type):
    '''Function that takes date start and end as strings, a geodataframe and the satellite sensor as string input to return a collection of available imagery'''
    max_cloud_cover = 5
    cloud_filter = {"eo:cloud_cover": {"lte": max_cloud_cover}}    
    time_range = date_start+'/'+ date_end
    client = initialize_client()
    search = client.search(collections=[sensor_type], 
                           intersects=geojson, 
                           datetime=time_range,
                           query=cloud_filter)
    items = search.get_all_items()
    return items

def sat_feature_properties(sat_collection, sat_feature_num):
    '''Function that takes in the satellite collection and the satellite feature number as integer input to return a dictionary
    for the feature properties
    '''
    sat_properties = sat_collection[sat_feature_num].properties
    return sat_properties

def view_collection_tiles(sat_collection):
    '''Function that takes the satellite collection and plots an interactive map of Sentinel 2 Tiles
    that overlap the ROI
    '''
    iteration = range(len(sat_collection))
    bbox_dictionary = {}
    for i in iteration:
        tile_id = sat_collection[i].properties['s2:mgrs_tile']
        bbox_polygon = sat_feature_geometry(sat_collection,i)
        bbox_dictionary[tile_id] = bbox_polygon

    # Extract all coordinates from bounding boxes
    all_coords = [coord for bbox in bbox_dictionary.values() for coord in bbox]

    # Compute the centroid of all bounding boxes
    center_lat = np.mean([coord[0] for coord in all_coords])
    center_lon = np.mean([coord[1] for coord in all_coords])
    map_center = [center_lat, center_lon]

    # Create a Folium map centered at an approximate location
    m = folium.Map(location=map_center, zoom_start=7)
    # Loop through bounding boxes and add them as FeatureGroups
    for name, bbox in bbox_dictionary.items():
        fg = folium.FeatureGroup(name=name)  # Create a toggleable layer
        folium.Polygon(
            locations=bbox, 
            color='blue', 
            weight=2, 
            fill=True, 
            fill_color='blue', 
            fill_opacity=0.2,
            tooltip=name  # Show name when hovering over polygon
        ).add_to(fg)
        fg.add_to(m)  # Add FeatureGroup to map

    # Add a layer control to toggle bounding boxes
    folium.LayerControl().add_to(m)

    # Adjust view to fit all bounding boxes
    all_coords = [coord for bbox in bbox_dictionary.values() for coord in bbox]
    m.fit_bounds(all_coords)
    return m

def min_cloud(sat_collection):
    '''Function that takes the satellite image collection and returns a sorted geopandas df with least cloudiness at the top'''
    df = gpd.GeoDataFrame.from_features(sat_collection.to_dict(), crs="epsg:4326")
    df_sorted = df.sort_values(by='eo:cloud_cover')
    return df_sorted

def min_cloud_index_list(sat_collection):
    '''Function that takes the satellite image collection and returns a sorted list of sat image feature IDs asscending with most cloudiness'''
    df = gpd.GeoDataFrame.from_features(sat_collection.to_dict(), crs="epsg:4326")
    df_sorted = df.sort_values(by='eo:cloud_cover')
    feature_numbers = df_sorted.index.tolist()
    return feature_numbers

def sat_preview(sat_collection,sat_feature_num):
    '''Function that takes the satellite collection, the satellite feature number to plot the image'''
    selected_img = sat_collection[sat_feature_num].assets
    return Image(url=selected_img["rendered_preview"].href, width=500)

def view_band(sat_collection, sat_feature_num,band):
    '''Function that takes the Satellite collection, the feature number as integer and the band number
    to view the band in low resolution. This function will plot
    '''
    selected_collection = sat_collection[sat_feature_num]
    img = rioxarray.open_rasterio(selected_collection.assets[band].href,
                                 overview_level=4).squeeze()
    img = img.plot(cmap="gray", add_colorbar=False)
    return img.axes.set_axis_off();

def transform_bounds(src_crs, dst_crs, west, south, east, north):
    """Transform bounding box coordinates between coordinate reference systems."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    west_t, north_t = transformer.transform(west, north)
    east_t, south_t = transformer.transform(east, south)
    return west_t, south_t, east_t, north_t

def view_band_interactive(sat_collection, sat_feature_num, band):
    '''Function that takes the Satellite collection, the feature number as integer and the band number
    to interactively view the band in low resolution. This function will plot'''
    selected_collection = sat_collection[sat_feature_num]
    
    # Open the rasterio dataset using the asset href
    try:
        ds = rioxarray.open_rasterio(selected_collection.assets[band].href,
                                     overview_level=4).squeeze()
    except Exception as e:
        print(f"Error opening raster: {e}")
        return None  # Or handle the error in a way that makes sense for your application

    # Normalize pixel values (scale 0-255 for visualization)
    array = ds.values
    array = (array - array.min()) / (array.max() - array.min()) * 255
    array = array.astype(np.uint8)

    # Convert array to an image using PIL
    img = Image.fromarray(array)

    # Save image to an in-memory buffer (Folium requires file or URL)
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_data = np.array(img)  # Convert PIL image back to NumPy for Folium

    # Get raster bounds in lat/lon.  Crucially, use *ds* for the bounds!
    bounds = transform_bounds(ds.rio.crs, "EPSG:4326", *ds.rio.bounds())


    # Create a Folium map centered at the raster location
    m = folium.Map(location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2], zoom_start=10)

    # Add raster as an overlay
    folium.raster_layers.ImageOverlay(
        image=img_data,  # NumPy array instead of PIL Image
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],  # [[south, west], [north, east]]
        opacity=1,
        interactive=True).add_to(m)

    # Display the map
    return m

def asset_table(sat_collection, sat_feature_num):
    '''Function that takes the satellite collection, the satellite feature number to return the asset attribute table'''
    table = rich.table.Table("Asset Key", "Description")
    selected_img = sat_collection[sat_feature_num].assets
    for asset_key, asset in selected_img.items():
        table.add_row(asset_key, asset.title)
    return table

def band_assets(sat_collection, sat_feature_num):
    ''''Function that takes the satellite collection and the sat feature number as integer and returns a dictionary of assets'''
    return sat_collection[sat_feature_num].assets.keys()

def band_img(band,sat_collection, sat_feature_num):
    '''Function that takes the band as string input, the satellite collection and the satellite feature number
    to return a numpy array
    '''
    band_url = sat_collection[sat_feature_num].assets[band].href
    with rasterio.open(band_url) as dataset:
        band = dataset.read()  # Read all bands
        band_reflectance = band.astype(float) / 65535
        band_reflectance = band_reflectance[0, :, :]
    return band_reflectance

def band_img_dask(band, sat_collection, sat_feature_num):
    '''Function that takes the band as string input, the satellite collection and the satellite feature number
    to return a dask array
    '''
    band_url = sat_collection[sat_feature_num].assets[band].href

    with rasterio.open(band_url) as dataset:
        dask_band = da.from_array(dataset.read(1), chunks=(1024, 1024)) / 65535
    return dask_band

def band_img_dask2(band, sat_collection, sat_feature_num):
    '''
    Function that takes the band as string input, the satellite collection,
    and the satellite feature number to return a
    xarray.core.dataarray.DataArray.
    '''
    band_url = sat_collection[sat_feature_num].assets[band].href

    if band == "SCL":
        print(f'Skipping normalization for {band}.')
        dask_band = rioxarray.open_rasterio(band_url).squeeze()
        print(f'{band} layer loaded.')
        return dask_band
    else:
        print(f'Normalizing Band {band}.')
        dask_band = rioxarray.open_rasterio(band_url).squeeze() / 65535
        dask_band = dask_band.fillna(0)
        print(f'Band {band} normalized.')
        return dask_band

def band_img_properties(sat_collection, sat_feature_num, band):
    '''Function that takes the satellite collection, satellite feature number as integar and and the band name as string to return a dictionary
    of band properties
    '''
    band_properties = sat_collection[sat_feature_num].assets[band].extra_fields
    return band_properties

def get_scl_stats(sat_collection, sat_feature_num):
    '''Function that takes the satellite collection and the satellite feature number as integar input
    and returns a dictionary'''
    stats = {}
    scl = band_img_dask2('SCL',sat_collection,sat_feature_num)
    labels = scl_labels()
    unique, counts = np.unique(scl, return_counts=True)
    for val, count in zip(unique,counts):
        stats[val] = (labels.get(val, 'Unknown'),count)
    return stats

def load_image(sat_collection,sat_feature_num, band, geopandas_df):
    '''Function that loads images into python as xarray.DataArray format'''
    asset = sat_collection[sat_feature_num].assets[band]
    img = rioxarray.open_rasterio(asset.href, masked=True, chunks=True).squeeze()
    img = img.rio.clip(geopandas_df.geometry.apply(mapping), geopandas_df.crs)
    return img

def format_time(seconds):
    '''Function that formats time to seconds, minute or hours'''
    if seconds < 60:
        return f"{seconds:.2f} sec"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} min"
    else:
        return f"{seconds / 3600:.2f} hr"


def make_grid(gpdf, grid_size, buffer_size=0, save: bool = False):
    '''Function that takes a geopandas dataframe and grid size as integer input
    to create grid cells.
    Includes the option to buffer the tiling by a specified amount.

    Args:
        gpdf: GeoPandas DataFrame representing the region of interest.
        grid_size: Size of the grid cells in the same units as the gpdf's CRS.
        buffer_size: Buffer distance in the same units as the gpdf's CRS (default: 0).

    Returns:
        A GeoPandas DataFrame containing the clipped grid cells.
    '''
    # Get the bounding box of the ROI
    minx, miny, maxx, maxy = gpdf.total_bounds

    # Define cell size (e.g., 10000m x 10000m)
    cell_size = grid_size  # Use the provided grid_size argument

    # Generate grid rows and columns
    rows = np.arange(miny, maxy, cell_size)
    cols = np.arange(minx, maxx, cell_size)

    grid_cells = []
    ids = []

    # Assign unique IDs
    for row_idx, y in enumerate(rows):
        row_letter = string.ascii_uppercase[row_idx]  # Convert index to letter (A, B, C...)
        for col_idx, x in enumerate(cols):
            grid_cell = box(x, y, x + cell_size, y + cell_size)
            if buffer_size > 0:
                # Scale the box to simulate buffering with sharp corners
                grid_cell = scale(grid_cell, xfact=1 + (buffer_size / (cell_size/2)), yfact=1 + (buffer_size / (cell_size/2)), origin='center')
                #alternative and more computationally intensive method
                #grid_cell = grid_cell.exterior.buffer(buffer_size, cap_style=3).intersection(grid_cell)  # Apply buffer with square cap

            grid_cells.append(grid_cell)
            unique_id = f"{row_letter}-{col_idx+1}"  # Format: A-1, A-2, B-1, etc.
            ids.append(unique_id)


    # Convert to a GeoDataFrame
    grid = gpd.GeoDataFrame({'id': ids, 'geometry': grid_cells}, crs=gpdf.crs)

    # Clip the grid to the ROI
    grid_clipped = gpd.overlay(grid, gpdf, how="intersection")

    if save:
        # Save to file
        grid_clipped.to_file("grid_cells.shp")
        return grid_clipped
    else:
        return grid_clipped

def scl_labels():
    '''Function that returns the a dictionary of scene classified layer labels'''
    scl_labels = {
    0: "No Data",
    1: "Saturated or Defective",
    2: "Dark Area Pixels",
    3: "Cloud Shadows",
    4: "Vegetation",
    5: "Not Vegetated",
    6: "Water",
    7: "Unclassified",
    8: "Cloud Medium Probability",
    9: "Cloud High Probability",
    10: "Thin Cirrus",
    11: "Snow or Ice"}
    return scl_labels

def save_scene(xarray_data_array, save_name):
    '''Function that takes only an xarray.dataArray and saves it as a geotif'''
    # Full scene
    array = xarray_data_array.rio.write_crs("EPSG:32760", inplace=True)
    array.rio.to_raster(
        f"{save_name}.tif",
        driver='GTiff',
	nodata=-9999,
        tiled=True,          # Optional: Enable tiling
        blockxsize=256, # Optional: Control tile size
        blockysize=256, # Optional: Control tile size
        compress='LZW',      # Optional: Add compression
        # windowed=True # is implicitly True when using dask
    )
    print('Successfully saved.')

def get_intersecting_grids(xarray_data_array,grid):
    '''
    Function that takes an xarray and a gridded geodataframe and returns a list of index for rows in the geodataframe that intersect
    cell values
    '''
    intersecting_cells = []

    if xarray_data_array.rio.crs != grid.crs:
        print("CRSs do not match. Reprojecting GeoDataFrame...")
        gdf = grid.to_crs(xarray_data_array.rio.crs)
        print(f"GeoDataFrame CRS after reprojection: {gdf.crs}\n")
    else:
        print("CRSs match.\n")

    grid_cell_ids = grid['id'].values

    for i in range(len(grid)):
        # Check crs and align
        try:
            grid_cell_polygon = gdf.iloc[[i]]
            # Crop
            clip = xarray_data_array.rio.clip(grid_cell_polygon.geometry.values, grid_cell_polygon.crs)
            print(f'Grid cell {grid_cell_ids[i]} intersects')
            intersecting_cells.append(i)
        except:
            # Code to run if ANY exception occurred in the try block
            print(f'No Data in cell {grid_cell_ids[i]}. Going to next available cell')

    return intersecting_cells

def crop_scene(xarray_data_array, gdf):
    '''Function that takes xarray and geodataframe and crops it to the grid'''
    # Check crs and align
    if xarray_data_array.rio.crs != gdf.crs:
        print("CRSs do not match. Reprojecting GeoDataFrame...")
        gdf = gdf.to_crs(xarray_data_array.rio.crs)
        #print(f"GeoDataFrame CRS after reprojection: {gdf.crs}\n")
    else:
        print("CRSs match.\n")

    # Crop
    clipped_array = xarray_data_array.rio.clip(gdf.geometry.values, gdf.crs)
    return clipped_array

def array_covers_polygon(xarray, grid):
    """
    Function that takes an xarray.DataArray and grid object to return a true or false response fi array is within grid
    """
    da_bounds = {
        "minx": float(xarray.x.min()),
        "maxx": float(xarray.x.max()),
        "miny": float(xarray.y.min()),
        "maxy": float(xarray.y.max())
    }
    
    da_box = box(da_bounds["minx"], da_bounds["miny"], da_bounds["maxx"], da_bounds["maxy"])

    if grid.crs != xarray.rio.crs:  # if using rioxarray
        gdf = grid.to_crs(xarray.rio.crs)
    polygon = gdf.geometry.iloc[0]
    is_covered = da_box.contains(polygon)
    return is_covered

def plotSCL(scene_classified_layer):
    """Function that takes a scene classified layer in the data format xarray.core.dataarray.DataArray and
    visualizes the layer
    """
    # Define your SCL class-to-label mapping (based on Sentinel-2 standards)
    scl_labels = {
        0: "No Data",
        1: "Saturated / Defective",
        2: "Dark Area Pixels",
        3: "Cloud Shadows",
        4: "Vegetation",
        5: "Bare Soils",
        6: "Water",
        7: "Clouds Low Probability",
        8: "Clouds Medium Probability",
        9: "Clouds High Probability",
        10: "Thin Cirrus",
        11: "Snow or Ice"
    }

    # Define custom colors for each class (in order 0–11)
    scl_colors = [
        "#000000",  # 0: No Data - black
        "#784212",  # 1: Saturated / Defective - brown
        "#363636",  # 2: Dark Area Pixels - dark grey
        "#646464",  # 3: Cloud Shadows - grey
        "#00FF00",  # 4: Vegetation - bright green
        "#A0522D",  # 5: Bare Soils - sienna
        "#0000FF",  # 6: Water - blue
        "#FFD700",  # 7: Clouds Low Prob - gold
        "#FFA500",  # 8: Clouds Med Prob - orange
        "#FF0000",  # 9: Clouds High Prob - red
        "#ADD8E6",  # 10: Thin Cirrus - light blue
        "#FFFFFF"   # 11: Snow or Ice - white
    ]

    # Create custom colormap
    custom_cmap = ListedColormap(scl_colors)

    # Plot
    plt.imshow(scene_classified_layer, cmap=custom_cmap, vmin=0, vmax=11)
    cbar = plt.colorbar(ticks=list(scl_labels.keys()))
    cbar.ax.set_yticklabels([scl_labels[i] for i in scl_labels])
    plt.title("Sentinel-2 Scene Classification Layer")
    plt.show()
