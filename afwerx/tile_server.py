import os
import numpy as np
import multiprocessing
from subprocess import Popen
import cv2
from PIL import Image
from scipy.interpolate import RectBivariateSpline as RBS
import xarray as xr
import warnings
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
from visualize import plot_xr, rain_palette


# NOTE: must be installed through conda
try:
    import xesmf as xe
except:
    warnings.warn("xesmf not installed")


NUM_CORES = multiprocessing.cpu_count()
N = 16384
TILE_SIZE = 256
ZOOM = 7

# Aurora size: 720, 1440

minTileX = 0
minTileY = 0
maxTileX = 6
maxTileY = 3

minX = minTileX * TILE_SIZE
minY = minTileY * TILE_SIZE
maxX = maxTileX * TILE_SIZE
maxY = maxTileY * TILE_SIZE

SKIP_PLOT = ["latitude", "longitude", "time", "batch", "history", "rollout_step", "level"]


def regrid(data : xr.Dataset, regrid_file : str) -> xr.Dataset:
    '''
    Interpolate 0.25 degree GFS/Aurora/Mercator data to GOES CONUS grid

    Parameters
    ----------
    data : xr.DataSet
        Xarray dataset to be regridded
    regrid_file : str
        Full path to the xESMF regridding file. 
        Likely GFS/Aurora/Mercator 0.25 degree to GOES CONUS Zoom 7
    '''



    pass




def save_tile(filename, data):
    '''
    Writes a [Alpha, R, G, B] tile to a file. 

    Parameters
    ----------
    filename : str
        Full path name of the file to write to.
    data : np.ndarray
        Data to be written to the file.

    '''

    # rgba --> bgra, which is the order for the RGBA 32-bit representation 
    bgra = [2, 1, 0, 3]

    tile = np.empty((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)

    # Remap colors for each channel individually
    for chan in range(tile.shape[2]):
        print("Channel: ", chan)   
        print("\tTile Channel Shape: ", type(tile), tile[:, :, chan].shape)
        print("\tData Shape: ", type(data), data.shape)
        print("\tData Channel Shape: ", type(data), data[:, :, chan].shape)
        print("\tRain Pallete Output: ", type(rain_palette[data[:, :, chan], bgra[chan]]), rain_palette[data[:, :, chan], bgra[chan]].shape)

        tile[:, :, chan] = rain_palette[data[:, :, chan], bgra[chan]]

    print("Saving image to: ", filename)

    file_dir = os.path.join(*filename.split('/')[:-1])
    if not os.path.exists(file_dir):
        print("Creating directory: ", file_dir)
        os.makedirs(file_dir)

    # CV write
    sts = cv2.imwrite(filename, tile)

    print("Status: ", sts)

    # PIL Image write
    # img = Image.fromarray(tile)
    # img.save(filename)


def tile_into_folders(serve_dir, noun, timestamp, image):
    '''
    Create folders and tiles from image for serving.
    
    Parameters
    ----------
    serve_dir : str
        Directory to serve tiles from.
    noun : str
        Name of the image.
    timestamp : str
        Timestamp of the image.
    image : np.ndarray
        Image to be tiled.

    '''

    # Iterate through minX to maxX tiles
    for x in np.arange(minTileX, maxTileX):
        # folder = serve_dir + noun + timestamp + '/'
        # sts = Popen('mkdir -p ' + folder + '{}/{}'.format(7, x), shell=True).wait()
        folder = os.path.join(serve_dir, noun, timestamp, f"{7}/{x}")
        os.makedirs(folder, exist_ok=True)

        for y in np.arange(minTileY, maxTileY):
            image = image.astype(np.uint8)

            print("Image before regridding: ", image.shape)

            image = regrid(image)

            print("Image after regridding: ", image.shape)

            # Save the regridded image
            cv2.imwrite("tiles/regridded.png", image)

            print("Image Tile: ", image[y*TILE_SIZE-minY : y*TILE_SIZE-minY+TILE_SIZE, \
                                        x*TILE_SIZE-minX : x*TILE_SIZE-minX+TILE_SIZE].shape)

            filename = f"{7}/{x}/{y}.png"
            file_path = os.path.join(folder, filename)

            print("File Path: ", file_path)

            sts = save_tile(file_path, \
                            image[y*TILE_SIZE-minY : y*TILE_SIZE-minY+TILE_SIZE, \
                                    x*TILE_SIZE-minX : x*TILE_SIZE-minX+TILE_SIZE])
                

if __name__ == '__main__':
    serve_dir = "serve"
    noun = "test"
    timestamp = '20240101T0000Z'
    ds = xr.open_dataset('tiles/test_preds.h5')

    # Print ranges for lat, lon, time, and level
    print("Lat Range: ", type(ds['latitude'].min().values), ds['latitude'].min().values, " - to - ", ds['latitude'].max().values)
    print("Lon Range: ", type(ds['longitude'].min().values), ds['longitude'].min().values, " - to - ", ds['longitude'].max().values)
    print("Time Range: ", type(ds['time'].min().values), ds['time'].min().values, " - to - ", ds['time'].max().values)
    print("Levels: ", ds['level'].values)

    # Print name and shape of each variable
    print("Variables: ")
    for var in ds.variables:
        print("\t", var, ds[var].shape)


    # Print lat lon arrays
    # lats = ds['latitude'].values
    # lons = ds['longitude'].values

    # print("Lats: ", lats)
    # print("Lons: ", lons)

    # Extract only CONUS data from the xarray dataset
    minLat = 55.0
    maxLat = 20.0
    minLon = 180.0
    maxLon = 360.0

    ds = ds.sel(latitude=slice(minLat, maxLat), longitude=slice(minLon, maxLon))

    print("New Lat Range: ", type(ds['latitude'].min().values), ds['latitude'].min().values, " - to - ", ds['latitude'].max().values)
    print("New Lon Range: ", type(ds['longitude'].min().values), ds['longitude'].min().values, " - to - ", ds['longitude'].max().values)

    # Plot plot some variables as a sanity check
    for v in ['atmos_v', 'surf_2t', 'static_z']:
        if v in SKIP_PLOT:
            continue

        print("Plotting variable: ", v)
        # Check if the variable is atmospheric, has levels
        if 'level' in ds[v].dims:
            for l in [1000, 700, 300]:
                plot_xr(ds, var=v, level=l)
                plot_xr(ds, var=v, level=l, with_crs=False)
        else:
            plot_xr(ds, var=v)
            plot_xr(ds, var=v, with_crs=False)




    # tile_into_folders(serve_dir, noun, timestamp, image)

