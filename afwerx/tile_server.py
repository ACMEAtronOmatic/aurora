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
from visualize import plot_xr, rain_palette, apply_colormap
import yaml
from datetime import datetime
from io import BytesIO
import boto3
from botocore.exceptions import ClientError
import ast



# NOTE: must be installed through conda
import xesmf as xe


wasabi_endpoint = os.environ['WASABI_ENDPOINT']
wasabi_bucket   = os.environ['WASABI_BUCKET']
idrive_endpoint = os.environ['IDRIVE_ENDPOINT']
idrive_bucket   = os.environ['IDRIVE_BUCKET']
wasabi_secret   = os.environ['WASABI_SECRET_KEY']
wasabi_key      = os.environ['WASABI_ACCESS_KEY']
idrive_secret   = os.environ['IDRIVE_ACCESS_SECRET']
idrive_key      = os.environ['IDRIVE_ACCESS_KEY']

print("Wasabi: ")
print("\tEndpoint: ", wasabi_endpoint)
print("\tBucket: ", wasabi_bucket)

print("IDrive: ")
print("\tEndpoint: ", idrive_endpoint)
print("\tBucket: ", idrive_bucket)


NUM_CORES = multiprocessing.cpu_count()
N = 16384
TILE_SIZE = 256
ZOOM = 7

CONUS_Y_SIZE = 4608
CONUS_X_SIZE = 7168

# Aurora size: 720, 1440
# CONUS Size: 4608, 7168

minTileX = 0
minTileY = 0
maxTileX = CONUS_X_SIZE // TILE_SIZE
maxTileY = CONUS_Y_SIZE // TILE_SIZE

minX = minTileX * TILE_SIZE
minY = minTileY * TILE_SIZE
maxX = maxTileX * TILE_SIZE
maxY = maxTileY * TILE_SIZE

SKIP_PLOT = ["latitude", "longitude", "time", "batch", "history", "rollout_step", "level"]


def regrid(data : xr.Dataset, mosaic_data : xr.Dataset, regrid_file : str, output_path : str) -> xr.Dataset:
    '''
    Interpolate 0.25 degree GFS/Aurora/Mercator data to GOES CONUS grid

    Parameters
    ----------
    data : xr.Dataset
        Xarray dataset to be regridded
    regrid_file : str
        Full path to the xESMF regridding file. 
        Likely GFS/Aurora/Mercator 0.25 degree to GOES CONUS Zoom 7
    '''

    regridder = xe.Regridder(ds_in=data, ds_out=mosaic_data, method='bilinear', filename=regrid_file, reuse_weights=True)

    data_regridded = regridder(data)

    # Print attributes of this dataset
    print("\n")
    print("Regridded Dataset: ", data_regridded.dims)

    minlon = data_regridded['longitude'].min().values
    maxlon = data_regridded['longitude'].max().values
    minlat = data_regridded['latitude'].min().values
    maxlat = data_regridded['latitude'].max().values

    print("Lons: ", minlon, " - to - ", maxlon)
    print("Lats: ", minlat, " - to - ", maxlat)

    return data_regridded


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

    num_data_channels = len(data.shape)

    # rgba --> bgra, which is the order for the RGBA 32-bit representation 
    bgra = [2, 1, 0, 3]

    # tile = np.empty((TILE_SIZE, TILE_SIZE, num_data_channels), dtype=np.uint8)

    # # Remap colors for each channel individually
    # for chan in range(num_data_channels):
    #     print("Channel: ", chan)   
    #     print("\tTile Channel Shape: ", type(tile), tile[:, :, chan].shape)
    #     print("\tData Shape: ", type(data), data.shape)
    #     print("\tData Channel Shape: ", type(data), data[:, :, chan].shape)

    #     # If appplying rain palette
    #     tile[:, :, chan] = rain_palette[data[:, :, chan], bgra[chan]]

    tile = apply_colormap(data[:, :, 0], cmap='turbo')

    print("Tile Shape: ", tile.shape)   
    print("Saving image to: ", filename)

    file_dir = os.path.join(*filename.split('/')[:-1])
    if not os.path.exists(file_dir):
        print("Creating directory: ", file_dir)
        os.makedirs(file_dir)

    # CV write
    sts = cv2.imwrite(filename, tile)

    print(f"{filename} Status: ", sts)

    # PIL Image write
    # img = Image.fromarray(tile)
    # img.save(filename)


def tile_into_folders(serve_dir : str, noun : str, timestamp : str, image : np.ndarray, remap : bool = True):
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

    # Check the image shape
    if len(image.shape) == 2:
        # Repeat the image 3 times
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    # Check if 3 dimensions, but only one channel
    elif len(image.shape) == 3 and image.shape[2] == 1:
        # Repeat the image 3 times
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    if remap:
        max = np.max(image)
        min = np.min(image)

        image = 255.0 * ((image - min) / (max - min))

    image = image.astype(np.uint8)

    # Save the image
    cv2.imwrite("tiles/pre-tiled-image.png", image)

    # Iterate through minX to maxX tiles
    for x in np.arange(minTileX, maxTileX):
        # folder = serve_dir + noun + timestamp + '/'
        # sts = Popen('mkdir -p ' + folder + '{}/{}'.format(7, x), shell=True).wait()

        folder = os.path.join(serve_dir, noun, timestamp, '{}/{}'.format(7, x))
        os.makedirs(folder, exist_ok=True)

        for y in np.arange(minTileY, maxTileY):

            y1 = y*TILE_SIZE-minY
            y2 = y*TILE_SIZE-minY+TILE_SIZE

            x1 = x*TILE_SIZE-minX
            x2 = x*TILE_SIZE-minX+TILE_SIZE

            print("Image Tile: ", image[y1 : y2, x1 : x2].shape)

            filename = f"{y}.png"
            file_path = os.path.join(folder, filename)

            print("File Path: ", file_path)

            sts = save_tile(file_path, image[y1 : y2, x1 : x2])
                

def process_aurora_preds(filename : str):
    ds = xr.open_dataset(filename, engine='netcdf4')

    # Print coordinates and lat/lon before processing
    # Lats: 90 to -90
    # Lons: 
    print("Coordinates before processing: ", ds.coords)
    print("Lats before processing: ", ds['latitude'].values[:10], ds['latitude'].values[-10:])
    print("Lons before processing: ", ds['longitude'].values[:10], ds['longitude'].values[-10:])

    # Attempt to pad Aurora
    ds = ds.pad({"latitude": (0, 1)}, mode='wrap')

    # Reassign latitude values from [90, -89.75] to [90.0, -90.0]
    new_lats = np.linspace(90.0, -90.0, 721)

    ds = ds.assign_coords(latitude=new_lats, longitude=ds.longitude)

    # ds = ds.assign_coords(latitude=ds.latitude, longitude=ds.longitude - 180.0)
    ds = ds.assign_attrs({'latitude': {'units': 'degrees_north'},
                        'longitude': {'units': 'degrees_east'}})
    
    # Print ranges for lat, lon, time, and level
    print("\n")
    print("Lats after processing: ", ds['latitude'].dtype, ds['latitude'].values[:10], ds['latitude'].values[-10:])
    print("Lons after processing: ", ds['longitude'].dtype, ds['longitude'].values[:10], ds['longitude'].values[-10:])
    print("Time Range: ", ds['time'].dtype, ds['time'].min().values, " - to - ", ds['time'].max().values)
    print("Levels: ", ds['level'].values)

    # Print name and shape of each variable
    print("Variables: ")
    for var in ds.variables:
        print("\t", var, ds[var].shape)
    
    return ds


def select_conus(ds):
    # Extract only CONUS data from the xarray dataset
    minLat = 55.0 - 0.25
    maxLat = 20.0
    minLon = -130.0
    maxLon = -60.0 - 0.25

    ds = ds.sel(latitude=slice(minLat, maxLat), longitude=slice(minLon, maxLon))

    print("New Lat Range: ", type(ds['latitude'].min().values), ds['latitude'].min().values, " - to - ", ds['latitude'].max().values)
    print("New Lon Range: ", type(ds['longitude'].min().values), ds['longitude'].min().values, " - to - ", ds['longitude'].max().values)

    return ds


def myradar_dataset(dtg : datetime, output_path : str, wasabi : bool = False): 

    # Check if the mosaic dataset already exists
    name = dtg.strftime('mosaic_CONUS_%Y_%m_%d_%H_%M.png')

    full_path = os.path.join(output_path, name)

    if os.path.exists(full_path):
        print("Mosaic already exists: ", full_path)
        
        # Read in the dataset
        im = Image.open(full_path)
        out = np.asarray(im, np.uint8) # Single channel
    else:
        print("Loading AWS S3 client...")

        if wasabi:
            my_bucket = wasabi_bucket
            s3 = boto3.client('s3',
                endpoint_url          = wasabi_endpoint,
                aws_access_key_id     = wasabi_key,
                aws_secret_access_key = wasabi_secret)
            
        else:
            my_bucket = idrive_bucket
            s3 = boto3.client('s3',
                endpoint_url          = idrive_endpoint,
                aws_access_key_id     = idrive_key,
                aws_secret_access_key = idrive_secret)
        
        print("S3 Client Loaded: ", s3)

        response = s3.list_buckets()
        print("Buckets: ")
        for bucket in response['Buckets']:
            print("\t", bucket['Name'])

        # response = s3.list_objects_v2(
        #     Bucket=my_bucket,
        #     Delimiter='/'
        # )
        # print(response)

        yy, mon, dd, hh, mm, _ = dtg.strftime('%Y %m %d %H %M %s').split()
        with BytesIO() as f:
            aa_path = dtg.strftime('mosaic/CONUS/%Y/%m/%d/%H/%M.png')
            print("AA File Path: ", aa_path)

            _ = s3.download_fileobj(my_bucket, aa_path, f)
            im = Image.open(f)

            out = np.asarray(im, np.uint8) # Single channel

    # Get some grid information from the metadata from the PNG file.
    im.getdata()
    tileb = im.info['tile_bounds']
    tileinfo = ast.literal_eval(tileb)
    lb = tileinfo['TileMinX'] ; rb = tileinfo['TileMaxX']
    bb = tileinfo['TileMinY'] ; tb = tileinfo['TileMaxY']
    zoom = tileinfo['Zoom']

    px = im.width // (rb - lb + 1) ; py = im.height // (tb - bb + 1)

    scale = np.power(2,zoom)

    xgrid = (np.float64(lb)*px + np.arange(0.,np.float64(im.width),1.0))/px
    ygrid = (np.float64(bb)*py + np.arange(0.,np.float64(im.height),1.0))/py

    xx,yy = np.meshgrid(xgrid,ygrid)

    longitude = ( xx / scale  * 360.0) - 180.0
    latitude = np.degrees(np.arctan( np.sinh( np.pi * (1.0  -  2.0 * yy / scale ) ) ) )

    data_vars = { 'refl':(['y', 'x'], out,
                            {'units': 'dBZ*3',
                            'long_name':'compressed radar mosaic'})}
    coords = {'longitude': (['y', 'x'], longitude),
                'latitude': (['y', 'x'], latitude)}

    attrs = { 'description' : 'MyRadar mosaic' }

    ds = xr.Dataset(data_vars = data_vars, coords = coords, attrs = attrs)

    # Print coordinates from myradar mosaic
    # Note that lat/lon is a meshgrid, with shape (y, x) for both
    print("Mosaic Dims: ", ds.dims)
    print("Mosaic Coords: ")
    print(ds.coords)

    im.close()

    return ds


if __name__ == '__main__':

    # Set variables
    serve_dir = "serve"
    noun = "test"
    timestamp = '20240101T0000Z'
    visualize = True

    # set to 'afwerx' if using vscode debugger
    dir = "afwerx"

    # Load configs
    config_path = os.path.join(dir, "configs", "configs.yml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Read in downloaded aurora predictions
    preds_path = os.path.join(dir, "tiles", "aurora_predictions_raw.nc")
    ds = process_aurora_preds(preds_path)

    mosaic_output_path = os.path.join(dir, "tiles")
    myradar_mosaic = myradar_dataset(dtg=datetime(2023, 1, 1, 0, 0), output_path=mosaic_output_path)

    plot_path = os.path.join(dir, "tiles")
    # if visualize:
    #     # Plot plot some variables as a sanity check
    #     for v in ['atmos_v', 'surf_2t', 'static_z']:
    #         if v in SKIP_PLOT:
    #             continue

    #         print("Plotting variable: ", v)
    #         # Check if the variable is atmospheric, has levels
    #         if 'level' in ds[v].dims:
    #             for l in [1000, 700, 300]:
    #                 plot_xr(ds, var=v, level=l, output_path=plot_path)
    #                 plot_xr(ds, var=v, level=l, with_crs=False, output_path=plot_path)
    #         else:
    #             plot_xr(ds, var=v, output_path=plot_path)
    #             plot_xr(ds, var=v, with_crs=False, output_path=plot_path)

    regrid_path = os.path.join(dir, "tiles")
    ds_regridded = regrid(data=ds, mosaic_data=myradar_mosaic, regrid_file=config['inference']['interpolation_file'], output_path=regrid_path)

    # Visualize the regridded dataset
    if visualize:
        # Plot plot some variables as a sanity check
        for v in ['atmos_v', 'surf_2t', 'static_z']:
            if v in SKIP_PLOT:
                continue

            print("Plotting variable: ", v)
            # Check if the variable is atmospheric, has levels
            if 'level' in ds_regridded[v].dims:
                for l in [1000, 700, 300]:
                    plot_xr(ds_regridded, var=v, level=l, conus=True, output_path=plot_path)
                    plot_xr(ds_regridded, var=v, level=l, with_crs=False, conus=True, output_path=plot_path)
            else:
                plot_xr(ds_regridded, var=v, conus=True, output_path=plot_path)
                plot_xr(ds_regridded, var=v, with_crs=False, conus=True, output_path=plot_path)

    # tile_into_folders(serve_dir, noun, timestamp, image)

