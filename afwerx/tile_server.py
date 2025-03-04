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


def regrid(data : xr.Dataset, regrid_file : str, conus : bool = False) -> xr.Dataset:
    '''
    Interpolate 0.25 degree GFS/Aurora/Mercator data to GOES CONUS grid

    Parameters
    ----------
    data : xr.Dataset
        Xarray dataset to be regridded
    regrid_file : str
        Full path to the xESMF regridding file. 
        Likely GFS/Aurora/Mercator 0.25 degree to GOES CONUS Zoom 7
    conus : bool = False
        Whether to regrid to the CONUS extent
    '''

    # Create a dummy target dataset of the CONUS extent
    # MRMS CONUS domain boundaries
    if conus:
        lat_min, lat_max = 20.0, 55.0  # degrees North
        lon_min, lon_max = -130.0, -60.0  # degrees West (converted to negative values)
    else:
        # global extent
        lat_min, lat_max = -90.0, 90.0  # degrees North
        lon_min, lon_max = -180.0, 180.0  # degrees West

    # Create coordinates at MRMS resolution (0.01°)
    lat = np.arange(lat_min, lat_max, 0.01)
    lon = np.arange(lon_min, lon_max, 0.01)

    # Create the xarray Dataset with proper attributes
    ds_dummy = xr.Dataset(
        coords={
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "lon": (["lon"], lon, {"units": "degrees_east"}),
        }
    )

    # Print the shapes of the data and the dummy dataset
    print("Data Lat Range: ", type(data['latitude'].min().values), data['latitude'].min().values, " - to - ", data['latitude'].max().values)
    print("Data Lon Range: ", type(data['longitude'].min().values), data['longitude'].min().values, " - to - ", data['longitude'].max().values)

    # Get shape of latitude and longitude
    print("Data Shape: ", len(data['latitude']), len(data['longitude']))

    print("Dummy Lat Range: ", type(ds_dummy['lat'].min().values), ds_dummy['lat'].min().values, " - to - ", ds_dummy['lat'].max().values)
    print("Dummy Lon Range: ", type(ds_dummy['lon'].min().values), ds_dummy['lon'].min().values, " - to - ", ds_dummy['lon'].max().values)
    print("Dummy Shape: ", len(ds_dummy['lat']), len(ds_dummy['lon']))


    # Open and inspect the weights file
    weights_ds = xr.open_dataset(regrid_file)

    print("Weights File: ")
    for var in weights_ds.variables:
        print("\t", var, weights_ds[var].shape)

    print("Weight rows range: ", type(weights_ds['row'].min().values),weights_ds['row'].min().values, " - to - ", weights_ds['row'].max().values)
    print("Weight cols range: ", type(weights_ds['col'].min().values), weights_ds['col'].min().values, " - to - ", weights_ds['col'].max().values)


    regridder = xe.Regridder(data, ds_dummy, method = 'bilinear', weights = regrid_file, reuse_weights=True)

    # Regrid
    ds_regridded = regridder(data)

    regridder.grid_in.destroy()
    regridder.grid_out.destroy()

    # Print new shape, latitude, and longitude range
    print("New Lat Range: ", type(ds_regridded['latitude'].min().values), ds_regridded['latitude'].min().values, " - to - ", ds_regridded['latitude'].max().values)
    print("New Lon Range: ", type(ds_regridded['longitude'].min().values), ds_regridded['longitude'].min().values, " - to - ", ds_regridded['longitude'].max().values)
    print("Shape: ", ds_regridded.shape)

    return ds_regridded




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
                

def process_aurora_preds(filename : str):
    ds = xr.open_dataset(filename, engine='netcdf4')

    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.assign_attrs({'latitude': {'units': 'degrees_north'},
                        'longitude': {'units': 'degrees_east'}})
    
    # Print ranges for lat, lon, time, and level
    print("Lat Range: ", type(ds['latitude'].min().values), ds['latitude'].min().values, " - to - ", ds['latitude'].max().values)
    print("Lon Range: ", type(ds['longitude'].min().values), ds['longitude'].min().values, " - to - ", ds['longitude'].max().values)
    print("Time Range: ", type(ds['time'].min().values), ds['time'].min().values, " - to - ", ds['time'].max().values)
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

    im.close()

    return ds


if __name__ == '__main__':
    serve_dir = "serve"
    noun = "test"
    timestamp = '20240101T0000Z'
    visualize = False

    with open("configs/configs.yml", 'r') as file:
        config = yaml.safe_load(file)

    ds = process_aurora_preds('tiles/test_preds.h5')

    myradar_mosaic = myradar_dataset(dtg=datetime(2023, 1, 1, 0, 0), output_path='tiles')

    # Print the latitude/longitude ranges and shapes of the myradar mosaic
    print("Mosaic Lat Range: ", type(myradar_mosaic['latitude'].min().values), myradar_mosaic['latitude'].min().values, " - to - ", myradar_mosaic['latitude'].max().values)
    print("Mosaic Lon Range: ", type(myradar_mosaic['longitude'].min().values), myradar_mosaic['longitude'].min().values, " - to - ", myradar_mosaic['longitude'].max().values)
    print("Mosaic Dims: ", myradar_mosaic.dims)

    # ds = select_conus(ds)

    if visualize:
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

    ds_regridded = regrid(data=ds, regrid_file=config['inference']['interpolation_file'], conus=False)

    # tile_into_folders(serve_dir, noun, timestamp, image)

