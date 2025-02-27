import os
import numpy as np
import multiprocessing
from subprocess import Popen
import cv2
from PIL import Image
from scipy.interpolate import RectBivariateSpline as RBS
import xarray as xr

# NOTE: must be installed through conda
# import xesmf as xe


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

rain_colors = \
[0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 
0x0, 0x0, 0x0, 0x0, 0x0, 0x2000000, 0x5000000, 0x7000000, 0xa010000, 0xc010100, 
0xf020100, 0x11030200, 0x14040300, 0x16050300, 0x19070400, 0x1b080500, 0x1e0a0700, 0x200c0800, 0x230e0900, 0x25100a00, 
0x28130c00, 0x2a150d00, 0x2c170f00, 0x2f1a1100, 0x311d1200, 0x34201500, 0x36231600, 0x39261900, 0x3b2a1b00, 0x3e2e1e00, 
0x40312000, 0x43352300, 0x45392500, 0x483e2800, 0x4a422b00, 0x4d472e00, 0x4f4b3100, 0x504c3100, 0x534f2f00, 0x57522d00, 
0x5a552900, 0x5d582600, 0x615c2300, 0x645f1f00, 0x67621a00, 0x6b651600, 0x6e681101, 0x716b0b01, 0x756f0501, 0x78720001, 
0x79007700, 0x7a007700, 0x7a007600, 0x7b007600, 0x7c007600, 0x7c007400, 0x7d007400, 0x7e007400, 0x7e007300, 0x7f007300, 
0x80017201, 0x80017101, 0x81007100, 0x82007000, 0x83007000, 0x83006f00, 0x84006e00, 0x85006e00, 0x85006d00, 0x86006c00, 
0x87006c00, 0x87006b00, 0x88006a00, 0x89006a00, 0x89006800, 0x8a006800, 0x8b006800, 0x8b006600, 0x8c006600, 0x8d006500, 
0x8d006400, 0x8e006300, 0x8f006200, 0x8f006100, 0x90006100, 0x91006000, 0x92005f00, 0x92005e00, 0x93005d00, 0x94005d00, 
0x94005b00, 0x95005a00, 0x96005a00, 0x96005800, 0x97005700, 0x98005700, 0x98005500, 0x99005400, 0x9a019598, 0x9b019499, 
0x9b019399, 0x9c01939a, 0x9c01929a, 0x9d01919b, 0x9e01919c, 0x9e018f9c, 0x9f018f9d, 0x9f018e9d, 0xa0018d9e, 0xa1018d9f, 
0xa1008c9f, 0xa2008ba0, 0xa20089a0, 0xa30089a1, 0xa40089a2, 0xa40087a2, 0xa50087a3, 0xa50086a3, 0xa60085a4, 0xa70085a5, 
0xa70083a5, 0xa80083a6, 0xa80081a6, 0xa90080a7, 0xa9007fa7, 0xaa007ea8, 0xab007ea9, 0xab007ca9, 0xac007caa, 0xac007aaa, 
0xad007aab, 0xae0078ac, 0xae0077ac, 0xaf0076ad, 0xaf0075ae, 0xb00074af, 0xb10074b0, 0xb10072b0, 0xb20072b1, 0xb20070b1, 
0xb3006fb2, 0xb4006eb3, 0xb4006cb3, 0xb5006cb4, 0xb5006ab4, 0xb60069b5, 0xb70000b4, 0xb80000b4, 0xb80000b3, 0xb90000b3, 
0xba0000b2, 0xba0000b1, 0xbb0000b1, 0xbc0000b0, 0xbc0000af, 0xbd0000af, 0xbe0000af, 0xbe0000ad, 0xbf0000ad, 0xc00000ad, 
0xc00000ab, 0xc10000ab, 0xc20000ab, 0xc20000a9, 0xc30000a9, 0xc40000a8, 0xc40000a7, 0xc50000a6, 0xc50000a5, 0xc60000a5, 
0xc70000a4, 0xc70000a3, 0xc80000a3, 0xc90000a2, 0xc90000a0, 0xca0000a0, 0xcb0000a0, 0xcb00009e, 0xcc00009d, 0xcd00009d, 
0xcd00009c, 0xce00009b, 0xcf00009a, 0xcf000099, 0xd0000098, 0xd1ce00ca, 0xd2cc00c8, 0xd3c900c5, 0xd3c500c2, 0xd4c300bf,
0xd5c000bc, 0xd6bd00ba, 0xd7bb00b7, 0xd7b700b4, 0xd8b400b2, 0xd9b200af, 0xdaaf00ac, 0xdbab00a9, 0xdba800a5, 0xdca500a3, 
0xdda200a0, 0xde9f009d, 0xde9c009a, 0xdf980097, 0xe0950093, 0xe1920091, 0xe28f008e, 0xe28b008a, 0xe3880087, 0xe4850084, 
0xe5810081, 0xe67e007d, 0xe67b007a, 0xe7770077, 0xe8740074, 0x0, 0x0, 0x0, 0x0, 0x0, 
0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 
0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 
0x0, 0x0, 0x0, 0x0]


def RGBAfromColor(ARGBcolor):
    blue = ARGBcolor & 255
    green = (ARGBcolor >> 8) & 255
    red = (ARGBcolor >> 16) & 255
    alpha = (ARGBcolor >> 24) & 255
    return [red, green, blue, alpha]

# 
rain_palette = np.empty((len(rain_colors), 4))
for i, color in enumerate(rain_colors):
    rain_palette[i] = RGBAfromColor(color)


def regrid(data):
    '''
    Interpolate 0.25 degree GFS/Aurora/Mercator data to GOES CONUS grid

    Parameters
    ----------
    data : np.ndarray
        Data to be regridded
    regrid_file : str
        Full path to the xESMF regridding file
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
    print("Lat Range: ", ds['latitude'].min().values, " - to - ", ds['latitude'].max().values)
    print("Lon Range: ", ds['longitude'].min().values, " - to - ", ds['longitude'].max().values)
    print("Time Range: ", ds['time'].min().values, " - to - ", ds['time'].max().values)
    print("Level Range: ", ds['level'].min().values, " - to - ", ds['level'].max().values)

    # Print name and shape of each variable
    print("Variables: ")
    for var in ds.variables:
        print("\t", var, ds[var].shape)
        
    # tile_into_folders(serve_dir, noun, timestamp, image)

