import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import cv2
from PIL import Image

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

rain_palette = np.empty((len(rain_colors), 4))
for i, color in enumerate(rain_colors):
    rain_palette[i] = RGBAfromColor(color)


def plot_xr(xr_data : xr.Dataset, var : str, level : int = None,
             with_crs : bool = True, conus : bool = False, output_path : str = ""):
    
    print(f"Plotting variable: {var} @ {level} hPa")

    ds = xr_data[var]

    # ds_pandas = ds.to_dataframe().reset_index()
    # df_fig = plt.figure(figsize=(12, 6))
    # ax = df_fig.add_subplot(111)
    # ax.scatter(x=ds_pandas["longitude"], y=ds_pandas["latitude"], c=ds_pandas[var], cmap='rainbow')
    # df_fig.savefig(os.path.join(output_path, f"df_test_{var}.png"))

    # print(f"Plotting variable: {type(ds)} {var}")
    # print(f"{var} dimensions: ", ds.dims)
    for v in ds.dims:
        print("\t", v, ds[v].shape)

    if conus:
        central_lon = -95.5
    else:
        central_lon = 0.0

    if "batch" in ds.dims:
        ds = ds.isel(batch=0)
    if "history" in ds.dims:
        ds = ds.isel(history=0)
    if "time" in ds.dims:
        ds = ds.isel(time=0)

    fig = plt.figure(figsize=(12, 6))

    if level is not None:
        fig.suptitle(f"{var} at {level} hPa", fontsize=16)
        ds = ds.sel(level=level)
    else:
        fig.suptitle(f"{var} at Surface", fontsize=16)
        level="surface"

    min, max = ds.min().values, ds.max().values

    # Print lon/lat range
    # print("Plotting Lon Range: ", ds['latitude'].dtype, ds['longitude'].values[:1], " - to - ", ds['longitude'].values[-1:])
    # print("Plotting Lat Range: ", ds['latitude'].dtype, ds['latitude'].values[:1], " - to - ", ds['latitude'].values[-1:])

    if with_crs:
        crs = ccrs.PlateCarree(central_longitude=central_lon)
        ax = fig.add_subplot(1, 1, 1, projection=crs)

        # Check if the coordinates are on a meshgrid
        if 'x' in ds.dims and 'y' in ds.dims:
            ds.plot.pcolormesh(x='x', y='y',vmin=min, vmax=max, cmap='turbo', ax=ax, transform=crs)
        else:
            im = ds.plot.imshow(vmin=min, vmax=max, x='longitude', y='latitude', cmap='turbo', ax=ax, transform=crs)

        ax.coastlines(linewidth=0.7)

        # Extra gridlines around CONUS
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                        xlocs=[-180, -165, -150, -140, -130, -120,
                                -110, -100, -90, -80, -70, -60, -45, 0, 60, 120, 180],
                        ylocs=[-90, -60, -30, 0, 20, 30, 40, 50, 60, 70, 80, 90],
                        linewidth=2, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        ax.add_feature(cfeature.LAKES, alpha=0.9)  
        ax.add_feature(cfeature.COASTLINE, zorder=10)

        if conus:
            name = f"{var}_{level}_crs_conus.png"
        else:
            name = f"{var}_{level}_crs.png"
    else:
        ax = fig.add_subplot(1, 1, 1)
        # Check if the coordinates are on a meshgrid
        if 'x' in ds.dims and 'y' in ds.dims:
            ds.plot.pcolormesh(x='x', y='y', vmin=min, vmax=max, cmap='turbo', ax=ax)
        else:
            im = ds.plot.imshow(vmin=min, vmax=max, x='longitude', y='latitude', cmap='turbo', ax=ax, transform=crs)
        if conus:
            name = f"{var}_{level}_conus.png"
        else:
            name = f"{var}_{level}.png"

    fig.tight_layout()

    full_path = os.path.join(output_path, name)

    fig.savefig(full_path)

    plt.close('all')


def apply_colormap(img, cmap='turbo'):
    # print("Input Dimensions: ", img.shape)
    colormap = plt.get_cmap(cmap)

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    # print("After cv2 Conversion: ", img.shape)

    # Put at half opacity
    colorized = colormap(img) * 255.0
    colorized[:, :, 3] = 128

    # print("After Colorization: ", colorized.shape)

    return colorized