import os
import sys

sys.path.append('../')

import yaml
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from inference.generate_outputs import generate_outputs
from data.era5_download import download_era5, make_batch, batch_to_xr
from aurora import Aurora


def main():
    DESCRIPTION = 'Aurora Module'
    parser = ArgumentParser(description = DESCRIPTION)

    parser.add_argument('yaml_file', help = 'YAML file with data & training guidelines.')
    parser.add_argument("-b", "--backfill", default=8, help= "Number of past ERA5 time steps to include in outputted dataset")
    parser.add_argument("-f", "--forecasts", default=8, help= "Number of rollout steps")

    args = parser.parse_args()

    with open(args.yaml_file, 'r') as file:
        config = yaml.safe_load(file)
        print("Configs loaded!")

    # Check devices
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
            device = "cpu"
        else:
            device = "mps"
    else:
        device = "cpu"

    print("Using device: ", device)
    torch.set_default_device(device)

    # Download ERA5 data      
    # NOTE: valid_time was renamed to time          
    static_path, surface_path, atmos_path = download_era5(config['data']['era5'])

    # Open these datasets
    static = xr.open_dataset(static_path, engine="netcdf4")
    surface = xr.open_dataset(surface_path, engine="netcdf4")
    atmos = xr.open_dataset(atmos_path, engine="netcdf4")

    print("ERA5 Timestamps: ")
    print(surface.time[np.argsort(surface.time)[-8:]])
    

    # TODO:Create an xarray dataset from the ERA5 data
    # Create batch, (step - 1) >= 0
    
    datasets = []
    curr_timestamp = datetime(2023, 1, 6, 0)
    for b in range(args.backfill):
        print("Current Time Step: ", curr_timestamp)
        print("Timestamp Index: ", np.where(surface.time.values == np.datetime64(curr_timestamp))[0])
        curr_batch = make_batch(static, surface, atmos, curr_timestamp)

        print("Batch Created: ")
        print(type(curr_batch), curr_batch.metadata.time)

        curr_xr = batch_to_xr(curr_batch)
        datasets.append(curr_xr)

        curr_timestamp += timedelta(hours=6)

    era_xr_dataset = xr.concat(datasets, dim='time').sortby('time')

    print("ERA XR Dataset: ")
    print(era_xr_dataset.dims)
    print(era_xr_dataset.time)

    era_xr_dataset.to_netcdf('tiles/era5_xr_dataset.nc')

    # Load model
    model_name = config['inference']['model']
    model_checkpoint = config['inference']['checkpoint']
    use_lora = config['inference']['use_lora']

    print("Loading model...")
    model = Aurora(use_lora=use_lora)
    model.load_checkpoint(model_name, model_checkpoint)
    print("Model loaded!")

    steps = config['inference']['steps']
    variable = config['inference']['variable']

    print("Generating outputs...")
    preds = generate_outputs(model, curr_batch, steps=steps, device=device)
    print("Outputs generated!")
    print(type(preds), type(preds[0]))

    print("\nSurface Vars: ")
    print(preds[0].surf_vars.keys())
    print("Tensor: ", preds[0].surf_vars['2t'].shape)

    print("\nStatic Vars: ")
    print(preds[0].static_vars.keys())
    print("Tensor: ", preds[0].static_vars['z'].shape)

    print("\nAtmos Vars: ")
    print(preds[0].atmos_vars.keys())
    print("Tensor: ", preds[0].atmos_vars['u'].shape)

    # Print the metadata of the batch
    # lat, lon, time, atmos_levels, rollout_step

    lats = preds[0].metadata.lat
    lons = preds[0].metadata.lon
    times = preds[0].metadata.time
    atmos_levels = preds[0].metadata.atmos_levels
    rollout_step = preds[0].metadata.rollout_step

    print("\nMetadata: ")
    print("Lat: ", lats[0], lats[-1])
    print("Lon: ", lons[0], lons[-1])
    print("Time: ", times[0], times[-1])
    print("Atmos Levels: ", atmos_levels)
    print("Rollout Step: ", rollout_step)

    # Convert to xarray dataset, save as HDF5 file
    # TODO: generate this dataset for each timestep in the rollout and concatenate them along the time dimension

    print("Converting to xarray...")
    aurora_ds = batch_to_xr(preds)

    aurora_ds.to_netcdf(os.path.join("tiles", "aurora_predictions_raw_multitime.nc"), mode = 'w',
                            format = 'NETCDF4')
    
    # Combined the era5 and aurora preds into a single xarray dataset
    era_xr_dataset = era_xr_dataset.sel(history=0, batch=0)
    aurora_ds = aurora_ds.sel(history=0, batch=0)

    combined_ds = xr.concat([era_xr_dataset, aurora_ds], dim='time').sortby('time')

    # Latitude needs to be (90, -90) and longitude needs to be (-180, 180)
    new_lons = np.linspace(-180.0, 180.0, 1440)

    combined_ds = combined_ds.reindex(latitude=combined_ds.latitude[::-1])

    combined_ds = combined_ds.assign_coords(
        latitude=combined_ds.latitude.values,
        longitude=new_lons
    )

    combined_ds.to_netcdf(os.path.join("tiles", "combined_era_aurora_raw_multitime.nc"), mode = 'w',
                            format = 'NETCDF4')

    # # TODO: change this to make it configurable
    # # R = u @ 1000 mb
    # # G = u @ 1000 mb
    # # B = t @ 1000 mb

    # # From atmos: ['t', 'u', 'v', 'q', 'z']
    # R_VAR = 'u'
    # G_VAR = 'v'
    # B_VAR = 'q'
    # A_VAR = 't'

    # # (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
    # R_LEVEL = 0
    # G_LEVEL = 0
    # B_LEVEL = 0
    # A_LEVEL = 0

    # # Atmos tensors take shape: [b, t, c, h, w] where c is the pressure level
    # # Surface tensors take shape: [b, t, h, w]
    # # Static tensors take shape: [h, w]

    # r = preds[0].atmos_vars[R_VAR][0, 0, R_LEVEL, :, :]
    # g = preds[0].atmos_vars[G_VAR][0, 0, G_LEVEL, :, :]
    # b = preds[0].atmos_vars[B_VAR][0, 0, B_LEVEL, :, :]
    # a = preds[0].atmos_vars[A_VAR][0, 0, A_LEVEL, :, :]


    # print("R: ", type(r), r.shape, r.min(), r.max())
    # print("G: ", type(g), g.shape, g.min(), g.max())
    # print("B: ", type(b), b.shape, b.min(), b.max())
    # print("A: ", type(a), a.shape, a.min(), a.max())

    # # Normalize the tensors to [0, 255]
    # r_min, r_max = r.min(), r.max()
    # g_min, g_max = g.min(), g.max()
    # b_min, b_max = b.min(), b.max()
    # a_min, a_max = a.min(), a.max()

    # r = (r - r_min) / (r_max - r_min) * 255.0
    # g = (g - g_min) / (g_max - g_min) * 255.0
    # b = (b - b_min) / (b_max - b_min) * 255.0
    # a = (a - a_min) / (a_max - a_min) * 255.0


    # print("After normalizing:")
    # print("R: ", type(r), r.shape, r.min(), r.max())
    # print("G: ", type(g), g.shape, g.min(), g.max())
    # print("B: ", type(b), b.shape, b.min(), b.max())
    # print("A: ", type(a), a.shape, a.min(), a.max())

    # rgba = np.dstack((r,g,b,a))

    # print("RGBA: ", type(rgba), rgba.shape, rgba.min(), rgba.max())

    # img = Image.fromarray(rgba.astype(np.uint8), mode='RGBA')

    # # Save the image
    # img.save("tiles/test_rgba.png")

    # # Plot the a variable without a CRS
    # fig = plt.figure(figsize=(12, 6))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(r, cmap='gist_ncar')
    # ax.set_xticks(np.arange(0, 1440, 120))
    # ax.set_yticks(np.arange(0, 720, 120))

    # plt.savefig("tiles/raw_aurora_image.png")


if __name__ == "__main__":
    main()