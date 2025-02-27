import yaml
import os
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch
import xarray as xr

from inference.generate_outputs import generate_outputs
from data.era5_download import download_era5, make_batch
from aurora import Aurora

def main():
    DESCRIPTION = 'Aurora Module'
    parser = ArgumentParser(description = DESCRIPTION)

    parser.add_argument('yaml_file', help = 'YAML file with data & training guidelines.')
    parser.add_argument("-g", "--gfs", action = "store_true", help = "Download GFS data to compare with ERA5")
    parser.add_argument("-l", "--level", default=1000, help= "Pressure level to use in visualizations")
    parser.add_argument("-v", "--visualize", action = "store_true", help = "Visualize predictions")

    args = parser.parse_args()

    use_gfs = args.gfs
    level = int(args.level)
    visualize = args.visualize

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

    era_level = "surface" if level==1000 else level
    print("ERA5 Level: ", era_level)

# Create batch, (step - 1) >= 0
    print("Making batch...")
    batch = make_batch(static_path, surface_path, atmos_path, 1)
    print("Batch created!")

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
    preds = generate_outputs(model, batch, steps=steps, device=device)
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
    print("Converting to xarray...")
    ds = xr.Dataset(
        {
            **{f"surf_{k}": (("batch", "history", "latitude", "longitude"), v) 
            for k, v in preds[0].surf_vars.items()},
            **{f"static_{k}": (("latitude", "longitude"), v) 
            for k, v in preds[0].static_vars.items()},
            **{f"atmos_{k}": (("batch", "history", "level", "latitude", "longitude"), v) 
            for k, v in preds[0].atmos_vars.items()},
        },
        coords={
            "latitude": preds[0].metadata.lat,
            "longitude": preds[0].metadata.lon,
            "time": list(preds[0].metadata.time),
            "level": list(preds[0].metadata.atmos_levels),
            "rollout_step": preds[0].metadata.rollout_step,
        },
    )

    # Save to HDF5 file
    ds.to_netcdf("tiles/test_preds.h5")


    # TODO: change this to make it configurable
    # R = u @ 1000 mb
    # G = u @ 1000 mb
    # B = t @ 1000 mb

    # From atmos: ['t', 'u', 'v', 'q', 'z']
    R_VAR = 'u'
    G_VAR = 'v'
    B_VAR = 'q'
    A_VAR = 't'

    # (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
    R_LEVEL = 0
    G_LEVEL = 0
    B_LEVEL = 0
    A_LEVEL = 0

    # Atmos tensors take shape: [b, t, c, h, w] where c is the pressure level
    # Surface tensors take shape: [b, t, h, w]
    # Static tensors take shape: [h, w]

    r = preds[0].atmos_vars[R_VAR][0, 0, R_LEVEL, :, :]
    g = preds[0].atmos_vars[G_VAR][0, 0, G_LEVEL, :, :]
    b = preds[0].atmos_vars[B_VAR][0, 0, B_LEVEL, :, :]
    a = preds[0].atmos_vars[A_VAR][0, 0, A_LEVEL, :, :]


    print("R: ", type(r), r.shape, r.min(), r.max())
    print("G: ", type(g), g.shape, g.min(), g.max())
    print("B: ", type(b), b.shape, b.min(), b.max())
    print("A: ", type(a), a.shape, a.min(), a.max())

    # Normalize the tensors to [0, 255]
    r_min, r_max = r.min(), r.max()
    g_min, g_max = g.min(), g.max()
    b_min, b_max = b.min(), b.max()
    a_min, a_max = a.min(), a.max()

    r = (r - r_min) / (r_max - r_min) * 255.0
    g = (g - g_min) / (g_max - g_min) * 255.0
    b = (b - b_min) / (b_max - b_min) * 255.0
    a = (a - a_min) / (a_max - a_min) * 255.0


    print("After normalizing:")
    print("R: ", type(r), r.shape, r.min(), r.max())
    print("G: ", type(g), g.shape, g.min(), g.max())
    print("B: ", type(b), b.shape, b.min(), b.max())
    print("A: ", type(a), a.shape, a.min(), a.max())

    rgba = np.dstack((r,g,b,a))

    print("RGBA: ", type(rgba), rgba.shape, rgba.min(), rgba.max())

    img = Image.fromarray(rgba.astype(np.uint8), mode='RGBA')

    # Save the image
    img.save("tiles/test_rgba.png")



if __name__ == "__main__":
    main()