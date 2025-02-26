import yaml
import os
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch

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

    # TODO: change this to make it configurable
    # R = u @ 1000 mb
    # G = u @ 1000 mb
    # B = t @ 1000 mb

    # From atmos: ['t', 'u', 'v', 'q', 'z']
    R_VAR = 't'
    G_VAR = 't'
    B_VAR = 't'

    # (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
    R_LEVEL = 0
    G_LEVEL = 2
    B_LEVEL = 5

    # Atmos tensors take shape: [b, t, c, h, w] where c is the pressure level
    # Surface tensors take shape: [b, t, h, w]
    # Static tensors take shape: [h, w]

    r = preds[0].atmos_vars[R_VAR][0, 0, R_LEVEL, :, :]
    g = preds[0].atmos_vars[G_VAR][0, 0, G_LEVEL, :, :]
    b = preds[0].atmos_vars[B_VAR][0, 0, B_LEVEL, :, :]

    print("R: ", type(r), r.shape, r.min(), r.max())
    print("G: ", type(g), g.shape, g.min(), g.max())
    print("B: ", type(b), b.shape, b.min(), b.max())

    # Normalize the tensors to [0, 255]
    r_min, r_max = r.min(), r.max()
    g_min, g_max = g.min(), g.max()
    b_min, b_max = b.min(), b.max()

    r = (r - r_min) / (r_max - r_min) * 255.0
    g = (g - g_min) / (g_max - g_min) * 255.0
    b = (b - b_min) / (b_max - b_min) * 255.0

    print("After normalizing:")
    print("R: ", type(r), r.shape, r.min(), r.max())
    print("G: ", type(g), g.shape, g.min(), g.max())
    print("B: ", type(b), b.shape, b.min(), b.max())

    rgb = np.dstack((r,g,b))

    print("RGB: ", type(rgb), rgb.shape, rgb.min(), rgb.max())

    img = Image.fromarray(rgb.astype(np.uint8))

    # Save the image
    img.save("temp_at_levels.png")



if __name__ == "__main__":
    main()