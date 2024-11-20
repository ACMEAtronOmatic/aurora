import yaml
from argparse import ArgumentParser
import numpy as np
import torch

from data.era5_download import download_era5, make_batch
from data.gfs_download import download_gfs, process_gfs
from inference.generate_outputs import generate_outputs, visualize_outputs, era5_comparison
from inference.check_configs import check_configs
from aurora import Aurora

def main():
    DESCRIPTION = 'Aurora Module'
    parser = ArgumentParser(description = DESCRIPTION)

    parser.add_argument('yaml_file', help = 'YAML file with data & training guidelines.')
    parser.add_argument("--gfs", action = "store_true", help = "Download GFS data to compare with ERA5")

    args = parser.parse_args()

    use_gfs = args.gfs

    with open(args.yaml_file, 'r') as file:
        config = yaml.safe_load(file)
        print(config)
        # TODO: add GFS data checks to the config_checker
        # config = check_configs(config)
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

    if use_gfs:
        # Download GFS data if it is not already downloaded
        download_gfs(config['data']['gfs'])

        # Process GFS data
        gfs_ds, gfs_varlist = process_gfs(config['data']['gfs'])

        # TODO: Compare GFS data with ERA5 data

    # Download ERA5 data                
    static_path, surface_path, atmos_path = download_era5(config['data']['era5'])

    if config['inference']['variable'] == "wind":
        comparison_variable = 'wind'
    elif config['inference']['variable'] == "2t":
        comparison_variable = 't2m'
    elif config['inference']['variable'] == "msl":
        comparison_variable = 'msl'
    else:
        raise ValueError(f"Don't know which ERA5 variable name to use for comparison from Aurora variable {config['inference']['variable']}")

    baseline = era5_comparison(steps=28, 
                          variable=comparison_variable, 
                          data_path=surface_path)

    print("ERA5 Baseline: ", baseline.keys(), baseline[list(baseline.keys())[0]].shape)

    
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

    print("Visualizing...")
    visualize_outputs(preds, steps=steps, variable=variable, comparison_data=baseline)
    print("Visualizations created!")

if __name__ == "__main__":
    main()

