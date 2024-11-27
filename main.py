import yaml
import os
from argparse import ArgumentParser
import numpy as np
import torch

from data.era5_download import download_era5, make_batch
from data.gfs_download import download_gfs, process_gfs
from inference.generate_outputs import generate_outputs, \
    visualize_outputs, era5_comparison, \
          gfs_comparison, visualize_gfs_era5, \
          visualize_tensor
from inference.check_configs import check_configs
from data.dataloader import GFSDataset, GFSDataModule, CHANNEL_MAP, LEVEL_MAP
from aurora import Aurora

# If using MPS, some operations not yet implemented
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = 1

# gfs_dict = {
#     "wind": ['u', 'v'],
#     "temp": ['t']
# }
# era_dict = {
#     "wind": {
#         "surface": ['u10', 'v10'],
#         "atmo": ['u', 'v']
#     },
#     "temp": {
#         "surface": ['t2m'],
#         "atmo": ['t']
#     }
# }
# aurora_dict = {
#     "wind": ['u', 'v'],
#     "temp": ['2t']
# }



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

    # Download ERA5 data      
    # NOTE: valid_time was renamed to time          
    static_path, surface_path, atmos_path = download_era5(config['data']['era5'])

    era_level = "surface" if level==1000 else level
    print("ERA5 Level: ", era_level)
    # era_variable = era_dict[config['inference']['variable']][era_level if era_level == "surface" else "atmo"]

    if visualize:
        era_data = surface_path if era_level == "surface" else atmos_path
        era5_baseline = era5_comparison(steps=config['inference']['steps'], 
                            variable=config['inference']['variable'], 
                            data_path=era_data, 
                            level=era_level)

        print("ERA5 Baseline: ", era5_baseline.keys(), era5_baseline[list(era5_baseline.keys())[0]].shape)

    if use_gfs:
        start = config['data']['gfs']['time']['start']
        end = config['data']['gfs']['time']['end']
        archDir = config['data']['gfs']['archive']

        gfs_path = os.path.join(archDir, f"gfs_{start}_{end}.nc")
        if not os.path.exists(gfs_path):
            # Download GFS data if it is not already downloaded
            download_gfs(config['data']['gfs'])

            # Process GFS data
            # NOTE: isobaricInhPa was renamed to pressure
            gfs_path = process_gfs(config['data']['gfs'])
        else:
            print("Processed GFS Data for this time range found")

        if visualize:
            gfs_baseline = gfs_comparison(gfs_path, 
                                        steps=config['inference']['steps'], 
                                        variable=config['inference']['variable'], level=level)
            
            print("GFS Baseline: ", gfs_baseline.keys(), gfs_baseline[list(gfs_baseline.keys())[0]].shape)

            visualize_gfs_era5(era5_data=era5_baseline, gfs_data=gfs_baseline,
                                steps=config['inference']['steps'], variable=config['inference']['variable'],
                                output_path="downloads", fps=4, format="mp4")
        
        # Test torch dataset
        print("Testing torch dataset...")
        torch_ds = GFSDataset(gfs_path=gfs_path, era_statics_path=static_path, configs=config)

        torch_tensor = torch_ds.__getitem__(0)

        print("Tensor Shape")
        print(torch_tensor.shape)

        # Extract data for one variable from the tensor
        # [channel, level, lat, lon]
        # q and level 850

        q_tensor = torch_tensor[CHANNEL_MAP['q'], LEVEL_MAP[850], :, :]

        visualize_tensor(q_tensor, output_path="test_q_tensor_visualization.png", variable="q")

        exit(0)

        # gfs_ds = GFSDataModule(configs=config)
        # gfs_ds.prepare_data()
        # gfs_ds.setup()

        # batch = next(iter(gfs_ds.train_dataloader()))

        # print(batch)
        
        exit(0)

    
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
    visualize_outputs(preds, steps=steps, variable=variable, comparison_data=era5_baseline)
    print("Visualizations created!")

if __name__ == "__main__":
    main()

