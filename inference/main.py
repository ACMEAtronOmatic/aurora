import yaml
from argparse import ArgumentParser

from data_download import download_era5, make_batch
from generate_outputs import generate_outputs, visualize_outputs
from aurora import Aurora

def main():
    DESCRIPTION = 'Aurora Module'
    parser = ArgumentParser(description = DESCRIPTION)

    parser.add_argument('yaml_file', help = 'YAML file with training guidelines.')

    args = parser.parse_args()

    with open(args.yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    # Download ERA5 data
    static_path, surface_path, atmos_path = download_era5(config)

    # Create batch, (step - 1) >= 0
    batch = make_batch(static_path, surface_path, atmos_path, 1)

    # Load model
    model_name = config['inference']['model']
    model_checkpoint = config['inference']['checkpoint']
    use_lora = config['inference']['use_lora']

    model = Aurora(use_lora=use_lora)
    model.load_checkpoint(model_name, model_checkpoint)

    steps = config['inference']['steps']
    variable = config['inference']['variable']

    preds = generate_outputs(model, batch, steps=steps)
    visualize_outputs(preds, steps=steps, variable=variable)


if __name__ == "__main__":
    main()

