'''
1. Download data if necessary
2. Instantiate DataLoader module
3. Instantiate Lightning module
4. Instantiate EarlyStopping and Checkpoint callbacks
5. Instantiate Trainer
6. trainer.fit()
7. Load best model and make predictions
8. Visualize GFS, ERA5, and predictions
'''
import os
import yaml
from argparse import ArgumentParser

from data.era5_download import download_era5, make_batch
from data.gfs_download import download_gfs, process_gfs
from data.dataloader import GFSDataModule
from data.model import LightningGFSUnbiaser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


def download_process_data(config):
    start = config['data']['gfs']['time']['start']
    end = config['data']['gfs']['time']['end']
    arch_dir = config['data']['gfs']['archive']

    era5_download_path = config['data']['era5']['download_path']
    static_tag = config['data']['era5']['static_tag']
    year = config['data']['era5']['year']
    month = config['data']['era5']['month']
    days = config['data']['era5']['days']

    gfs_path = os.path.join(arch_dir, f"gfs_{start}_{end}.nc")

    static_path = f"{era5_download_path}/static_{static_tag}.nc"
    surface_path = f"{era5_download_path}/{year}_{month:02d}_{days[0]:02d}-{days[-1]:02d}_surface.nc"
    atmos_path = f"{era5_download_path}/{year}_{month:02d}_{days[0]:02d}-{days[-1]:02d}_atmospheric.nc"

    if not os.path.exists(gfs_path):
        # Download GFS data if it is not already downloaded
        download_gfs(config['data']['gfs'])

        # Process GFS data
        # NOTE: isobaricInhPa was renamed to pressure
        gfs_path = process_gfs(config['data']['gfs'])
    else:
        print("Processed GFS Data for this time range found")


    if not os.path.exists(static_path) or not os.path.exists(surface_path) or not os.path.exists(atmos_path):
        static_path, surface_path, atmos_path = download_era5(config['data']['era5'])
    else:
        print("ERA5 Data for this time range found")

def main():
    DESCRIPTION = 'GFS-ERA5 Converter Module'
    parser = ArgumentParser(description = DESCRIPTION)

    parser.add_argument('yaml_file', help = 'YAML file with data & training guidelines.')
    parser.add_argument("-v", "--visualize", action = "store_true", help = "Visualize predictions")

    args = parser.parse_args()

    with open(args.yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    checkpoint_save_path = config['inference']['save_path']
    logs_save_path = config['inference']['logs_path']

    if not os.path.exists(checkpoint_save_path):
        os.makedirs(checkpoint_save_path)

    if not os.path.exists(logs_save_path):
        os.makedirs(logs_save_path)

    # Download GFS and ERA5 data
    download_process_data(config)

    # Instantiate dataloader
    dm = GFSDataModule(config)
    dm.prepare_data()
    dm.setup()

    # How many channels are being input and output?
    input_channels = dm.shape[0]

    # Need 9 output channels: 2t, 10u, 10v, msl, t, u, v, q, z
    # Output should be for every level, although surface data will only use the first level (1000)
    output_channels = 9

    print("Input Channels: ", input_channels, "Output Channels: ", output_channels)

    # # Instantiate Lightning module
    model = LightningGFSUnbiaser(in_channels=input_channels, out_channels=output_channels)

    # Instantiate Callbacks & Loggers
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=2, verbose=False, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=checkpoint_save_path,
        filename="gfs_converter-epoch-{epoch:02d}-val_loss-{val_loss:.2f}",
    )

    progress_callback = RichProgressBar()

    tensorboard_logger = TensorBoardLogger(logs_save_path, name="gfs_converter_unet")

    # Instantiate Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=15,
        callbacks=[early_stop_callback, checkpoint_callback, progress_callback],
        logger=tensorboard_logger,
    )


if __name__ == '__main__':
    main()
