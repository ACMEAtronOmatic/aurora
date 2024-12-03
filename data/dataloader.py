import math
import os

import numpy as np
import pytorch_lightning as pl
import xarray as xr
import torch
from PIL import Image

from .gfs_download import download_gfs, process_gfs
from .era5_download import download_static_era5

# Channel Map after concatenating ERA5 Static Variables
CHANNEL_MAP = {
    't': 0, 
    'u': 1,
    'v': 2,
    'r': 3, 
    'q': 4,
    'mslet': 5, 
    'slt': 6, 
    'gh': 7,
    'orog': 8, 
    'lsm': 9,
    'tv': 10,
    'theta': 11,
    'ns2': 12, 
    'z_era': 13, 
    'lsm_era': 14, 
    'slt_era': 15
}

# 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
LEVEL_MAP = {
    1000: 0,
    925: 1,
    850: 2,
    700: 3,
    600: 4,
    500: 5,
    400: 6,
    300: 7,
    250: 8,
    200: 9,
    150: 10,
    100: 11,
    50: 12
}


class GFSDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        start = config['data']['gfs']['time']['start']
        end = config['data']['gfs']['time']['end']
        arch_dir = config['data']['gfs']['archive']

        era5_download_path = config['data']['era5']['download_path']
        static_tag = config['data']['era5']['static_tag']
        year = config['data']['era5']['year']
        month = config['data']['era5']['month']
        days = config['data']['era5']['days']

        self.gfs_path = os.path.join(arch_dir, f"gfs_{start}_{end}.nc")

        self.static_path = f"{era5_download_path}/static_{static_tag}.nc"
        self.surface_path = f"{era5_download_path}/{year}_{month:02d}_{days[0]:02d}-{days[-1]:02d}_surface.nc"
        self.atmos_path = f"{era5_download_path}/{year}_{month:02d}_{days[0]:02d}-{days[-1]:02d}_atmospheric.nc"

        # Load the datasets
        self.gfs = xr.open_dataset(self.gfs_path, engine="netcdf4")
        self.times = self.gfs.time.values
        self.levels = self.gfs.level.values

        era_static = xr.open_dataset(self.static_path, engine="netcdf4").isel(time=0) # load into tensor
        self.era_surface = xr.open_dataset(self.surface_path, engine="netcdf4")
        self.era_atmos = xr.open_dataset(self.atmos_path, engine="netcdf4")

        self.era_static_tensor = torch.from_numpy(era_static.to_array().values).to(dtype=torch.float32)


        # Print some info about the datasets
        # print("GFS Shape: ", {dim: self.gfs.sizes[dim] for dim in self.gfs.dims})
        # print("GFS Variables: ", self.gfs.data_vars.keys())
        # print("ERA Statics Shape: ", {dim: era_static.sizes[dim] for dim in era_static.dims})
        # print("ERA Surface Shape: ", {dim: self.era_surface.sizes[dim] for dim in self.era_surface.dims})
        # print("ERA Atmos Shape: ", {dim: self.era_atmos.sizes[dim] for dim in self.era_atmos.dims})
        # print("ERA Statics Variables: ", era_static.data_vars.keys())
        # print("ERA Surface Variables: ", self.era_surface.data_vars.keys())
        # print("ERA Atmos Variables: ", self.era_atmos.data_vars.keys())


    def __len__(self):
        return len(self.times)

    def __getitem__(self, index):
        # Return data at all levels and all variables
        # Assume that time steps across ERA and GFS are aligned

        gfs_slice = self.gfs.isel(time=index)

        # Dimensions: [channel, level, lat, lon]
        gfs_tensor = torch.from_numpy(gfs_slice.to_array().values).to(dtype=torch.float32)

        # Need to combine channel and level dimensions: [channel*level, lat, lon]
        gfs_tensor = gfs_tensor.view(gfs_tensor.shape[0]*gfs_tensor.shape[1],
                                     gfs_tensor.shape[2], gfs_tensor.shape[3])

        # print("GFS Tensor Shape: ", gfs_tensor.shape)
        # print("ERA Statics Tensor Shape: ", self.era_static_tensor.shape)

        input_tensor = torch.cat((gfs_tensor, self.era_static_tensor), dim=0)

        # Need to combine channel and level dimensions: [channel*level, lat, lon]
        # input_tensor = input_tensor.view(input_tensor.shape[0]*input_tensor.shape[1],
        #                                   input_tensor.shape[2], input_tensor.shape[3])

        print("Input Tensor Shape: ", input_tensor.shape)

        self.input_shape = input_tensor.shape

        era_surface_slice = self.era_surface.isel(time=index)
        era_atmos_slice = self.era_atmos.isel(time=index)

        # Surface: [channel, lat, lon]
        era_surface_tensor = torch.from_numpy(era_surface_slice.to_array().values).to(dtype=torch.float32)

        # Atmos: [channel, level, lat, lon]
        era_atmos_tensor = torch.from_numpy(era_atmos_slice.to_array().values).to(dtype=torch.float32)

        # New Atmos: [channel*level, lat, lon]
        era_atmos_tensor = era_atmos_tensor.view(era_atmos_tensor.shape[0] * era_atmos_tensor.shape[1],
                                                  era_atmos_tensor.shape[2], era_atmos_tensor.shape[3])

        # Static: [channel, lat, lon]
        print("ERA Static Tensor Shape: ", self.era_static_tensor.shape)
        print("ERA Surface Tensor Shape: ", era_surface_tensor.shape)
        print("ERA Atmos Tensor Shape: ", era_atmos_tensor.shape)

        # Final truth tensor should have shape: [channel, level, lat, lon]
        # [9, 13, 721, 1440]
        # If combining channel*level:
        # [117, 721, 1440]

        # Repeat surface tensor for each level
        # NOTE: not needed if combining channel and level dimensions
        # era_surface_tensor = era_surface_tensor.unsqueeze(1).repeat(1, len(self.levels), 1, 1)
        truth_tensor = torch.cat((era_surface_tensor, era_atmos_tensor), dim=0)


        self.output_shape = truth_tensor.shape

        print("Truth Tensor Shape: ", truth_tensor.shape)

        return input_tensor, truth_tensor


class GFSDataModule(pl.LightningDataModule):
    '''
    Should the data module download ERA5 data as well?
    '''

    def __init__(self, configs, train_size=0.9, batch_size=4):
        super().__init__()
        self.configs = configs
        self.train_size = train_size
        self.batch_size = batch_size

        start = self.configs['data']['gfs']['time']['start']
        end = self.configs['data']['gfs']['time']['end']
        archDir = self.configs['data']['gfs']['archive']
        static_tag = self.configs['data']['era5']['static_tag']

        self.gfs_path = os.path.join(archDir, f"gfs_{start}_{end}.nc")
        self.static_path = f"{archDir}/static_{static_tag}.nc"



    def prepare_data(self):
        if not os.path.exists(self.gfs_path):
            # Download GFS data if it is not already downloaded
            print("Downloading GFS Data: ", self.gfs_path)
            download_gfs(self.configs['data']['gfs'])

            # Process GFS data
            # NOTE: isobaricInhPa was renamed to pressure
            process_gfs(self.configs['data']['gfs'])
        else:
            print("Processed GFS Data for this time range found")


        if not os.path.exists(self.static_path):
            # Download ERA5 static data if it is not already downloaded
            print("Downloading ERA5 Static Data: ", self.static_path)
            download_static_era5(self.configs['data']['gfs'])

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        gfs_dataset = GFSDataset(config=self.configs)
        training_size = math.floor(len(gfs_dataset) * self.train_size)
        val_size = len(gfs_dataset) - training_size


        train_dataset, val_dataset = torch.utils.data.random_split(
            gfs_dataset, [training_size, val_size]
        )

        gfs_dataset.__getitem__(0) # saves shape
        self.input_shape = gfs_dataset.input_shape
        self.output_shape = gfs_dataset.output_shape

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        del gfs_dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            self.batch_size,
            num_workers=16,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            self.batch_size,
            num_workers=16,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, self.batch_size)