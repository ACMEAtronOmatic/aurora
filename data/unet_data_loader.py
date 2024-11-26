import math
import os

import numpy as np
import pytorch_lightning as pl
import xarray as xr
import torch
from PIL import Image

from .gfs_download import download_gfs, process_gfs
from .era5_download import download_static_era5

class GFSDataset(torch.utils.data.Dataset):
    def __init__(self, gfs_path, era_statics_path, configs):
        super().__init__()
        self.gfs_path = gfs_path
        self.era_statics_path = era_statics_path

        # Load the datasets
        self.gfs = xr.open_dataset(self.gfs_path, engine="netcdf4")
        self.times = self.gfs.time.values
        self.levels = self.gfs.level.values

        self.era_static = xr.open_dataset(self.era_statics_path, engine="netcdf4").isel(time=0)
        self.era_tensor = torch.from_numpy(self.era_static.to_array().values).to(dtype=torch.float32)
        # Add a dimension for levels
        # Dimensions: ('variable', 'time', 'level', 'lat', 'lon')
        self.era_tensor = self.era_tensor.unsqueeze(1).repeat(1, len(self.levels), 1, 1)

        # Print some info about the datasets
        print("GFS Shape: ", {dim: self.gfs.sizes[dim] for dim in self.gfs.dims})
        print("GFS Levels: ", self.gfs.level.values)
        print("ERA Statics Shape: ", {dim: self.era_static.sizes[dim] for dim in self.era_static.dims})


    def __len__(self):
        return len(self.times)

    def __getitem__(self, index):
        # TODO: reimplement to get a batch of info from GFS + ERA at a particular time step
        # Return data at all levels and all variables

        gfs_slice = self.gfs.isel(time=index)

        # Dimensions: [time, level, lat, lon]
        # Which dimension is for variable?
        gfs_tensor = torch.from_numpy(gfs_slice.to_array().values).to(dtype=torch.float32)

        print("GFS Tensor Shape: ", gfs_tensor.shape)
        print("ERA Statics Tensor Shape: ", self.era_tensor.shape)

        combined_tensor = torch.cat((gfs_tensor, self.era_tensor), dim=0)

        return combined_tensor


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
        dataset = GFSDataset(gfs_path=self.gfs_path,
                              era_statics_path=self.static_path, configs=self.configs)
        training_size = math.floor(len(dataset) * self.train_size)
        val_size = len(dataset) - training_size

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device).manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [training_size, val_size], generator=generator
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

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