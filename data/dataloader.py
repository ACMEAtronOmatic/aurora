import math
import os

import numpy as np
import pytorch_lightning as pl
import xarray as xr
import torch
from PIL import Image
import torch.nn.functional as F

from data.utils import print_debug, check_gpu_memory
from data.gfs_download import download_gfs, process_gfs
from data.era5_download import download_static_era5

ERA5_GLOBAL_RANGES = {
    # Surface Variables
    't2m': {'min': 170, 'max': 350},  # 2m Temperature [K]
    'msl': {'min': 85000, 'max': 110000},  # Mean Sea Level Pressure [Pa]
    'u10': {'min': -110, 'max': 110},  # 10m U-wind [m/s]
    'v10': {'min': -80, 'max': 80},  # 10m V-wind [m/s]
    
    # Atmospheric Variables (at all pressure levels)
    't': {'min': 160, 'max': 350},  # Temperature [K]
    'q': {'min': 0, 'max': 0.04},  # Specific Humidity [kg/kg]
    'u': {'min': -110, 'max': 110},  # U-wind [m/s]
    'v': {'min': -100, 'max': 100},  # V-wind [m/s]
    'z': {'min': -5000, 'max': 225000},  # Geopotential Height [m]
}

GFS_GLOBAL_RANGES = {
    # Atmospheric Variables (at all pressure levels)
    't': {'min': 160, 'max': 350},  # Temperature [K]
    'q': {'min': 0, 'max': 0.04},  # Specific Humidity [kg/kg]
    'u': {'min': -110, 'max': 110},  # U-wind [m/s]
    'v': {'min': -100, 'max': 100},  # V-wind [m/s]
    'z': {'min': -5000, 'max': 225000},  # Geopotential Height [m]

    # Surface Variables
    'lsm': {'min': 0, 'max': 1},  # Land Sea Mask
    'mslet': {'min': 85000, 'max': 110000},  # Mean Sea Level Pressure,
    'slt': {'min': 0, 'max': 16}, # Soil Type
    'orog': {'min': -500, 'max': 6000} # Orography (Elevation)
}

# Distinguish between GFS surface and atmospheric variables
# To minimize total number of channels
GFS_LAND_CHANNELS = ['lsm', 'mslet', 'slt', 'orog']
ERA_LAND_CHANNELS = ['u10', 'v10', 't2m', 'msl']

# Map from ERA5 to analogous GFS variables
ERA5_TO_GFS = {
    't': 't',
    'q': 'q',
    'u': 'u',
    'v': 'v',
    'z': 'gh',
    'msl': 'mslet',
    'u10': 'u',
    'v10': 'v',
    't2m': 't',
}


# Denote ERA5 Surface Channels, which can be roughly aligned with GFS Level 1000 channels


def interpolate_missing_values(tensor, threshold = 0.1):

    tensor = torch.nan_to_num(tensor, posinf=1e30, neginf=-1e30)

    # Check for NaN and Inf values
    mask = torch.isnan(tensor)

    # Check if there is a large number of missing values
    missing_ratio = mask.sum() / tensor.numel()
    if missing_ratio >= threshold:
        print("WARNING: tensor contains too many missing values: ", missing_ratio.item())

    # Interpolate missing values
    if missing_ratio > 0:
        interpolated_values = F.interpolate(tensor.unsqueeze(0), size=tensor.shape[1:], mode='nearest').squeeze(0)
        tensor[mask] = interpolated_values[mask]

    # Check for NaN and Inf values again
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print("\n***WARNING: tensor contains NaN or Inf values after interpolation***\n")

    return tensor

def normalize_tensor(tensor, var, dataset='GFS'):
    '''
    Normalize data based on global ranges for the specific variables/datasets

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to normalize
    var : str
        Variable name
    dataset : str
        Dataset name ['GFS', 'ERA5']

    Returns
    -------
    torch.Tensor
        Normalized tensor

    '''
    if dataset == 'GFS':
        min_val = GFS_GLOBAL_RANGES[var]['min']
        max_val = GFS_GLOBAL_RANGES[var]['max']
    else:
        min_val = ERA5_GLOBAL_RANGES[var]['min']
        max_val = ERA5_GLOBAL_RANGES[var]['max']

    tensor = torch.clamp((tensor - min_val) / (max_val - min_val), 0, 1)

    return tensor

def denormalize_tensor(tensor, var, dataset='GFS'):
    '''
    Returns normalized data to its orignal data space

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to denormalize
    var : str
        Variable name
    dataset : str
        Dataset name ['GFS', 'ERA5']

    Returns
    -------
    torch.Tensor
        Denormalized tensor
    '''
    if dataset == 'GFS':
        min_val = GFS_GLOBAL_RANGES[var]['min']
        max_val = GFS_GLOBAL_RANGES[var]['max']
    else:
        min_val = ERA5_GLOBAL_RANGES[var]['min']
        max_val = ERA5_GLOBAL_RANGES[var]['max']

    tensor = tensor * (max_val - min_val) + min_val

    return tensor

class GFSDataset(torch.utils.data.Dataset):
    '''
    Torch Dataset for the GFS Unbiasing UNet. 
    Input data is GFS surface and atmospheric variables at defined levels
    Target data is ERA5 surface and atmoshperic data at defined levels
    The output data can also be the residuals between ERA5 and GFS, training the UNet
    to only predict the changes necessary to unbias GFS

    Parameters
    ----------
    config : dict
        Configuration dictionary
    debug : bool
        Debug mode
    '''
    def __init__(self, config, debug=False):
        super().__init__()

        self.debug = debug
        self.normalize = config['data']['normalize']
        self.interpolate = config['data']['interpolate']
        self.residuals = config['data']['residuals']

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
        self.era_static_tensor = torch.from_numpy(era_static.to_array().values).to(dtype=torch.float32)

        self.era_surface = xr.open_dataset(self.surface_path, engine="netcdf4")
        self.era_atmos = xr.open_dataset(self.atmos_path, engine="netcdf4")

        # Get levels and number of channels for the atmos dataset, channels from the surface dataset
        self.era_atmos_levels = list(self.era_atmos.level.values)
        self.era_atmos_levels = [int(level) for level in self.era_atmos_levels]
        self.era_atmos_channels = list(self.era_atmos.data_vars.keys())
        self.era_surface_channels = list(self.era_surface.data_vars.keys())

        self.gfs_atmos_levels = list(self.gfs.level.values)
        self.gfs_atmos_levels = [int(level) for level in self.gfs_atmos_levels]

        # GFS surface and atmospheric channels will be together, we need to extract them
        # Extract land variables
        self.gfs_surface = self.gfs[GFS_LAND_CHANNELS].isel(time=0)
        self.gfs_surface_channels = list(self.gfs_surface.keys())
        self.gfs_atmos = self.gfs.drop_vars(GFS_LAND_CHANNELS)
        self.gfs_atmos_channels = list(self.gfs_atmos.keys())

        print_debug(self.debug, "ERA Surface Channels: ", self.era_surface_channels)
        print_debug(self.debug, "ERA Atmos Channels: ", self.era_atmos_channels)
        print_debug(self.debug, "ERA Atmos Levels: ", self.era_atmos_levels)

        print_debug(self.debug, "GFS Atmos Channels: ", self.gfs_atmos_channels)
        print_debug(self.debug, "GFS Atmos Levels: ", self.gfs_atmos_levels)
        print_debug(self.debug, "GFS Surface Channels: ", self.gfs_surface_channels)

        # Print some info about the datasets
        # print_debug(self.debug, "GFS Shape: ", {dim: self.gfs.sizes[dim] for dim in self.gfs.dims})
        # print_debug(self.debug, "GFS Variables: ", self.gfs.data_vars.keys())
        # print_debug(self.debug, "ERA Statics Shape: ", {dim: era_static.sizes[dim] for dim in era_static.dims})
        # print_debug(self.debug, "ERA Surface Shape: ", {dim: self.era_surface.sizes[dim] for dim in self.era_surface.dims})
        # print_debug(self.debug, "ERA Atmos Shape: ", {dim: self.era_atmos.sizes[dim] for dim in self.era_atmos.dims})
        # print_debug(self.debug, "ERA Statics Variables: ", era_static.data_vars.keys())
        # print_debug(self.debug, "ERA Surface Variables: ", self.era_surface.data_vars.keys())
        # print_debug(self.debug, "ERA Atmos Variables: ", self.era_atmos.data_vars.keys())

    def gfs_idx_to_variable(self, tensor_idx):
        '''
        Channel mapper from the channel index to the variable and level

        Parameters
        ----------
        tensor_idx : int
            Channel index

        Returns
        -------
        str
            Variable name
        str
            Level
        '''
        # Return the variable and level based off the channel index in the tensor
        # 4 land variables, 6 atmos variables, 13 levels
        surface_len = len(self.gfs_surface_channels)
        atmo_len = (len(self.gfs_atmos_channels) * len(self.gfs_atmos_levels))

        # NOTE: starts with atmos, ends with surface
        if tensor_idx < atmo_len:
            # Directly index
            return self.gfs_atmos_channels[tensor_idx//len(self.gfs_atmos_levels)], self.gfs_atmos_levels[tensor_idx%len(self.gfs_atmos_levels)]
        
        elif atmo_len <= tensor_idx < atmo_len + surface_len:
            # Surface
            tensor_idx -= atmo_len
            return self.gfs_surface_channels[tensor_idx], 'surface'
        
        else:
            # It is one of the ERA5 static variables
            return self.era_static_tensor[tensor_idx - atmo_len - surface_len], 'surface'

    def era_idx_to_variable(self, tensor_idx):
        '''
        Channel mapper from the channel index to the variable and level

        Parameters
        ----------
        tensor_idx : int
            Channel index

        Returns
        -------
        str
            Variable name
        str
            Level
        '''
        # Example: idx 9, channel 2, level 1 of 3 channels, 4 levels
        # To get channel 2 from idx 9: 9//4 == 2
        # To get level 1 from idx 9: 9%4 == 1
        surface_len = len(self.era_surface_channels)

        # NOTE: starts with surface, then has atmos
        if tensor_idx >= surface_len:
            # Subtracting number of surface channels
            # Since their information is stored in separate tensors
            tensor_idx -= surface_len
            return self.era_atmos_channels[tensor_idx//len(self.era_atmos_levels)], self.era_atmos_levels[tensor_idx%len(self.era_atmos_levels)]
        else:
            return self.era_surface_channels[tensor_idx], 'surface'

    def __len__(self):
        return len(self.times)

    def __getitem__(self, index):
        '''
        Returns a single timestep of (input, target/truth) data using the given configs
        Concatenates surface & atmospheric data together, interpolates missing values, and normalizes
        '''
        # Return data at all levels and all variables
        # Assume that time steps across ERA and GFS are aligned

        gfs_slice = self.gfs_atmos.isel(time=index)

        # Print the order of the variables in gfs slice
        for i, var in enumerate(gfs_slice.data_vars.keys()):
            print_debug(self.debug, f"GFS Variable {i}: {var}", "- Land" if var in GFS_LAND_CHANNELS else "")

        # Print order of levels in the gfs slice
        for i, level in enumerate(gfs_slice.level.values):
            print_debug(self.debug, f"GFS Level {i}: {level}")

        # Print the order of the variables in gfs slice
        print_debug(self.debug, "After Extracting Land Variables:")
        for i, var in enumerate(gfs_slice.data_vars.keys()):
            print_debug(self.debug, f"GFS Variable {i}: {var}", "- Land" if var in GFS_LAND_CHANNELS else "")

        # Dimensions: [channel, level, lat, lon]
        gfs_tensor = torch.from_numpy(gfs_slice.to_array().values).to(dtype=torch.float32)

        # Need to combine channel and level dimensions: [channel*level, lat, lon]
        gfs_tensor = gfs_tensor.view(gfs_tensor.shape[0]*gfs_tensor.shape[1],
                                     gfs_tensor.shape[2], gfs_tensor.shape[3])
        
        print_debug(self.debug, "GFS Tensor Shape before Surface: ", gfs_tensor.shape)
        
        # Now add the land variables
        gfs_tensor = torch.cat((gfs_tensor, torch.from_numpy(self.gfs_surface.to_array().values).to(dtype=torch.float32)), dim=0)

        print_debug(self.debug, "GFS Tensor Shape: ", gfs_tensor.shape)
        print_debug(self.debug, "ERA Statics Tensor Shape: ", self.era_static_tensor.shape)

        input_tensor = torch.cat((gfs_tensor, self.era_static_tensor), dim=0)

        # Need to combine channel and level dimensions: [channel*level, lat, lon]
        # input_tensor = input_tensor.view(input_tensor.shape[0]*input_tensor.shape[1],
        #                                   input_tensor.shape[2], input_tensor.shape[3])

        print_debug(self.debug, "Input Tensor Shape: ", input_tensor.shape)

        self.input_shape = input_tensor.shape

        era_surface_slice = self.era_surface.isel(time=index)
        era_atmos_slice = self.era_atmos.isel(time=index)

        # Print the order of the variables in era slice
        for i, var in enumerate(era_surface_slice.data_vars.keys()):
            print_debug(self.debug, f"ERA Surface Variable {i}: {var} | Min: {era_surface_slice[var].min().values} | Median: {era_surface_slice[var].median().values} | Max: {era_surface_slice[var].max().values}")

        for i, var in enumerate(era_atmos_slice.data_vars.keys()):
            print_debug(self.debug, f"ERA Atmos Variable {i}: {var} | Min: {era_atmos_slice[var].min().values} | Median: {era_atmos_slice[var].median().values} | Max: {era_atmos_slice[var].max().values}")

        # Print the order of levels in the atmos slice
        for i, level in enumerate(era_atmos_slice.level.values):
            print_debug(self.debug, f"ERA Atmos Level {i}: {level}")


        # Surface: [channel, lat, lon]
        era_surface_tensor = torch.from_numpy(era_surface_slice.to_array().values).to(dtype=torch.float32)

        # Atmos: [channel, level, lat, lon]
        era_atmos_tensor = torch.from_numpy(era_atmos_slice.to_array().values).to(dtype=torch.float32)

        # New Atmos: [channel*level, lat, lon]
        # Will be ordered by channel, then level s.t.
        # Index 1 == (channel=0, level=0), Index 2 == (channel=0, level=1), etc
        era_atmos_tensor = era_atmos_tensor.view(era_atmos_tensor.shape[0] * era_atmos_tensor.shape[1],
                                                  era_atmos_tensor.shape[2], era_atmos_tensor.shape[3])

        # Static: [channel, lat, lon]
        print_debug(self.debug, "ERA Static Tensor Shape: ", self.era_static_tensor.shape)
        print_debug(self.debug, "ERA Surface Tensor Shape: ", era_surface_tensor.shape)
        print_debug(self.debug, "ERA Atmos Tensor Shape: ", era_atmos_tensor.shape)

        truth_tensor = torch.cat((era_surface_tensor, era_atmos_tensor), dim=0)

        self.output_shape = truth_tensor.shape

        print_debug(self.debug, "Truth Tensor Shape: ", truth_tensor.shape)

        # Check if there are any nans/infs in the tensor and correct them
        if self.interpolate:
            print_debug(self.debug, "Interpolating Missing Values...")

            input_tensor = interpolate_missing_values(input_tensor)
            truth_tensor = interpolate_missing_values(truth_tensor)

            print_debug(self.debug, "Missing Values Interpolated", input_tensor.shape, truth_tensor.shape)

        if self.normalize:
            print_debug(self.debug, "Normalizing Data...")

            for i in range(input_tensor.input_shape[0]):
                # GFS
                var, level = self.gfs_idx_to_variable(i)
                input_tensor[i, :, :, :] = normalize_tensor(input_tensor[i, :, :, :], var, level, 'GFS')

            for i in range(truth_tensor.output_shape[0]):
                # ERA
                var, level = self.era_idx_to_variable(i)
                truth_tensor[i, :, :, :] = normalize_tensor(truth_tensor[i, :, :, :], var, level, 'ERA')

            print_debug(self.debug, "Data Normalized")

        if self.residuals:
            # Return the difference between ERA5 and GFS as the truth tensor
            # Need to align based on variables

            # Go through each ERA5 channel individually
            # Subtract the analogous channel from GFS, including level
            for i in range(truth_tensor.output_shape[0]):
                # ERA
                var, level = self.era_idx_to_variable(i)
                
                # Get the analogous GFS variable
                var = ERA5_TO_GFS[var]

                # If the level is 'surface', use 1000 in GFS
                if level == 'surface':
                    level = 1000

                gfs_index = self.gfs_variable_to_idx[(var, level)]

                # Subtract this GFS channel from the ERA5 data
                truth_tensor[i, :, :, :] = truth_tensor[i, :, :, :] - input_tensor[gfs_index, :, :, :]


        return input_tensor, truth_tensor


class GFSDataModule(pl.LightningDataModule):
    '''
    Should the data module download ERA5 data as well?
    '''

    def __init__(self, configs, train_size=0.9, batch_size=4, debug=False):
        super().__init__()
        self.configs = configs
        self.train_size = train_size
        self.batch_size = batch_size
        self.debug = debug

        start = self.configs['data']['gfs']['time']['start']
        end = self.configs['data']['gfs']['time']['end']
        archDir = self.configs['data']['gfs']['archive']
        static_tag = self.configs['data']['era5']['static_tag']

        self.gfs_path = os.path.join(archDir, f"gfs_{start}_{end}.nc")
        self.static_path = f"{archDir}/static_{static_tag}.nc"



    def prepare_data(self):
        if not os.path.exists(self.gfs_path):
            # Download GFS data if it is not already downloaded
            print_debug(self.debug, "Downloading GFS Data: ", self.gfs_path)
            download_gfs(self.configs['data']['gfs'])

            # Process GFS data
            # NOTE: isobaricInhPa was renamed to pressure
            process_gfs(self.configs['data']['gfs'])
        else:
            print_debug(self.debug, "Processed GFS Data for this time range found")


        if not os.path.exists(self.static_path):
            # Download ERA5 static data if it is not already downloaded
            print_debug(self.debug, "Downloading ERA5 Static Data: ", self.static_path)
            download_static_era5(self.configs['data']['gfs'])

        print(f"Memory after Preparing Data: {check_gpu_memory():.2f} GB")

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        gfs_dataset = GFSDataset(config=self.configs, debug=self.debug)
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

        # Create a dictionary of indices to era variables
        self.era_idx_to_variable = {}
        for i in range(gfs_dataset.output_shape[0]):
            self.era_idx_to_variable[i] = gfs_dataset.era_idx_to_variable(i)

        # Create the reverse dictionary as well (variable, level): index
        self.era_variable_to_idx = {v:k for k, v in self.era_idx_to_variable.items()}

        self.gfs_idx_to_variable = {}
        print("GFS Size: ", gfs_dataset.input_shape[0])
        for i in range(gfs_dataset.input_shape[0]):
            self.gfs_idx_to_variable[i] = gfs_dataset.gfs_idx_to_variable(i)

        # Create the reverse dictionary (variable, level): index
        self.gfs_variable_to_idx = {v:k for k, v in self.gfs_idx_to_variable.items()}


        print(f"Memory after Data Setup: {check_gpu_memory():.2f} GB")


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            self.batch_size,
            num_workers=4,
            shuffle=True,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            self.batch_size,
            num_workers=4,
            shuffle=False,
            persistent_workers=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, self.batch_size)