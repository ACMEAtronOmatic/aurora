from subprocess import list2cmdline
import cdsapi
from pathlib import Path
import xarray as xr
from aurora import Batch, Metadata
import torch
import os
from datetime import datetime, timedelta
import numpy as np

def download_static_era5(configs):
    DOWNLOAD_PATH = configs['download_path']
    STATIC_VARIABLES = configs['static_variables']
    STATIC_TAG = configs['static_tag']
    SURFACE_VARIABLES = configs['surface_variables']
    ATMO_VARIABLES = configs['atmo_variables']
    PRESSURES = configs['pressures']
    TIMES = configs['times']
    YEAR = configs['year']
    MONTH = configs['month']
    DAYS = configs['days']

    static_path = f"{DOWNLOAD_PATH}/static_{STATIC_TAG}_raw.nc"
    processed_static_path = static_path = f"{DOWNLOAD_PATH}/static_{STATIC_TAG}.nc"

    if not os.path.exists(processed_static_path):
            
            c = cdsapi.Client()

            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [v for v in STATIC_VARIABLES],
                    "year": "2023",
                    "month": "01",
                    "day": "01",
                    "time": "00:00",
                    "format": "netcdf",
                },
                str(static_path),
            )

    print("Static variables downloaded!")

    # Read in data and rename dimensions
    static_ds = xr.open_dataset(static_path, engine="netcdf4")
    print("ERA Statics Shape: ", {dim: static_ds.sizes[dim] for dim in static_ds.dims})
    static_ds = static_ds.rename({"valid_time": "time"})
    static_ds.to_netcdf(processed_static_path)

    return processed_static_path



def download_era5(configs):
    DOWNLOAD_PATH = configs['download_path']
    STATIC_VARIABLES = configs['static_variables']
    STATIC_TAG = configs['static_tag']
    SURFACE_VARIABLES = configs['surface_variables']
    ATMO_VARIABLES = configs['atmo_variables']
    PRESSURES = configs['pressures']
    time_interval = configs['time_interval']
    YEAR = configs['year']
    MONTHS = configs['months']
    start_day, end_day = configs['days'] # inclusive or exclusive?
    delete_raw = configs['delete_raw']

    # Static variables only need one time step
    # Surface & Atmospheric Variables must be at every time step
    # NOTE: in order to make a prediction, Aurora needs [t-1, t] to predict [t+1]

    static_path = f"{DOWNLOAD_PATH}/static_{STATIC_TAG}_raw.nc"
    surface_path = f"{DOWNLOAD_PATH}/y{YEAR}_m{MONTHS[0]:02d}-{MONTHS[-1]:02d}_d{start_day:02d}-{end_day:02d}_surface_raw.nc"
    atmos_path = f"{DOWNLOAD_PATH}/y{YEAR}_m{MONTHS[0]:02d}-{MONTHS[-1]:02d}_d{start_day:02d}-{end_day:02d}_atmospheric_raw.nc"

    processed_static_path = f"{DOWNLOAD_PATH}/static_{STATIC_TAG}.nc"
    processed_surface_path = f"{DOWNLOAD_PATH}/y{YEAR}_m{MONTHS[0]:02d}-{MONTHS[-1]:02d}_d{start_day:02d}-{end_day:02d}_surface.nc"
    processed_atmos_path = f"{DOWNLOAD_PATH}/y{YEAR}_m{MONTHS[0]:02d}-{MONTHS[-1]:02d}_d{start_day:02d}-{end_day:02d}_atmospheric.nc"

    # Check if all necessary data already exists
    if os.path.exists(processed_static_path) and \
        os.path.exists(processed_surface_path) and \
            os.path.exists(processed_atmos_path):
        return processed_static_path, processed_surface_path, processed_atmos_path
    
    if not os.path.isdir(DOWNLOAD_PATH):
        os.mkdir(DOWNLOAD_PATH)

    # else instantiate the api client
    print("Instantiating API Client...")
    c = cdsapi.Client()
    print("API Client Instantiated!")

    if not os.path.exists(processed_static_path):

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [v for v in STATIC_VARIABLES],
                "year": "2023",
                "month": "01",
                "day": "01",
                "time": "00:00",
                "format": "netcdf",
            },
            str(static_path),
        )
    print("Static variables downloaded!")

    print("Attempting Download: ",             {
                "product_type": "reanalysis",
                "variable": [v for v in SURFACE_VARIABLES],
                "year": str(YEAR),
                "month": [str(m).zfill(2) for m in MONTHS],
                "day": [str(d).zfill(2) for d in range(start_day, end_day + 1)],
                "time": [f"{h:02d}:00" for h in range(0, 24, time_interval)],
                "format": "netcdf",
            },)

    # Download the surface-level variables.
    if not os.path.exists(processed_surface_path):

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [v for v in SURFACE_VARIABLES],
                "year": str(YEAR),
                "month": [str(m).zfill(2) for m in MONTHS],
                "day": [str(d).zfill(2) for d in range(start_day, end_day + 1)],
                "time": [f"{h:02d}:00" for h in range(0, 24, time_interval)],
                "format": "netcdf",
            },
            str(surface_path),
        )
    print("Surface-level variables downloaded!")

    # Download the atmospheric variables.
    if not os.path.exists(processed_atmos_path):

        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": [v for v in ATMO_VARIABLES],
                "pressure_level": [str(x) for x in PRESSURES],
                "year": str(YEAR),
                "month": [str(m).zfill(2) for m in MONTHS],
                "day": [str(d).zfill(2) for d in range(start_day, end_day + 1)],
                "time": [f"{h:02d}:00" for h in range(0, 24, time_interval)],
                "format": "netcdf",
            },
            str(atmos_path),
        )
    print("Atmospheric variables downloaded!")


    # Read in data and rename dimensions
    static_ds = xr.open_dataset(static_path, engine="netcdf4")
    print("ERA Statics Shape: ", {dim: static_ds.sizes[dim] for dim in static_ds.dims})
    static_ds = static_ds.rename({"valid_time": "time"})
    static_ds.to_netcdf(processed_static_path)

    surface_ds = xr.open_dataset(surface_path, engine="netcdf4")
    print("ERA Surface Shape: ", {dim: surface_ds.sizes[dim] for dim in surface_ds.dims})
    surface_ds = surface_ds.rename({"valid_time": "time"})
    surface_ds.to_netcdf(processed_surface_path)

    atmos_ds = xr.open_dataset(atmos_path, engine="netcdf4")
    print("ERA Atmos Shape: ", {dim: atmos_ds.sizes[dim] for dim in atmos_ds.dims})
    atmos_ds = atmos_ds.rename({"valid_time": "time", "pressure_level": "level"})
    atmos_ds.to_netcdf(processed_atmos_path)

    if delete_raw:
        print("Deleting raw ERA5data...")
        os.remove(static_path)
        os.remove(surface_path)
        os.remove(atmos_path)

    return processed_static_path, processed_surface_path, processed_atmos_path

def make_batch(static_ds : str | xr.Dataset, surface_ds : str | xr.Dataset, atmos_ds : str | xr.Dataset, step : int | datetime):
    '''
    Use netcdf files to creat batch at specified time step
    Note that step is the CURRENT time step
    Data at [step-1, step] will be used
    to create the batch object for predicting step+1
    '''

    if not (isinstance(static_ds, xr.Dataset) and isinstance(surface_ds, xr.Dataset) and isinstance(atmos_ds, xr.Dataset)):
        # Paths have been passed in 
        if not (isinstance(static_ds, str) and isinstance(surface_ds, str) and isinstance(atmos_ds, str)):
            raise ValueError("Parameters to make_batch() must be all xr.Dataset or all string paths")

        static_ds = xr.open_dataset(static_ds, engine="netcdf4")
        surface_ds = xr.open_dataset(surface_ds, engine="netcdf4")
        atmos_ds = xr.open_dataset(atmos_ds, engine="netcdf4")

    # Check if the step is an int or a timestamp
    if isinstance(step, int):
        # Check that this and the previous step exist in the given datasets
        if step not in surface_ds.time.values or step - 1 not in surface_ds.time.values:
            raise ValueError(f"Step {step} does not exist in the given datasets")
        
        steps = [step-1, step]
    
    elif isinstance(step, datetime):
        # TODO: Check that it is a viable timestamp
        prev_step = np.datetime64(step - timedelta(hours=6))
        step = np.datetime64(step)

        if step not in surface_ds.time or prev_step not in surface_ds.time:
            print("Dataset Timestamps: ")
            print(surface_ds.time)
            raise ValueError(f"Timestep {step} or the previous timestep do not exist in the given datasets")
        
        # If they exist, get the equivalent steps for these timestamps
        prev_step = np.where(surface_ds.time.values == prev_step)[0][0]
        step = np.where(surface_ds.time.values == step)[0][0]

        steps = [prev_step, step]
        

    print("Batch Steps: ", steps)
    # print("Batch with presure levels: ", tuple(int(level) for level in atmos_ds.level.values))
    # print("Batch at time: ", (surface_ds.time.values.astype("datetime64[s]").tolist()[step],))
    # print("Batch latitudes: ", atmos_ds['latitude'].values[:5], atmos_ds['latitude'].values[-5:])
    # print("Batch longitudes: ", atmos_ds['longitude'].values[:5], atmos_ds['longitude'].values[-5:])
    # print("Batch atmos shape: ", atmos_ds["t"].shape)

    batch = Batch(
        surf_vars={
            # First select time points `i` and `i - 1`. Afterwards, `[None]` inserts a
            # batch dimension of size one.
            "2t": torch.from_numpy(surface_ds["t2m"].values[steps][None]).to(dtype=torch.float32),
            "10u": torch.from_numpy(surface_ds["u10"].values[steps][None]).to(dtype=torch.float32),
            "10v": torch.from_numpy(surface_ds["v10"].values[steps][None]).to(dtype=torch.float32),
            "msl": torch.from_numpy(surface_ds["msl"].values[steps][None]).to(dtype=torch.float32),
        },
        static_vars={
            # The static variables are constant, so we just get them for the first time.
            "z": torch.from_numpy(static_ds["z"].values[0]).to(dtype=torch.float32),
            "slt": torch.from_numpy(static_ds["slt"].values[0]).to(dtype=torch.float32),
            "lsm": torch.from_numpy(static_ds["lsm"].values[0]).to(dtype=torch.float32),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_ds["t"].values[steps][None]).to(dtype=torch.float32),
            "u": torch.from_numpy(atmos_ds["u"].values[steps][None]).to(dtype=torch.float32),
            "v": torch.from_numpy(atmos_ds["v"].values[steps][None]).to(dtype=torch.float32),
            "q": torch.from_numpy(atmos_ds["q"].values[steps][None]).to(dtype=torch.float32),
            "z": torch.from_numpy(atmos_ds["z"].values[steps][None]).to(dtype=torch.float32),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surface_ds.latitude.values).to(dtype=torch.float32),
            lon=torch.from_numpy(surface_ds.longitude.values).to(dtype=torch.float32),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s. Note that this needs to be a tuple of length one:
            # one value for every batch element.
            time=(surface_ds.time.values.astype("datetime64[s]").tolist()[step],),
            atmos_levels=tuple(int(level) for level in atmos_ds.level.values),
        ),
    )

    return batch

def batch_to_xr(batch: Batch | list) -> xr.Dataset:
    if isinstance(batch, Batch):
        ds = xr.Dataset(
            {
                **{f"surf_{k}": (("batch", "history", "latitude", "longitude"), v) 
                for k, v in batch.surf_vars.items()},
                **{f"static_{k}": (("latitude", "longitude"), v) 
                for k, v in batch.static_vars.items()},
                **{f"atmos_{k}": (("batch", "history", "level", "latitude", "longitude"), v) 
                for k, v in batch.atmos_vars.items()},
            },
            coords={
                "latitude": batch.metadata.lat,
                "longitude": batch.metadata.lon,
                "time": list(batch.metadata.time),
                "level": list(batch.metadata.atmos_levels),
                "rollout_step": batch.metadata.rollout_step,
            },
        )

        return ds
    
    else:
        # Iterate through the steps in the rollout
        datasets = []
        for b in batch:
            ds = xr.Dataset(
                {
                    **{f"surf_{k}": (("batch", "history", "latitude", "longitude"), v) 
                    for k, v in b.surf_vars.items()},
                    **{f"static_{k}": (("latitude", "longitude"), v) 
                    for k, v in b.static_vars.items()},
                    **{f"atmos_{k}": (("batch", "history", "level", "latitude", "longitude"), v) 
                    for k, v in b.atmos_vars.items()},
                },
                coords={
                    "latitude": b.metadata.lat,
                    "longitude": b.metadata.lon,
                    "time": list(b.metadata.time),
                    "level": list(b.metadata.atmos_levels),
                    "rollout_step": b.metadata.rollout_step,
                },
            )

            datasets.append(ds)

        combined_ds = xr.concat(datasets, dim='time').sortby('time')

        return combined_ds

if __name__ == "__main__":
    # Download ERA5 data
    import yaml
    configs = yaml.safe_load(open("../configs/configs.yml", "r"))
    static_path, surface_path, atmos_path = download_era5(configs['data']['era5'])

    # Create batch, (step - 1) >= 0
    batch = make_batch(static_path, surface_path, atmos_path, 1)
    print("Batch created!")
    # print(batch)
