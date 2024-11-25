import cdsapi
from pathlib import Path
import xarray as xr
from aurora import Batch, Metadata
import torch
import os



def download_era5(configs):
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

    # Download the static variables.
    c = cdsapi.Client()

    # Static variables only need one time step
    # Surface & Atmospheric Variables must be at every time step
    # NOTE: in order to make a prediction, Aurora needs [t-1, t] to predict [t+1]

    static_path = f"{DOWNLOAD_PATH}/static_{STATIC_TAG}_raw.nc"
    surface_path = f"{DOWNLOAD_PATH}/{YEAR}_{MONTH:02d}_{DAYS[0]:02d}-{DAYS[-1]:02d}_surface_raw.nc"
    atmos_path = f"{DOWNLOAD_PATH}/{YEAR}_{MONTH:02d}_{DAYS[0]:02d}-{DAYS[-1]:02d}_atmospheric_raw.nc"

    processed_static_path = static_path = f"{DOWNLOAD_PATH}/static_{STATIC_TAG}.nc"
    processed_surface_path = surface_path = f"{DOWNLOAD_PATH}/{YEAR}_{MONTH:02d}_{DAYS[0]:02d}-{DAYS[-1]:02d}_surface.nc"
    processed_atmos_path = atmos_path = f"{DOWNLOAD_PATH}/{YEAR}_{MONTH:02d}_{DAYS[0]:02d}-{DAYS[-1]:02d}_atmospheric.nc"

    all_downloaded = True

    if not os.path.exists(processed_static_path):
        all_downloaded = False

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

    # Download the surface-level variables.
    if not os.path.exists(processed_surface_path):
        all_downloaded = False

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [v for v in SURFACE_VARIABLES],
                "year": str(YEAR),
                "month": str(MONTH),
                "day": [str(d).zfill(2) for d in DAYS],
                "time": [f"{h:02d}:00" for h in TIMES],
                "format": "netcdf",
            },
            str(surface_path),
        )
    print("Surface-level variables downloaded!")

    # Download the atmospheric variables.
    if not os.path.exists(processed_atmos_path):
        all_downloaded = False

        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": [v for v in ATMO_VARIABLES],
                "pressure_level": [str(x) for x in PRESSURES],
                "year": str(YEAR),
                "month": str(MONTH),
                "day": [str(d).zfill(2) for d in DAYS],
                "time": [f"{h:02d}:00" for h in TIMES],
                "format": "netcdf",
            },
            str(atmos_path),
        )
    print("Atmospheric variables downloaded!")

    if all_downloaded:
        return processed_static_path, processed_surface_path, processed_atmos_path

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
    atmos_ds = atmos_ds.rename({"valid_time": "time"})
    atmos_ds.to_netcdf(processed_atmos_path)

    return processed_static_path, processed_surface_path, processed_atmos_path

def make_batch(static_path, surface_path, atmos_path, step):
    '''
    Use netcdf files to creat batch at specified time step
    Note that step is the CURRENT time step
    Data at [step-1, step] will be used
    to create the batch object for predicting step+1
    '''
    static_ds = xr.open_dataset(static_path, engine="netcdf4")
    surface_ds = xr.open_dataset(surface_path, engine="netcdf4")
    atmos_ds = xr.open_dataset(atmos_path, engine="netcdf4")

    batch = Batch(
        surf_vars={
            # First select time points `i` and `i - 1`. Afterwards, `[None]` inserts a
            # batch dimension of size one.
            "2t": torch.from_numpy(surface_ds["t2m"].values[[step - 1, step]][None]).to(dtype=torch.float32),
            "10u": torch.from_numpy(surface_ds["u10"].values[[step - 1, step]][None]).to(dtype=torch.float32),
            "10v": torch.from_numpy(surface_ds["v10"].values[[step - 1, step]][None]).to(dtype=torch.float32),
            "msl": torch.from_numpy(surface_ds["msl"].values[[step - 1, step]][None]).to(dtype=torch.float32),
        },
        static_vars={
            # The static variables are constant, so we just get them for the first time.
            "z": torch.from_numpy(static_ds["z"].values[0]).to(dtype=torch.float32),
            "slt": torch.from_numpy(static_ds["slt"].values[0]).to(dtype=torch.float32),
            "lsm": torch.from_numpy(static_ds["lsm"].values[0]).to(dtype=torch.float32),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_ds["t"].values[[step - 1, step]][None]).to(dtype=torch.float32),
            "u": torch.from_numpy(atmos_ds["u"].values[[step - 1, step]][None]).to(dtype=torch.float32),
            "v": torch.from_numpy(atmos_ds["v"].values[[step - 1, step]][None]).to(dtype=torch.float32),
            "q": torch.from_numpy(atmos_ds["q"].values[[step - 1, step]][None]).to(dtype=torch.float32),
            "z": torch.from_numpy(atmos_ds["z"].values[[step - 1, step]][None]).to(dtype=torch.float32),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surface_ds.latitude.values).to(dtype=torch.float32),
            lon=torch.from_numpy(surface_ds.longitude.values).to(dtype=torch.float32),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s. Note that this needs to be a tuple of length one:
            # one value for every batch element.
            time=(surface_ds.valid_time.values.astype("datetime64[s]").tolist()[step],),
            atmos_levels=tuple(int(level) for level in atmos_ds.pressure_level.values),
        ),
    )

    return batch


if __name__ == "__main__":
    # Download ERA5 data
    import yaml
    configs = yaml.safe_load(open("../configs/configs.yml", "r"))
    static_path, surface_path, atmos_path = download_era5(configs['data']['era5'])

    # Create batch, (step - 1) >= 0
    batch = make_batch(static_path, surface_path, atmos_path, 1)
    print("Batch created!")
    # print(batch)
