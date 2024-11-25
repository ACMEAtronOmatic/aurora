#!/usr/bin/env python3
#
#  Code to parse .idx files from NOMADS so that
#  I can extract relevant fields for N2N
#

import os ; import sys
from pathlib import Path

import time
import requests
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime, timedelta
import yaml
import copy
from argparse import ArgumentParser
from tempfile import NamedTemporaryFile

import xarray as xr
import cfgrib
import dask

RD = 287.052874247 # Dry Air Gas Constant
CP = 1003.5 # Specific Heat Dry Air
RDDCP = RD/CP # Gas Constant / Specific Heat
CPRRD = CP/RD # Specific Heat / Gas Constant
P00 = 100000. # Reference Pressure

# variables to exclude from gfs 
VAREXCLUSIONLIST = ['time', 'step', 'isobaricInhPa', 'latitude', 'longitude', 'valid_time', 'atmosphereSingleLayer', 'surface']


def derived_gfs_fields(ds, varlist):
    '''
    Calculate derived variables from GFS fields
    '''
    pressures = np.float32(ds['isobaricInhPa'].values * 100.) # Convert from hPa to Pa
    ds['tv'] = ds['t'] * (1 + 0.608 * ds['q']) # Virtual Temperature
    ds['theta'] = ds['tv'] * (P00 / pressures[:, None, None])**RDDCP # Potential Temperature
    ds['ns2'] = -1.0 * RD * ds['tv'] / pressures[:,None,None] * np.log(ds['theta']).diff("isobaricInhPa", 1, label = 'lower') / np.diff(pressures, 1)[:,None,None]
    ds['ns2'][-1,:,:] = ds['ns2'][-2,:,:] # Brunt-Vaisala Frequency

    varlist.append('tv')
    varlist.append('theta')
    varlist.append('ns2')

    return ds, varlist

def datetime_range(start=None, end=None, deltat=3600.):
    '''
    Returns a list of datetimes, [start, end, deltat]

    Parameters
    ----------
    start: datetime
        Beginning of range, inclusive
    end: datetime
        End of range, exclusive
    deltat: float, int, timedelta = 3600.0
        Time step in seconds

    Returns
    -------
    list of datetimes

    '''
    if not isinstance(start, datetime):
        raise TypeError("start needs to be a datetime object")
    if not isinstance(end, datetime):
        raise TypeError("end needs to be a datetime object")

    span = end - start
    totalSeconds = span.total_seconds()
    dt = []
    if isinstance(deltat, (float, int)):
        for i in range(int(totalSeconds/deltat) + 1):
            dt.append(start + timedelta(seconds = i * int(deltat)))
    elif isinstance(deltat, timedelta):
        tds = deltat.total_seconds()
        for i in range(int(totalSeconds/tds) + 1):
            dt.append(start + i*deltat)
    else:
        raise TypeError("deltat needs to be either a float/int or a timedelta object.")
    return dt

def datetime_to_pieces(dtg):
    '''
    Separates datetime into year, month, day, hour, minute, second

    Parameters
    ----------
    dtg: datetime
        Datetime object

    Returns
    -------
    tuple
        (year, month, day, hour, minute, second)
    '''
    yy  = dtg.strftime('%Y')
    mon = dtg.strftime('%m')
    dd  = dtg.strftime('%d')
    hh  = dtg.strftime('%H')
    mm  = dtg.strftime('%M')
    ss  = dtg.strftime('%s')
    return yy, mon, dd, hh, mm, ss

def get_lookup_index(idxFile):
    '''
    Retrieves an Index File from idxFile's URL, which are 
    NOAA GFS data on AWS. If it returns the request, write the data
    to a temporary file to re-read and format into a dictionary. If the request
    returns 302, 503, or 500, it will continue to retry.
    '''
    idxTmp = NamedTemporaryFile(suffix='.idx')
    fail = True
    for tryNum in range(3):
        try:
            req = requests.get(idxFile)
            # 302 - redirect, 503 - service unavailable, 500 - internal server error
            if req.status_code in [302, 503, 500]:
                print(req.content, req.headers)
                print(f"Failed with status {req.status_code}, trying again...")
                time.sleep(0.3)
                continue
            # 404 - file not found
            elif req.status_code == 404:
                print(f"404 status on: {idxFile}")
                return 404
            print(f"{req.status_code} {idxFile}")
            # 200 - OK
            if req.status_code != 200:
                return None
            with open(idxTmp.name, 'wb') as f:
                f.write(req.content)
            fail = False
            break
        except:
            traceback.print_exc()
            time.sleep(1)

    if fail: return None

    # Open the temporary file and format into a dictionary
    with open(idxTmp.name, 'r') as f:
        lines = f.readlines()
        lookup = {}
        for i in range(len(lines)):
            ls = lines[i].split(':')
            startByte = int(ls[1])
            varName = ls[3]
            level = ls[4]
            fcst = ls[5]
            if i+1 >= len(lines):
                endByte = ''
            else:
                endByte = int(lines[i+1].split(":")[1])
            byteString = f"bytes={startByte}-{endByte}"
            if varName not in lookup.keys():
                lookup[varName] = {}
            if level not in lookup[varName].keys():
                lookup[varName][level] = {}
            lookup[varName][level][fcst] = byteString

    return lookup

def build_archive(dtg, forecastHours, archDir, config, noclobber, model='gfs'):
    '''
    Downloads and archives NOAA GFS data. 

    Parameters
    ----------
    dtg: datetime
        Datetime object
    forecastHours: list
        List of forecast hours
    archDir: Path
        Path save downloaded data
    config: dict
        Dictionary of config
    noclobber: bool
        Do not clobber existing files
    model: str
        Either gfs or enkf
    '''
    model = model.lower()
    yy, mon, dd, hh, mm, ss = datetime_to_pieces(dtg)
    fullday = f'{yy}{mon}{dd}'
    dtgPath = archDir / fullday
    if not dtgPath.is_dir(): dtgPath.mkdir(parents=True, exist_ok = True)
    variables = config['variables']

    for fhr in forecastHours:
        # Requests, saves to temp file, and formats to dictionary for each forecast hour
        removeAtmos = False
        outputFileName = f'gfs_select.t{hh}z.pgrb2.0p25.f{fhr:03d}.grb2'
        fullPath = dtgPath / outputFileName
        if fullPath.exists() and noclobber:
            print(f"{fullPath} exists and noclobber set; skipping.")
            continue
        idxFile = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{fullday}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{fhr:03d}.idx"
        lookup = get_lookup_index(idxFile)
        if lookup == None: continue
        if lookup == 404:
            # If 404, try without atmos in the URL
            removeAtmos = True
            idxFile = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{fullday}/{hh}/gfs.t{hh}z.pgrb2.0p25.f{fhr:03d}.idx"
            lookup = get_lookup_index(idxFile)
        if lookup == None or lookup == 404: continue
        if fhr == 0:
            fcstLookup = 'anl'
        else:
            fcstLookup = f'{fhr} hour fcst'

        if removeAtmos:
            remoteGRIB2 = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{fullday}/{hh}/gfs.t{hh}z.pgrb2.0p25.f{fhr:03d}"
        else:
            remoteGRIB2 = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{fullday}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{fhr:03d}"
        
        with open(fullPath, 'wb') as f:
            for var in variables.keys():
                for level in variables[var]:
                    print("Day:", fullday, "IC:", hh, "FHR:", fhr, "Var:", var, "Level:", level, "FCST:", fcstLookup)
                    # print(fullday, fhr, var, level, fcstLookup)
                    # print(lookup[var][level][fcstLookup])

                    header = { "Range" : lookup[var][level][fcstLookup] }
                    for tryNum in  range(3):
                        try:
                            req = requests.get(remoteGRIB2, headers=header)
                            if req.status_code in [302, 503, 500]:
                                print(req.content, req.headers)
                                print(f"Failed with status {req.status_code}, trying again...")
                                time.sleep(1)
                                continue
                            if req.status_code not in [200, 206]:
                                print(f"Bad status: {req.status_code}")
                                break
                            f.write(req.content)
                            time.sleep(1.)
                            break
                        except:
                            print("Unknown failure, retrying.")
                            time.sleep(1)

def download_gfs(config, nproc = 4, noclobber = False):
    '''
    Download GFS data from NOAA AWS, based on YAML configuration files.
    Runs parallel build_archive jobs to download and archive files.

    Paramaters
    ----------
    configs : dict
        Loaded config yaml describing the levels, date range, and variables requested from GFS.
    no_clobber : bool = False
        Do not clobber existing directories.
    nproc : int = 4
        Number of threads for parallelization.

    Returns
    -------
    

    '''

    start = config['time']['start']
    end = config['time']['end']
    archDir = Path(config['archive'])

    # Check if the processed gfs data already exists in the archive folder
    if os.path.exists(os.path.join(archDir, f"gfs_{start}_{end}.nc")):
        print("Processed GFS data already found")
        return

    sdtg = datetime.strptime(str(config['time']['start']), '%Y%m%d%H')
    edtg = datetime.strptime(str(config['time']['end']), '%Y%m%d%H')
    interval = int(config['time']['interval'])

    dtgrange = datetime_range(sdtg, edtg, interval*3600)

    forecastStart = config['time']['forecast_start']
    forecastEnd = config['time']['forecast_end']
    forecastInterval = config['time']['forecast_interval']

    forecastHours = list(range(forecastStart, forecastEnd + forecastInterval, forecastInterval))

    # Check if thes GFS data is already downloaded
    all_dir_paths = [os.path.join(archDir, d.strftime('%Y%m%d')) for d in dtgrange]

    gfs_data_missing = False
    for d in all_dir_paths:
        print(d)
        if not os.path.isdir(d):
            print(f"At least one day of GFS data '{d}' is missing. Downloading...")
            gfs_data_missing = True
            break

    if not gfs_data_missing:
        print("All GFS data is already downloaded.")
        return

    if not archDir.is_dir(): archDir.mkdir(parents=True, exist_ok = True)

    Parallel(n_jobs=nproc)(delayed(build_archive)(d, forecastHours, archDir, config, noclobber) for d in tqdm(dtgrange, desc = 'DTG'))

    return

def process_gfs(config):
    data_path = Path(config['archive'])
    start_date = config['time']['start']
    end_date = config['time']['end']

    if not data_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {data_path}")
    
    gribFiles = sorted(data_path.glob('*/*grb2'))
    numFiles = len(gribFiles)

    gribFileFirst = gribFiles[0]
    varList = []

    print("Extracting values from first grib file.")
    with cfgrib.open_dataset(gribFileFirst) as ds:
        ny = ds.sizes['latitude']
        nx = ds.sizes['longitude']
        nz = ds.sizes['isobaricInhPa']
        pressures = np.float32(ds['isobaricInhPa'].values * 100.)
        for var in ds.variables:
            if var not in VAREXCLUSIONLIST:
                varList.append(var)

    print("All GFS Dimensions: ", ds.dims)
    print("Sizes (x, y, z): ", nx, ny, nz)
    print("Pressures: ", pressures)
    print("Var: ", varList)

    print("\nOpening whole dataset...")

    try:
        ds = xr.open_mfdataset(gribFiles, concat_dim = 'time', combine = 'nested',  engine = 'cfgrib',  parallel = True)
    except:
        raise Exception(f"Failed to open dataset with xr.open_mfdataset")
    
    try:
        datasets = []
        for file in gribFiles:
                ds = xr.open_dataset(file)
                datasets.append(ds)

        ds = xr.concat(datasets, dim='time')
    except:
        print(f"Failed to open dataset with xr.open_dataset")
        exit(0)


    # Calculate derived variables
    print("Calculating derived variables...")
    ds, varList = derived_gfs_fields(ds, varList)
    ds = ds.rename({"isobaricInhPa": "level"})

    save_path = os.path.join(data_path, f"gfs_{start_date}_{end_date}.nc")
    ds.to_netcdf(save_path)

    print("New GFS Vars: ", varList)
    print("Saved to: ", save_path)

    return save_path



if __name__ == '__main__':
    import pprint
    DESCRIPTION = 'GFS Download Module'
    parser = ArgumentParser(description = DESCRIPTION)

    parser.add_argument('yaml_file', help = 'YAML file with data & training guidelines.')

    args = parser.parse_args()

    with open(args.yaml_file, 'r') as file:
        config = yaml.safe_load(file)
        pprint.pprint(config)

    download_gfs(config=config['data']['gfs'])
    process_gfs(config=config['data']['gfs'])
