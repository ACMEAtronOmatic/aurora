import torch

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


def check_gpu_memory():
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        free = mem - allocated
        gb = free * 1e-9
        return gb
    else:
        return None


def print_debug(debug=True, *args):
    if debug:
        msg = " ".join(map(str, args))
        print(msg)


def normalize_tensor():
    pass