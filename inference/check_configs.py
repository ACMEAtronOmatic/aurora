import os 
import time
from datetime import datetime

def check_configs(configs):
    '''
    data:
        download_path: "../downloads"
        static_variables: ["geopotential", "land_sea_mask", "soil_type"]
        static_tag: gls
        surface_variables: ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure"]
        atmo_variables: ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"]
        pressures: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        times: [0, 6, 12, 18]
        year: 2023
        month: 1
        days: [1, 2, 3, 4, 5, 6, 7]
    '''

    if not os.path.exists(configs['data']['download_path']):
        print("WARNING: the download path specified does not exist.")
        print("Will create a new directory at specified path in 5 seconds, press ctrl+c to cancel.")
        time.sleep(5)
        os.makedirs(configs['data']['download_path'])

    if not isinstance(configs['data']['static_variables'], list) or len(configs['data']['static_variables']) < 1:
        raise ValueError("The static_variables field must be a list of >=1 static variables to download.")
    
    if not isinstance(configs['data']['static_variables'][0], str):
        raise ValueError("Static variables must be strings.")
    
    if not isinstance(configs['data']['static_tag'], str):
        raise ValueError("The static_tag field must be a string.")
    
    if not isinstance(configs['data']['surface_variables'], list) or len(configs['data']['surface_variables']) < 1:
        raise ValueError("The surface_variables field must be a list of >=1 surface variables to download.")
    
    if not isinstance(configs['data']['surface_variables'][0], str):
        raise ValueError("Surface variables must be strings.")
    
    if not isinstance(configs['data']['atmo_variables'], list) or len(configs['data']['atmo_variables']) < 1:
        raise ValueError("The atmo_variables field must be a list of >=1 atmospheric variables to download.")
    
    if not isinstance(configs['data']['atmo_variables'][0], str):
        raise ValueError("Atmospheric variables must be strings.")
    
    if not isinstance(configs['data']['pressures'], list) or len(configs['data']['pressures']) < 1: 
        raise ValueError("The pressures field must be a list of pressures to download, with at least one pressure specified.")
    
    if not isinstance(configs['data']['pressures'], list) or len(configs['data']['pressures']) < 1: 
        raise ValueError("The pressures field must be a list of pressures to download, with at least one pressure specified.")
    
    if not isinstance(configs['data']['pressures'][0], int) or max(configs['data']['pressures']) > 1000:
        raise ValueError("Pressures must be integers with a valid pressure. ERA5 has pressures up to 1000hPa.")
    
    if not isinstance(configs['data']['times'], list) or len(configs['data']['times']) < 1: 
        raise ValueError("The times field must be a list of times to download, with at least one time specified.")
    
    if not isinstance(configs['data']['times'][0], int) or max(configs['data']['times']) > 23:
        raise ValueError("Times must be integers with a valid time of the day.")

    if not isinstance(configs['data']['days'], list) or len(configs['data']['days']) < 1: 
        raise ValueError("The days field must be a list of days to download, with at least one day specified.")
    
    if not isinstance(configs['data']['days'][0], int) or max(configs['data']['days']) > 31:
        raise ValueError("Days must be integers with a valid day of the month.")
    
    if not isinstance(configs['data']['year'], int):
        raise ValueError("The year field must be an integer.")
    
    if configs['data']['year'] < 1940 or configs['data']['year'] > datetime.now().year:
        raise ValueError("The year field must be between 1940 and and the current year.")
    
    if not isinstance(configs['data']['month'], int) or configs['data']['month'] < 1 or configs['data']['month'] > 12:
        raise ValueError("The month field must be an integer between 1 and 12.")
    

    '''
    inference:
        model: "microsoft/aurora"
        checkpoint: "aurora-0.25-pretrained.ckpt"
        use_lora: false
        steps: 28
        variable: "2t"
    
    '''

    if not isinstance(configs['inference']['model'], str):
        raise ValueError("The model field must be a string.")
    
    if not isinstance(configs['inference']['checkpoint'], str):
        raise ValueError("The checkpoint field must be a string.")
    
    if not isinstance(configs['inference']['use_lora'], bool):
        raise ValueError("The use_lora field must be a boolean.")
    
    if not isinstance(configs['inference']['steps'], int):
        raise ValueError("The steps field must be an integer.")
    
    if not configs['inference']['variable'] in ['2t', 'wind']:
        raise ValueError("The variable field must be either '2t' or 'wind'.")

    
    return configs

