from aurora import Aurora, rollout
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import cv2
import torch
import numpy as np
import os
import xarray as xr

def generate_outputs(model, batch, steps=28, device="cuda"):
    model.eval()
    model = model.to(device)

    with torch.inference_mode():
        preds = [pred.to("cpu") for pred in rollout(model, batch, steps=steps)]


    return preds


def visualize_outputs(preds, steps=28, output_path="", fps=6, variable="2t", format='mp4', comparison_data=None):
    # Get all images
    images = []
    crs = ccrs.PlateCarree(central_longitude=180)

    lon = np.arange(-180, 180, 0.25)
    lat = np.arange(-90, 90, 0.25)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]   

    for i in range(steps):
        # Your plotting code here   
        pred = preds[i]              

        if comparison_data is not None:
            fig = plt.figure(figsize=(10, 6))
            ax1 = fig.add_subplot(1, 2, 1, projection=crs) # truth
            ax1.set_aspect('auto')

            ax2 = fig.add_subplot(1, 2, 2, projection=crs) # pred
            ax2.set_aspect('auto')

            if variable == "2t":
                truth = np.array(comparison_data["t2m"][i])

                ax1.imshow(truth - 273.15, vmin=-50, vmax=50, cmap='turbo', transform=crs, extent=extent)
                ax2.imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50, cmap='turbo', transform=crs, extent=extent)

            elif variable == "wind":
                u = pred.surf_vars["10u"][0, 0].numpy()
                v = pred.surf_vars["10v"][0, 0].numpy()

                u_truth = np.array(comparison_data["u10"][i])
                v_truth = np.array(comparison_data["v10"][i])

                # print("u", u.shape, "\t v", v.shape)
                # print("truth data type: ", type(u_truth))
                # print("u_truth", u_truth.shape, "\t v_truth", v_truth.shape)

                # Calculate wind speed
                speed = np.sqrt(u**2 + v**2)
                speed_truth = np.sqrt(u_truth**2 + v_truth**2)

                # print("speed", speed.shape, "speed_truth", speed_truth.shape)

                # Create wind barbs
                # ax.barbs(lon, lat, u, v, transform=crs, length=5, barbcolor='black', barb_increments=dict(half=2.5, full=5, flag=25, nan=0))
                ax1.imshow(speed, vmin=0, vmax=30, cmap='turbo', transform=crs, extent=extent)
                ax2.imshow(speed_truth, vmin=0, vmax=30, cmap='turbo', transform=crs, extent=extent)

            elif variable == "msl":
                msl = pred.surf_vars["msl"][0, 0].numpy()
                msl_truth = np.array(comparison_data["msl"][i])

                ax1.imshow(msl, vmin=850, vmax=1050, cmap='turbo', transform=crs, extent=extent)
                ax2.imshow(msl_truth, vmin=850, vmax=1050, cmap='turbo', transform=crs, extent=extent)


            ax1.coastlines(linewidth=0.7)
            ax2.coastlines(linewidth=0.7)

            ax1.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
            # ax1.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='black')

            ax2.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
            # ax2.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='black')

            ax1.set_title("ERA5")
            ax2.set_title("Aurora")

            ax1.set_ylabel(str(pred.metadata.time[0]))
            ax1.set_xticks([])
            ax1.set_yticks([])

            ax2.set_ylabel(str(pred.metadata.time[0]))
            ax2.set_xticks([])
            ax2.set_yticks([])

            # Save plot to memory
            plt.tight_layout()
            plt.savefig(f"temporary_{i}.jpg", bbox_inches='tight')
            images.append(f"temporary_{i}.jpg")
            # Clear the current figure
            plt.close()  



        else:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(1, 1, 1, projection=crs)
            ax.set_aspect('auto')

            if variable == "2t":
                ax.imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50, cmap='turbo', transform=crs, extent=extent)

            elif variable == "wind":
                u = pred.surf_vars["10u"][0, 0].numpy()
                v = pred.surf_vars["10v"][0, 0].numpy()

                # Calculate wind speed
                speed = np.sqrt(u**2 + v**2)

                # Create wind barbs
                # ax.barbs(lon, lat, u, v, transform=crs, length=5, barbcolor='black', barb_increments=dict(half=2.5, full=5, flag=25, nan=0))
                ax.imshow(speed, vmin=-50, vmax=50, cmap='turbo', transform=crs, extent=extent)

            ax.coastlines(linewidth=0.7)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
            ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='black')
            ax.set_ylabel(str(pred.metadata.time[0]))
            ax.set_title(f"Step {i}")
            ax.set_xticks([])
            ax.set_yticks([])

            
            # Save plot to memory
            plt.tight_layout()
            plt.savefig(f"temporary_{i}.jpg", bbox_inches='tight')
            images.append(f"temporary_{i}.jpg")
            # Clear the current figure
            plt.close()        

    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    
    # Set up video writer
    if format == 'gif':
        fourcc = cv2.VideoWriter_fourcc(*'GIF ')  # Note: GIF support is limited
    elif format == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        raise ValueError(f"Invalid format: {format}")
    
    # Create output filename with correct extension
    output_file = output_path + f"aurora_{variable}" + ('.gif' if format == 'gif' else '.mp4')
    
    # Initialize video writer
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Write frames
    for image in images:
        frame = cv2.imread(image)
        out.write(frame)
    
    # Release everything
    out.release()

    # Delete temporary images
    for i in range(steps):
        os.remove(f"temporary_{i}.jpg")


def era5_comparison(data_path, steps=28, variable="temp", level=1000):
    '''
    Generate arrays for each timestep from ERA5 data of the desired variable
    If data is surface, variables: t2m, msl, wind
    If data is atmo, variables: t, wind, q, z

    '''
    ds = xr.open_dataset(data_path, engine="netcdf4")

    print("ERA5 Shape: ", {dim: ds.sizes[dim] for dim in ds.dims})

    data = {}

    if level == "surface":
        if variable == "wind":
            data['u'] = ds['u10'].values[:steps]
            data['v'] = ds['v10'].values[:steps]
        elif variable == "temp":
            data['t'] = ds['t2m'].values[:steps]
        else:
            raise ValueError(f"Variable {variable} not supported")


    else: # pressure level, atmospheric data
        if variable == "wind":
            data['u'] = ds['u'].sel(level=level).values[:steps]
            data['v'] = ds['v'].sel(level=level).values[:steps]
        elif variable == "temp":
            data['t'] = ds['t'].sel(level=level).values[:steps]
        else:
            raise ValueError(f"Variable {variable} not supported")


        
    return data


def gfs_comparison(data_path, steps=28, variable="t2m", level=1000):
    '''
    Extract data for comparisons
    Variables: t, u, v, r, q, cape, cin, pwat
    Pressure Levels: 1000, ...
    '''        
    ds = xr.open_dataset(data_path, engine="netcdf4")

    print("GFS Shape: ", {dim: ds.sizes[dim] for dim in ds.dims})
    print("GFS Times: ", ds.time.values)

    data = {}

    if variable == "wind":
        data['u'] = ds['u'].sel(isobaricInhPa=level).values[:steps]
        data['v'] = ds['v'].sel(isobaricInhPa=level).values[:steps]
    elif variable == "temp":
        # t, r, q, cape, cin, pwat
        data['t'] = ds['t'].sel(isobaricInhPa=level).values[:steps]
    else:
        raise ValueError(f"Variable {variable} not supported")

        
    return data


def visualize_gfs_era5(era5_data, gfs_data, steps=28, variable="wind", output_path="", fps=4, format="gif"):
    # Get all images
    images = []
    crs = ccrs.PlateCarree(central_longitude=180)

    lon = np.arange(-180, 180, 0.25)
    lat = np.arange(-90, 90, 0.25)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]   

    # Assumes the data is already filtered for the correct levels and time ranges
    for t in range(steps):
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection=crs) # ERA
        ax1.set_aspect('auto')

        ax2 = fig.add_subplot(1, 2, 2, projection=crs) # GFS
        ax2.set_aspect('auto')

        if variable == "wind":
            u_era5 = era5_data['u'][t]
            v_era5 = era5_data['v'][t]
            u_gfs = gfs_data['u'][t]
            v_gfs = gfs_data['v'][t]

            # Get composite wind speed for ERA and GFS
            speed_era5 = np.sqrt(u_era5**2 + v_era5**2)
            speed_gfs = np.sqrt(u_gfs**2 + v_gfs**2)

            # Create wind barbs

            # Creat imshow color plot
            ax1.imshow(speed_era5, vmin=0, vmax=30, cmap='turbo', transform=crs, extent=extent)
            ax2.imshow(speed_gfs, vmin=0, vmax=30, cmap='turbo', transform=crs, extent=extent)


        elif variable == "temp":
            t_era = era5_data['t'][t]
            t_gfs = gfs_data['t'][t]
            ax1.imshow(t_era - 273.15, vmin=-50, vmax=50, cmap='turbo', transform=crs, extent=extent)
            ax2.imshow(t_gfs - 273.15, vmin=-50, vmax=50, cmap='turbo', transform=crs, extent=extent)

        else:
            raise ValueError(f"Variable {variable} not supported")

        ax1.coastlines(linewidth=0.7)
        ax2.coastlines(linewidth=0.7)

        ax1.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
        # ax1.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='black')

        ax2.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
        # ax2.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='black')

        fig.suptitle(f"{variable} Step {t}")
        ax1.set_title("ERA5")
        ax2.set_title("GFS")

        # ax1.set_ylabel(str(pred.metadata.time[0]))
        ax1.set_xticks([])
        ax1.set_yticks([])

        # ax2.set_ylabel(str(pred.metadata.time[0]))
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Save plot to memory
        plt.tight_layout()
        plot_path = os.path.join(output_path, f"temporary_{t}.jpg")
        plt.savefig(plot_path, bbox_inches='tight')
        images.append(plot_path)
        # Clear the current figure
        plt.close()  


    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    
    # Set up video writer
    if format == 'gif':
        fourcc = cv2.VideoWriter_fourcc(*'GIF ')  # Note: GIF support is limited
    elif format == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        raise ValueError(f"Invalid format: {format}")
    
    # Create output filename with correct extension
    output_file = output_path + "/" + f"aurora_{variable}" + ('.gif' if format == 'gif' else '.mp4')
    
    # Initialize video writer
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Write frames
    for image in images:
        print(image)
        frame = cv2.imread(image)
        out.write(frame)
    
    # Release everything
    out.release()

    # Delete temporary images
    for i in images:
        os.remove(i)


        
        



if __name__ == "__main__":
    from data_download import download_era5, make_batch

    static_path, surface_path, atmos_path = download_era5()
    batch = make_batch(static_path, surface_path, atmos_path, 1)

    model = Aurora(use_lora=False)
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

    preds = generate_outputs(model, batch, steps=28)
    visualize_outputs(preds, steps=28, variable="2t")