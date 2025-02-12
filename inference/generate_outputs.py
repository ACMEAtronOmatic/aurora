from aurora import Aurora, rollout
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
from numpy.fft import fft2, fftshift
import cv2
import torch
import numpy as np
import os
import xarray as xr
import pyshtools as pysh
from scores.probability import crps_cdf

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
                temp_truth = np.array(comparison_data["t2m"][i])
                temp_pred = pred.surf_vars["2t"][0, 0].numpy()

                print("Temp Range: ", np.min(temp_truth), np.max(temp_truth))

                ax1.imshow(temp_truth - 273.15, vmin=-50, vmax=50, cmap='turbo', transform=crs, extent=extent)
                ax2.imshow(temp_pred - 273.15, vmin=-50, vmax=50, cmap='turbo', transform=crs, extent=extent)

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

                print("Wind Speed Range: ", np.min(speed_truth), np.max(speed_truth))

                # print("speed", speed.shape, "speed_truth", speed_truth.shape)

                # Create wind barbs
                # ax.barbs(lon, lat, u, v, transform=crs, length=5, barbcolor='black', barb_increments=dict(half=2.5, full=5, flag=25, nan=0))
                ax1.imshow(speed, vmin=0, vmax=30, cmap='turbo', transform=crs, extent=extent)
                ax2.imshow(speed_truth, vmin=0, vmax=30, cmap='turbo', transform=crs, extent=extent)

            elif variable == "msl":
                msl = pred.surf_vars["msl"][0, 0].numpy()
                msl_truth = np.array(comparison_data["msl"][i])

                print("MSL Range: ", np.min(msl_truth), np.max(msl_truth))

                ax1.imshow(msl, vmin=7500, vmax=11000, cmap='turbo', transform=crs, extent=extent)
                ax2.imshow(msl_truth, vmin=7500, vmax=11000, cmap='turbo', transform=crs, extent=extent)


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
            plt.close('all')



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
            plt.close('all')

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


def era5_comparison(data_path, steps=28, variable="temp", level="surface"):
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
        elif variable == "msl":
            data['msl'] = ds['msl'].values[:steps]
        else:
            raise ValueError(f"Variable {variable} not supported")


    else: # pressure level, atmospheric data
        if variable == "wind":
            data['u'] = ds['u'].sel(pressure_level=level).values[:steps]
            data['v'] = ds['v'].sel(pressure_level=level).values[:steps]
        elif variable == "temp":
            data['t'] = ds['t'].sel(pressure_level=level).values[:steps]
        elif variable == "specific_humidity":
            data['q'] = ds['q'].sel(pressure_level=level).values[:steps]
        else:
            raise ValueError(f"Variable {variable} not supported")


        
    return data


def gfs_comparison(data_path, steps=28, variable="temp", level=1000):
    '''
    Extract data for comparisons
    Variables: 't', 'u', 'v', 'r', 'q', 
    'meanSea', 'mslet', 'slt', 'gh', 
    'orog', 'lsm', 'cape', 'cin', 'pwat',
    'tv', 'theta', 'ns2'
    Pressure Levels: which of these are surface vs. at pressure levels?
    '''        
    ds = xr.open_dataset(data_path, engine="netcdf4")

    data = {}

    if variable == "wind":
        data['u'] = ds['u'].sel(isobaricInhPa=level).values[:steps]
        data['v'] = ds['v'].sel(isobaricInhPa=level).values[:steps]
    elif variable == "temp":
        # t, r, q, cape, cin, pwat
        data['t'] = ds['t'].sel(isobaricInhPa=level).values[:steps]
    elif variable == "specific_humidity":
        data['q'] = ds['q'].sel(isobaricInhPa=level).values[:steps]
    elif variable == "msl":
        data['msl'] = ds['mslet'].values[:steps]
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

            print("Wind Range: ", np.min(speed_gfs), np.max(speed_gfs))

            # Creat imshow color plot
            ax1.imshow(speed_era5, vmin=90000, vmax=110000, cmap='turbo', transform=crs, extent=extent)
            ax2.imshow(speed_gfs, vmin=90000, vmax=110000, cmap='turbo', transform=crs, extent=extent)


        elif variable == "temp":
            t_era = era5_data['t'][t]
            t_gfs = gfs_data['t'][t]

            print("Temp Range: ", np.min(t_gfs), np.max(t_gfs))

            ax1.imshow(t_era - 273.15, vmin=-50, vmax=50, cmap='turbo', transform=crs, extent=extent)
            ax2.imshow(t_gfs - 273.15, vmin=-50, vmax=50, cmap='turbo', transform=crs, extent=extent)

        elif variable == "specific_humidity":
            q_era = era5_data['q'][t]
            q_gfs = gfs_data['q'][t]

            print("Q Range: ", np.min(q_gfs), np.max(q_gfs))

            ax1.imshow(q_era, vmin=0, vmax=0.05, cmap='turbo', transform=crs, extent=extent)
            ax2.imshow(q_gfs, vmin=0, vmax=0.05, cmap='turbo', transform=crs, extent=extent)

        elif variable == "msl":
            msl_era = era5_data['msl'][t]
            msl_gfs = gfs_data['msl'][t]

            print("MSL Range: ", np.min(msl_gfs), np.max(msl_gfs))

            ax1.imshow(msl_era, vmin=90000, vmax=110000, cmap='turbo', transform=crs, extent=extent)
            ax2.imshow(msl_gfs, vmin=90000, vmax=110000, cmap='turbo', transform=crs, extent=extent)


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
        plt.close('all')


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


def visualize_tensor(tensor, channel, channel_mapper, output_path="", format='mp4', fps=6):
    print("Tensor Shape: ", tensor.shape)

    crs = ccrs.PlateCarree(central_longitude=180)

    lon = np.arange(-180, 180, 0.25)
    lat = np.arange(-90, 90, 0.25)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]   

    valid_tensor = torch.where(torch.isnan(tensor), torch.tensor(0, device=tensor.device), tensor)

    variable, level = channel_mapper[channel]

    tensor_slice = valid_tensor[0, channel, :, :]

    min = torch.amin(tensor_slice).item()
    max = torch.amax(tensor_slice).item()
    med = torch.median(tensor_slice).item()

    # min = torch.amin(valid_tensor[CHANNEL_MAP[variable], :, :, :]).item()
    # max = torch.amax(valid_tensor[CHANNEL_MAP[variable], :, :, :]).item()
    # med = torch.median(valid_tensor[CHANNEL_MAP[variable], :, :, :]).item()

    print("Tensor Shape: ", tensor.shape)
    print("Tensor Range: ", min, max)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=crs) # ERA

    # Create a numpy array from this slice?
    tensor_values = np.asarray(tensor_slice.detach().cpu().numpy())


    im = ax.imshow(tensor_values, vmin=min, vmax=max, cmap='turbo', transform=crs, extent=extent)

    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label(variable)
    cbar.set_ticks([min, med, max])

    ax.coastlines(linewidth=0.7)

    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    # ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='black')

    ax.set_title(f"C{channel}-{variable}-{level}")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    name = os.path.join(output_path, f"C{channel}-{variable}-{level}")
    plt.savefig(name, bbox_inches='tight')
    # images.append(temp)
    plt.close('all')



def generate_mp4(images, output_path, fps=6, format='mp4'):
    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    
    # Set up video writer
    if format == 'gif':
        raise ValueError(f"Invalid format: {format}")
    elif format == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        raise ValueError(f"Invalid format: {format}")
    
    # Initialize video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
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


def compare_all_tensors(gfs_tensor, era_tensor, output_tensor,
                         channel, variable, level, output_path="",
                           format='mp4', fps=6):
    # The GFS data has 85 channels
    # ERA5 & the model output data has 69 channels that are aligned

    crs = ccrs.PlateCarree(central_longitude=180)   

    lon = np.arange(-180, 180, 0.25)
    lat = np.arange(-90, 90, 0.25)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]   

    valid_gfs = torch.where(torch.isnan(gfs_tensor), torch.tensor(0, device=gfs_tensor.device), gfs_tensor).cpu().numpy()
    valid_era = torch.where(torch.isnan(era_tensor), torch.tensor(0, device=era_tensor.device), era_tensor).cpu().numpy()
    valid_output = torch.where(torch.isnan(output_tensor), torch.tensor(0, device=output_tensor.device), output_tensor).cpu().numpy()

    min = np.amin(valid_gfs)
    max = np.amax(valid_gfs)
    med = np.median(valid_gfs)
    print("Tensor Range: ", min, max)

    fig = plt.figure(figsize=(12, 6))

    # Display three plots horizontally
    ax1 = fig.add_subplot(1, 3, 1, projection=crs)
    ax2 = fig.add_subplot(1, 3, 2, projection=crs)
    ax3 = fig.add_subplot(1, 3, 3, projection=crs)
    # Create a numpy array from this slice?

    im1 = ax1.imshow(valid_gfs, vmin=min, vmax=max, cmap='turbo', transform=crs, extent=extent)
    im2 = ax2.imshow(valid_era, vmin=min, vmax=max, cmap='turbo', transform=crs, extent=extent)
    im3 = ax3.imshow(valid_output, vmin=min, vmax=max, cmap='turbo', transform=crs, extent=extent)

    ax1.coastlines(linewidth=0.7)
    ax2.coastlines(linewidth=0.7)
    ax3.coastlines(linewidth=0.7)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax3.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')

    fig.suptitle(f"C{channel}-{variable}-{level}")

    ax1.set_title("GFS")
    ax2.set_title("ERA5")
    ax3.set_title("Model Output")

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])

    # cbar = plt.colorbar(im3, ax=ax3, shrink=0.6)
    # cbar.set_label(variable)
    # cbar.set_ticks([min, med, max])

    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])
    cbar = fig.colorbar(im3, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(variable)
    cbar.set_ticks([min, med, max])

    plt.tight_layout()
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    name = os.path.join(output_path, f"C{channel}-{variable}-{level}")
    plt.savefig(name, bbox_inches='tight')
    plt.close('all')


def compare_all_tensors_spectral(gfs_tensor, era_tensor, output_tensor,
                         channel, variable, level, output_path="",
                           format='mp4', fps=6):
    
    # Ensure non inf non null values
    valid_gfs = torch.where(torch.isnan(gfs_tensor), torch.tensor(0, device=gfs_tensor.device), gfs_tensor).cpu().numpy()
    valid_era = torch.where(torch.isnan(era_tensor), torch.tensor(0, device=era_tensor.device), era_tensor).cpu().numpy()
    valid_output = torch.where(torch.isnan(output_tensor), torch.tensor(0, device=output_tensor.device), output_tensor).cpu().numpy()

    # For Driscoll-Healy Grids, the 90 degree S latitude should not be included
    valid_gfs = valid_gfs[:-1, :]
    valid_era = valid_era[:-1, :]
    valid_output = valid_output[:-1, :]

    # Plot the spectrum
    fig_spec = plt.figure(figsize=(12, 6))
    axs = fig_spec.add_subplot(1, 1, 1)

    # Use pysh to compute the power spectrum for each of these datasets
    gfs_grid = pysh.SHGrid.from_array(valid_gfs, grid='DH')
    era_grid = pysh.SHGrid.from_array(valid_era)
    output_grid = pysh.SHGrid.from_array(valid_output)

    gfs_coeffs = gfs_grid.expand()
    era_coeffs = era_grid.expand()
    output_coeffs = output_grid.expand()

    gfs_coeffs.plot_spectrum(ax=axs, legend="GFS")
    era_coeffs.plot_spectrum(ax=axs, legend="ERA")
    output_coeffs.plot_spectrum(ax=axs, legend="Output")

    fig_spec.suptitle(f"Spectrum for C{channel}-{variable}-{level} Datasets")
    fig_spec.savefig(f"testing_viz/spectrum_{channel}_{variable}_{level}.png")

    # Plot some comparison statistics
    fig_comp = plt.figure(figsize=(18, 6))
    ax1 = fig_comp.add_subplot(1, 3, 1)
    ax2 = fig_comp.add_subplot(1, 3, 2)
    ax3 = fig_comp.add_subplot(1, 3, 3)

    era_coeffs.plot_cross_spectrum(output_coeffs, ax=ax1)
    era_coeffs.plot_admittance(output_coeffs, ax=ax2)
    era_coeffs.plot_correlation(output_coeffs, ax=ax3)

    ax1.set_title("Cross Spectrum")
    ax2.set_title("Admittance")
    ax3.set_title("Correlation")

    fig_comp.suptitle(f"Spherical Harmonics Comparisons Between Outputs v. ERA5 for C{channel}-{variable}-{level} Datasets")

    fig_comp.savefig(f"testing_viz/coeffs_comparison_{channel}_{variable}_{level}.png")

    plt.close('all')

    # # CRPS for output and ERA5
    # output_xr = xr.DataArray(data=valid_output)
    # era_xr = xr.DataArray(data=valid_era)

    # crps = crps_cdf(output_xr, era_xr)

    # print(f"CRPS: {crps}")


def visualize_residuals(gfs_tensor, era_tensor, output_tensor, 
                        channel, variable, level, output_path=""):
    
    # Ensure non inf non null values
    valid_gfs = torch.where(torch.isnan(gfs_tensor), torch.tensor(0, device=gfs_tensor.device), gfs_tensor).cpu().numpy()
    valid_era = torch.where(torch.isnan(era_tensor), torch.tensor(0, device=era_tensor.device), era_tensor).cpu().numpy()
    valid_output = torch.where(torch.isnan(output_tensor), torch.tensor(0, device=output_tensor.device), output_tensor).cpu().numpy()

    # Create the composite output
    composite = valid_gfs + valid_output

    # Get the stats from each of these datasets
    min = np.amin([np.amin(valid_gfs), np.amin(valid_era), np.amin(composite)])
    max = np.amax([np.amax(valid_gfs), np.amax(valid_era), np.amax(composite)])
    med = np.median([np.median(valid_gfs), np.median(valid_era), np.median(composite)])

    # Set the projection
    crs = ccrs.PlateCarree(central_longitude=180)   

    lon = np.arange(-180, 180, 0.25)
    lat = np.arange(-90, 90, 0.25)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]   

    fig = plt.figure(figsize=(12, 6))

    # Display three plots horizontally
    ax1 = fig.add_subplot(2, 2, 1, projection=crs) # GFS
    ax2 = fig.add_subplot(2, 2, 2, projection=crs) # ERA
    ax3 = fig.add_subplot(2, 2, 3, projection=crs) # Output
    ax4 = fig.add_subplot(2, 2, 4, projection=crs) # Composite

    im1 = ax1.imshow(valid_gfs, vmin=min, vmax=max, cmap='turbo', transform=crs, extent=extent)
    im2 = ax2.imshow(valid_era, vmin=min, vmax=max, cmap='turbo', transform=crs, extent=extent)
    im3 = ax3.imshow(valid_output, vmin=min, vmax=max, cmap='turbo', transform=crs, extent=extent)
    im4 = ax4.imshow(composite, vmin=min, vmax=max, cmap='turbo', transform=crs, extent=extent)

    ax1.coastlines(linewidth=0.7)
    ax2.coastlines(linewidth=0.7)
    ax3.coastlines(linewidth=0.7)
    ax4.coastlines(linewidth=0.7)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax3.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax4.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')

    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])
    cbar = fig.colorbar(im4, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(variable)
    cbar.set_ticks([min, med, max])

    fig.suptitle(f"All Datasets: C{channel}-{variable}-{level}")

    ax1.set_title("GFS")
    ax2.set_title("ERA5")
    ax3.set_title("Model Output Residuals")
    ax4.set_title("Composite: GFS + Residuals")

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])

    plt.tight_layout()
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    name = os.path.join(output_path, f"all_datasets_c{channel}-{variable}-{level}.png")
    plt.savefig(name, bbox_inches='tight')
    plt.close('all')



def visualize_input_target(input_tensor, target_tensor, variable, level, output_path=""):
    crs = ccrs.PlateCarree(central_longitude=180)   

    lon = np.arange(-180, 180, 0.25)
    lat = np.arange(-90, 90, 0.25)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]   

    input_tensor = torch.where(torch.isnan(input_tensor), torch.tensor(0, device=input_tensor.device), input_tensor).cpu().numpy()
    target_tensor = torch.where(torch.isnan(target_tensor), torch.tensor(0, device=target_tensor.device), target_tensor).cpu().numpy()

    in_min = np.amin(input_tensor)
    in_max = np.amax(input_tensor)
    in_med = np.median(input_tensor)

    tar_min = np.amin(target_tensor)
    tar_max = np.amax(target_tensor)
    tar_med = np.median(target_tensor)

    print("Tensor Range: ", in_min, in_max)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection=crs)
    ax2 = fig.add_subplot(1, 2, 2, projection=crs)

    im1 = ax1.imshow(input_tensor, vmin=in_min, vmax=in_max, cmap='turbo', transform=crs, extent=extent)
    im2 = ax2.imshow(target_tensor, vmin=tar_min, vmax=tar_max, cmap='coolwarm', transform=crs, extent=extent)

    ax1.coastlines(linewidth=0.7)
    ax2.coastlines(linewidth=0.7)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')

    fig.suptitle(f"{variable}-{level}")

    ax1.set_title("GFS")
    ax2.set_title("Residuals")

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    cbar1 = fig.colorbar(im1, orientation="horizontal")
    cbar1.set_label(variable)
    cbar1.set_ticks([in_min, in_med, in_max])

    cbar2 = fig.colorbar(im2, orientation="horizontal")
    cbar2.set_label(variable)
    cbar2.set_ticks([tar_min, tar_med, tar_max])

    plt.tight_layout()
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    name = os.path.join(output_path, f"{variable}-{level}-datacheck.png")
    plt.savefig(name, bbox_inches='tight')
    plt.close('all')


def visualize_input_era_target(input_tensor, era_tensor, target_tensor, variable, level, output_path=""):
    crs = ccrs.PlateCarree(central_longitude=180)   

    lon = np.arange(-180, 180, 0.25)
    lat = np.arange(-90, 90, 0.25)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]   

    input_tensor = torch.where(torch.isnan(input_tensor), torch.tensor(0, device=input_tensor.device), input_tensor).cpu().numpy()
    target_tensor = torch.where(torch.isnan(target_tensor), torch.tensor(0, device=target_tensor.device), target_tensor).cpu().numpy()
    era_tensor = torch.where(torch.isnan(era_tensor), torch.tensor(0, device=era_tensor.device), era_tensor).cpu().numpy()

    cbar_min = np.amin([np.amin(era_tensor), np.amin(target_tensor)])
    cbar_med = np.median([np.median(era_tensor), np.median(target_tensor)])
    cbar_max = np.amax([np.amax(era_tensor), np.amax(target_tensor)])

    tar_min = np.amin(target_tensor)
    tar_max = np.amax(target_tensor)
    tar_med = np.median(target_tensor)

    print(f"{variable} - {level} Tensor Range: ", cbar_min, cbar_max)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 3, 1, projection=crs)
    ax2 = fig.add_subplot(1, 3, 2, projection=crs)
    ax3 = fig.add_subplot(1, 3, 3, projection=crs)

    im1 = ax1.imshow(input_tensor, vmin=cbar_min, vmax=cbar_max, cmap='turbo', transform=crs, extent=extent)
    im2 = ax2.imshow(era_tensor, vmin=cbar_min, vmax=cbar_max, cmap='turbo', transform=crs, extent=extent)
    im3 = ax3.imshow(target_tensor, vmin=tar_min, vmax=tar_max, cmap='coolwarm', transform=crs, extent=extent)

    ax1.coastlines(linewidth=0.7)
    ax2.coastlines(linewidth=0.7)
    ax3.coastlines(linewidth=0.7)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax3.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')

    fig.suptitle(f"{variable}-{level}")

    ax1.set_title("GFS")
    ax2.set_title("ERA")
    ax3.set_title("Residuals")

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])

    cbar1 = fig.colorbar(im1, orientation="horizontal")
    cbar1.set_label(variable)
    cbar1.set_ticks([cbar_min, cbar_med, cbar_max])

    cbar2 = fig.colorbar(im2, orientation="horizontal")
    cbar2.set_label(variable)
    cbar2.set_ticks([cbar_min, cbar_med, cbar_max])

    cbar3 = fig.colorbar(im3, orientation="horizontal")
    cbar3.set_label(variable)
    cbar3.set_ticks([tar_min, tar_med, tar_max])

    plt.tight_layout()
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    name = os.path.join(output_path, f"{variable}-{level}-datacheck.png")
    plt.savefig(name, bbox_inches='tight')
    plt.close('all')


def visualize_residual_outputs(input_tensor, era_tensor, target_tensor, output_tensor, variable, level, batch, output_path=""):
    crs = ccrs.PlateCarree(central_longitude=180)   

    lon = np.arange(-180, 180, 0.25)
    lat = np.arange(-90, 90, 0.25)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]   

    input_tensor = torch.where(torch.isnan(input_tensor), torch.tensor(0, device=input_tensor.device), input_tensor).cpu().numpy() # GFS
    era_tensor = torch.where(torch.isnan(era_tensor), torch.tensor(0, device=era_tensor.device), era_tensor).cpu().numpy() # ERA
    target_tensor = torch.where(torch.isnan(target_tensor), torch.tensor(0, device=target_tensor.device), target_tensor).cpu().numpy() # True Residuals
    output_tensor = torch.where(torch.isnan(output_tensor), torch.tensor(0, device=output_tensor.device), output_tensor).cpu().numpy() # Model Residuals

    # Combine GFS & Model Residuals
    composite_tensor = input_tensor + output_tensor

    full_min = np.amin([np.amin(composite_tensor), np.amin(era_tensor)])
    full_med = np.median([np.median(composite_tensor), np.median(era_tensor)])
    full_max = np.amax([np.amax(composite_tensor), np.amax(era_tensor)])

    res_min = np.amin([np.amin(target_tensor), np.amin(output_tensor)])
    res_med = np.median([np.median(target_tensor), np.median(output_tensor)])
    res_max = np.amax([np.amax(target_tensor), np.amax(output_tensor)])

    # Difference between the two residuals
    diff_tensor = np.abs(target_tensor - output_tensor) / (res_max - res_min)
    diff_med = np.median(diff_tensor)

    # print(f"{variable} - {level} Tensor Range: ", cbar_min, cbar_max)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(2, 3, 1, projection=crs) # GFS
    ax2 = fig.add_subplot(2, 3, 2, projection=crs) # ERA5
    ax3 = fig.add_subplot(2, 3, 3, projection=crs) # Composite: GFS + Model Residuals

    ax4 = fig.add_subplot(2, 3, 4, projection=crs) # True Residuals
    ax5 = fig.add_subplot(2, 3, 5, projection=crs) # Diff Residuals
    ax6 = fig.add_subplot(2, 3, 6, projection=crs) # Model Residuals

    im1 = ax1.imshow(input_tensor, vmin=full_min, vmax=full_max, cmap='turbo', transform=crs, extent=extent) # GFS
    im2 = ax2.imshow(era_tensor, vmin=full_min, vmax=full_max, cmap='turbo', transform=crs, extent=extent) # ERA
    im3 = ax3.imshow(composite_tensor, vmin=full_min, vmax=full_max, cmap='turbo', transform=crs, extent=extent) # Composite

    im4 = ax4.imshow(target_tensor, vmin=res_min, vmax=res_max, cmap='bwr', transform=crs, extent=extent) # True Residuals
    im5 = ax5.imshow(diff_tensor, vmin=0, vmax=1, cmap='RdPu', transform=crs, extent=extent) # Diff Residuals
    im6 = ax6.imshow(output_tensor, vmin=res_min, vmax=res_max, cmap='bwr', transform=crs, extent=extent) # Model Residuals

    ax1.coastlines(linewidth=0.7)
    ax2.coastlines(linewidth=0.7)
    ax3.coastlines(linewidth=0.7)
    ax4.coastlines(linewidth=0.7)
    ax5.coastlines(linewidth=0.7)
    ax6.coastlines(linewidth=0.7)

    ax1.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax3.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax4.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax5.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax6.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')

    fig.suptitle(f"{variable}-{level}")

    ax1.set_title("GFS")
    ax2.set_title("ERA5")
    ax3.set_title("Composite: GFS + Model Residuals")
    ax4.set_title("True Residuals")
    ax5.set_title("Residuals Absolute Error")
    ax6.set_title("Model Residuals")

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax6.set_xticks([])
    ax6.set_yticks([])

    cbar1 = fig.colorbar(im1, orientation="horizontal", shrink=0.6)
    cbar1.set_label(variable)
    cbar1.set_ticks([full_min, full_med, full_max])

    cbar2 = fig.colorbar(im2, orientation="horizontal", shrink=0.6)
    cbar2.set_label(variable)
    cbar2.set_ticks([full_min, full_med, full_max])

    cbar3 = fig.colorbar(im3, orientation="horizontal", shrink=0.6)
    cbar3.set_label(variable)
    cbar3.set_ticks([full_min, full_med, full_max])

    cbar4 = fig.colorbar(im4, orientation="horizontal", shrink=0.6)
    cbar4.set_label(variable)
    cbar4.set_ticks([res_min, res_med, res_max])

    cbar5 = fig.colorbar(im5, orientation="horizontal", shrink=0.6)
    cbar5.set_label(variable)
    cbar5.set_ticks([0, diff_med, 1])    

    cbar6 = fig.colorbar(im6, orientation="horizontal", shrink=0.6)
    cbar6.set_label(variable)
    cbar6.set_ticks([res_min, res_med, res_max])

    plt.tight_layout()
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    name = os.path.join(output_path, f"residual-outputs-{variable}-L{level}-{batch}.png")
    plt.savefig(name, bbox_inches='tight')
    plt.close('all')


if __name__ == "__main__":
    from data_download import download_era5, make_batch

    static_path, surface_path, atmos_path = download_era5()
    batch = make_batch(static_path, surface_path, atmos_path, 1)

    model = Aurora(use_lora=False)
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

    preds = generate_outputs(model, batch, steps=28)
    visualize_outputs(preds, steps=28, variable="2t")