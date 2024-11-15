from aurora import Aurora, rollout
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import cv2
import torch
import numpy as np
import os
import xarray as xr

def generate_outputs(model, batch, steps=28):
    model.eval()
    model = model.to("cuda")

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
                truth = comparison_data["t2m"]

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
                ax1.imshow(speed, vmin=0, vmax=50, cmap='turbo', transform=crs, extent=extent)
                ax2.imshow(speed_truth, vmin=0, vmax=50, cmap='turbo', transform=crs, extent=extent)


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


def era5_comparison(steps=28, variable="t2m", data_path=""):
    '''
    Generate arrays for each timestep from ERA5 data of the desired variable
    '''
    ds = xr.open_dataset(data_path, engine="netcdf4")

    data = {}

    if variable == "wind":
        data['u10'] = ds['u10'].values[:steps]
        data['v10'] = ds['v10'].values[:steps]
    else:
        data[variable] = ds[variable].values[:steps]

    # data = []

    # for i in range(steps):
    #     if variable == "t2m":
    #         data.append(ds["t2m"].values[i])
    #     elif variable == "u10":
    #         data.append(ds["u10"].values[i])
    #     elif variable == "v10":
    #         data.append(ds["v10"].values[i])
    #     elif variable == "msl":
    #         data.append(ds["msl"].values[i])
    #     else:
    #         raise ValueError(f"Invalid variable: {variable}")
        
    return data

if __name__ == "__main__":
    from data_download import download_era5, make_batch

    static_path, surface_path, atmos_path = download_era5()
    batch = make_batch(static_path, surface_path, atmos_path, 1)

    model = Aurora(use_lora=False)
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

    preds = generate_outputs(model, batch, steps=28)
    visualize_outputs(preds, steps=28, variable="2t")