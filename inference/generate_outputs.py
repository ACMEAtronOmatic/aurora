from aurora import Aurora, rollout
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import torch
import numpy as np
import imageio.v3 as iio
import os

def generate_outputs(model, batch, steps=28):
    model.eval()
    model = model.to("cuda")

    with torch.inference_mode():
        preds = [pred.to("cpu") for pred in rollout(model, batch, steps=steps)]


    return preds


def visualize_outputs(preds, steps=28, output_path="aurora.gif", fps=6):
# Create and save individual plot images
    images = []
    crs = ccrs.PlateCarree(central_longitude=180)

    lon = np.arange(-180, 180, 0.25)
    lat = np.arange(-90, 90, 0.25)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]   

    for i in range(steps):
        # Your plotting code here   
        pred = preds[i]  

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=crs)
        ax.set_aspect('auto')
        ax.imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50, cmap='turbo', transform=crs, extent=extent)
        ax.coastlines(linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='black')
        ax.set_ylabel(str(pred.metadata.time[0]))
        ax.set_title(f"Step {i}")
        ax.set_xticks([])
        ax.set_yticks([])

        
        # Save plot to memory
        plt.savefig(f"temp_{i}.jpg", bbox_inches='tight')
        # Clear the current figure
        plt.close()
        
        # Read the saved image
        images.append(iio.imread(f"temp_{i}.jpg"))
    
    # Create GIF
    iio.imwrite(output_path, images, loop=0, fps=fps)

    # Delete temporary images
    for i in range(steps):
        os.remove(f"temp_{i}.jpg")


if __name__ == "__main__":
    from data_download import download_era5, make_batch

    static_path, surface_path, atmos_path = download_era5()
    batch = make_batch(static_path, surface_path, atmos_path, 1)

    model = Aurora(use_lora=False)
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

    preds = generate_outputs(model, batch, steps=28)
    visualize_outputs(preds, steps=28)