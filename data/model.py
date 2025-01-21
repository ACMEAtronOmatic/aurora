'''
UNet for unbiasing GFS ICs to be used as inputs to Aurora, which is trained on ERA5 data

Loss: 
- MSE between unbiased GFS ICs and actual ERA5 data
- SSIM between unbiased GFS ICs and actual ERA5 data
- MAE between unbiased GFS ICs and actual ERA5 data
- Should MSE and MAE be pixel weighted?

Encoder:
ConvNext with SE blocks and Cross-dimension attention

Decoder:
Transposed convolutions to mirror encoder, depth to space
Has skip connections from analagous encoder layer
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import pytorch_lightning as pl
from pytorch_msssim import MS_SSIM

import matplotlib.pyplot as plt

from data.utils import print_debug, check_gpu_memory
from inference.generate_outputs import visualize_tensor

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


class Loss(nn.Module):
    def __init__(self, channel_mapper, remap=True,
                  exponent=5.0, power=3.0, constant=1.0,
                    channels=16, debug=False,
                     mse_weight=1.0, mae_weight=0.0, ssim_weight=0.0):
        super().__init__()

        self.debug = debug
        self.remap = remap
        self.exponent = exponent
        self.power = power
        self.constant = constant
        self.channel_mapper = channel_mapper

        # NOTE: data range must be consistent, which requires remapping the channels
        # To a consistent range of [0, 1]
        # Need to determine global min and max for each channel that would work for GFS and ERA5
        self.ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=channels)

    # x = UNet output, y = actual ERA5 data
    def forward(self, x, y):
        # Use remap if data is not already in [0, 1]
        if self.remap:
            # For each channel in x and y, remap to [0, 1]
            # Map the predicted tensor based on predicted variable/level
            _, channels, _, _ = x.size()

            for c in range(channels):
                # Get the associated variable
                # Using the GFS channel mapper to get the variable and 
                var, level = self.channel_mapper[c]
                min_val = ERA5_GLOBAL_RANGES[var]['min']
                max_val = ERA5_GLOBAL_RANGES[var]['max']

                print_debug(self.debug, "\nVariable: ", var, "Level: ", level)
                print_debug(self.debug, "X:", x[:, c, :, :].min().item(), x[:, c, :, :].median().item(), x[:, c, :, :].max().item())
                print_debug(self.debug, "Y:", y[:, c, :, :].min().item(), y[:, c, :, :].median().item(), y[:, c, :, :].max().item())

                if self.debug:
                    visualize_tensor(y, c, self.channel_mapper, output_path="debugging")

                # Remap to [0, 1]
                x[:, c, :, :] = torch.clamp((x[:, c, :, :] - min_val) / (max_val - min_val), 0, 1)
                y[:, c, :, :] = torch.clamp((y[:, c, :, :] - min_val) / (max_val - min_val), 0, 1)

                print_debug(self.debug, "X REMAPPED:", x[:, c, :, :].min().item(), x[:, c, :, :].median().item(), x[:, c, :, :].max().item())
                print_debug(self.debug, "Y REMAPPED:", y[:, c, :, :].min().item(), y[:, c, :, :].median().item(), y[:, c, :, :].max().item())

                # Check for NaN and Inf value in these slices
                if torch.isnan(x[:, c, :, :]).any() or torch.isinf(x[:, c, :, :]).any():
                    raise ValueError("Input tensor contains NaN or Inf values after remapping")

                if torch.isnan(y[:, c, :, :]).any() or torch.isinf(y[:, c, :, :]).any():
                    raise ValueError("Output tensor contains NaN or Inf values after remapping")


        # MS SSIM is always non-negative
        ssim_loss = 1 - self.ms_ssim_module(x, y)

        weighting = torch.full_like(y, 1.0, dtype=torch.float32)
        weighting = torch.exp(self.exponent * torch.pow(y, self.power)) + self.constant

        mae_loss = torch.mean(torch.multiply(weighting, torch.abs(x - y)))
        mse_loss = torch.mean(torch.multiply(weighting, torch.square(x - y)))

        return mse_loss, mae_loss, ssim_loss


class SEAttnBlock(nn.Module):
    '''
    Squeeze-Excitation block to determine channel weightings
    Uses global average pooling to squeeze info into lower dimensional
    representation, and learns weightings (excites) for each channel
    through a linear layer

    Often used after a convolutional layer to refine feature representations
    
    '''
    def __init__(self, c_in, r=8, debug=False):
        super().__init__()
        
        self.debug = debug
        # Define the functions/layers that will be used

        # Adaptive pooling allows you to define the size of the output
        self.pool = nn.AdaptiveAvgPool2d(1) # Global Average Pooling

        # Define the linear layer
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // r, bias=False), # reduce
            nn.GELU(), # act fxn
            nn.Linear(c_in // r, c_in, bias=False), # expand
            nn.Sigmoid() # output weightings
        )


    def forward(self, x):
        print_debug(self.debug, f"\t\tSE - Input: {x.shape}")

        # TODO: why is there sometimes a batch dimension?

        batches, channels, _, _ = x.size()

        # GAP, compresses to a 2D tensor bxc (H & W are 1)
        y = self.pool(x).view(batches, channels)

        print_debug(self.debug, f"\t\tSE - GAP: {y.shape}")

        # Apply Excitement Layer, reshape to tensor bxcx1x1
        y = self.fc(y).view(batches, channels, 1, 1)

        print_debug(self.debug, f"\t\tSE - Excitement: {y.shape}")

        # expand_as will match y to the shape of x
        return x * y.expand_as(x)


class Conv2DBlock(nn.Module):
    '''
    Convolutional 2d block for spatio-temporal relationships
    Can use SE blocks for channel attention
    '''

    def __init__(self, c_in, c_out, samples, height, width, use_se=True, r=8, double=False, debug=False):
        super().__init__()

        self.debug = debug

        # Define convolutional layer
        # Input: [samples, c_in, levels, height, width]
        # Output: [samples, c_out, levels, height, width]
        if double:
            self.conv = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1),
                nn.LayerNorm([c_out, height, width]),
                nn.GELU(),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, stride=1),
                nn.LayerNorm([c_out, height, width]),
                nn.GELU()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1),
                nn.LayerNorm([c_out, height, width]),
                nn.GELU()
            )

        # SE block
        self.se = SEAttnBlock(c_out, r=r) if use_se else nn.Identity()


    def forward(self, x):
        print_debug(self.debug, f"\t\tCONV - Input: {x.shape}")
        x = self.conv(x)
        print_debug(self.debug, f"\t\tCONV - Post Conv2D: {x.shape}")
        x = self.se(x)
        print_debug(self.debug, f"\t\tCONV -Post SE: {x.shape}")
        return x


class Encoder(nn.Module):
    '''
    Conv blocks with SE and MaxPooling for downsampling
    '''

    def __init__(self, c_in, samples, height, width,
                 use_se=True, r=8, feature_dims=[64, 128, 256, 512], double=False, debug=False):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsampling = nn.ModuleList()

        self.debug = debug

        c = c_in

        for i, f in enumerate(feature_dims):
            # Decrease in spatial resolution, increase in channels
            self.downsampling.append(
                Conv2DBlock(c_in=c, c_out=f, samples=samples,
                             height=height//(2**i), width=width//(2**i),
                              use_se=use_se, r=r, double=double, debug=debug)
            )

            print_debug(self.debug, f"Conv2D Block: {c} -> {f}, H={height//(2**i)}, W={width//(2**i)}")

            c = f


    def forward(self, x):
        skip_connections = [] # will be returned to use in decoder
        for downsampling in self.downsampling:
            print_debug(self.debug, f"\n\tENC - Input: {x.shape}")
            x = downsampling(x)            
            print_debug(self.debug, f"\tENC - Post Downsampling: {x.shape}")
            skip_connections.append(x)
            x = self.pool(x) # reduces spatial resolution by half
            print_debug(self.debug, f"\tENC - Post Pooling: {x.shape}")


        return x, skip_connections


class Decoder(nn.Module):
    '''
    Transposed Convolution for upsampling, followed by Conv3D
    to refine features and reduce potential artifacts

    # Typical decoder block
    self.upconv = nn.ConvTranspose3d(in_channels, mid_channels, kernel_size=2, stride=2)
    self.refine_conv = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)

    NOTE: the final output will be handled by the UNet
    Only focus on the feature dimensions here


    '''

    def __init__(self, samples, height, width,
                 feature_dims=[64, 128, 256, 512],
                  use_se=True, r=8, double=False, skip_method="add", debug=False):
        super().__init__()

        self.skip_method = skip_method
        self.debug = debug

        self.upsampling = nn.ModuleList()
        self.dec = nn.ModuleList()

        # We assume that the height and width passed in are of the output
        # The height and width of the bottleneck will be
        # H // 2**len(feature_dims), W // 2**len(feature_dims)


        # Assumes that feature_dims will be in the same order as Encoder
        # Starts with input from bottleneck, which is 2*largest feature dims
        for i, f in enumerate(feature_dims[::-1]):
            layer = len(feature_dims) - i - 1

            if i == len(feature_dims) - 1:
                f_out = f # last layer
            else:
                f_out = f//2

            # Increase in spatial resolution
            self.upsampling.append(
                nn.ConvTranspose2d(f, f, kernel_size=2, stride=2)
            )

            print_debug(self.debug, f"Conv Transposed 2D: {f} -> {f}")

            # Refine upsampling by decreasing channels
            self.dec.append(
                Conv2DBlock(c_in=f, c_out=f_out, 
                            samples=samples, height=height//(2**layer),
                              width=width//(2**layer), use_se=use_se, r=r, double=double, debug=debug)
            )

            print_debug(self.debug, f"Conv2D Block: {f} -> {f_out}, H={height//(2**layer)}, W={width//(2**layer)}")


    def forward(self, x, skip_connections):

        skip_connections = skip_connections[::-1]
        for i, (upsampling, dec) in enumerate(zip(self.upsampling, self.dec)):
            print_debug(self.debug, f"\n\tDEC - Input: {x.shape}")
            
            x = upsampling(x) # transposed conv
            print_debug(self.debug, f"\tDEC - Upsampling: {x.shape}")
            print_debug(self.debug, f"\tDEC - Skip Connection: {skip_connections[i].shape}")

            if self.skip_method == "add":
                # May need to upsample the skip connection as well
                # Typically only if f//2 does not exactly match the next feature dim
                # Which can happen in the very last layer (e.g. 720 vs 721)
                if x.shape != skip_connections[i].shape:
                    print_debug(self.debug, f"DEC - Shape Mismatch: X = {x.shape} | Skip = {skip_connections[i].shape}")

                    # Check if 3D or 4D tensor
                    if len(x.shape) == 3:
                        # Need to add an extra dimension to use this function
                        # Requires min. 4D
                        x = F.interpolate(
                            x.unsqueeze(0),
                            size=skip_connections[i].shape[2:],  # Match spatial dimensions of the skip connection
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)
                    elif len(x.shape) == 4:
                        x = F.interpolate(
                            x,
                            size=skip_connections[i].shape[2:],  # Match spatial dimensions of the skip connection
                            mode='bilinear', 
                            align_corners=False
                        )

                x = x + skip_connections[i]
            elif self.skip_method == "cat":
                # NOTE: will increase the number of channels, 
                # which needs to get reflected in the decoder layer channels
                x = torch.cat((x, skip_connections[i]), dim=1)

            print_debug(self.debug, f"DEC - Skip Added: {x.shape}")

            x = dec(x) # refinement'
            print_debug(self.debug, f"DEC - Refined: {x.shape}")

        return x


class GFSUnbiaser(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 samples, height, width,
                  feature_dims=[64, 128, 256, 512],
                   use_se=True, r=8, double=False, debug=False):
        super().__init__()

        self.debug = debug

        print_debug(self.debug, "Instantiating Encoder...")
        self.encoder = Encoder(c_in=in_channels, samples=samples, height=height, width=width,
                                  use_se=use_se, r=r, feature_dims=feature_dims, double=double, debug=self.debug)
        
        print_debug(self.debug, "Instantiating Decoder...")
        self.decoder = Decoder(feature_dims=feature_dims,
                                samples=samples, height=height, width=width, r=r,
                                    use_se=use_se, double=double, skip_method="add", debug=self.debug)
   
        self.final_conv = nn.Conv2d(feature_dims[0], out_channels, kernel_size=1)

        print_debug(self.debug, "Done!")

    def forward(self, x):
        print_debug(self.debug, f"MODEL - Input: {x.shape}")
        x, skip_connections = self.encoder(x)
        print_debug(self.debug, f"MODEL - Post Encoder: {x.shape}")
        x = self.decoder(x, skip_connections)
        print_debug(self.debug, f"MODEL - Post Decoder: {x.shape}")
        x = self.final_conv(x)
        print_debug(self.debug, f"MODEL - After Final Conv: {x.shape}")

        return x
    

class LightningGFSUnbiaser(pl.LightningModule):
    def __init__(self, in_channels, out_channels, channel_mapper, 
                 samples, height, width,
                  feature_dims=[64, 128, 256, 512],
                   use_se=True, r=8, double=False, debug=False,
                     mse_weight=1.0, mae_weight=0.0, ssim_weight=0.0):
        super().__init__()
        self.save_hyperparameters()

        if len(channel_mapper) != out_channels:
            raise ValueError(f"Length of channel mapper must be equal to number of output channels. Got {len(self.channel_mapper)} vs {out_channels}")
        
        if len(feature_dims) < 2:
            raise ValueError(f"Feature dimensions must be at least 2. Got {len(feature_dims)}")
        
        self.channel_mapper = channel_mapper
        self.debug = debug

        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.ssim_weight = ssim_weight

        self.model = GFSUnbiaser(in_channels, out_channels, 
                                 feature_dims=feature_dims,
                                 samples=samples, height=height, width=width,
                                   use_se=use_se, r=r, double=double, debug=debug)
        
        print_debug(self.debug, "Model Summary:")
        summary(self.model, (samples, in_channels, height, width))
        
        self.loss_fxn = Loss(channel_mapper=self.channel_mapper,remap=True, exponent=5.0, power=3.0,
                              constant=1.0, channels=out_channels, debug=debug)
        

    def forward(self, x):
        return self.model(x)
    

    def evaluate(self, batch, stage=None):
        x, y = batch
        pred = self.forward(x)

        mse_loss, mae_loss, ssim_loss = self.loss_fxn.forward(pred, y)

        composite_loss = (self.mse_weight * mse_loss) + (self.mae_weight * mae_loss) + (self.ssim_weight * ssim_loss)

        if stage is not None:
            self.log_dict({f"{stage}_ssim_loss": ssim_loss,
                            f"{stage}_mse_loss": mse_loss, 
                            f"{stage}_mae_loss": mae_loss,
                            f"{stage}_composite_loss": composite_loss},
                            prog_bar=True)
        else:
            self.log_dict({"val_ssim_loss": ssim_loss,
                            "val_mse_loss": mse_loss,
                            "val_mae_loss": mae_loss,
                            "val_composite_loss": composite_loss},
                            prog_bar=True)

    def training_step(self, batch, batch_idx, stage=None):
        x, y = batch
        # Check if there are nan or inf values in the input tensor
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input tensor contains NaN or Inf values")

        logits = self.forward(x)

        # Check if there are nan or inf values in the output tensor
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("Output tensor contains NaN or Inf values")

        mse_loss, mae_loss, ssim_loss = self.loss_fxn.forward(logits, y)

        # Check if there are nan or inf values in the loss tensor
        if torch.isnan(ssim_loss).any() or torch.isinf(ssim_loss).any():
            raise ValueError("SSIM loss tensor contains NaN or Inf values")
        
        if torch.isnan(mse_loss).any() or torch.isinf(mse_loss).any():
            raise ValueError("MSE loss tensor contains NaN or Inf values")
        
        if torch.isnan(mae_loss).any() or torch.isinf(mae_loss).any():
            raise ValueError("MAE loss tensor contains NaN or Inf values")

        composite_loss = (self.mse_weight * mse_loss) + (self.mae_weight * mae_loss) + (self.ssim_weight * ssim_loss)

        if stage is not None:
            self.log_dict({f"{stage}_ssim_loss": ssim_loss,
                            f"{stage}_mse_loss": mse_loss,
                            f"{stage}_mae_loss": mae_loss, 
                            f"{stage}_composite_loss": composite_loss},
                            prog_bar=True)
        else:
            self.log_dict({"train_ssim_loss": ssim_loss,
                            "train_mse_loss": mse_loss,
                            "train_mae_loss": mae_loss, 
                            "train_composite_loss": composite_loss},
                            prog_bar=True)

        return composite_loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        return pred

    def configure_optimizers(self, config=None):
        lr = 1e-3
        algo = "AdamW"
        decay = 0.01
        if config is not None:
            lr = config["lr"]
            algo = config["optimizer"]
            decay = config["weight_decay"]

        if algo == "AdamW":
            return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=decay)
        else:
            return torch.optim.Adam(self.parameters(), lr=lr)

