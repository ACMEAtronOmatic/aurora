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

from data.dataloader import CHANNEL_MAP, LEVEL_MAP
from data.era5_download import download_era5

ERA5_GLOBAL_RANGES = {
    # Surface Variables
    't2m': {'min': 170, 'max': 350},  # 2m Temperature [K]
    'msl': {'min': 85000, 'max': 110000},  # Mean Sea Level Pressure [Pa]
    'u10': {'min': -110, 'max': 110},  # 10m U-wind [m/s]
    'v10': {'min': -80, 'max': 80},  # 10m V-wind [m/s]
    
    # Atmospheric Variables (at all pressure levels)
    't': {'min': 170, 'max': 350},  # Temperature [K]
    'q': {'min': 0, 'max': 0.04},  # Specific Humidity [kg/kg]
    'u': {'min': -110, 'max': 110},  # U-wind [m/s]
    'v': {'min': -100, 'max': 100},  # V-wind [m/s]
    'z': {'min': -1000, 'max': 60000},  # Geopotential Height [m]
    
}

def get_model_memory_usage(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return {
        'Parameters (MB)': param_size / 1024 / 1024,
        'Buffers (MB)': buffer_size / 1024 / 1024,
        'Total (MB)': (param_size + buffer_size) / 1024 / 1024
    }

class Loss(nn.Module):
    def __init__(self, remap=True, exponent=5.0, power=3.0, constant=1.0, channels=16):
        super().__init__()

        self.remap = remap
        self.exponent = exponent
        self.power = power
        self.constant = constant

        # NOTE: data range must be consistent, which requires remapping the channels
        # To a consistent range of [0, 1]
        # Need to determine global min and max for each channel that would work for GFS and ERA5
        self.ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=channels)

    # x = UNet output, y = actual ERA5 data
    def forward(self, x, y):
        # Use remap if data is not already in [0, 1]
        if self.remap:
            # For each channel in x and y, remap to [0, 1]
            for c in ERA5_GLOBAL_RANGES.keys():
                for l in LEVEL_MAP.keys():
                    x[c, l, :, :] = torch.clamp((x[c, l, :, :] - ERA5_GLOBAL_RANGES[c]['min']) / (ERA5_GLOBAL_RANGES[c]['max'] - ERA5_GLOBAL_RANGES[c]['min']), 0, 1)
                    y[c, l, :, :] = torch.clamp((y[c, l, :, :] - ERA5_GLOBAL_RANGES[c]['min']) / (ERA5_GLOBAL_RANGES[c]['max'] - ERA5_GLOBAL_RANGES[c]['min']), 0, 1)

        # MS SSIM is always non-negative
        ssim_loss = 1 - self.ms_ssim_module(x, y)

        weighting = torch.full_like(y, 1.0, dtype=torch.float32)
        weighting = torch.exp(self.exponent * torch.pow(y, self.power)) + self.constant

        pixel_loss = (0.5 * torch.mean(torch.multiply(weighting, torch.abs(x - y)))) + \
                     (0.5 * torch.mean(torch.multiply(1 - weighting, torch.square(x - y))))

        total_loss = ssim_loss + pixel_loss

        return total_loss


class SEAttnBlock(nn.Module):
    '''
    Squeeze-Excitation block to determine channel weightings
    Uses global average pooling to squeeze info into lower dimensional
    representation, and learns weightings (excites) for each channel
    through a linear layer

    Often used after a convolutional layer to refine feature representations
    
    '''
    def __init__(self, c_in, r=8):
        super().__init__()
        
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
        batches, channels, _, _ = x.size()

        # GAP, compresses to a 2D tensor bxc (H & W are 1)
        y = self.pool(x).view(batches, channels)

        # Apply Excitement Layer, reshape to tensor bxcx1x1
        y = self.fc(y).view(batches, channels, 1, 1)

        # expand_as will match y to the shape of x
        return x * y.expand_as(x)


class Conv2DBlock(nn.Module):
    '''
    Convolutional 3D block for spatio-temporal relationships
    Can use SE blocks for channel attention
    '''

    def __init__(self, c_in, c_out, samples, levels, height, width, use_se=True, r=8, double=False):
        super().__init__()

        # Define convolutional layer
        # Input: [samples, c_in, levels, height, width]
        # Output: [samples, c_out, levels, height, width]
        if double:
            self.conv = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1),
                nn.LayerNorm([levels, c_out, height, width]),
                nn.GELU(),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, stride=1),
                nn.LayerNorm([levels, c_out, height, width]),
                nn.GELU()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1),
                nn.LayerNorm([levels, c_out, height, width]),
                nn.GELU()
            )

        # SE block
        self.se = SEAttnBlock(c_out, r=r) if use_se else nn.Identity()


    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return x


class Encoder(nn.Module):
    '''
    Conv blocks with SE and MaxPooling for downsampling
    '''

    def __init__(self, c_in, samples, levels, height, width,
                 use_se=True, r=8, feature_dims=[64, 128, 256, 512], double=False):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsampling = nn.ModuleList()

        c = c_in

        for i, f in enumerate(feature_dims):
            # Decrease in spatial resolution, increase in channels
            self.downsampling.append(
                Conv2DBlock(c_in=c, c_out=f, samples=samples,
                             levels=levels, height=height//(2**i), width=width//(2**i),
                              use_se=use_se, r=r, double=double)
            )

            print(f"Conv2D Block: {c} -> {f}, H={height//(2**i)}, W={width//(2**i)}")

            c = f


    def forward(self, x):
        skip_connections = [] # will be returned to use in decoder
        for downsampling in self.downsampling:
            x = downsampling(x)            
            skip_connections.append(x)
            x = self.pool(x) # reduces spatial resolution by half


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

    def __init__(self, samples, levels, height, width,
                 feature_dims=[64, 128, 256, 512],
                  use_se=True, r=8, double=False, skip_method="add"):
        super().__init__()

        self.skip_method = skip_method

        self.upsampling = nn.ModuleList()
        self.dec = nn.ModuleList()

        # We assume that the height and width passed in are of the output
        # The height and width of the bottleneck will be
        # H // 2**len(feature_dims), W // 2**len(feature_dims)


        # Assumes that feature_dims will be in the same order as Encoder
        # Starts with input from bottleneck, which is 2*largest feature dims
        for i, f in enumerate(reversed(feature_dims)):
            layer = len(feature_dims) - i - 1

            # Increase in spatial resolution
            self.upsampling.append(
                nn.ConvTranspose2d(f, f, kernel_size=2, stride=2)
            )

            print(f"Conv Transposed 2D: {f} -> {f}")

            # Refine upsampling by decreasing channels
            self.dec.append(
                Conv2DBlock(c_in=f, c_out=f//2, 
                            samples=samples, levels=levels, height=height//(2**layer),
                              width=width//(2**layer), use_se=use_se, r=r, double=double)
            )

            print(f"Conv2D Block: {f} -> {f//2}, H={height//(2**layer)}, W={width//(2**layer)}")


    def forward(self, x, skip_connections):
        print(f"Decoder Input: {x.shape}")
        for i, (upsampling, dec) in enumerate(zip(self.upsampling, self.dec)):
            x = upsampling(x) # transposed conv
            print(f"Upsampling: {x.shape}")
            print(f"Skip Connection: {skip_connections[i].shape}")

            if self.skip_method == "add":
                # May need to upsample the skip connection as well
                # Typically only if f//2 does not exactly match the next feature dim
                # Which can happen in the very last layer (e.g. 720 vs 721)
                if x.shape != skip_connections[i].shape:
                    skip_connections[i] = F.interpolate(
                        skip_connections[i], 
                        size=x.shape[2:],  # Match spatial dimensions of upsampled input
                        mode='bilinear', 
                        align_corners=False
                    )

                x = x + skip_connections[i]
            elif self.skip_method == "cat":
                # NOTE: will increase the number of channels
                x = torch.cat((x, skip_connections[i]), dim=1)

            print(f"Concatenated: {x.shape}")
            x = dec(x) # refinement'
            print(f"Refined: {x.shape}")

        return x


class GFSUnbiaser(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 samples, levels, height, width,
                  feature_dims=[64, 128, 256, 512],
                   use_se=True, r=8, double=False):
        super().__init__()

        print("Instantiating Encoder...")
        self.encoder = Encoder(c_in=in_channels, samples=samples,
                                levels=levels, height=height, width=width,
                                  use_se=use_se, r=r, feature_dims=feature_dims, double=double)
        
        print("Encoder Summary:")
        summary(self.encoder, (levels, in_channels, height, width))
        
        print("Instantiating Decoder...")
        self.decoder = Decoder(feature_dims=feature_dims,
                                samples=samples, levels=levels,
                                  height=height, width=width, r=r,
                                    use_se=use_se, double=double, skip_method="add")
        
        print("Decoder Summary:")
        # NOTE: all skip connections are passed to decoder. Need tensor for every layer
        # Should be ordered from least to most channels
        bottleneck_input = torch.randn(levels, feature_dims[-1], height//2**len(feature_dims), width//2**len(feature_dims))
        test_skip_connections = [torch.randn(levels, feature_dims[0], height//2**0, width//2**0), 
                                  torch.randn(levels, feature_dims[1], height//2**1, width//2**1),
                                  torch.randn(levels, feature_dims[2], height//2**2, width//2**2),
                                  torch.randn(levels, feature_dims[3], height//2**3, width//2**3)]
        
        summary(self.decoder, input_data=[bottleneck_input, test_skip_connections[::-1]])
        # summary(self.decoder, (feature_dims[0], height//2**len(feature_dims), width//2**len(feature_dims)))
   
        self.final_conv = nn.Conv2d(feature_dims[0], out_channels, kernel_size=1)

        print("Done!")

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, reversed(skip_connections))
        x = self.final_conv(x)

        return x
    

class LightningGFSUnbiaser(pl.LightningModule):
    def __init__(self, in_channels, out_channels, 
                 samples, levels, height, width,
                  feature_dims=[64, 128, 256, 512],
                   use_se=True, r=8, double=False):
        super().__init__()

        self.model = GFSUnbiaser(in_channels, out_channels, 
                                 feature_dims=feature_dims,
                                 samples=samples, levels=levels,
                                   height=height, width=width,
                                   use_se=use_se, r=r, double=double)
        
        self.loss_fxn = Loss(remap=True, exponent=5.0, power=3.0,
                              constant=1.0, channels=out_channels)

    def forward(self, x):
        return self.model(x)
    

    def evaluate(self, batch, stage=None):
        x, y = batch
        pred = self.forward(x)

        loss = self.loss_fxn.forward(pred, y)

        if stage is not None:
            self.log(f"{stage}_loss", loss, prog_bar=True)


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fxn.forward(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

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

