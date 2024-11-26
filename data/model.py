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

import torch
import torch.nn as nn
import math
import pytorch_lightning as pl


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
        self.pool = nn.AdaptiveAvgPool2d(1)

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


class Conv3DBlock(nn.Module):
    '''
    Convolutional 3D block for spatio-temporal relationships
    Can use SE blocks for channel attention
    May need some form of channel mixing?
    '''

    def __init__(self, c_in, c_out, use_se=True, r=8, double=False):
        super().__init__()


        # Define convolutional layer
        if double:
            self.conv = nn.Sequential(
                nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, stride=1),
                nn.LayerNorm(c_out),
                nn.GELU(),
                nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, stride=1),
                nn.LayerNorm(c_out),
                nn.GELU()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, stride=1),
                nn.LayerNorm(c_out),
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

    def __init__(self, c_in, use_se=True, r=8, feature_dims=[64, 128, 256, 512]):
        super().__init__()

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.downsampling = nn.ModuleList()

        c = c_in

        for f in feature_dims:
            # Decrease in spatial resolution, increase in channels
            self.downsampling.append(
                Conv3DBlock(c, f, use_se=use_se, r=r, double=True)
            )

            c = f


    def forward(self, x):
        skip_connections = [] # will be returned to use in decoder

        for downsampling in self.downsampling:
            x = downsampling(x)
            skip_connections.append(x)
            x = self.pooling(x)

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

    def __init__(self, feature_dims=[64, 128, 256, 512], use_se=True, r=8, double=True):
        super().__init__()

        self.upsampling = nn.ModuleList()
        self.dec = nn.ModuleList()

        # Assumes that feature_dims will be in the same order as Encoder
        # Starts with input from bottleneck, which is 2*largest feature dims
        for f in reversed(feature_dims):
            # Increase in spatial resolution, decrease in channels
            self.upsampling.append(
                nn.ConvTranspose3d(f*2, f, kernel_size=2, stride=2)
            )

            # Refine upsampling
            self.dec.append(
                Conv3DBlock(f*2, f, use_se=use_se, r=r, double=double)
            )


    def forward(self, x, skip_connections):
        for i, (upsampling, dec) in enumerate(zip(self.upsampling, self.dec)):
            x = upsampling(x) # transposed conv
            x = torch.cat((x, skip_connections[i]), dim=1) # skip connection
            x = dec(x) # refinement

        return x


class GFSUnbiaser(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 feature_dims=[64, 128, 256, 512], use_se=True, r=8):
        super().__init__()

        self.encoder = Encoder(in_channels, use_se=use_se, r=r, feature_dims=feature_dims)
        self.decoder = Decoder(feature_dims=feature_dims)
        self.final_conv = nn.Conv3d(feature_dims[0], out_channels, kernel_size=1)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        x = self.final_conv(x)

        return x
    

class LightningGFSUnbiaser(pl.LightningModule):
    def __init__(self, in_channels, out_channels, 
                 feature_dims=[64, 128, 256, 512], use_se=True, r=8):
        super().__init__()

        self.model = GFSUnbiaser(in_channels, out_channels, 
                                 feature_dims=feature_dims,
                                   use_se=use_se, r=r)

    def forward(self, x):
        return self.model(x)

    # def evaluate(self, batch, stage=None):
    #     x, y = batch
    #     logits = self.forward(x)

    #     loss = F.binary_cross_entropy_with_logits(logits, y)
    #     acc = accuracy(torch.sigmoid(logits), y.long())
    #     f1 = f1_score(torch.sigmoid(logits), y.long())

    #     if stage is not None:
    #         self.log(f"{stage}_loss", loss, prog_bar=True)
    #         self.log(f"{stage}_acc", acc, prog_bar=True)
    #         self.log(f"{stage}_f1", f1, prog_bar=True)

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self.forward(x)
    #     loss = F.binary_cross_entropy_with_logits(logits, y)
    #     self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     self.evaluate(batch, "val")

    # def test_step(self, batch, batch_idx):
    #     self.evaluate(batch, "test")

    # def predict_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self.forward(x)
    #     return logits

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=1e-3)