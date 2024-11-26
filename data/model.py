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

    def __init__(self, c_in, c_out, use_se=True, r=8):
        super().__init__()


        # Define convolutional layer
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
    pass

class Decoder(nn.Module):
    pass

class GFSUnbiaser(nn.Module):
    pass

