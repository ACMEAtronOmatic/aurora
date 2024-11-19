'''
UNet for unbiasing GFS ICs to be used as inputs to Aurora, which is trained on ERA5 data

Loss: 
- MSE between unbiased GFS ICs and actual ERA5 data
- SSIM between unbiased GFS ICs and actual ERA5 data
- MAE between unbiased GFS ICs and actual ERA5 data
- Should MSE and MAE be pixel weighted?

Encoder:
Convolutions with progressive downsampling, space to depth
GroupNorm, LeakReLu

Decoder:
Transposed convolutions to mirror encoder, depth to space
Has skip connections from analagous encoder layer
'''