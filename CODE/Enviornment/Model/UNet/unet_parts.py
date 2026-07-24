import torch
import torch.nn as nn
class DoubleConv(nn.Module):
    #input channels are the number of values describing each input pixel
    #Ex. RGB = 3 channels Red, Green, Blue
    #Grayscale = 1 chanell
    #out channel is the sane logic
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            #we need padding so that we can have kernels on the corner
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),
            #number of chanells as paramters
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            """without padding the dimensions would shrink ex form 284^2 it 282^2 since Kernels take 1 pixel off from each side"""
        )

        def forward(self, input):
            return self.conv_op(input)

"""Creating the DownSample Class"""   
#Downsampling takes channels from previous layers and the reduces its dimensions
"""
Ex. If you downscale for a 2x2 in each 2x2 it downscales it takes the max.
Ex.
[2 3
0 5] -> 5
It allows for the model to learn more complex features
"""
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        #halves the height and width
        self.pool = nn.MaxPool2d(kernel_size = 2, stride =2)

    def forward(self, input):
        down = self.conv(input)
        p = self.pool(down)
        return down, p 