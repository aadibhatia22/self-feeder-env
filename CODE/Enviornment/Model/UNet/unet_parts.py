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
            nn.ReLU(inplace=True)

            """without padding the dimensions would shrink ex form 284^2 it 282^2 since Kernels take 1 pixel off from each side"""
        )

        def forward(self, input):
            return self.conv_op(input)


