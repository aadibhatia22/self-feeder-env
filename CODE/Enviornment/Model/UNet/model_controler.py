import os
import torch as T
import torch.nn.functional as F
import numpy as np
from unet import UNet

class model_controler:
    def __init__(self, in_channels, num_classes, checkpoint_dir = "tmp/UNet", learning_rate=10e-3)