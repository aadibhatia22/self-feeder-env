import numpy as np


class Model_Constants:
    """Public configuration values used by Enviornment_Randomizer."""

    def __init__(self):
        self.output_x_dim = 256
        self.output_y_dim = 144
        self.max_reward = 1
        self.sigma = 2