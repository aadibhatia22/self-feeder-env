import numpy as np


class Model_Constants:
    """Public configuration values used by Enviornment_Randomizer."""

    def __init__(self):
        self.input_x_dim = 256
        self.input_y_dim = 144
        self.output_x_dim = 256
        self.output_y_dim = 144
        self.max_reward = 1
        self.sigma = 2
        self.learning_rate=10e-3