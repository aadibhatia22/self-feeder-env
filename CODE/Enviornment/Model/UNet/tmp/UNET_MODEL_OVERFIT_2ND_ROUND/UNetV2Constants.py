from Model.model_constants import Model_Constants


class UNetV2Constants(Model_Constants):
    """Model constants for the second U-Net fine-tuning stage."""

    def __init__(self):
        super().__init__()
        self.learning_rate = 2e-5
