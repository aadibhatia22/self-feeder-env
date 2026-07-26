import torch 
import torch.nn as nn
import torch.nn.functional as F
from Model.UNet.unet_parts import DoubleConv, DownSample, UpSample
import os
import torch.optim as optim
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, checkpoint_dir = "tmp/UNet", learning_rate=10e-3):
        super().__init__()

        """UNET ALGORITHIM"""

        self.down_conv1 = DownSample(in_channels, 64)
        self.down_conv2 = DownSample(64,128)
        self.down_conv3 = DownSample(128,256)
        self.down_conv4 = DownSample(256,512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_conv1 = UpSample(1024,512)
        self.up_conv2 = UpSample(512,256)
        self.up_conv3 = UpSample(256,128)
        self.up_conv4 = UpSample(128,64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        # Start heatmap probabilities near 0.1 so background pixels do not
        # overwhelm the first optimizer updates.
        nn.init.constant_(self.out.bias, -2.19)

        self.checkpoint_dir = checkpoint_dir
        self.learning_rate = learning_rate
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir,
            "latest_UNet.pt",
        )


        #to use apple GPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        #checking for GPU
        print(f"Created UNet on: {self.device}")

        #moves the model to the GPU
        self.to(self.device)

        #using AdamW in one place helps the model explore
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)

    def forward(self, x):
        down1, p1 = self.down_conv1(x)
        down2, p2 = self.down_conv2(p1)
        down3, p3 = self.down_conv3(p2)
        down4, p4 = self.down_conv4(p3)

        b = self.bottle_neck(p4)

        up1 = self.up_conv1(b, down4)
        up2 = self.up_conv2(up1, down3)
        up3 = self.up_conv3(up2, down2)
        up4 = self.up_conv4(up3, down1)

        out = self.out(up4)
        return out
    def save_checkpoint(self, checkpoint_dir=None):
        checkpoint_dir = checkpoint_dir or self.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(
            checkpoint_dir,
            "latest_UNet.pt",
        )
        torch.save(self.state_dict(), self.checkpoint_file)

    #this helps us go back to a checkpoint
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        return self

    #alpha supresses easy pred
    #beta reduces punishment near the correct center
    def heat_map_loss(self, prediction, ground_truth, N, alpha = 2.0, beta = 4.0):
        #scale the predicition to be between 0-1
        prediction_probability = torch.sigmoid(prediction)

        #ground truth heatmaps from the environment begin as NumPy arrays
        ground_truth = torch.as_tensor(
            ground_truth,
            device=prediction.device,
            dtype=prediction.dtype
        )

        #applying (1 - prediction(x))^a * log(prediction(x)) for ground_truth = 1
        # else (1-ground_truth(x))^b * prediction ^a log(1-prediction(x))
        positive_pixels = ground_truth == 1
        negative_pixels = (ground_truth < 1) & (ground_truth >= 0)

        positive_loss = (
            ((1 - prediction_probability) ** alpha)
            * F.logsigmoid(prediction)
        )
        negative_loss = (
            ((1 - ground_truth) ** beta)
            * (prediction_probability ** alpha)
            * F.logsigmoid(-prediction)
        )

        loss = -(
            torch.where(
                positive_pixels,
                positive_loss,
                torch.zeros_like(positive_loss),
            ).sum()
            + torch.where(
                negative_pixels,
                negative_loss,
                torch.zeros_like(negative_loss),
            ).sum()
        )
        
        #computing the 1/N part of the formula
        normalizer = max(float(N), 1.0)
        return 1.0/normalizer * loss

    def train_step(self, prediction, ground_truth, N, alpha = None, beta = None):
        # 1. Clear gradients
        self.optimizer.zero_grad()

        #prediction comes from UNet as (1, 1, H, W)
        #remove the batch and channel dimensions to match ground_truth (H, W)
        prediction = prediction.squeeze(0).squeeze(0)
        
        # 2. Compute loss
        loss = -1
        if alpha is not None and beta is not None:
            loss = self.heat_map_loss(prediction= prediction, ground_truth= ground_truth, N = N, alpha= alpha, beta= beta)
        else:
            loss = self.heat_map_loss(prediction= prediction, ground_truth= ground_truth, N = N)
        
        # 3. Backpropagate (This attaches gradients to the model parameters)
        loss.backward()
        
        # 4. Update the weights
        self.optimizer.step()
        
        return loss.item() # Returns the raw number for logging
        

if __name__ == "__main__":
    double_conv = DoubleConv
    print(double_conv)

    input_image = torch.rand((1,3,512,512))
    model = UNet(3,10)
    input_image = input_image.to(model.device)
    output = model(input_image)
    print(output.size())
