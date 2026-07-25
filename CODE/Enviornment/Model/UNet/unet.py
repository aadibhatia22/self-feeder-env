import torch 
import torch.nn as nn
from unet_parts import DoubleConv, DownSample, UpSample
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

        self.checkpoint_dir = checkpoint_dir
        self.learning_rate = learning_rate
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 'UNet')


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
        checkpoint_file = os.path.join(checkpoint_dir, self.name+'_td3')
        torch.save(self.state_dict(), checkpoint_file)

    #this helps us go back to a checkpoint
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))

    #alpha supresses easy pred
    #beta reduces punishment near the correct center
    def heat_map_loss(self, prediction, ground_truth, N, alpha = 2.0, beta = 4.0):
        #scale the predicition to be between 0-1
        prediction = torch.sigmoid(prediction)
        #keep logarithms away from log(0)
        prediction = torch.clamp(prediction, min=1e-4, max=1.0-1e-4)

        #ground truth heatmaps from the environment begin as NumPy arrays
        ground_truth = torch.as_tensor(
            ground_truth,
            device=prediction.device,
            dtype=prediction.dtype
        )

        loss = prediction.new_tensor(0.0)
        #applying (1 - prediction(x))^a * log(prediction(x)) for ground_truth = 1
        # else (1-ground_truth(x))^b * prediction ^a log(1-prediction(x))

        for i in range(prediction.shape[0]):
            for j in range(prediction.shape[1]):
                ground_truth_value = ground_truth[i][j]
                prediction_value = prediction[i][j]
                if ground_truth_value == 1:
                    loss -= ((1-prediction_value) ** alpha) * torch.log(prediction_value)
                elif ground_truth_value < 1 and ground_truth_value >= 0:
                    loss -= ((1-ground_truth_value) ** beta) * (prediction_value ** alpha) * torch.log(1-prediction_value)
        
        #computing the 1/N part of the formula
        normalizer = max(float(N), 1.0)
        return 1.0/normalizer * loss
        

if __name__ == "__main__":
    double_conv = DoubleConv
    print(double_conv)

    input_image = torch.rand((1,3,512,512))
    model = UNet(3,10)
    input_image = input_image.to(model.device)
    output = model(input_image)
    print(output.size())
