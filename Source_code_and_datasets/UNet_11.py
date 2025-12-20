import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2): # For up1, the input x1 and x2 sizes should be [512,1,1] and [256,2,2]
        x1 = self.up(x1) # [32，256, 2, 2]
        # #Dealing with size mismatches
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2]) # [32，256, 2, 2] 
        x = torch.cat([x2, x1], dim=1) # [32，512, 2, 2]
        return self.conv(x)  # [32，256, 2, 2]


class UNet(nn.Module):
    def __init__(self, n_channels=6, n_classes=5):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # Decoder
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)

        #Output layer 
        # self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.fc = nn.Linear(64, n_classes)  # Fully connected layer

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)  # (32, 64, 11, 11)
        x2 = self.down1(x1)  # (32, 128, 5, 5)
        x3 = self.down2(x2)  # (32, 256, 2, 2)
        x4 = self.down3(x3)  # (32, 512, 1, 1)

        # Dealing with situations where the size is too small
        if x4.size()[2] == 0:
            x4 = F.interpolate(x4, size=(1, 1), mode='bilinear', align_corners=True)

        # Decoder
        x = self.up1(x4, x3)  # (32, 256, 2, 2)
        x = self.up2(x, x2)  # (32, 128, 5, 5)
        x = self.up3(x, x1)  # (32, 64, 11, 11)

        # output
        # logits = self.outc(x)  # (32, n_classes, 1, 1)
        # return logits
        x = self.global_pool(x)  # [32, 64, 1, 1]
        x = x.view(x.size(0), -1)  # [32, 64]
        return self.fc(x)  # [32, 5]


# test network
if __name__ == "__main__":


    # Initialize the model, n_channels can be 3,6,9,18
    model = UNet(n_channels=18, n_classes=5)


