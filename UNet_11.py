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

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配的情况
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


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

        # 输出层 (调整输出尺寸为输入尺寸)
        # self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Linear(64, n_classes)  # 全连接层输出7类

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)  # (32, 64, 11, 11)
        x2 = self.down1(x1)  # (32, 128, 5, 5)
        x3 = self.down2(x2)  # (32, 256, 3, 3)
        x4 = self.down3(x3)  # (32, 512, 1, 1) → 实际会向下取整为1x1

        # 处理尺寸过小的情况
        if x4.size()[2] == 0:
            x4 = F.interpolate(x4, size=(1, 1), mode='bilinear', align_corners=True)

        # Decoder
        x = self.up1(x4, x3)  # (32, 256, 1, 1)
        x = self.up2(x, x2)  # (32, 128, 3, 3)
        x = self.up3(x, x1)  # (32, 64, 7, 7)

        # 输出
        # logits = self.outc(x)  # (32, n_classes, 11, 11)
        # return logits
        x = self.global_pool(x)  # [32, 64, 1, 1]
        x = x.view(x.size(0), -1)  # [32, 64]
        return self.fc(x)  # [32, 7]


# 测试网络
if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建输入数据 (batch_size=32, channels=33, height=11, width=11)
    # x = torch.randn(32, 33, 11, 11).to(device)

    # 初始化模型
    model = UNet(n_channels=6, n_classes=5)

    # # 前向传播
    # with torch.no_grad():
    #     output = model(x)
    #
    # print(f"输入尺寸: {x.shape}")
    # print(f"输出尺寸: {output.shape}")  # 应为 (32, 1, 11,11)