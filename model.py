import torch
import torch.nn as nn
import torch.nn.functional as F


# Define U-Net building blocks

# Double convolution block
class DoubleConv(nn.Module):
    """(Conv => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

# Downsampling block (Encoder Block)
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Upsampling block (Decoder Block)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW (CHW means (Channels, Height, Width) which in our case is (Channel, Frequency, Time))
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Output convolution block
class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        return self.conv(x) # no maxpool here because it's the output layer

# Define U-Net constructors
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_c=64, bilinear=False):
        super(UNet, self).__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c*2)
        self.down2 = Down(base_c*2, base_c*4)
        self.down3 = Down(base_c*4, base_c*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c*8, base_c*16 // factor)
        self.up1 = Up(base_c*16, base_c*8 // factor, bilinear)
        self.up2 = Up(base_c*8, base_c*4 // factor, bilinear)
        self.up3 = Up(base_c*4, base_c*2 // factor, bilinear)
        self.up4 = Up(base_c*2, base_c, bilinear)
        self.outc = OutConv(base_c, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits