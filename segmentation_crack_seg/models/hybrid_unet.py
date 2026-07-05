import torch
import torch.nn as nn
from models.unet import DoubleConv


class GraphUNet(nn.Module):
    """Hybrid UNet: UNet encoder/decoder with graph-like conv at the bottleneck."""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.name = "GraphUNet"
        self.downs = nn.ModuleList([
            DoubleConv(in_channels, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
        ])
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(512, 1024)
        self.graph_conv = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, 2, 2), DoubleConv(1024, 512),
            nn.ConvTranspose2d(512, 256, 2, 2), DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, 2, 2), DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, 2, 2), DoubleConv(128, 64),
        ])
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        x = self.graph_conv(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            x = torch.cat([skips[i // 2], x], dim=1)
            x = self.ups[i + 1](x)
        return self.out(x)
