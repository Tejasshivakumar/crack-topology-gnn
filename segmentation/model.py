import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_layers.torch_vertex import Grapher
from graph_layers.vig import FFN


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EnhancedGraphUNet(nn.Module):
    """
    Hybrid CNN-GNN segmentation model for road crack detection.

    Architecture:
        Encoder  : 4x (DoubleConv -> MaxPool2d)   [32 -> 64 -> 128 -> 256 channels]
        Bottleneck:
            Grapher(k=9, dilation=1, conv='edge')
            -> FFN(x4 expansion)
            -> Grapher(k=9, dilation=2, conv='edge')   # multi-scale graph receptive field
            -> FFN(x4 expansion)
            -> Dropout2d(0.1)
        Decoder  : 4x (ConvTranspose2d + skip-concat -> DoubleConv)
        Head     : 1x1 Conv -> num_classes

    The dual Grapher+FFN bottleneck captures multi-scale crack topology.
    dilation=2 in the second Grapher enlarges the graph receptive field.
    Lighter feature widths [32,64,128,256] prevent overfitting on crack datasets.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        features: tuple = (32, 64, 128, 256),
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = nn.Sequential(
            Grapher(ch, kernel_size=9, dilation=1, conv='edge',
                    act='relu', bias=True, drop_path=drop_path),
            FFN(ch, ch * 4, act='relu', drop_path=drop_path),
            Grapher(ch, kernel_size=9, dilation=2, conv='edge',
                    act='relu', bias=True, drop_path=drop_path),
            FFN(ch, ch * 4, act='relu', drop_path=drop_path),
        )
        self.bottleneck_drop = nn.Dropout2d(dropout)

        rev = list(reversed(features))
        for i in range(len(rev) - 1):
            self.ups.append(nn.ConvTranspose2d(rev[i], rev[i + 1], 2, 2))
            self.ups.append(DoubleConv(rev[i + 1] + rev[i], rev[i + 1]))
        self.ups.append(nn.ConvTranspose2d(rev[-1], rev[-1], 2, 2))
        self.ups.append(DoubleConv(rev[-1] * 2, rev[-1]))

        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.bottleneck_drop(x)

        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((x, skip), dim=1)
            x = self.ups[i + 1](x)

        return self.final_conv(x)
