"""
Phase 1 GraphUNet — scratch-trained UNet with Conv2d bottleneck.

Architecture (from Aryan's original codebase):
  Encoder : 4 DoubleConv blocks at 64 → 128 → 256 → 512 channels
             MaxPool2d downsampling between each block
  Bottleneck:
      DoubleConv(512, 1024)
      Conv2d(1024, 1024, 3, padding=1) → BatchNorm2d → ReLU  ← the "graph_conv"
  Decoder : 4 ConvTranspose2d upsamplings with skip connections + DoubleConv
  Head    : Conv2d(64, out_channels, 1)

This is reproduced from Phase 1 to run on crack_seg_clean for a fair baseline.
No pretrained weights; all parameters trained from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class GraphUNet(nn.Module):
    """
    Phase 1 architecture: UNet + Conv2d bottleneck (the "graph_conv" block).

    The Conv2d at the bottleneck operates on a spatially compressed feature map
    (32x downsampled), approximating graph-like reasoning over local feature
    regions before decoding. Not a true GNN — the name reflects the intent.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        features: list = None,
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.downs = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck: DoubleConv → graph_conv
        bn_ch = features[-1] * 2
        self.bottleneck = nn.Sequential(
            DoubleConv(features[-1], bn_ch),
            nn.Conv2d(bn_ch, bn_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(bn_ch),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up_convs  = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for f in reversed(features):
            self.up_convs.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.up_blocks.append(DoubleConv(f * 2, f))

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for i, (up, block) in enumerate(zip(self.up_convs, self.up_blocks)):
            x = up(x)
            skip = skips[-(i + 1)]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = block(x)

        return self.head(x)

    def encoder_params(self):
        return []

    def decoder_params(self):
        return list(self.parameters())
