import timm
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


class HybridGraphUNet(nn.Module):
    """
    Pretrained CNN encoder (timm) + GNN bottleneck + UNet decoder.

    Architecture:
        Encoder  : timm pretrained model (default: resnet34d, ImageNet weights)
                   Returns 5 multi-scale feature maps at strides 2,4,8,16,32.
        Bottleneck:
            Grapher(k=9, dilation=1) -> FFN
            -> Grapher(k=9, dilation=2) -> FFN
            -> Dropout2d(0.1)
        Decoder  : 4x UNet blocks with skip connections + 1 final upsample to input res

    Why this beats the scratch-trained encoder:
        - ImageNet features already encode edges, textures, and curves.
        - With only 374 training samples the scratch encoder underfits badly.
        - The GNN bottleneck still captures crack topology; the pretrained encoder
          handles all low/mid-level feature extraction.
    """

    def __init__(
        self,
        encoder_name: str = 'resnet34d',
        pretrained: bool = True,
        out_channels: int = 2,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)
        enc_chs = [f['num_chs'] for f in self.encoder.feature_info]
        # resnet34d: [64, 64, 128, 256, 512]

        bn_ch = enc_chs[-1]

        self.bottleneck = nn.Sequential(
            Grapher(bn_ch, kernel_size=9, dilation=1, conv='edge',
                    act='relu', bias=True, drop_path=drop_path),
            FFN(bn_ch, bn_ch * 4, act='relu', drop_path=drop_path),
            Grapher(bn_ch, kernel_size=9, dilation=2, conv='edge',
                    act='relu', bias=True, drop_path=drop_path),
            FFN(bn_ch, bn_ch * 4, act='relu', drop_path=drop_path),
        )
        self.bottleneck_drop = nn.Dropout2d(dropout)

        # Build decoder dynamically from encoder channels
        rev_enc_chs = list(reversed(enc_chs[:-1]))   # e.g. [256, 128, 64, 64]
        self.ups   = nn.ModuleList()
        self.convs = nn.ModuleList()

        ch = bn_ch
        for skip_ch in rev_enc_chs:
            out_ch = max(ch // 2, 32)
            self.ups.append(nn.ConvTranspose2d(ch, ch // 2, 2, 2))
            self.convs.append(DoubleConv(ch // 2 + skip_ch, out_ch))
            ch = out_ch

        # Final upsample to original input resolution (no skip at this level)
        self.final_up    = nn.ConvTranspose2d(ch, ch // 2, 2, 2)
        self.final_block = DoubleConv(ch // 2, 16)
        self.head        = nn.Conv2d(16, out_channels, 1)

    def encoder_params(self):
        return list(self.encoder.parameters())

    def decoder_params(self):
        return [p for name, p in self.named_parameters() if not name.startswith('encoder')]

    def forward(self, x):
        features = self.encoder(x)
        x        = self.bottleneck(features[-1])
        x        = self.bottleneck_drop(x)
        skips    = list(reversed(features[:-1]))

        for up, conv, skip in zip(self.ups, self.convs, skips):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:])
            x = conv(torch.cat([x, skip], dim=1))

        x = self.final_up(x)
        x = self.final_block(x)
        return self.head(x)


class EnhancedGraphUNet(nn.Module):
    """Original scratch-trained model. Kept for backward compatibility."""

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
