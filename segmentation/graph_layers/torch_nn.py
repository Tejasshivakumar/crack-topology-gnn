import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm, nc):
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


class MLP(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
        super(MLP, self).__init__(*m)


class BasicConv(nn.Module):
    def __init__(self, channel_sizes, act='relu', norm=None, bias=True, groups=1):
        super(BasicConv, self).__init__()
        layers = []
        for i in range(len(channel_sizes) - 1):
            layers.append(nn.Conv2d(channel_sizes[i], channel_sizes[i+1], kernel_size=1, bias=bias, groups=groups))
            if norm:
                layers.append(nn.BatchNorm2d(channel_sizes[i+1]))
            if act == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif act == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.01, inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


def batched_index_select(x, idx):
    B, C, N, _ = x.shape
    _, _, K = idx.shape
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)
    x = x.transpose(2, 1)
    x = x.reshape(B * N, C)
    selected = x[idx]
    selected = selected.view(B, N, K, C).permute(0, 3, 1, 2)
    return selected
