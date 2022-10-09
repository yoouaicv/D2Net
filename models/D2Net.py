# Copyright (c) 2022 Yoouaicv.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return torch.Tensor.permute(x, self.dims)


class Mlp(nn.Module):
    def __init__(self, dim):
        super(Mlp, self).__init__()
        self.mlp = nn.Sequential(
            LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    def forward(self, input):
        x = self.mlp(input)
        return x

class DDConv(nn.Module):
    def __init__(self, dim):
        super(DDConv, self).__init__()
        self.norm = LayerNorm(dim, eps=1e-6)
        self.act = nn.GELU()
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.dwconv_3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1, groups=dim)
        self.dwconv_5 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim)
        self.dwconv_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3, groups=dim)
        self.pwconv = nn.Linear(dim, dim)
    def forward(self, x):
        x = self.norm(x)
        x = x.permute([0, 3, 1, 2])
        # x = self.dwconv(x)
        # x = torch.cat([self.dwconv_7(x), self.dwconv_5(x), self.dwconv_3(x)], dim=1)
        x = self.dwconv_7(x) + self.dwconv_5(x) + self.dwconv_3(x)
        x = self.act(x)
        x = x.permute([0, 2, 3, 1])
        x = self.pwconv(x)
        return x
class D2Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super(D2Block, self).__init__()
        self.attention = DDConv(dim)
        self.mlp = Mlp(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x.permute([0, 2, 3, 1])
        x = x + self.attention(x)
        x = x + self.drop_path(self.mlp(x))
        x = x.permute([0, 3, 1, 2])
        return x

class D2Net(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[24, 48, 96, 192], drop_path_rate=0.0,
                 layer_scale_init_value=1e-6, head_init_scale=1.0
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, padding=0),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[D2Block(dim=dims[i], drop_path=dp_rates[cur + j],
                         layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.avg(x).view(x.size(0), -1)
        x = self.norm(x)
        return x  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# class LayerNorm2d(nn.LayerNorm):
#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1)
#         x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, eps=1e-6)
#         x = x.permute(0, 3, 1, 2)
#         return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


@register_model
def D2Net_xxs(pretrained=False, **kwargs):
    model = D2Net(depths=[2, 2, 6, 2], dims=[24, 48, 96, 192], **kwargs)
    # bs=1024 input_size=256 color_jitter=0.4 hf=0.5 droppath=0.0 epochs=300 lr=0.006 wd=0.05
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url='../log/D2Net_xxs/D2Net_xxs.pth', map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def D2Net_xs(pretrained=False, **kwargs):
    model = D2Net(depths=[2, 2, 6, 2], dims=[32, 64, 128, 256], **kwargs)
    return model

@register_model
def D2Net_s(pretrained=False, **kwargs):
    model = D2Net(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

if __name__ == '__main__':
    from torchsummary import summary
    net = D2Net_xxs()
    # net.cuda()
    summary(net, (3, 224, 224), device='cpu', batch_size=1)