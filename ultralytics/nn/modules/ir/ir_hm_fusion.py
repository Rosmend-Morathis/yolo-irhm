import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .transformer_fusion import Transformer_Fusion_M
from .position_encoding import PositionEmbeddingSine


BN_MOMENTUM = 0.1

# class BaseBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, dilation=1):
#         super(BaseBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
#         self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
#         self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
#         self.stride = stride

#     def forward(self, x, residual=None):
#         if residual is None:
#             residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         out += residual
#         out = self.relu(out)

#         return out


class InfraredHeatmapFusion(nn.Module):
    def __init__(self, c1, c2=16):
        super(InfraredHeatmapFusion, self).__init__()
        self.channel = c2
        self.base_layer = nn.Sequential(
            nn.Conv2d(1, c2, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(c2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.pre_img_layer = nn.Sequential(
            nn.Conv2d(1, c2, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(c2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.pre_hm_layer = nn.Sequential(
            nn.Conv2d(1, c2, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(c2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.fusion = Transformer_Fusion_M(d_model=16, nhead=4, num_fusion_encoder_layers=1, dim_feedforward=64)
        self.pos_encoding = PositionEmbeddingSine(num_pos_feats=16//2, sine_type='lin_sine', avoid_aliazing=True, max_spatial_resolution=60)
        self.patchembed = nn.Conv2d(16, 16, kernel_size=16, stride=16)


    def forward(self, x):
        # print(x.shape)
        img, pre_img, pre_hm = torch.split(x, 1, dim=1)
        # print(img.shape, pre_img.shape, pre_hm.shape)
        img = self.base_layer(img)

        pre_img = self.pre_img_layer(pre_img)
        pre_hm = self.pre_hm_layer(pre_hm)

        img_p = self.patchembed(img)
        pre_img_p = self.patchembed(pre_img)
        pre_hm_p = self.patchembed(pre_hm)

        img_token = self.feature2token(img_p)
        pre_img_token = self.feature2token(pre_img_p)
        pre_hm_token = self.feature2token(pre_hm_p)
        pos_embed = self.get_positional_encoding(img_p)
        pos_embed_token = self.feature2token(pos_embed)

        x_f = self.fusion(img_token, pre_img_token, pre_hm_token, pos_embed_token)
        [B,C,H,W] = img.shape
        [b,c,h,w] = img_p.shape
        x_p = self.token2feature(x_f, h, w)
        out = F.interpolate(x_p, size=(H, W), mode='bilinear', align_corners=True) + img
        return out

    def feature2token(self, x):
        B,C,W,H = x.shape
        L = W*H
        token = x.view(B,C,L).permute(2, 0, 1).contiguous()
        return token


    def token2feature(self, tokens, h, w):
        L,B,D = tokens.shape
        H,W = h,w
        x = tokens.permute(1, 2, 0).view(B,D,H,W).contiguous()
        return x
    

    def get_positional_encoding(self, feat):
        b, _, h, w = feat.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
        pos = self.pos_encoding(mask)
        return pos.reshape(b, -1, h, w)
    