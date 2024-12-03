源码理解
DLA 类的 __init__ 函数解释
参数说明
- levels: 一个列表，表示每个层级的层数。例如 [1, 1, 1, 2, 2, 1] 表示有 6 个层级，每个层级的层数分别为 1, 1, 1, 2, 2, 1。
- channels: 一个列表，表示每个层级的通道数。例如 [16, 32, 64, 128, 256, 512] 表示有 6 个层级，每个层级的通道数分别为 16, 32, 64, 128, 256, 512。
- num_classes: 整数，表示分类任务的类别数，默认为 1000。
- block: 模块类型，默认为 BasicBlock。可以是 BasicBlock, Bottleneck, 或 BottleneckX。
- residual_root: 布尔值，表示是否在根节点使用残差连接，默认为 False。
- linear_root: 布尔值，表示是否在根节点使用线性层，默认为 False。
- opt: 配置选项，默认为 None。

初始化过程
1. 初始化基本属性
self.channels = channels
self.num_classes = num_classes


2. 定义基础层
self.base_layer = nn.Sequential(
    nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
    nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
    nn.ReLU(inplace=True)
)

- nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False): 输入通道数为 3（RGB 图像），输出通道数为 channels[0]，卷积核大小为 7x7，步长为 1，填充为 3，不使用偏置。
- nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM): 批归一化层，动量为 BN_MOMENTUM。
- nn.ReLU(inplace=True): ReLU 激活函数，原地操作。

3. 定义预处理层（可选）
if opt.pre_img:
    self.pre_img_layer = nn.Sequential(
        nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
        nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True)
    )
if opt.pre_hm:
    self.pre_hm_layer = nn.Sequential(
        nn.Conv2d(1, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
        nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True)
    )

- pre_img_layer 和 pre_hm_layer 分别用于处理前一帧的图像和热图，结构与 base_layer 相同。

4. 定义层级
self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root)
self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root)
self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root)

- _make_conv_level 方法用于创建卷积层。
- Tree 类用于创建树状结构的层级，每个层级可以包含多个子层级。

5. 定义融合模块
self.fusion = Transformer_Fusion(d_model=16, nhead=4, num_fusion_encoder_layers=1, dim_feedforward=64)
self.fusion_m = Transformer_Fusion_M(d_model=16, nhead=4, num_fusion_encoder_layers=1, dim_feedforward=64)
self.pos_encoding = PositionEmbeddingSine(num_pos_feats=16//2, sine_type='lin_sine', avoid_aliazing=True, max_spatial_resolution=60)
self.patchembed = nn.Conv2d(16, 16, kernel_size=16, stride=16)

- Transformer_Fusion 和 Transformer_Fusion_M 是自定义的融合模块。
- PositionEmbeddingSine 用于生成位置编码。
- patchembed 是一个卷积层，用于将特征图转换为 patch。

6. 辅助方法
def _make_level(self, block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            nn.MaxPool2d(stride, stride=stride),
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample=downsample))
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)

def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
    modules = []
    for i in range(convs):
        modules.extend([
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1, padding=dilation, bias=False, dilation=dilation),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        ])
        inplanes = planes
    return nn.Sequential(*modules)

- _make_level 方法用于创建带有下采样的层级。
- _make_conv_level 方法用于创建多个卷积层。

7. 辅助方法（特征转换）
def token2feature(self, tokens, h, w):
    L, B, D = tokens.shape
    x = tokens.permute(1, 2, 0).view(B, D, H, W).contiguous()
    return x

def feature2token(self, x):
    B, C, W, H = x.shape
    L = W * H
    tokens = x.view(B, C, L).permute(2, 0, 1).contiguous()
    return tokens

def get_positional_encoding(self, feat):
    b, _, h, w = feat.shape
    mask = torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
    pos = self.pos_encoding(mask)
    return pos.reshape(b, -1, h, w)

- token2feature 将 token 转换为特征图。
- feature2token 将特征图转换为 token。
- get_positional_encoding 生成位置编码。

辅助方法函数解释

1. token2feature(self, tokens, h, w)
这个函数的作用是将 token 转换为特征图。
- 参数:
  - tokens: 形状为 (L, B, D) 的 tensor，其中 L 是 token 数量，B 是 batch 大小，D 是每个 token 的维度。
  - h: 特征图的高度。
  - w: 特征图的宽度。
- 步骤:
  1. 获取 tokens 的形状 (L, B, D)。
  2. 设置 H 和 W 为传入的高度和宽度。
  3. 使用 permute 方法将 tokens 的形状从 (L, B, D) 转换为 (B, D, L)。
  4. 使用 view 方法将形状从 (B, D, L) 转换为 (B, D, H, W)。
  5. 使用 contiguous 方法确保 tensor 在内存中是连续的，以便后续操作。
- 返回:
  - 形状为 (B, D, H, W) 的特征图。
def token2feature(self, tokens, h, w):
    L, B, D = tokens.shape
    H, W = h, w
    x = tokens.permute(1, 2, 0).view(B, D, H, W).contiguous()
    return x

2. feature2token(self, x)
这个函数的作用是将特征图转换为 token。
- 参数:
  - x: 形状为 (B, C, W, H) 的特征图，其中 B 是 batch 大小，C 是通道数，W 和 H 分别是特征图的宽度和高度。
- 步骤:
  1. 获取 x 的形状 (B, C, W, H)。
  2. 计算 L 为 W * H，即 token 的数量。
  3. 使用 view 方法将形状从 (B, C, W, H) 转换为 (B, C, L)。
  4. 使用 permute 方法将形状从 (B, C, L) 转换为 (L, B, C)。
  5. 使用 contiguous 方法确保 tensor 在内存中是连续的，以便后续操作。
- 返回:
  - 形状为 (L, B, C) 的 token。
def feature2token(self, x):
    B, C, W, H = x.shape
    L = W * H
    tokens = x.view(B, C, L).permute(2, 0, 1).contiguous()
    return tokens

3. get_positional_encoding(self, feat)
这个函数的作用是生成位置编码。

- 参数:
  - feat: 形状为 (b, c, h, w) 的特征图，其中 b 是 batch 大小，c 是通道数，h 和 w 分别是特征图的高度和宽度。
- 步骤:
  1. 获取 feat 的形状 (b, c, h, w)。
  2. 创建一个形状为 (b, h, w) 的全零 mask，数据类型为布尔值，设备与 feat 相同。
  3. 使用 self.pos_encoding 生成位置编码，输入为 mask。
  4. 使用 reshape 方法将位置编码的形状从 (b, -1, h, w) 转换为 (b, -1, h, w)。
- 返回:
  - 形状为 (b, -1, h, w) 的位置编码。
def get_positional_encoding(self, feat):
    b, _, h, w = feat.shape
    mask = torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
    pos = self.pos_encoding(mask)
    return pos.reshape(b, -1, h, w)

总结
token2feature 和 feature2token 是互逆的操作，分别用于将 token 转换为特征图和将特征图转换为 token。
get_positional_encoding 用于生成位置编码，通常在 transformer 模型中用于提供位置信息。


DLA 类的 forward 函数解释

这个函数定义了一个模型的前向传播过程。它接受多个输入，并通过一系列操作生成输出。以下是详细的步骤解释：

参数
- x_rgb: RGB 图像的输入特征图。
- x_t: 热红外图像的输入特征图。
- pre_img_rgb: 前一帧的 RGB 图像特征图（可选）。
- pre_img_t: 前一帧的热红外图像特征图（可选）。
- pre_hm: 前一帧的热图（可选）。

步骤
1. 初始化输出列表:
y = []
2. 基础层处理:
  - 对 x_rgb 和 x_t 进行基础层处理。
x_rgb = self.base_layer(x_rgb)
x_t = self.base_layer(x_t)
3. 前一帧特征图处理:
  - 对前一帧的特征图进行预处理。
pre_img_rgb = self.pre_img_layer(pre_img_rgb)
pre_img_t = self.pre_img_layer(pre_img_t)
pre_hm = self.pre_hm_layer(pre_hm)
4. Patch Embedding:
  - 将特征图转换为 patch embedding。
x_rgb_p = self.patchembed(x_rgb)
x_t_p = self.patchembed(x_t)
pre_x_rgb_p = self.patchembed(pre_img_rgb)
pre_x_t_p = self.patchembed(pre_img_t)
pre_hm_p = self.patchembed(pre_hm)
5. 特征图转 token:
  - 将 patch embedding 转换为 token。
x_rgb_token = self.feature2token(x_rgb_p)
x_t_token = self.feature2token(x_t_p)
pre_x_rgb_token = self.feature2token(pre_x_rgb_p)
pre_x_t_token = self.feature2token(pre_x_t_p)
pre_hm_token = self.feature2token(pre_hm_p)
6. 位置编码:
  - 获取位置编码并将其转换为 token。
pos_embed = self.get_positional_encoding(x_rgb_p)
pos_embed_token = self.feature2token(pos_embed)
7. 融合操作:
  - 使用 fusion_m 方法对当前帧和前一帧的 token 进行融合。
x_rgb_all = self.fusion_m(x_rgb_token, pre_x_rgb_token, pre_hm_token, pos_embed_token)
x_t_all = self.fusion_m(x_t_token, pre_x_t_token, pre_hm_token, pos_embed_token)
8. 最终融合:
  - 使用 fusion 方法对融合后的 token 进行最终融合。
[B, C, H, W] = x_rgb.shape
result = self.fusion(x_rgb_all, x_t_all, pos_embed_token)
9. Token 转特征图:
  - 将融合结果转换回特征图。
[b, c, h, w] = x_rgb_p.shape
x_p = self.token2feature(result, h, w)
10. 上采样和残差连接:
  - 对特征图进行上采样，并与原始输入特征图进行残差连接。
x = F.interpolate(x_p, size=(H, W), mode='bilinear', align_corners=True) + x_rgb + x_t
11. 多尺度特征提取:
  - 通过多个层级的特征提取模块，生成多尺度特征图。
for i in range(6):
    x = getattr(self, 'level{}'.format(i))(x)
    y.append(x)
12. 返回结果:
  - 返回多尺度特征图列表。
return y

总结
- 输入: RGB 图像特征图、热红外图像特征图、前一帧的 RGB 和热红外图像特征图、前一帧的热图。
- 处理: 通过基础层、前一帧特征图处理、patch embedding、特征图转 token、位置编码、融合操作、最终融合、Token 转特征图、上采样和残差连接、多尺度特征提取等步骤。
- 输出: 多尺度特征图列表。

维度变换

For prime image (if set C = 16)
  initial shape: [B, 3, H, W] === base_layer ===> [B, 16, H, W]
  [B, 16, H, W] === patchembed ===> [B, 16, H//16, W//16]
  [B, 16, H//16, W//16] === feature2token ===> [(H//16)*(W//16), B, 16]
  ( 
  [B, 16, H//16, W//16] === get_positional_encoding ===> [B, 16, H//16, W//16]
  [B, 16, H//16, W//16] === feature2token ===> [(H//16)*(W//16), B, 16]
   )
  [(H//16)*(W//16), B, 16] === fusion_m ===>  [(H//16)*(W//16), B, 16]
  [(H//16)*(W//16), B, 16] === token2feature ===> [B, 16, H//16, W//16]
  [B, 16, H//16, W//16] === F.interpolate ===> [B, 16, H, W]

For the previous image, same as primes

For the heatmap  (if set C = 16)
initial shape: [B, 1, H, W] === base_layer ===> [B, 16, H, W]
others same as primes

函数梳理
def feature2token(self, x)
    B, C, W, H = x.shape    # batch size, channel, height, width
    L = W * H    # patches
    tokens = x.view(B, C, L).permute(2, 0, 1).contiguous()
        # view : reshape x[B,C,W,H] as x[B,C,L], 
        #        which combine height and width into one dimension.
        # permute: rearrange x[B,C,L] as X[L,B,C]
        # contiguous: make sure Tensor is contiguous in memory
        
def token2feature(self,tokens,h,w):
    L,B,D=tokens.shape    # patches, batch size, feature dimention(Channel)
    H,W=h,w
    x = tokens.permute(1, 2, 0).view(B, D, H, W).contiguous()
        # permute: rearrange tokens[L,B,D] as tokens[B,D,L]
        # view: reshape last dimension into 2 dim{Height, Width}
        # contiguous: make sure Tensor is contiguous in memory
    return x
fusion_m 梳理
fusion_m call:
  d_model = 16
  nhead = 4
  num_fusion_encoder_layers  1
  dim_feedforward = 64
call fusion_m(img, pre, hm, pos_embed)

---
Transformer_Fusion_M
  d_model = 16
  nhead = 4
  encoder_layer = TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=64, dropout=0.1, activation=”relu”, normalize_before=False)
  encoder = TransformerEncoder(encoder_layer, num_fusion_encoder_layers =1, encoder_norm = None)

---
Transformer_Fusion_M. Forward()           call encoder(img, pre, hm, pos_embed)
TransformerEncoder
  layers = _get_clones(encoder_layer, num_layers = 1) = nn.ModuleList(TransformerEncoderLayer)
  output = layers
TransformerEncoderLayer
  d_model = 16
  nhead = 4
  dim_feedforward = 64
  dropout=0.1
  activation=”relu”
  
  multihead_attn = nn.MultiheadAttention(16, 4, dropout=0.1)
  linear1 = nn.Linear(16, 64)
  linear2 = nn.Linear(64, 16)
  dropout = dropout1 = dropout2 = nn.Dropout(0.1)
  norm1 = norm2 = nn.LayerNorm(16)
  activation = F.relu

as call encoder(img, pre, hm, pos_embed)
  src = img
  pre_src = pre
  pre_hm = hm
  pos = pos_embed

get_positional_encoding 梳理
mask   torch.zeros(( b, h, w))        # all-zero

---
PositionEmbeddingSine
  num_pos_feats = 8
  sine_type = ‘lin_sine’
  avoid_aliazing = True
  max_spatial_resolution = 60
  temperature = 10000
  sine = NerfPositionalEncoding(depth = 4, sine_type = ‘lin_sine’, avoid_aliazing = True, max_spatial_resolution = 60)
  
  ~mask     # all-one
  ……
  pos   torch.stack([x_embed, y_embed], dim=-1)     dimension: [b, h, w, 2]
  out = self.sine(pos).permute(0, 3, 1, 2)

---
call sine(pos)
sine(pos) return a Tensor of (b, h, w, 16), which will be permuted as (b, 16, h, w) 

---
NerfPosifionalEncoding
  depth = 4
  sine_type = ‘lin_sine’
  avoid_aliazing = True
  max_spatial_resolution = 60
  bases = [1, 2, 3, 4]
  factor = 60/4 = 15

  out = torch.cat([torch.sin(i * self.factor * math.pi * inputs) for i in self.bases] +
                          [torch.cos(i * self.factor * math.pi * inputs) for i in self.bases], axis=-1)
          # [b, h, w, 2*depth*2 = 16]


YOLO 代码
数据处理
由于YOLO实时检测的特点，适合对单张图像进行处理。红外图像的时序信息尝试通过维度堆叠的方式融合到单张图像中。
假设红外图像img : [1, h, w] , 前帧图像 pre_img : [1, h, w]，通过pre_img 生成前帧热图 pre_hm : [1, h, w]
合成图像 x : [3, h, w] ，堆叠方式 3 ==> {img, pre_img, pre_hm}

单幅合成图像中包含时序信息，不需自定义 Dataset 类。运行单独的脚本提前处理数据集。
测试部分还没想好
模块编写
在源码DLA类基础上修改，编写InfraredHeatmapFusion类，计划作为网络主干的输入层。
模块 InfraredHeatmapFusion , 导入同目录下的类 
  from .transformer_fusion import Transformer_Fusion_M
  from .position_encoding import PositionEmbeddingSine

ir_hm_fusion.py
从PFTrack项目代码中梳理出来从输入图像到融合结果的变换过程，输入3通道合成图，拆分成三个单通道图像分别处理后进行特征融合。
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .transformer_fusion import Transformer_Fusion_M
from .position_encoding import PositionEmbeddingSine

BN_MOMENTUM = 0.1

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
        img, pre_img, pre_hm = torch.split(x, 1, dim=1)
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
    
position_encoding.py
这里自定义m_cumsum函数来替换源码的torch.cumsum调用。原因是torch.cumsum调用方式与cuda内核相关，会产生下面的警告，采用自定义函数明确化cumsum计算，运算速度有少许下降。
  /home/rody/code/ultralytics/ultralytics/nn/modules/ir/position_encoding.py:52:  UserWarning: cumsum_cuda_kernel does not have a deterministic  implementation, but you set 'torch.use_deterministic_algorithms(True,  warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues   to help us prioritize adding deterministic support for this operation.  (Triggered internally at  /opt/conda/conda-bld/pytorch_1720538438750/work/aten/src/ATen/Context.cpp:83.)   x_embed = not_mask.cumsum(2, dtype=torch.float32)

import math

import numpy as np
import torch
from numpy import dtype
from torch import nn

class NerfPositionalEncoding(nn.Module):
    def __init__(self, depth=10, sine_type='lin_sine', avoid_aliasing=False, max_spatial_resolution=None):
        '''
        out_dim = in_dim * depth * 2
        '''
        super().__init__()
        if sine_type == 'lin_sine':
            self.bases = [i+1 for i in range(depth)]
        elif sine_type == 'exp_sine':
            self.bases = [2**i for i in range(depth)]
        print(f'using {sine_type} as positional encoding')

        if avoid_aliasing and (max_spatial_resolution is None):
            raise ValueError('Please specify the maxima spatial resolution (h, w) of the feature map')
        elif avoid_aliasing:
            self.factor = max_spatial_resolution/depth
        else:
            self.factor = 1.

    @torch.no_grad()
    def forward(self, inputs):
        out = torch.cat([torch.sin(i * self.factor * math.pi * inputs) for i in self.bases] +
                        [torch.cos(i * self.factor * math.pi * inputs) for i in self.bases], axis=-1)
        #pdb.set_trace()
        assert torch.isnan(out).any() == False
        return out


def m_cumsum(input_tensor, dim, dtype=torch.float16):

    # print(input_tensor.dim())
    if input_tensor.dim() != 3:
        return torch.cumsum(input_tensor, dim)

    input_tensor = input_tensor.clone().detach().requires_grad_(False)
    result = torch.zeros_like(input_tensor, dtype=dtype)
    # print(result)
    if dim == 1:
        for i in range(input_tensor.shape[dim]):
            if i == 0:
                result[:, i, :] = input_tensor[:, i, :]
            else:
                result[:, i, :] = result[:, i - 1, :] + input_tensor[:, i, :]
    elif dim == 2:
        for i in range(input_tensor.shape[dim]):
            if i == 0:
                result[:, :, i] = input_tensor[:, :, i]
            else:
                result[:, :, i] = result[:, :, i - 1] + input_tensor[:, :, i]
    return result


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, sine_type='lin_sine',
                 avoid_aliazing=False, max_spatial_resolution=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.sine = NerfPositionalEncoding(num_pos_feats//2, sine_type, avoid_aliazing, max_spatial_resolution)

    @torch.no_grad()
    def forward(self, mask):

        assert mask is not None
        not_mask = ~mask
        # y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # x_embed = not_mask.cumsum(2, dtype=torch.float32)
        y_embed = m_cumsum(not_mask, 1, dtype=torch.float16)
        x_embed = m_cumsum(not_mask, 2, dtype=torch.float16)
        # print(y_embed.dtype, x_embed.dtype)
        eps = 1e-6
        y_embed = (y_embed-0.5) / (y_embed[:, -1:, :] + eps)
        x_embed = (x_embed-0.5) / (x_embed[:, :, -1:] + eps)
        pos = torch.stack([x_embed, y_embed], dim=-1)
        out=self.sine(pos).permute(0, 3, 1, 2)
        return out
transformer_fusion.py 
yolov8 默认启用 AWP（Automatic Mixed Precision），TransformerEncoderLayer.forward入参在训练过程中被 amp 自动转换为torch.float16，为保持张量正常匹配计算，把涉及多头注意力层计算的张量都进行手动转换。另外在训练前须指定参数dtype 为 torch.float32。
# --------------------------------------------------------------------*/
# This file includes code from https://github.com/facebookresearch/detr/blob/main/models/detr.py
# --------------------------------------------------------------------*/
#


import copy
import torch
import torch.nn.functional as F
from torch import nn
import pdb


class Transformer_Fusion_M(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_fusion_encoder_layers=2, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False, return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_fusion_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead       
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    def forward(self, src, pre_src,pre_hm, pos_embed):

        output = self.encoder(src, pre_src, pre_hm, pos=pos_embed)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm
    def forward(self, src, pre_src, pre_hm, pos=None):
        output = src
        pre_output = pre_src
        for layer in self.layers:
            output= layer(output, pre_output, pre_hm, pos=pos)
        return output

        
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()    
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)        
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
         
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pre_src, pre_hm, pos=None, c=1, n=2):
        q = self.with_pos_embed(src, pos)
        k = self.with_pos_embed(pre_src, pos)
        if pos is not None: pos.to(dtype=torch.float16)
        # print("src: {}, pre_src: {}, pre_hm: {}, q: {}, k: {}, v: {}, pos: {}".format(src.dtype, pre_src.dtype, pre_hm.dtype, q.dtype, k.dtype, pre_src.dtype, pos.dtype))
        src1 = self.multihead_attn(q, k, value=pre_src)[0]
        src = src + self.dropout(src1)
        src = self.norm1(src)
        src = src + pre_hm
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer_fusion(args):
    return Transformer_Fusion_M(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu/glu, not {activation}.")
ultralytics/nn/tasks.py 修改 parse_model(c, ch, verbose=True)
导入ir包，添加对 InfraredHeatmapFusion 模块的解析。
from ultralytics.nn.modules.ir.ir_hm_fusion import InfraredHeatmapFusion as IrHmFusion

def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {
            Classify,Conv,ConvTranspose,GhostConv,Bottleneck,GhostBottleneck,SPP,SPPF,C2fPSA,C2PSA,DWConv,Focus,BottleneckCSP,C1,C2,C2f,C3k2,RepNCSPELAN4,ELAN1,ADown,AConv,SPPELAN,C2fAttn,C3,C3TR,C3Ghost,nn.ConvTranspose2d,DWConvTranspose2d,C3x,RepC3,PSA,SCDown,C2fCIB,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads

            args = [c1, c2, *args[1:]]
            if m in {
                BottleneckCSP,C1,C2,C2f,C3k2,C2fAttn,C3,C3TR,C3Ghost,C3x,RepC3,C2fPSA,C2fCIB,C2PSA,
            }:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}:
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m is IrHmFusion:
            c2 = args[0]
            args = [c2, *args]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


训练测试
yaml文件 
把第0层换成 InfraredHeatmapFusion 模块。
模型结构
                   from  n    params  module                                       arguments                     
using lin_sine as positional encoding
  0                  -1  1     71280  ultralytics.nn.modules.ir.ir_hm_fusion.InfraredHeatmapFusion[16, 16]                      
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 22        [15, 18, 21]  1    751702  ultralytics.nn.modules.head.Detect           [2, [64, 128, 256]]           
YOLOv8-irhm summary: 251 layers, 3,082,054 parameters, 3,082,038 gradients



在本机上训练：
model.to(dtype=torch.float32, device='cuda')
result = model.train(data='vtuav-irhm.yaml',
                     name='yolov8_irhm',
                     epochs=50,
                     workers=8,
                     batch=1)

结果：
tm
Validating runs/detect/yolov8_irhm6/weights/best.pt...
WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.
Ultralytics 8.3.35 🚀 Python-3.8.20 torch-2.4.0 CUDA:0 (NVIDIA GeForce GTX 1650 Ti, 3897MiB)
YOLOv8-irhm summary (fused): 195 layers, 3,076,870 parameters, 0 gradients
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 17/17 [00:05<00:00,  3.12it/s]
                   all         67        915     0.0128      0.144      0.008    0.00161
                person         67        612    0.00422    0.00327    0.00212   0.000212
                   car         42        303     0.0215      0.284     0.0139      0.003
Speed: 0.3ms preprocess, 78.1ms inference, 0.0ms loss, 0.7ms postprocess per image
Results saved to runs/detect/yolov8_irhm6


Dataset类

12.02 继承torch.util.data.Dataset 的自定义Dataset类实现，未测试。准备再修改，到YOLO中使用。

继承YOLODataset类，重写__getitem__。重写BaseDataset中的 load_image，在这一级完成 image concat。
  
  image concat 必需入参： idx， ann_path 
  img2label_paths 把图像文件路径转换为对应的标签文件路径 。仿写此函数。
  
  img_path = '/home/rody/datas/irhm1128/train'
  im_files 是str list , 储存路径 如 ['/home/rody/datas/irhm1128/train/images/000001.jpg', ... ]
  label_files 是str list , 储存路径 如 ['/home/rody/datas/irhm1128/train/images/000001.txt', ... ]
  
  self.label_files[i]


ultralytics/data/dataset.py
  自定义类YOLOIRHMDataset , 继承 YOLODataset, 重写 load_image, 完成hm generate 与 image concat.
import math

class YOLOIRHMDataset(YOLODataset):
    def __init__(self, *args, data=None, task="detect", **kwargs):
        super().__init__( *args, data=data, task=task, **kwargs)

    def _load_ann(self, filename):
        with open(filename, 'r') as anns_file:
            return [ann.strip() for ann in anns_file.readlines()]
    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            else:  # read image
                im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            if i > 0:
                prev_i = i - 1
                prev_im = cv2.imread(self.im_files[prev_i], cv2.IMREAD_GRAYSCALE)
                prev_ann = self.label_files[i - 1]
            else:
                prev_im = im
                prev_ann = self.label_files[i]
            # check similarity
            similarity = self.compare_histograms(im, prev_im)
            if similarity < 0.9:
                prev_im, prev_ann = im, self.label_files[i]

            prev_hm, _ = self.generate_heatmap(prev_im, self._load_ann(prev_ann))

            # im = torch.from_numpy(im).permute(2, 0, 1).float()
            # prev_im = torch.from_numpy(prev_im).permute(2, 0, 1).float()
            # prev_hm = torch.from_numpy(prev_hm).float()

            assert im.shape == prev_im.shape == prev_hm.shape
            im = np.stack([im, prev_im, prev_hm], axis=-1)

            # im = torch.cat([im, prev_im, prev_hm], dim=0)

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def generate_heatmap(self, img, anns):
        hm_h, hm_w = img.shape
        pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32)
        pre_cts = []

        for ann in anns:
            class_id, x, y, w, h = map(float, ann.split(' '))
            # transfer yolo format 'xywh' to 'xyxy'
            bbox = [(x - w / 2) * hm_w, (y - h / 2) * hm_h, (x + w / 2) * hm_w, (y + h / 2) * hm_h]
            bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            # calculate gaussian radius

            max_rad = 1
            if bbox_h > 0 and bbox_w > 0:
                radius = self.gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
                radius = max(0, int(np.sqrt(bbox_h * bbox_w) / 2))
                max_rad = max(max_rad, radius)
                # print(radius, max_rad)
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                pre_cts.append(ct)
                pre_hm[0] = self.draw_umich_gaussian(pre_hm[0], ct_int, radius, 255)

        return pre_hm[0], pre_cts

    def compare_histograms(self, img1, img2):
        if img1.shape != img2.shape:
            return 0
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    # @numba.jit(nopython=True, nogil=True)
    def gaussian2d(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        # y, x = np.arange(-m, m + 1).reshape(-1, 1), np.arange(-n, n + 1).reshape(1, -1)
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    # @numba.jit(nopython=True, nogil=True)
    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        # import pdb; pdb.set_trace()
        diameter = 2 * radius + 1
        gaussian = self.gaussian2d((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        # import pdb; pdb.set_trace()
        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap
        

ultralytics/data/build.py
from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset, YOLOIRHMDataset

def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """Build YOLO Dataset."""
    # dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    try:
        dataset = YOLOIRHMDataset
    except Exception as e:
        dataset = YOLOMultiModalDataset if multi_modal else YOLODataset

    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )
