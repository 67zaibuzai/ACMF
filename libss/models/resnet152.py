import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from torch.nn.utils import fusion

from .triplet import TripletAttention
from .TackleStyle import get_color_feature, gram_matrix, StyleDownsample, FiLMFusion, StyleStats, Router
from .register import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, LayerType, create_attn, \
    get_attn, get_act_layer, get_norm_layer, create_classifier, create_aa, to_ntuple

def get_padding(kernel_size: int, stride: int, dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class BasicBlock(nn.Module):
    """Basic residual block for ResNet.

    This is the standard residual block used in ResNet-18 and ResNet-34.
    """
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1, #  Number of convolution groups
            base_width: int = 64,
            reduce_first: int = 1, #  Reduction factor for first convolution output width of residual blocks.
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None, # Anti-aliasing layer class.
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
    ) -> None:
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        # 相当于下采样
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)
        self.triplet_attention = None

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self) -> None:
        """Initialize the last batch norm layer weights to zero for better convergence."""
        if getattr(self.bn2, 'weight', None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x

class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet.

    This is the bottleneck block used in ResNet-50, ResNet-101, and ResNet-152.
    """
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
            use_attn: bool = True,
    ) -> None:
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path
        self.use_attn = use_attn

        if self.use_attn:
            self.triplet_attention = TripletAttention(planes * 4, 16)
        else:
            self.triplet_attention = None

    def zero_init_last(self) -> None:
        """Initialize the last batch norm layer weights to zero for better convergence."""
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.triplet_attention is not None:
            x = self.triplet_attention(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.act3(x)

        return x

def downsample_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    # 如果 stride=1，不下采样 → 用 1×1 卷积做通道变换
    # 如果 stride>1，需要下采样 → 用 3×3（或更大）卷积 + padding + dilation
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])
def downsample_avg(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])
def drop_blocks(drop_prob: float = 0.) -> List[Optional[partial]]:
    """Create DropBlock layer instances for each stage.

    Args:
        drop_prob: Drop probability for DropBlock.

    Returns:
        List of DropBlock partial instances or None for each stage.
    """
    # 对应ResNet四层
    # 由于层不同，深层特征图尺寸小，丢掉一个 block 就可能丢掉太多像素
    return [
        None, None,
        # 丢掉大块，但概率低
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]

def make_blocks(
        block_fns: Tuple[Union[Type[BasicBlock], Type[Bottleneck]], ...],
        channels: Tuple[int, ...],
        block_repeats: Tuple[int, ...],
        inplanes: int,
        reduce_first: int = 1,
        output_stride: int = 32,
        down_kernel_size: int = 1,
        avg_down: bool = False,
        drop_block_rate: float = 0.,
        drop_path_rate: float = 0.,
        use_attn: bool = False,
        **kwargs,
) -> Tuple[List[Tuple[str, nn.Module]], List[Dict[str, Any]]]:
    """Create ResNet stages with specified block configurations.

    Args:
        block_fns: Block class to use for each stage.
        channels: Number of channels for each stage.
        block_repeats: Number of blocks to repeat for each stage.
        inplanes: Number of input channels.
        reduce_first: Reduction factor for first convolution in each stage.
        output_stride: Target output stride of network.
        down_kernel_size: Kernel size for downsample layers.
        avg_down: Use average pooling for downsample.
        drop_block_rate: DropBlock drop rate.
        drop_path_rate: Drop path rate for stochastic depth.
        **kwargs: Additional arguments passed to block constructors.

    Returns:
        Tuple of stage modules list and feature info list.
    """
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (block_fn, planes, num_blocks, db) in enumerate(zip(block_fns, channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get('norm_layer'),
            )
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes,
                planes,
                stride,
                downsample,
                first_dilation=prev_dilation,
                use_attn=use_attn,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None,
                **block_kwargs,
            ))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info

class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    """

    def __init__(
            self,
            block: Union[BasicBlock, Bottleneck],
            layers: Tuple[int, ...],
            in_chans_img: int = 3,
            in_chans_text: int = 3,
            act_layer : LayerType = nn.ReLU,
            norm_layer : LayerType = nn.BatchNorm2d,
            aa_layer : Optional[Type[nn.Module]] = None,
            spec = None,
    ):
        """
        Args:
            block (nn.Module): class for the residual block. Options are BasicBlock, Bottleneck.
            layers (List[int]) : number of layers in each block
            num_classes (int): number of classification classes (default 1000)
            in_chans_img (int): number of input (color) channels. (default 3)
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            cardinality (int): number of convolution groups for 3x3 conv in Bottleneck. (default 1)
            base_width (int): bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): number of channels in stem convolutions (default 64)
            stem_type (str): The type of stem (default ''):
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool (bool): replace stem max-pooling layer with a 3x3 stride-2 convolution
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            avg_down (bool): use avg pooling for projection skip connection between stages/downsample (default False)
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            aa_layer (nn.Module): anti-aliasing layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default 0.)
            drop_block_rate (float): Drop block rate (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module
        """
        output_stride = spec.OUTPUT_STRIDE
        cardinality = spec.CARDINALITY
        base_width = spec.BASE_WIDTH
        stem_width = spec.STEM_WIDTH
        stem_type = spec.STEM_TYPE
        replace_stem_pool = spec.REPLACE_STEM_POOL
        block_reduce_first = spec.BLOCK_REDUCE_FIRST
        down_kernel_size = spec.DOWN_KERNEL_SIZE
        avg_down = spec.AVG_DOWN
        channels = spec.CHANNELS # Optional[Tuple[int, ...]] = (64, 128, 256, 512),
        drop_rate = spec.DROP_RATE # float = 0.0,
        drop_path_rate = spec.DROP_PATH_RATE # float = 0.,
        drop_block_rate = spec.DROP_BLOCK_RATE # float = 0.,
        zero_init_last = spec.ZERO_INIT_LAST # bool = True,
        block_args = spec.BLOCK_ARGS # Optional[Dict[str, Any]] = None,
        output_dim = spec.OUTPUT_DIM
        hidden_mlp = output_dim // 8
        nmb_prototypes = spec.NMB_PROTOTYPES
        num_classes = spec.NUM_CLASSES
        use_attn = spec.USE_ATTN

        super(ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans_img, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0], eps=1e-3),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1], eps=1e-3),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans_img, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes, eps=1e-3)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [
                nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
                norm_layer(inplanes),
                act_layer(inplace=True),
            ]))
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(*[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2)])
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        block_fns = to_ntuple(len(channels))(block)
        stage_modules, stage_feature_info = make_blocks(
            block_fns,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            use_attn = use_attn,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **block_args,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        self.avg_pool = nn.Sequential(
            nn.Conv2d(channels[-1] * block.expansion, output_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(1)
        )

        self.style_encoders = nn.ModuleList([
            StyleStats(),
            StyleStats(),
            StyleStats()
        ])

        D_out = output_dim

        self.style_downlinears = nn.ModuleList([
            nn.Linear(channels[0] * block.expansion * 2, D_out),
            nn.Linear(channels[2] * block.expansion * 2, D_out),
            nn.Linear(channels[3] * block.expansion * 2, D_out),
        ])

        self.router = Router(D_out * 3, 3, )
        self.weight_net = nn.Sequential(
            nn.Linear(D_out, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  # 输出每个特征的权重
        )

        self.filmfusion = FiLMFusion(D_out, 18)

        # TEXT
        text_hidden_mlp = spec.TEXT_HIDDEN_MLP
        if text_hidden_mlp == 0:
            self.text_mlp = nn.Sequential(
                nn.Linear(in_chans_text, D_out),
            )
        else:
            self.text_mlp = nn.Sequential(
                nn.Linear(in_chans_text, text_hidden_mlp),
                nn.BatchNorm1d(text_hidden_mlp),
                nn.ReLU(),
                nn.Linear(text_hidden_mlp, D_out)
            )

        self.l2norm = True

        self.router_fusion = Router(D_out * 3, 3)

        self.classifier = nn.Sequential(
            nn.Linear(D_out, D_out//8),
            nn.ReLU(),
            nn.Linear(D_out//8, num_classes),
        )

        self.init_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
    def init_weights(self, zero_init_last: bool = True) -> None:
        """Initialize model weights.

        Args:
            zero_init_last: Zero-initialize the last BN in each residual branch.
        """

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 初始化适合 ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                # Xavier 初始化适合线性层
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # 所有 BatchNorm 类型都初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            # 可以添加其他类型的层初始化
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # zero_init_last 通常用于 ResNet 的最后一个 BN 层
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last') and callable(m.zero_init_last):
                    m.zero_init_last()

    def forward_features(self, x: torch.Tensor, batchnum) -> torch.Tensor:
        """Forward pass through feature extraction layers."""
        # 检查输入
        self.check_tensor("img", x, [x.min(), x.max(), x.mean(), x.std()])

        x1 = self.conv1(x)
        self.check_tensor("stem layer1", x1, [batchnum, x.min(), x.max(), x.mean(), x.std()])  # 移除了括号
        x = self.bn1(x1)
        self.check_tensor("stem layer2", x, self.get_layer_weights(self.conv1))  # 移除了括号
        x = self.act1(x)
        self.check_tensor("stem layer3", x, ([self.bn1.weight, x]))  # 移除了括号
        x = self.maxpool(x)
        self.check_tensor("stem layer4", x)  # 移除了括号


        x = self.layer1(x)
        style1_x = x
        self.check_tensor("layer 1", x, self.get_layer_weights(self.layer1))
        x = self.layer2(x)
        self.check_tensor("layer 2", x, self.get_layer_weights(self.layer2))
        x = self.layer3(x)
        style2_x = x
        self.check_tensor("layer 3", x, self.get_layer_weights(self.layer3))
        x = self.layer4(x)
        style3_x = x
        self.check_tensor("layer 4", x, self.get_layer_weights(self.layer4))

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # [B, C]
        return x, [style1_x, style2_x, style3_x]

    def get_layer_weights(self, layer):
        """获取层的权重信息（用于调试）"""
        weights_info = []
        for name, module in layer.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weights_info.append(f"{name}.weight: ={[module.weight.max(), module.weight.min()]}")
            if hasattr(module, 'bias') and module.bias is not None:
                weights_info.append(f"{name}.bias: ={[module.bias.max(), module.bias.min()]}")
        return "\n".join(weights_info)

    def check_tensor(self, name, t, last=None):
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"[⚠️ NaN/Inf detected] {name}")
            if last is not None:
                print(f"Last weights: {last}")
            raise ValueError(f"{name}: shape={tuple(t.shape)}, min={t.min().item():.4e}, "
                             f"max={t.max().item():.4e}, mean={t.mean().item():.4e}")

    def forward_style(self, img, img_styles, verbose: bool = True):
        color_feature = get_color_feature(img)  # [B, D_color]
        self.check_tensor("color_feature", color_feature)

        style_feature = []
        for i, style in enumerate(img_styles):
            G = self.style_encoders[i](style)  # [B, 2*C_i]
            G = self.style_downlinears[i](G)  # [B, D_out]
            self.check_tensor(f"style_{i}", style)
            self.check_tensor(f"风格矩阵{i}", G)
            style_feature.append(G)  # [B, 1, C, C]

        style_feature = torch.stack(style_feature, dim=1)

        # 为每个特征生成自适应权重
        weights = self.weight_net(style_feature)  # [B, N, 1]
        self.check_tensor("adaptive_weights", weights)

        # 应用softmax确保权重和为1
        weights = torch.softmax(weights, dim=1)  # [B, N, 1]
        self.check_tensor("normalized_weights", weights)

        # 加权求和
        style_feature = torch.sum(weights * style_feature, dim=1)  # [B, D_out]
        self.check_tensor("style_feature", style_feature)

        # Film 融合
        img_style = self.filmfusion(style_feature, color_feature)
        self.check_tensor("img_style", img_style)

        return img_style

    def forward_text(self, text_feature) -> torch.Tensor:
        x = self.text_mlp(text_feature)
        return x

    def forward_head(self, x):
        return self.classifier(x)

    def forward(self, x: torch.Tensor, batch_num) -> torch.Tensor:
        """Forward pass."""
        """if not isinstance(x, list):
            raise ValueError("x is not a list: [image, text]")"""

        img = x[0]
        text_feature = x[1]

        img_emo_feature, img_styles = self.forward_features(img, batch_num)
        text_feature = self.forward_text(text_feature)

        img_style_feature = self.forward_style(img, img_styles)

        # 收集多模态特征
        features = [img_emo_feature, img_style_feature, text_feature]  # list of [B, D]
        features = [
            F.normalize(feature, dim=1, eps=1e-8) for feature in features
        ]
        feature_tensor = torch.stack(features, dim=1)  # [B, N, D], N=3

        # 路由融合，得到每个模态的权重
        weights = self.router_fusion(feature_tensor)  # [B, N]
        fusion_feature = torch.sum(feature_tensor * weights.unsqueeze(-1), dim=1)  # [B, D]
        fusion_feature = F.normalize(fusion_feature, dim=1, eps=1e-8)

        # 加入融合特征
        all_features = features + [fusion_feature]  # list of 4 features

        # 拼接送入分类头
        concat_features = torch.cat(all_features, dim=0)  # [4B, D]
        classifier_logits = self.classifier(concat_features)

        return concat_features, classifier_logits, weights


class Relu(nn.Module):
    def __init__(self, inplace=True):
        super(Relu, self).__init__()
        self.activation = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.activation(x)

class QuickGELU(nn.Module):
    def __init__(self, approximate: str = "none", inplace: bool = False):
        super().__init__()
        self.approximate = approximate
        self.inplace = inplace

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


@register_model
def get_resnet152_model(config, **kwargs):
    model_spec = config.MODEL.SPEC
    resnet = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        in_chans_img=config.DATASET.IMAGE_CHANS,
        in_chans_text=config.DATASET.TEXT_CHANNEL,
        num_classes=config.MODEL.NUM_CLASSES,
        act_layer=QuickGELU,
        spec=model_spec
    )

    if config.MODEL.INIT_WEIGHTS:
        resnet.init_weights(
            zero_init_last=model_spec.ZERO_INIT_LAST
        )

    return resnet

