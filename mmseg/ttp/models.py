import copy
from typing import List, Tuple, Optional
import torch.nn.functional as F
import einops
import torch
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks.transformer import PatchEmbed, FFN, build_transformer_layer
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from peft import get_peft_config, get_peft_model
from torch import Tensor, nn
# from mmdet.utils import OptConfigType, MultiConfig
from mmpretrain.models import resize_pos_embed
from mmpretrain.models.backbones.vit_sam import Attention, window_partition, window_unpartition
from mmseg.models import BaseSegmentor, EncoderDecoder
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.utils import OptConfigType, MultiConfig
from opencd.registry import MODELS

from mmpretrain.models import build_norm_layer as build_norm_layer_mmpretrain


@MODELS.register_module()
class MMPretrainSamVisionEncoder(BaseModule):
    def __init__(
            self,
            encoder_cfg,
            peft_cfg=None,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        vision_encoder = MODELS.build(encoder_cfg)
        vision_encoder.init_weights()
        if peft_cfg is not None and isinstance(peft_cfg, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                'target_modules': ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_cfg)
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder
            # freeze the vision encoder
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.vision_encoder(x)


@MODELS.register_module()
class MLPSegHead(BaseDecodeHead):
    def __init__(
            self,
            out_size,
            interpolate_mode='bilinear',
            **kwargs
    ):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)
        self.out_size = out_size
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=self.out_size,
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.cls_seg(out)
        return out


@MODELS.register_module()
class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

@MODELS.register_module()
class SequentialNeck(BaseModule):
    def __init__(self, necks):
        super().__init__()
        self.necks = nn.ModuleList()
        for neck in necks:
            self.necks.append(MODELS.build(neck))

    def forward(self, *args, **kwargs):
        for neck in self.necks:
            args = neck(*args, **kwargs)
        return args


@MODELS.register_module()
class SimpleFPN(BaseModule):
    def __init__(self,
                 backbone_channel: int,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(self.backbone_channel // 2,
                               self.backbone_channel // 4, 2, 2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, input: Tensor) -> tuple:
        # build FPN
        inputs = []
        inputs.append(self.fpn1(input))
        inputs.append(self.fpn2(input))
        inputs.append(self.fpn3(input))
        inputs.append(self.fpn4(input))

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)


@MODELS.register_module()
class TimeFusionTransformerEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 use_rel_pos: bool = False,
                 window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.window_size = window_size

        self.ln1 = build_norm_layer_mmpretrain(norm_cfg, self.embed_dims)

        self.attn = Attention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else
            (window_size, window_size),
        )

        self.ln2 = build_norm_layer_mmpretrain(norm_cfg, self.embed_dims)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

        if self.window_size > 0:
            in_channels = embed_dims * 2
            self.down_channel = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, bias=False)
            self.down_channel.weight.data.fill_(1.0/in_channels)

            self.soft_ffn = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1),
                nn.GELU(),
                nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1),
            )

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x):
        shortcut = x
        x = self.ln1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x

        x = self.ffn(self.ln2(x), identity=x)
        # # time phase fusion
        if self.window_size > 0:
            x = einops.rearrange(x, 'b h w d -> b d h w')  # 2B, C, H, W
            x0 = x[:x.size(0)//2]
            x1 = x[x.size(0)//2:]  # B, C, H, W
            x0_1 = torch.cat([x0, x1], dim=1)
            activate_map = self.down_channel(x0_1)
            activate_map = torch.sigmoid(activate_map)
            x0 = x0 + self.soft_ffn(x1 * activate_map)
            x1 = x1 + self.soft_ffn(x0 * activate_map)
            x = torch.cat([x0, x1], dim=0)
            x = einops.rearrange(x, 'b d h w -> b h w d')
        return x