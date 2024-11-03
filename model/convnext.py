import dill
from functools import partial

from model.patch_embed import PatchEmbed
from model.layers import DropPath, efficient_drop_path  # TODO: Fix DropPath import here

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import (
    trunc_normal_,
    AvgPool2dSame,
    Mlp,
    LayerNorm2d,
    LayerNorm,
    create_conv2d,
    get_act_layer,
    to_ntuple,
    NormMlpClassifierHead,
)


def resample_kernel(ckpt, target_size, kernel_name="stem.0.weight", mode="nearest"):
    kernel = ckpt[kernel_name]
    kernel = torch.mean(kernel, dim=1, keepdim=True)
    kernel = F.interpolate(kernel, size=(target_size, target_size), mode=mode)
    ckpt[kernel_name] = kernel
    return ckpt


class Downsample(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, dilation=1):
        super().__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = (
                AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            )
            self.pool = avg_pool_fn(
                2, avg_stride, ceil_mode=True, count_include_pad=False
            )
        else:
            self.pool = nn.Identity()

        if in_chs != out_chs:
            self.conv = create_conv2d(in_chs, out_chs, 1, stride=1)
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    """

    def __init__(
        self,
        in_chs,
        out_chs=None,
        kernel_size=7,
        stride=1,
        dilation=(1, 1),
        mlp_ratio=4,
        conv_bias=True,
        ls_init_value=1e-6,
        act_layer="gelu",
        norm_layer=None,
        drop_path=0.0,
    ):
        """
        Args:
            in_chs (int): Block input channels.
            out_chs (Optional[int]): Block output channels (defaults to in_chs if None).
            kernel_size (int): Depthwise convolution kernel size.
            stride (int): Stride of depthwise convolution.
            dilation (Union[int, Tuple[int, int]]): Dilation configuration for block.
            mlp_ratio (float): MLP expansion ratio.
            conv_bias (bool): If True, apply bias to convolution layers.
            ls_init_value (Optional[float]): Initial value for layer scaling; applies if not None.
            act_layer (Union[str, Callable]): Activation layer.
            norm_layer (Optional[Callable]): Normalization layer (defaults to LayerNorm if not specified).
            drop_path (float): Dropout probability for stochastic depth.
        """
        super().__init__()
        out_chs = out_chs or in_chs
        dilation = to_ntuple(2)(dilation)
        act_layer = get_act_layer(act_layer)
        if not norm_layer:
            norm_layer = LayerNorm
        self.conv_dw = create_conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation[0],
            depthwise=True,
            bias=conv_bias,
        )
        self.norm = norm_layer(out_chs)
        self.mlp = Mlp(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)
        self.gamma = (
            nn.Parameter(ls_init_value * torch.ones(out_chs))
            if ls_init_value is not None
            else None
        )
        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = Downsample(
                in_chs, out_chs, stride=stride, dilation=dilation[0]
            )
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    # This forward should be fast
    # def efficient_drop_path_forward(self, x):
    #     def residual_mlp_func(x):
    #         x = self.norm(self.conv_dw(x).permute(0, 2, 3, 1))
    #         x = self.mlp(x).permute(0, 3, 1, 2)
    #         if self.gamma is not None:
    #             x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    #         return x

    #     x = self.shortcut(x) + efficient_drop_path(
    #         x,
    #         residual_mlp_func,
    #         drop_ratio=(
    #             self.drop_path.drop_prob
    #             if isinstance(self.drop_path, DropPath)
    #             else 0.0
    #         ),
    #         training=self.training,
    #     )

    # This one should be slow
    def forward(self, x):
        shortcut = x
        x = self.mlp(self.norm(self.conv_dw(x).permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        x = self.drop_path(x) + self.shortcut(shortcut)
        return x


class ConvNeXtStage(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size=7,
        stride=2,
        depth=2,
        dilation=(1, 1),
        drop_path_rates=None,
        ls_init_value=1.0,
        conv_bias=True,
        act_layer="gelu",
        norm_layer=None,
        norm_layer_cl=None,
    ):
        super().__init__()

        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = (
                "same" if dilation[1] > 1 else 0
            )  # same padding needed if dilation used
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                create_conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=ds_ks,
                    stride=stride,
                    dilation=dilation[0],
                    padding=pad,
                    bias=conv_bias,
                ),
            )
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.0] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(
                ConvNeXtBlock(
                    in_chs=in_chs,
                    out_chs=out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation[1],
                    drop_path=drop_path_rates[i],
                    ls_init_value=ls_init_value,
                    conv_bias=conv_bias,
                    act_layer=act_layer,
                    norm_layer=norm_layer_cl,
                )
            )
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        output_stride=32,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        kernel_sizes=7,
        ls_init_value=1e-6,
        patch_size=4,
        head_init_scale=1.0,
        head_hidden_size=None,
        conv_bias=True,
        act_layer="gelu",
        drop_rate=0.0,
        drop_path_rate=0.0,
        pretrained_weights=None,
        strict_loading=False,
        interpolation_mode="nearest",
        kernel_name="stem.0.weight",
    ):
        """
        Args:
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            global_pool: Global pooling type.
            output_stride: Output stride of network, one of (8, 16, 32).
            depths: Number of blocks at each stage.
            dims: Feature dimension at each stage.
            kernel_sizes: Depthwise convolution kernel-sizes for each stage.
            ls_init_value: Init value for Layer Scale, disabled if None.
            patch_size: Stem patch size for patch stem.
            head_init_scale: Init scaling value for classifier weights and biases.
            head_hidden_size: Size of MLP hidden layer in head if not None.
            conv_bias: Use bias layers w/ all convolutions.
            act_layer: Activation layer type.
            drop_rate: Head pre-classifier dropout rate.
            drop_path_rate: Stochastic depth drop rate.
        """
        super().__init__()
        self.patch_size = patch_size
        self.interpolation_mode = interpolation_mode
        self.kernel_name = kernel_name

        assert output_stride in (8, 16, 32)
        kernel_sizes = to_ntuple(4)(kernel_sizes)

        norm_layer = partial(LayerNorm2d, eps=1e-6)
        norm_layer_cl = partial(LayerNorm, eps=1e-6)

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        self.stem = PatchEmbed(
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            in_chans=in_chans,
            embed_dim=dims[0],
            img_size=None,
            norm_layer=norm_layer(dims[0]),
        )
        stem_stride = patch_size

        self.stages = nn.Sequential()
        dp_rates = [
            x.tolist()
            for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)
        ]
        stages = []
        prev_chs = dims[0]
        curr_stride = stem_stride
        dilation = 1
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]
            stages.append(
                ConvNeXtStage(
                    prev_chs,
                    out_chs,
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    dilation=(first_dilation, dilation),
                    depth=depths[i],
                    drop_path_rates=dp_rates[i],
                    ls_init_value=ls_init_value,
                    conv_bias=conv_bias,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    norm_layer_cl=norm_layer_cl,
                )
            )
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [
                dict(num_chs=prev_chs, reduction=curr_stride, module=f"stages.{i}")
            ]
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs

        self.norm_pre = nn.Identity()
        self.head = NormMlpClassifierHead(
            self.num_features,
            num_classes,
            hidden_size=head_hidden_size,
            pool_type=global_pool,
            drop_rate=self.drop_rate,
            norm_layer=norm_layer,
            act_layer="gelu",
        )
        self.head.fc.weight.data.mul_(head_init_scale)
        self.head.fc.bias.data.mul_(head_init_scale)

        self.apply(self._init_weights)

        self.init_weights(pretrained_weights, strict_loading)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def init_weights(self, pretrained_weights, strict_loading):
        """Initialize weights for the model"""
        if pretrained_weights is None:
            return None

        with open(pretrained_weights, "rb") as f:
            checkpoint = torch.load(
                f, pickle_module=dill, map_location=lambda storage, loc: storage
            )
            # remove weights that have changed sizes.
            remove_layers = ["head.fc.weight", "head.fc.bias"]
            checkpoint = {
                k: v
                for k, v in checkpoint.items()
                if not any(name in k for name in remove_layers)
            }
            # Resample patch size
            checkpoint = resample_kernel(
                checkpoint,
                self.patch_size,
                kernel_name=self.kernel_name,
                mode=self.interpolation_mode,
            )

            self.load_state_dict(checkpoint, strict=strict_loading)
            return

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward(self, batch_image_data):
        x = batch_image_data
        x = self.forward_features(x)
        x = self.head(x)
        return {
            "model_output": x,
            "model_output_prob": torch.nn.functional.softmax(x, dim=1),
        }


class ConvNextTiny(ConvNeXt):
    def __init__(self, *args, dims=[96, 192, 384, 768], depths=[3, 3, 9, 3], **kwargs):
        super(ConvNextTiny, self).__init__(*args, dims=dims, depths=depths, **kwargs)


class ConvNextSmall(ConvNeXt):
    def __init__(self, *args, dims=[96, 192, 384, 768], depths=[3, 3, 27, 3], **kwargs):
        super(ConvNextSmall, self).__init__(*args, dims=dims, depths=depths, **kwargs)


class ConvNextBase(ConvNeXt):
    def __init__(
        self, *args, dims=[128, 256, 512, 1024], depths=[3, 3, 27, 3], **kwargs
    ):
        super(ConvNextBase, self).__init__(*args, dims=dims, depths=depths, **kwargs)


class ConvNextLarge(ConvNeXt):
    def __init__(
        self, *args, dims=[192, 384, 768, 1536], depths=[3, 3, 27, 3], **kwargs
    ):
        super(ConvNextLarge, self).__init__(*args, dims=dims, depths=depths, **kwargs)
