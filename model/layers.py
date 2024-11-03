import torch
import torch.nn as nn

from timm.layers import DropPath, drop_path


def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def get_subset_index_and_scale_factor(x, drop_ratio=0.0):
    # random selection of the subset of the batch
    B, _, _ = x.shape
    selected_subset_size = max(int(B * (1 - drop_ratio)), 1)
    selected_indicies = (torch.randperm(B, device=x.device))[:selected_subset_size]

    return selected_indicies, B / selected_subset_size


def apply_residual(x, selected_indicies, residual, residual_scale_factor):
    """_summary_

    Args:
        x (torch.Tensor): _description_
        selected_indicies (torch.Tensor): _description_
        residual (torch.Tensor): _description_
        residual_scale_factor (float): _description_

    Returns:
        torch.Tensor: _description_
    """
    residual = residual.to(dtype=x.dtype)
    x_flat, residual_flat = x.flatten(1), residual.flatten(1)

    return torch.index_add(
        x_flat, 0, selected_indicies, residual_flat, alpha=residual_scale_factor
    ).view_as(x)


def efficient_drop_path(x, func, drop_ratio=0.0, training=False):
    """Efficient Drop Path implementation.

    Ref impl: https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/block.py

    Args:
         x (torch.Tensor): input tensor
         func (Callable[[torch.Tensor], torch.Tensor]): function to calculate residual
         drop_ratio (float, optional): Drop ratio. Defaults to 0.0.
         training (bool, optional): training mode. Defaults to False.

    Returns:
         torch.Tensor: output tensor
    """

    if not training or drop_ratio == 0.0:
        return func(x)

    # there is an overhead of using fast drop block for small drop ratio
    if drop_ratio <= 0.1:
        return drop_path(func(x), drop_ratio, training=training)

    # extract subset of the batch
    selected_indicies, residual_scale_factor = get_subset_index_and_scale_factor(
        x, drop_ratio=drop_ratio
    )

    # apply residual
    residual = func(x[selected_indicies])

    return apply_residual(x, selected_indicies, residual, residual_scale_factor)


#  class EfficientDropPathBlock(nn.Module):
#      def __init__(
#              self,
#              dim,
#              num_heads,
#              mlp_ratio=4.,
#              qkv_bias=False,
#              qk_norm=False,
#              proj_drop=0.,
#              attn_drop=0.,
#              init_values=None,
#              drop_path=0.,
#              act_layer=nn.GELU,
#              norm_layer=nn.LayerNorm,
#              mlp_layer=Mlp,
#              attn_class=Attention,
#      ):
#          super().__init__()
#          self.norm1 = norm_layer(dim)
#          self.attn = attn_class(
#              dim,
#              num_heads=num_heads,
#              qkv_bias=qkv_bias,
#              qk_norm=qk_norm,
#              attn_drop=attn_drop,
#              proj_drop=proj_drop,
#              norm_layer=norm_layer,
#          )
#          self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#          self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#          self.norm2 = norm_layer(dim)
#          self.mlp = mlp_layer(
#              in_features=dim,
#              hidden_features=int(dim * mlp_ratio),
#              act_layer=act_layer,
#              drop=proj_drop,
#          )
#          self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#          self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#      @torch.jit.ignore(drop=True) # This is for ViT
#      def efficient_drop_path_forward(self, x):
#          def residual_attn_func(x):
#              return self.ls1(self.attn(self.norm1(x)))

#          def residual_mlp_func(x):
#              return self.ls2(self.mlp(self.norm2(x)))

#          x = efficient_drop_path(
#              x, residual_attn_func,
#              drop_ratio=self.drop_path1.drop_prob if isinstance(self.drop_path1, DropPath) else 0.0,
#              training=self.training
#          )

#          x = efficient_drop_path(
#              x, residual_mlp_func,
#              drop_ratio=self.drop_path2.drop_prob if isinstance(self.drop_path2, DropPath) else 0.0,
#              training=self.training
#          )

#          return x

#      def forward(self, x):
#          if not torch.jit.is_scripting():
#              x = self.efficient_drop_path_forward(x)
#          else:
#              x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
#              x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

#          return x
