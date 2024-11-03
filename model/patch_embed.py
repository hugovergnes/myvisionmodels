import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def resample_patch_embed(ckpt, kernel_name, target_size, mode="nearest"):
    kernel = ckpt[kernel_name]
    kernel = F.interpolate(kernel, size=target_size, mode=mode)
    if kernel.shape[1] == 3:
        kernel = torch.mean(kernel, dim=1, keepdim=True)
    ckpt[kernel_name] = kernel
    return ckpt


class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size=(7, 7),
        stride=(4, 4),
        padding=(3, 3),
        in_chans=3,
        embed_dim=768,
        img_size=None,
        norm_layer=None,
    ):
        """_summary_

        Args:
            kernel_size (Tuple): kernel size of the projection layer.
                Defaults to (7, 7).
            stride (Tuple): stride of the projection layer. Defaults to (4, 4).
            padding (Tuple): padding size of the projection layer. Defaults to (3, 3).
            in_chans (int): Number of input image channels. Defaults to 3.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
                Defaults to 768.
            img_size (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.norm = norm_layer if norm_layer is not None else nn.Identity()

        if img_size is not None:
            grid_size = (np.array(img_size) + np.array(padding)) / np.array(stride)
            self.grid_size = grid_size.astype(int)
            self.num_patches = int(self.grid_size.prod())

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: Output tensor has shape B, H_tokens, W_tokens, C
        """
        return self.norm(self.proj(x))
