from torch.autograd import profiler


from model.convnext import ConvNextTiny, ConvNextSmall, ConvNextBase
from model.inception import InceptionNextTiny, InceptionNextSmall, InceptionNextBase

import torch


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    inp = torch.randn(3, 1, 224, 224)
    model = ConvNextBase(
        in_chans=1,
        num_classes=2,
        patch_size=8,
        drop_path_rate=0.3,
        head_init_scale=0.005,
    )
    # model = InceptionNextTiny(
    #     in_chans=1,
    #     num_classes=2,
    #     patch_size=4,
    #     weights="/Users/hugovergnes/Documents/checkoints/inception_next/inceptionnext_tiny.pth",
    # )
    # model = MetaNeXt(in_chans=1, num_classes=2)
    out = model(inp)
    out = out["model_output"]
    # print(f"done, out shape {out.shape}")

    n_param = count_trainable_parameters(model)
    print(f"{n_param/10**6:.1f}M parameters")

    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            _ = model(inp)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
