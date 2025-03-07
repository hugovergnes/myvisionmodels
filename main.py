from torch.autograd import profiler


from model.convnext import ConvNextTiny, ConvNextSmall, ConvNextBase
from model.inception import InceptionNextTiny, InceptionNextSmall, InceptionNextBase

import torch
import torch.nn as nn


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    inp = torch.randn(3, 1, 1280, 768)
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
    out_features = model.forward_features(inp)

    pool = nn.AdaptiveAvgPool2d((10, 6))
    out_features = pool(out_features)

    out_features = out_features.view(3, 1024, -1)
    seq_len, feature_dim, flattened_dim = out_features.shape
    lstm = nn.LSTM(
        input_size=1024,
        hidden_size=512,
        num_layers=1,
        batch_first=True,
        bidirectional=True,
    )
    lstm_input = out_features.permute(0, 1, 2).reshape(seq_len, 1, -1)

    output, (hn, cn) = lstm(lstm_input)

    print(output.shape)  # Output shape: (seq_len, batch, hidden_size)

    # out = out["model_output"]
    # print(f"done, out shape {out.shape}")

    n_param = count_trainable_parameters(model)
    print(f"{n_param/10**6:.1f}M parameters")

    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            _ = model(inp)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
