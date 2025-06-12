import torch
from torch import nn

from hepattn.models.pooling import Pooling
from hepattn.models.dense import Dense


def test_pooling():
    batch_size = 100
    seq_len = 30
    dim = 128

    pooling_layer = Pooling(
        input_object="x",
        output_object="y",
        dim=dim,
        pool_net=Dense(dim, dim),
    )

    inputs = {
        "x_embed": torch.randn(batch_size, seq_len, dim),
        "x_valid": torch.randn(batch_size, seq_len, dim) >= 0.5,
    }

    outputs = pooling_layer(inputs)

    assert outputs["y_embed"].shape == (batch_size, dim)
