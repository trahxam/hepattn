import torch

from hepattn.components import Dense, Pooling


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
        "x_valid": torch.randn(batch_size, seq_len) >= 0.5,
    }

    outputs = pooling_layer(inputs["x_embed"], inputs["x_valid"])

    assert outputs.shape == (batch_size, dim)
