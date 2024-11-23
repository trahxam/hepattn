import pytest
import torch
from torch import Tensor, nn

from hepattn.models import DropPath, EncoderLayer, LayerNorm, LayerScale, Residual, Transformer


# Fixtures for common inputs
@pytest.fixture
def input_tensor():
    return torch.rand(8, 16, 64)  # (batch_size, seq_len, dim)


@pytest.fixture
def bool_mask():
    return torch.ones(8, 16, dtype=torch.bool)  # (batch_size, seq_len)


# Tests for DropPath
def test_droppath_no_drop(input_tensor):
    model = DropPath(drop_prob=0.0)
    model.eval()  # Ensure not training
    output = model(input_tensor)
    assert torch.equal(output, input_tensor)


def test_droppath_with_drop(input_tensor):
    model = DropPath(drop_prob=0.5)
    model.train()  # Ensure training mode
    output = model(input_tensor)
    assert output.shape == input_tensor.shape
    assert not torch.equal(output, input_tensor)  # Values should differ due to drop


# Tests for LayerScale
def test_layerscale(input_tensor):
    model = LayerScale(dim=input_tensor.shape[-1], init_value=0.1)
    output = model(input_tensor)
    assert output.shape == input_tensor.shape
    assert torch.allclose(output, input_tensor * 0.1)


# Tests for Residual
def test_residual(input_tensor):
    fn = nn.Linear(input_tensor.shape[-1], input_tensor.shape[-1])
    model = Residual(fn=fn, norm=LayerNorm, layer_scale=1e-5, drop_path=0.0, dim=input_tensor.shape[-1])
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


# Tests for EncoderLayer
def test_encoderlayer(input_tensor):
    dim = input_tensor.shape[-1]
    model = EncoderLayer(dim=dim, drop_path=0.0, layer_scale=1e-5)
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


def test_encoderlayer_with_kwargs(input_tensor):
    dim = input_tensor.shape[-1]
    model = EncoderLayer(dim=dim, drop_path=0.1, attn_kwargs={"num_heads": 4})
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


# Tests for Transformer
def test_transformer_forward(input_tensor):
    model = Transformer(num_layers=3, dim=input_tensor.shape[-1])
    output, _ = model(input_tensor)
    assert isinstance(output, Tensor)
    # assert isinstance(mask, BoolTensor)  # noqa: ERA001
    assert output.shape == input_tensor.shape


def test_transformer_with_dict_inputs():
    batch_size = 8
    seq_len = 16
    dim = 64

    x = {"part1": torch.rand(batch_size, seq_len // 2, dim), "part2": torch.rand(batch_size, seq_len // 2, dim)}

    model = Transformer(num_layers=2, dim=dim)
    output, _ = model(x)
    assert output.shape == (batch_size, seq_len, dim)


# Edge cases
def test_transformer_empty_input():
    model = Transformer(num_layers=1, dim=64)
    x = torch.empty(0, 0, 64)
    mask = torch.empty(0, 0, dtype=torch.bool)
    output, output_mask = model(x, mask)
    assert output.shape == x.shape
    assert output_mask.shape == mask.shape
