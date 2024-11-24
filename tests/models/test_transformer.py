import pytest
import torch
from torch import Tensor, nn

from hepattn.models import DropPath, Encoder, EncoderLayer, LayerNorm, LayerScale, Residual


# Fixtures for common inputs
@pytest.fixture
def input_tensor():
    return torch.rand(8, 130, 128, device="cuda")  # (batch_size, seq_len, dim)


# Tests for DropPath
def test_droppath_no_drop(input_tensor):
    model = DropPath(drop_prob=0.0).cuda()
    model.eval()  # Ensure not training
    output = model(input_tensor)
    assert torch.equal(output, input_tensor)


def test_droppath_with_drop(input_tensor):
    model = DropPath(drop_prob=0.5).cuda()
    model.train()  # Ensure training mode
    output = model(input_tensor)
    assert output.shape == input_tensor.shape
    assert not torch.equal(output, input_tensor)  # Values should differ due to drop


# Tests for LayerScale
def test_layerscale(input_tensor):
    model = LayerScale(dim=input_tensor.shape[-1], init_value=0.1).cuda()
    output = model(input_tensor)
    assert output.shape == input_tensor.shape
    assert torch.allclose(output, input_tensor * 0.1)


# Tests for Residual
def test_residual(input_tensor):
    fn = nn.Linear(input_tensor.shape[-1], input_tensor.shape[-1]).cuda()
    model = Residual(fn=fn, norm=LayerNorm, layer_scale=1e-5, drop_path=0.0, dim=input_tensor.shape[-1]).cuda()
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


# Tests for EncoderLayer
def test_encoderlayer(input_tensor):
    dim = input_tensor.shape[-1]
    model = EncoderLayer(dim=dim, drop_path=0.0, layer_scale=1e-5).cuda()
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


def test_encoderlayer_with_kwargs(input_tensor):
    dim = input_tensor.shape[-1]
    model = EncoderLayer(dim=dim, drop_path=0.1, attn_kwargs={"num_heads": 4}, window_size=10).cuda()
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


# Tests for Encoder
def test_encoder_forward(input_tensor):
    model = Encoder(num_layers=3, dim=input_tensor.shape[-1]).cuda()
    output = model(input_tensor)
    assert isinstance(output, Tensor)
    # assert isinstance(mask, BoolTensor)  # noqa: ERA001
    assert output.shape == input_tensor.shape
    assert output.sum() != 0
    assert not torch.isnan(output).any()
