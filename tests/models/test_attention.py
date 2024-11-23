import pytest
import torch
from torch import nn

from hepattn.models import Attention


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [8])
@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("flex", [True, False])
def test_attention_consistency(batch_size, seq_len, dim, num_heads, bias, flex):
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Generate random input tensors
    q = torch.randn(batch_size, seq_len, dim, dtype=torch.float16, device="cuda", requires_grad=True)
    k = torch.randn(batch_size, seq_len, dim, dtype=torch.float16, device="cuda", requires_grad=True)
    v = torch.randn(batch_size, seq_len, dim, dtype=torch.float16, device="cuda", requires_grad=True)

    # Initialize attention layers
    attention_layer = Attention(dim=dim, num_heads=num_heads, bias=bias, flex=flex).cuda().half()
    mha_layer = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=bias, batch_first=True).cuda().half()

    # Synchronize weights for comparison
    attention_layer.q_proj.weight.data = mha_layer.in_proj_weight[:dim, :]
    attention_layer.k_proj.weight.data = mha_layer.in_proj_weight[dim : 2 * dim, :]
    attention_layer.v_proj.weight.data = mha_layer.in_proj_weight[2 * dim :, :]
    attention_layer.out_proj.weight.data = mha_layer.out_proj.weight
    if bias:
        attention_layer.q_proj.bias.data = mha_layer.in_proj_bias[:dim]
        attention_layer.k_proj.bias.data = mha_layer.in_proj_bias[dim : 2 * dim]
        attention_layer.v_proj.bias.data = mha_layer.in_proj_bias[2 * dim :]
        attention_layer.out_proj.bias.data = mha_layer.out_proj.bias

    # Compute outputs
    custom_out = attention_layer(q, k, v)
    mha_out, _ = mha_layer(q, k, v)

    # Compare outputs
    torch.testing.assert_close(custom_out, mha_out, atol=1e-3, rtol=1e-3)
