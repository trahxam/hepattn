import torch
from torch import nn

from hepattn.models import Dense  # Update the import to match your module's filename


def test_dense_default_parameters():
    """Test Dense network with default parameters."""
    model = Dense(input_size=10)
    x = torch.randn(5, 10)  # Batch of 5, input size of 10
    output = model(x)
    assert output.shape == (
        5,
        10,
    ), "Output shape should match batch size and output size"


def test_dense_custom_parameters():
    """Test Dense network with custom parameters."""
    model = Dense(
        input_size=10,
        output_size=5,
        hidden_layers=[20, 15],
        hidden_dim_scale=3,
        activation=nn.ReLU,
        final_activation=nn.Sigmoid,
        dropout=0.1,
        bias=False,
    )
    x = torch.randn(4, 10)  # Batch of 4, input size of 10
    output = model(x)
    assert output.shape == (
        4,
        5,
    ), "Output shape should match batch size and output size"
    assert isinstance(model.net[0], nn.Linear), "First layer should be a linear layer"
    assert isinstance(model.net[1], nn.ReLU), "Second layer should be a ReLU activation"
    assert isinstance(model.net[2], nn.Dropout), "Third layer should be a dropout layer"
    assert isinstance(model.net[-2], nn.Linear), "Fourth layer should be a linear layer"
    assert isinstance(model.net[-1], nn.Sigmoid), "Last layer should be a Sigmoid"


def test_dense_no_hidden_layers():
    """Test Dense network with no hidden layers."""
    model = Dense(input_size=8, output_size=4, hidden_layers=[])
    x = torch.randn(3, 8)
    output = model(x)
    assert output.shape == (
        3,
        4,
    ), "Output shape should match batch size and output size"
    assert len(model.net) == 1, "Network should contain only one layer"


def test_dense_forward_propagation():
    """Test Dense forward propagation for expected behavior."""
    model = Dense(input_size=6, output_size=6, hidden_layers=[12, 8])
    x = torch.ones(2, 6)  # Batch of 2, input size of 6
    output = model(x)
    assert torch.all(output != 0), "Output should not be all zeros"
