from torch import Tensor, nn

from hepattn.models.activation import SwiGLU


class Dense(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        hidden_layers: list[int] | None = None,
        hidden_dim_scale: int = 2,
        activation: nn.Module = SwiGLU,
        final_activation: nn.Module | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """A fully connected feed forward neural network, which can take
        in additional contextual information.

        Parameters
        ----------
        input_size : int
            Input size
        output_size : int
            Output size. If not specified this will be the same as the input size.
        hidden_layers : list, optional
            Number of nodes per layer, if not specified, the network will have
            a single hidden layer with size `input_size * hidden_dim_scale`.
        hidden_dim_scale : int, optional
            Scale factor for the hidden layer size.
        activation : nn.Module
            Activation function for hidden layers.
        final_activation : nn.Module, optional
            Activation function for the output layer.
        dropout : float, optional
            Apply dropout with the supplied probability.
        bias : bool, optional
            Whether to use bias in the linear layers.
        """
        super().__init__()

        if output_size is None:
            output_size = input_size
        if hidden_layers is None:
            hidden_layers = [input_size * hidden_dim_scale]

        self.input_size = input_size
        self.output_size = output_size
        gate = activation == SwiGLU

        layers = []
        node_list = [input_size, *hidden_layers]
        for i in range(len(node_list) - 1):
            in_dim = node_list[i]
            proj_dim = node_list[i + 1]
            proj_dim = proj_dim * 2 if gate else proj_dim

            # inner projection and activation
            layers.append(nn.Linear(in_dim, proj_dim, bias=bias))
            layers.append(activation())

            # maybe dropout
            if dropout:
                layers.append(nn.Dropout(dropout))

        # final projection and activation
        layers.append(nn.Linear(node_list[-1], output_size, bias=bias))
        if final_activation:
            layers.append(final_activation())

        # build the net
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
