import torch
from torch import Tensor, nn


class HitFilter(nn.Module):
    """Transformer model for per-hit classification (e.g. noise filtering).

    Embeds, optionally sorts, encodes, then runs classification tasks on each hit.
    """

    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module,
        tasks: nn.ModuleList,
        sorter: nn.Module | None = None,
    ):
        """Initialize HitFilter.

        Args:
            input_nets: List of input embedding modules, one per input type.
            encoder: Encoder module applied to the merged hit embeddings.
            tasks: List of per-hit task modules.
            sorter: Optional sorter module for ordering hits before encoding.
        """
        super().__init__()

        self.input_nets = input_nets
        self.encoder = encoder
        self.sorter = sorter
        self.tasks = tasks

    @property
    def input_names(self) -> list[str]:
        """Names of all registered input types."""
        return [input_net.input_name for input_net in self.input_nets]

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Embed, encode, and run tasks on input hits.

        Args:
            inputs: Dictionary of input tensors keyed by ``{input_name}_{field}``.

        Returns:
            Dictionary with a ``'final'`` key containing per-task output dicts.
        """
        x = {}

        # Embed the input constituents
        for input_net in self.input_nets:
            input_name = input_net.input_name
            x[input_name + "_embed"] = input_net(inputs)
            x[input_name + "_valid"] = inputs[input_name + "_valid"]

            device = inputs[input_name + "_valid"].device
            x[f"key_is_{input_name}"] = torch.cat(
                [torch.full((inputs[i + "_valid"].shape[-1],), i == input_name, device=device, dtype=torch.bool) for i in self.input_names], dim=-1
            )

        # Merge the input constituents and the padding mask into a single set
        x["key_embed"] = torch.concatenate([x[input_name + "_embed"] for input_name in self.input_names], dim=-2)
        x["key_valid"] = torch.concatenate([x[input_name + "_valid"] for input_name in self.input_names], dim=-1)

        # Sort inputs if a sorter is provided
        if self.sorter is not None:
            x[f"key_{self.sorter.input_sort_field}"] = torch.concatenate(
                [inputs[input_name + "_" + self.sorter.input_sort_field] for input_name in self.input_names], dim=-1
            )
            for input_name in self.input_names:
                field = f"{input_name}_{self.sorter.input_sort_field}"
                x[field] = inputs[field]
            x = self.sorter.sort_inputs(x, self.input_names)

        # Pass merged input constituents through the encoder
        if self.encoder is not None:
            x["key_embed"] = self.encoder(x["key_embed"], kv_mask=x.get("key_valid"))

        # Unmerge the updated features back into the separate input types
        for input_name in self.input_names:
            x[input_name + "_embed"] = x["key_embed"][..., x[f"key_is_{input_name}"], :]

        outputs = {"final": {}}
        for task in self.tasks:
            outputs["final"][task.name] = task(x)

        return outputs

    def predict(self, outputs: dict) -> dict:
        """Convert raw task outputs to predictions.

        Args:
            outputs: Raw outputs from ``forward``.

        Returns:
            Dictionary with a ``'final'`` key containing per-task predictions.
        """
        preds = {"final": {}}
        for task in self.tasks:
            preds["final"][task.name] = task.predict(outputs["final"][task.name])
        return preds

    def loss(self, outputs: dict, targets: dict) -> tuple[dict, dict, dict]:
        """Compute per-task losses.

        Args:
            outputs: Raw outputs from ``forward``.
            targets: Ground-truth target dictionary.

        Returns:
            Tuple of (outputs, targets, losses) where losses is keyed by ``'final'``
            and then by task name.
        """
        losses = {"final": {}}
        for task in self.tasks:
            losses["final"][task.name] = task.loss(outputs["final"][task.name], targets)
        return outputs, targets, losses
