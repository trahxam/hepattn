import torch
from torch import Tensor, nn


class HitFilter(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module,
        tasks: nn.ModuleList,
        sorter: nn.Module | None = None,
    ):
        super().__init__()

        self.input_nets = input_nets
        self.encoder = encoder
        self.sorter = sorter
        self.tasks = tasks

    @property
    def input_names(self) -> list[str]:
        return [input_net.input_name for input_net in self.input_nets]

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
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
        preds = {"final": {}}
        for task in self.tasks:
            preds["final"][task.name] = task.predict(outputs["final"][task.name])
        return preds

    def loss(self, outputs: dict, targets: dict) -> tuple[dict, dict, dict]:
        losses = {"final": {}}
        for task in self.tasks:
            losses["final"][task.name] = task.loss(outputs["final"][task.name], targets)
        return outputs, targets, losses
