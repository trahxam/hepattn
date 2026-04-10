from typing import Any

import torch
from torch import Tensor, nn

from hepattn.models.decoder import MaskFormerDecoder
from hepattn.utils.model_utils import unmerge_inputs


class MaskFormer(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module,
        decoder: MaskFormerDecoder,
        target_object: str = "particle",
        pooling: nn.Module | None = None,
        matcher: nn.Module | None = None,
        sorter: nn.Module | None = None,
    ):
        """MaskFormer: a modular transformer architecture for multi-task object reconstruction.

        Args:
            input_nets: A list of input modules, each responsible for embedding a specific constituent type.
            encoder: Encoder module that processes merged constituent embeddings.
            decoder: Decoder module containing decoder layers, tasks, and query initialization.
            target_object: The target object name used to mark valid/invalid objects during matching.
            pooling: Optional pooling module for aggregating features from the input constituents.
            matcher: Module for matching predictions to targets (e.g., Hungarian algorithm).
            sorter: Optional sorter module for reordering inputs before processing.
        """
        super().__init__()

        self.input_nets = input_nets
        self.encoder = encoder
        self.decoder = decoder
        self.pooling = pooling
        self.target_object = target_object
        self.matcher = matcher
        self.sorter = sorter

        assert "key" not in self.input_names, "'key' input name is reserved."
        assert "query" not in self.input_names, "'query' input name is reserved."
        assert not any("_" in name for name in self.input_names), "Input names cannot contain underscores."

    @property
    def input_names(self) -> list[str]:
        """Names of all registered input constituent types."""
        return [input_net.input_name for input_net in self.input_nets]

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, dict[str, dict[str, Tensor]]]:
        """Embed, encode, decode, and run all tasks on the inputs.

        Args:
            inputs: Dictionary of input tensors keyed by ``{input_name}_{field}``.

        Returns:
            Nested dict keyed by stage (``'encoder'``, ``'layer_N'``, ``'final'``)
            then by task name containing raw task output dicts.
        """
        batch_size = inputs[self.input_names[0] + "_valid"].shape[0]
        x = {"inputs": inputs}

        # Track per-input slices into the merged key tensor.
        # Used to keep dynamic_queries compatible with unified_decoding.
        key_slices: dict[str, slice] = {}
        key_start = 0

        # Embed the input constituents
        for input_net in self.input_nets:
            input_name = input_net.input_name
            x[input_name + "_embed"] = input_net(inputs)
            x[input_name + "_valid"] = inputs[input_name + "_valid"]

            n_objects = x[input_name + "_embed"].shape[-2]
            key_slices[input_name] = slice(key_start, key_start + n_objects)
            key_start += n_objects

            # These slices can be used to pick out specific
            # objects after we have merged them all together
            if not self.decoder.unified_decoding:
                device = inputs[input_name + "_valid"].device
                mask = torch.cat([torch.full((inputs[i + "_valid"].shape[-1],), i == input_name, device=device) for i in self.input_names], dim=-1)
                x[f"key_is_{input_name}"] = mask.unsqueeze(0).expand(batch_size, -1)

        # Merge the input constituents and the padding mask into a single set
        x["key_embed"] = torch.concatenate([x[input_name + "_embed"] for input_name in self.input_names], dim=-2)
        x["key_valid"] = torch.concatenate([x[input_name + "_valid"] for input_name in self.input_names], dim=-1)
        # Preserve a non-None version for downstream logic that expects a tensor mask.
        x["key_valid_full"] = x["key_valid"]

        # If all key_valid are true, then we can just set it to None, however,
        # if we are using flash-varlen, we have to always provide a kv_mask argument
        if batch_size == 1 and x["key_valid"].all() and self.encoder.attn_type != "flash-varlen":
            x["key_valid"] = None

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
        x["key_embed"] = self.encoder(x["key_embed"], kv_mask=x.get("key_valid"))

        # Keep dynamic query initialization compatible with unified decoding by ensuring
        # source_embed/source_valid refer to *post-encoder* features.
        if self.decoder.dynamic_queries and self.decoder.unified_decoding:
            if self.sorter is not None:
                raise ValueError("dynamic_queries with unified_decoding is not supported when sorter is enabled")
            dqs = self.decoder.dynamic_query_source
            if dqs not in key_slices:
                raise ValueError(f"dynamic_queries=True requires an input named '{dqs}'")
            source_slice = key_slices[dqs]
            x[f"{dqs}_embed"] = x["key_embed"][:, source_slice, :]
            x[f"{dqs}_valid"] = x["key_valid_full"][:, source_slice]

        # Unmerge the updated features back into the separate input types only if not doing unified decoding
        if not self.decoder.unified_decoding:
            x = unmerge_inputs(x, self.input_names)

        # Run encoder tasks
        outputs = {"encoder": {}}
        for task in self.decoder.encoder_tasks:
            outputs["encoder"][task.name] = task(x)

        # Pass through decoder layers
        x, decoder_outputs = self.decoder(x, self.input_names)
        outputs["encoder"].update(decoder_outputs.pop("encoder", {}))
        outputs.update(decoder_outputs)

        # Do any pooling if desired
        if self.pooling is not None:
            x_pooled = self.pooling(x[f"{self.pooling.input_name}_embed"], x[f"{self.pooling.input_name}_valid"])
            x[f"{self.pooling.output_name}_embed"] = x_pooled

        # Get the final outputs
        outputs["final"] = {}
        for task in self.decoder.tasks:
            outputs["final"][task.name] = task(x, outputs=outputs["final"])

        # store info about the input sort field for each input type
        if self.sorter is not None:
            sort = self.sorter.input_sort_field
            sort_dict = {f"{name}_{sort}": inputs[f"{name}_{sort}"] for name in self.input_names}
            outputs["final"][sort] = sort_dict

        return outputs

    def predict(self, outputs: dict) -> dict:
        """Takes the raw model outputs and produces predictions."""
        preds: dict[str, dict[str, Any]] = {}

        # Get query_mask from encoder outputs for masking padded queries in predictions
        query_mask = outputs.get("encoder", {}).get("query_mask")

        for layer_name, layer_outputs in outputs.items():
            if layer_name.startswith("_"):
                continue

            preds[layer_name] = {}

            if layer_name == "encoder":
                for task in self.decoder.encoder_tasks:
                    if task.name not in layer_outputs:
                        continue
                    preds[layer_name][task.name] = task.predict(layer_outputs[task.name])
            else:
                for task in self.decoder.tasks:
                    if task.name not in layer_outputs:
                        continue
                    preds[layer_name][task.name] = task.predict(layer_outputs[task.name], query_mask=query_mask)

        return preds

    def _prepare_targets_and_outputs(self, outputs: dict, targets: dict) -> tuple[dict, dict, dict]:
        """Split outputs into encoder/decoder parts and augment targets with query mask."""
        encoder_outputs = {"encoder": outputs["encoder"]} if "encoder" in outputs else {}
        decoder_outputs = {k: v for k, v in outputs.items() if k != "encoder"}

        if "encoder" in outputs and "query_mask" in outputs["encoder"] and "query_mask" not in targets:
            targets = targets.copy()
            targets["query_mask"] = outputs["encoder"]["query_mask"]

        # Sort targets if using a sorter
        if self.sorter is not None:
            targets = self.sorter.sort_targets(targets, decoder_outputs["final"][self.sorter.input_sort_field], self.input_names)

        return targets, encoder_outputs, decoder_outputs

    def _compute_encoder_losses(self, encoder_outputs: dict, targets: dict) -> dict[str, dict[str, Tensor]]:
        """Compute losses for encoder tasks (no matching required)."""
        losses: dict[str, dict[str, Tensor]] = {}
        for layer_name, layer_outputs in encoder_outputs.items():
            losses[layer_name] = {}
            for task in self.decoder.encoder_tasks:
                if task.name not in layer_outputs:
                    continue
                losses[layer_name][task.name] = task.loss(layer_outputs[task.name], targets, layer_outputs=layer_outputs)
        return losses

    def _compute_decoder_costs(self, decoder_outputs: dict, targets: dict) -> dict[str, Tensor]:
        """Compute costs for decoder layers by aggregating task costs."""
        costs = {}

        for layer_name, layer_outputs in decoder_outputs.items():
            layer_costs = None

            for task in self.decoder.tasks:
                if task.name not in layer_outputs:
                    continue

                task_costs = task.cost(layer_outputs[task.name], targets)

                for cost in task_costs.values():
                    if layer_costs is None:
                        layer_costs = cost
                    else:
                        layer_costs += cost

            if layer_costs is not None:
                layer_costs = layer_costs.detach()

            costs[layer_name] = layer_costs

        return costs

    def _match_and_permute_outputs(self, decoder_outputs: dict, costs: dict[str, Tensor], targets: dict) -> None:
        """Perform optimal matching and permute decoder outputs accordingly."""
        layer_names = list(costs.keys())
        num_layers = len(layer_names)

        if num_layers > 0:
            stacked_costs = torch.stack([costs[name] for name in layer_names], dim=0)
            batch_size = stacked_costs.shape[1]
            num_pred = stacked_costs.shape[2]
            num_target = stacked_costs.shape[3]

            stacked_costs = stacked_costs.reshape(num_layers * batch_size, num_pred, num_target)

            target_valid = targets[f"{self.target_object}_valid"]
            stacked_target_valid = target_valid.unsqueeze(0).expand(num_layers, -1, -1).reshape(num_layers * batch_size, -1)

            query_mask = targets.get("query_mask")
            stacked_query_valid = None
            if query_mask is not None:
                stacked_query_valid = query_mask.unsqueeze(0).expand(num_layers, -1, -1).reshape(num_layers * batch_size, -1)

            stacked_pred_idxs = self.matcher(stacked_costs, stacked_target_valid, stacked_query_valid)
            stacked_pred_idxs = stacked_pred_idxs.view(num_layers, batch_size, num_pred)

            batch_idxs_expanded = torch.arange(batch_size, device=stacked_pred_idxs.device).unsqueeze(1)

            for layer_idx, layer_name in enumerate(layer_names):
                pred_idxs = stacked_pred_idxs[layer_idx]

                for task in self.decoder.tasks:
                    if not task.should_permute_outputs(layer_name, decoder_outputs[layer_name]):
                        continue

                    for output_name in task.outputs:
                        output_tensor = decoder_outputs[layer_name][task.name][output_name]
                        decoder_outputs[layer_name][task.name][output_name] = output_tensor[batch_idxs_expanded, pred_idxs]

    def _compute_decoder_losses(self, decoder_outputs: dict, targets: dict) -> dict[str, dict[str, Tensor]]:
        """Compute final losses for decoder tasks using permuted outputs."""
        losses: dict[str, dict[str, Tensor]] = {}

        for layer_name, layer_outputs in decoder_outputs.items():
            losses[layer_name] = {}

            for task in self.decoder.tasks:
                if task.name not in layer_outputs:
                    continue

                task_losses = task.loss(layer_outputs[task.name], targets, layer_outputs=layer_outputs)
                losses[layer_name][task.name] = task_losses

        return losses

    def loss(self, outputs: dict, targets: dict) -> tuple[dict, dict, dict]:
        """Computes the loss using Hungarian matching to align predictions with targets."""
        targets, encoder_outputs, decoder_outputs = self._prepare_targets_and_outputs(outputs, targets)
        losses = self._compute_encoder_losses(encoder_outputs, targets)
        costs = self._compute_decoder_costs(decoder_outputs, targets)
        self._match_and_permute_outputs(decoder_outputs, costs, targets)
        decoder_losses = self._compute_decoder_losses(decoder_outputs, targets)
        losses.update(decoder_losses)
        return outputs, targets, losses
