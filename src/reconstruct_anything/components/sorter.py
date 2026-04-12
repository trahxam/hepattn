import torch
from torch import Tensor, nn


class Sorter(nn.Module):
    """Sort input constituents by a specified field before encoding.

    This module reorders input embeddings and associated tensors along the
    sequence dimension so that constituents are arranged by the given field
    (e.g. ``phi``).  The sort order is recorded so that target masks can be
    reordered consistently via :meth:`sort_targets`.

    Args:
        input_sort_field: Name of the field used as the sort key
            (e.g. ``"phi"``).
    """

    def __init__(self, input_sort_field: str):
        super().__init__()
        self.input_sort_field = input_sort_field

    def sort_inputs(self, inputs: dict[str, Tensor], input_names: list[str]) -> dict[str, Tensor]:
        """Sort input tensors by the configured field for each input type.

        Args:
            inputs: Dictionary of input tensors.
            input_names: List of input type names to sort.

        Returns:
            The same ``inputs`` dict with all tensors reordered.
        """
        all_names = [*input_names, "key"]
        sort_idxs = {}

        for input_name in all_names:
            sort_idx = torch.argsort(inputs[f"{input_name}_{self.input_sort_field}"], dim=-1)
            sort_idxs[input_name] = sort_idx

            for key, x in inputs.items():
                if x is None or input_name not in key:
                    continue

                # embeddings
                if key == f"{input_name}_embed":
                    sort_dim = 1
                    this_sort_idx = sort_idx.unsqueeze(-1).expand_as(x)

                # input type masks
                elif "key_is_" in key:
                    if input_name != "key":
                        continue
                    sort_dim = 1
                    this_sort_idx = sort_idx

                # normal inputs
                elif key.startswith(input_name):
                    sort_dim = 1
                    this_sort_idx = sort_idx

                else:
                    raise ValueError(f"Unexpected key {key} for input type {input_name}")

                shape_before = x.shape
                inputs[key] = torch.gather(x, sort_dim, this_sort_idx)
                assert inputs[key].shape == shape_before, f"Shape mismatch after sorting: {inputs[key].shape} != {shape_before} for key {key}"

        return inputs

    def sort_targets(self, targets: dict, sort_fields: dict[str, Tensor], input_names: list[str]) -> dict:
        """Sort target tensors to match the ordering applied to inputs.

        Only 3D hit-assignment masks (shape ``[B, tracks, hits]``) are sorted.
        2D targets are skipped — per-track scalars (e.g. ``sudo_num_pix``)
        don't depend on hit ordering, and per-hit fields (e.g. ``pix_valid``)
        are already sorted by :meth:`sort_inputs`.

        Args:
            targets: Dictionary of target tensors.
            sort_fields: Dictionary containing the sort-field values used to derive the sort order.
            input_names: List of input type names whose targets should be sorted.

        Returns:
            The same ``targets`` dict with all tensors reordered.
        """
        for input_name in input_names:
            sort_idx = torch.argsort(sort_fields[f"{input_name}_{self.input_sort_field}"], dim=-1)

            for key, x in targets.items():
                if x is None or input_name not in key:
                    continue

                # Only sort 3D hit-assignment masks [B, tracks, hits] along the hits dim
                if x.ndim != 3:
                    continue

                this_sort_idx = sort_idx.unsqueeze(1).expand_as(x)

                shape_before = x.shape
                targets[key] = torch.gather(x, 2, this_sort_idx)
                assert targets[key].shape == shape_before, f"Shape mismatch after sorting: {targets[key].shape} != {shape_before} for key {key}"

        return targets
