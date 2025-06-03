import numpy as np
import torch
from torch import Tensor


def concat_tensors(tensors: list[Tensor]) -> Tensor:
    x = []

    for tensor in tensors:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        x.append(tensor)

    return torch.concatenate(x, dim=-1)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    # Handle device transfer if tensor is on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    if tensor.dtype == torch.bfloat16:
        return tensor.to(torch.float16).numpy().astype(np.float16)

    if tensor.dtype == torch.float16:
        return tensor.numpy().astype(np.float16)

    return tensor.numpy()


def pad_to_size(x: torch.Tensor, target_shape: tuple, pad_value: float):
    """
    Pads a tensor `x` to exactly match `target_shape`, using `pad_value`.
    Works even if some dimensions of `x` are zero. If x is already the
    right shape, returns x unchanged. If any dimension of x is bigger
    than target_shape, raises a ValueError.

    Args:
        x           (torch.Tensor): any shaped tensor
        target_shape (tuple[int]):   desired shape (must have same length as x.dim())
        pad_value    (float): default fill for the padded region

    Returns:
        torch.Tensor of shape `target_shape`, where the upper left block is x
        and the rest is `pad_value`.

    Raises:
        ValueError if len(target_shape) != x.dim() or if any target < current
    """
    current_shape = tuple(x.shape)
    if len(target_shape) != x.dim():
        raise ValueError(f"Target shape must have the same number of dimensions as x: {current_shape} vs {target_shape}")

    # Check if any target dimension is smaller than x
    for i, (cur, tgt) in enumerate(zip(current_shape, target_shape, strict=False)):
        if cur > tgt:
            raise ValueError(f"Cannot pad: dimension {i} of x is {cur}, which is larger than target {tgt}.")

    # If x is already the correct shape, just return it
    if current_shape == target_shape:
        return x

    # Make a new tensor of exactly target_shape, filled with pad_value
    new_tensor = x.new_full(target_shape, pad_value)

    # Build a tuple of slice objects to index the upper left corner
    index_slices = tuple(slice(0, cur) for cur in current_shape)

    # Copy x into that region
    new_tensor[index_slices] = x

    return new_tensor
