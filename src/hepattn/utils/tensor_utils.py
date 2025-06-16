import numpy as np
import torch

from torch import Tensor


def concat_tensors(tensors: list[Tensor]) -> Tensor:
    """
    Concatenates a list of tensors along the last dimension, ensuring 3D shape.
    Each tensor is checked for dimensionality. If a tensor is 2D, an extra dimension
    is added at the end to make it 3D before concatenation. All tensors are then concatenated
    along the last dimension.

    Parameters
    ----------
    tensors : list of torch.Tensor
        List of tensors to concatenate. Each tensor must be at least 2D.
        Tensors with shape (N, M) will be reshaped to (N, M, 1) before concatenation.

    Returns
    -------
    torch.Tensor
        A tensor resulting from concatenating the input tensors along the last dimension.
        The output will have shape (N, M, K), where K is the number of input tensors
        (assuming all have matching first two dimensions).
    """
    x = []

    for tensor in tensors:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        x.append(tensor)

    return torch.concatenate(x, dim=-1)


def tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array, handling device transfer and dtype conversion.
    Also handles bfloat16 correctly.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to convert.

    Returns
    -------
    numpy.ndarray
        A NumPy array with the same data as the input tensor.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    if tensor.dtype == torch.bfloat16:
        return tensor.to(torch.float16).numpy().astype(np.float16)

    if tensor.dtype == torch.float16:
        return tensor.numpy().astype(np.float16)

    return tensor.numpy()


def pad_to_size(x: Tensor, target_shape: tuple, pad_value: float | int) -> Tensor:
    """
    Pads a tensor to a specified target shape with a constant value.
    The function creates a new tensor of shape target_shape, fills it with pad_value,
    and copies the values from the input tensor x into the upper-left corner
    (i.e., starting from index 0 along each dimension). The shape of x must not
    exceed the target shape in any dimension.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to pad.
    target_shape : tuple of int
        The desired shape of the output tensor. Must have the same number of dimensions as x.
    pad_value : float or int
        The constant value to use for padding.

    Returns
    -------
    torch.Tensor
        A tensor of shape target_shape, with x copied into the top-left region and
        remaining elements filled with pad_value.
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


def pad_and_concat(items: list[Tensor], target_size: tuple[int], pad_value: float | int) -> Tensor:
    """
    Pads and concatenates a list of tensors to a uniform target size.
    Each tensor in the input list is padded to match the specified target_size,
    then all padded tensors are concatenated along a new leading dimension.

    Parameters
    ----------
    items : list of torch.Tensor
        List of tensors to be padded and concatenated.
    target_size : tuple of int
        The target size (excluding the new leading dimension) that each tensor should be padded to.
    pad_value : float or int
        The value to use for padding.

    Returns
    -------
    torch.Tensor
        A single tensor of shape (N, *target_size), where N is the number of tensors in items.
    """
    return torch.cat([pad_to_size(item, (1, *target_size), pad_value) for item in items], dim=0)
