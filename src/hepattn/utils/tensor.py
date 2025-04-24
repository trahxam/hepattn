import torch
import numpy as np


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    # Handle device transfer if tensor is on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Handle tensor types and convert accordingly
    if tensor.dtype == torch.float32:
        return tensor.numpy().astype(np.float32)
    elif tensor.dtype == torch.float64:
        return tensor.numpy().astype(np.float64)
    elif tensor.dtype == torch.float16:
        return tensor.numpy().astype(np.float16)
    elif tensor.dtype == torch.bfloat16:
        return tensor.to(torch.float16).numpy().astype(np.float16)
    elif tensor.dtype == torch.int64:
        return tensor.numpy().astype(np.int64)
    elif tensor.dtype == torch.int32:
        return tensor.numpy().astype(np.int32)
    elif tensor.dtype == torch.int16:
        return tensor.numpy().astype(np.int16)
    elif tensor.dtype == torch.int8:
        return tensor.numpy().astype(np.int8)
    elif tensor.dtype == torch.bool:
        return tensor.numpy().astype(bool)
    else:
        raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")


def pad_to_size(x, target_shape, pad_value):
    """
    Pads a tensor to match the given target shape.
    """
    # Get the current shape of the tensor
    current_shape = x.shape
    
    if len(target_shape) != x.dim():
        raise ValueError(f"Target size must match input tensor dimensions: {x.shape} vs {target_shape}")
    
    # Calculate padding for each dimension
    padding = []
    for i in range(len(current_shape)):
        # Don't pad this dimension
        if target_shape[i] == -1 or current_shape[i] == target_shape[i]:
            padding.append((0, 0))

        # Need padding to match the target size
        elif current_shape[i] < target_shape[i]:
            padding.append((0, target_shape[i] - current_shape[i]))

        elif current_shape[i] > target_shape[i]:
            raise ValueError(f"Target size {target_shape[i]} smaller than current size {current_shape[i]} at dimension {i}")
    
    # Apply padding to the tensor (pad in the reverse order)
    padding = [item for sublist in reversed(padding) for item in sublist]
    
    return torch.nn.functional.pad(x, padding, value=pad_value)
