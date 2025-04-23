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
        return tensor.numpy().astype(np.bool)
    else:
        raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")
