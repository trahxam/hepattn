from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature


def sliding_window_mask(window_size: int) -> _mask_mod_signature:
    def mask_fn(b, h, q_idx, kv_idx):  # noqa: ARG001
        return (q_idx - kv_idx <= window_size // 2) & (kv_idx - q_idx <= window_size // 2)

    return mask_fn


def sliding_window_mask_wrapped(window_size: int, q_len: Tensor) -> _mask_mod_signature:
    def mask_fn(b, h, q_idx, kv_idx):  # noqa: ARG001
        diagonal = (q_idx - kv_idx <= window_size // 2) & (kv_idx - q_idx <= window_size // 2)
        wrap = ((q_idx - kv_idx + q_len[0]) <= window_size // 2) | ((kv_idx - q_idx + q_len[0]) <= window_size // 2)
        return diagonal | wrap

    return mask_fn
