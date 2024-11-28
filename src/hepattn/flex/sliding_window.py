import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature

from hepattn.flex.utils import visualize_attention_scores


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


def main(device: str = "cpu"):
    """Visualize the attention scores of sliding window mask mod.

    Args:
        device (str): Device to use for computation. Defaults
    """

    def make_tensor(q_len):
        return torch.ones(1, 1, q_len, 8, device=device)

    q_len = torch.tensor([24])
    query, key = make_tensor(q_len[0]), make_tensor(q_len[0])

    mask = sliding_window_mask(4)
    visualize_attention_scores(query, key, mask_mod=mask, device=device, name="sliding_window_mask")
    mask = sliding_window_mask_wrapped(6, q_len=q_len)
    visualize_attention_scores(query, key, mask_mod=mask, device=device, name="sliding_window_mask_wrap")
    query, key = make_tensor(48), make_tensor(48)
    q_len[0] = 48
    visualize_attention_scores(query, key, mask_mod=mask, device=device, name="sliding_window_mask_wrap2")


if __name__ == "__main__":
    main()
