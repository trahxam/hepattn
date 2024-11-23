import torch
from torch.nn.attention.flex_attention import _mask_mod_signature

from hepattn.flex.utils import visualize_attention_scores


def sliding_window_mask(window_size: int) -> _mask_mod_signature:
    def mask_fn(b, h, q_idx, kv_idx):  # noqa: ARG001
        return (q_idx - kv_idx <= window_size // 2) & (kv_idx - q_idx <= window_size // 2)

    return mask_fn


def main(device: str = "cpu"):
    """Visualize the attention scores of sliding window mask mod.

    Args:
        device (str): Device to use for computation. Defaults
    """

    def make_tensor():
        return torch.ones(1, 1, 24, 8, device=device)

    query, key = make_tensor(), make_tensor()

    mask = sliding_window_mask(4)
    visualize_attention_scores(query, key, mask_mod=mask, device=device, name="sliding_window_mask")


if __name__ == "__main__":
    main()
