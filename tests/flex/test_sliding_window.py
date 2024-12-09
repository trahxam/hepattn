import torch

from hepattn.flex import sliding_window_mask, sliding_window_mask_wrapped
from hepattn.flex.utils import visualize_attention_scores


def test_sliding_window(device: str = "cpu"):
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
