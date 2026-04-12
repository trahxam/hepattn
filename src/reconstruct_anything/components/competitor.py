"""CompetitorFormer modules for reducing inter-query competition/duplicates.

Implements three plug-and-play modules from:
"CompetitorFormer: Competitor Transformer for 3D Instance Segmentation"
(arXiv:2411.14179)

- QCL (Query Competition Layer): Identifies competing queries via predicted mask
  IoU and fuses leader/laggard embeddings to suppress duplicates
- RRE (Relative Relationship Encoding): Adds competition-aware bias to
  self-attention based on rank x IoU between query predictions
- RCA (Rank Cross Attention): Min-max normalizes cross-attention scores along
  query dimension to amplify dominant queries
"""

import torch
from torch import Tensor, nn


class QueryCompetitionLayer(nn.Module):
    """Query Competition Layer (QCL).

    Before each decoder layer (except the first), identifies competing query
    pairs by computing IoU between their predicted masks from the previous layer.
    Creates leader/laggard embeddings and fuses them into the query embeddings.

    Args:
        dim: Query embedding dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.fuse_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.query_update = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        query_embed: Tensor,
        pred_mask_logits: Tensor,
        pred_class_prob: Tensor,
    ) -> Tensor:
        """Update query embeddings based on competition with other queries.

        Args:
            query_embed: [B, Q, D] current query embeddings
            pred_mask_logits: [B, Q, N_hits] predicted mask logits from previous layer
            pred_class_prob: [B, Q] predicted object probability from previous layer

        Returns:
            Updated query embeddings [B, Q, D]
        """
        _B, _Q, D = query_embed.shape

        # Compute predicted masks (hard) and pairwise IoU
        pred_masks = pred_mask_logits > 0  # [B, Q, N]
        mask_float = pred_masks.float()

        # Pairwise IoU: [B, Q, Q]
        intersection = torch.bmm(mask_float, mask_float.transpose(1, 2))
        counts = mask_float.sum(-1)  # [B, Q]
        union = counts.unsqueeze(-1) + counts.unsqueeze(-2) - intersection
        iou = intersection / union.clamp(min=1)

        # For each query, find its strongest competitor (highest IoU, excluding self)
        iou_no_self = iou.clone()
        iou_no_self.diagonal(dim1=-2, dim2=-1).zero_()
        competitor_idx = iou_no_self.argmax(dim=-1)  # [B, Q]

        # Gather competitor embeddings
        competitor_embed = torch.gather(
            query_embed,
            1,
            competitor_idx.unsqueeze(-1).expand(-1, -1, D),
        )  # [B, Q, D]

        # For leaders: fuse with laggard (competitor) embedding
        # For laggards: fuse with leader (competitor) embedding
        # In both cases, we concatenate self + competitor
        fused = self.fuse_mlp(torch.cat([query_embed, competitor_embed], dim=-1))

        # Update queries
        return self.query_update(torch.cat([query_embed, fused], dim=-1))


class RelativeRelationshipEncoding(nn.Module):
    """Relative Relationship Encoding (RRE).

    Computes a competition-aware bias for self-attention based on the
    product of rank (±1) and IoU between query predictions.

    Args:
        dim: Attention head dimension.
        num_heads: Number of attention heads.
        table_size: Size of the relationship encoding lookup table.
    """

    def __init__(self, dim: int, num_heads: int, table_size: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.table_size = table_size

        # Learnable relationship encoding table: maps quantized state -> bias per head
        self.rel_table = nn.Embedding(table_size, num_heads)

    def forward(
        self,
        pred_mask_logits: Tensor,
        pred_class_prob: Tensor,
    ) -> Tensor:
        """Compute relative relationship bias for self-attention.

        Args:
            pred_mask_logits: [B, Q, N_hits] predicted mask logits
            pred_class_prob: [B, Q] predicted object probability

        Returns:
            rel_bias: [B, num_heads, Q, Q] bias to add to self-attention scores
        """
        # Compute IoU
        pred_masks = pred_mask_logits > 0
        mask_float = pred_masks.float()
        intersection = torch.bmm(mask_float, mask_float.transpose(1, 2))
        counts = mask_float.sum(-1)
        union = counts.unsqueeze(-1) + counts.unsqueeze(-2) - intersection
        iou = intersection / union.clamp(min=1)

        # Rank: +1 if query i has higher prob than j, -1 otherwise
        rank = torch.sign(pred_class_prob.unsqueeze(-1) - pred_class_prob.unsqueeze(-2))

        # Competition state = rank * IoU, range [-1, 1]
        state = rank * iou  # [B, Q, Q]

        # Quantize to indices in [0, table_size)
        indices = ((state + 1) / 2 * (self.table_size - 1)).long().clamp(0, self.table_size - 1)

        # Look up bias: [B, Q, Q, num_heads] -> [B, num_heads, Q, Q]
        return self.rel_table(indices).permute(0, 3, 1, 2)


class RankCrossAttention:
    """Rank Cross Attention (RCA) normalization.

    Replaces standard softmax in cross-attention with a rank-normalized version:
    1. Compute similarity X = Q @ K^T
    2. Min-max normalize X along query dimension (per key position)
    3. Element-wise multiply: X * X_norm
    4. Apply softmax over key dimension

    This amplifies dominant query-feature matches while suppressing weaker ones.
    """

    @staticmethod
    def apply(attn_scores: Tensor) -> Tensor:
        """Apply rank normalization to attention scores before softmax.

        Args:
            attn_scores: [B, H, Q, K] raw attention scores (Q @ K^T / sqrt(d))

        Returns:
            Modified attention scores [B, H, Q, K]
        """
        # Min-max normalize along query dimension (dim=-2)
        score_min = attn_scores.min(dim=-2, keepdim=True).values
        score_max = attn_scores.max(dim=-2, keepdim=True).values
        score_range = (score_max - score_min).clamp(min=1e-6)
        normalized = (attn_scores - score_min) / score_range

        # Element-wise multiply: amplify dominant matches
        return attn_scores * normalized
