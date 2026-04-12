"""Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former.
"""

import torch
from torch import Tensor, nn

from reconstruct_anything.components.decoder import DecoderLayer
from reconstruct_anything.components.flex.local_ca import sliding_window_mask_strided, sliding_window_mask_strided_wrapped, transpose_blockmask
from reconstruct_anything.components.posenc import pos_enc
from reconstruct_anything.utils.local_ca import auto_local_ca_mask
from reconstruct_anything.utils.model_utils import unmerge_inputs


class MaskFormerDecoder(nn.Module):
    def __init__(
        self,
        tasks: nn.ModuleList,
        num_queries: int,
        decoder_layer_config: dict,
        num_decoder_layers: int,
        encoder_tasks: nn.ModuleList | None = None,
        mask_attention: bool = True,
        posenc: dict[str, float] | None = None,
        local_strided_attn: bool = False,
        window_size: int = 512,
        window_wrap: bool = True,
        unified_decoding: bool = False,
        unmask_all_false: bool = True,
        dynamic_queries: bool = False,
        dynamic_query_source: str | None = None,
    ):
        """MaskFormer decoder that handles multiple decoder layers and task integration.

        Args:
            tasks: List of task modules for producing predictions from decoder outputs.
            num_queries: The number of object-level queries.
            decoder_layer_config: Configuration dictionary used to initialize each DecoderLayer.
            num_decoder_layers: The number of decoder layers to stack.
            encoder_tasks: Optional list of tasks to run after the encoder (before decoder).
            mask_attention: If True, attention masks will be used to control which input constituents are attended to.
            posenc: Optional positional encoding config with alpha and base parameters.
            local_strided_attn: If True, uses local strided window attention.
            window_size: The size of the window for local strided window attention.
            window_wrap: If True, wraps the window for local strided window attention.
            unified_decoding: If True, inputs remain merged for task processing instead of being unmerged after each layer.
            unmask_all_false: If True, queries with all-false attention masks will be unmasked to attend everywhere.
            dynamic_queries: If True, queries are initialized dynamically.
            dynamic_query_source: Name of the input type to use as the source for dynamic query initialization.
        """
        super().__init__()

        self.decoder_layers = nn.ModuleList([DecoderLayer(depth=i, **decoder_layer_config) for i in range(num_decoder_layers)])
        self.dim = decoder_layer_config["dim"]
        self.tasks = tasks
        self.encoder_tasks = encoder_tasks or nn.ModuleList()
        self._num_queries = num_queries
        self.mask_attention = mask_attention
        self.posenc = posenc
        self.local_strided_attn = local_strided_attn
        self.attn_type = decoder_layer_config.get("attn_kwargs", {}).get("attn_type", "torch")
        self.window_size = window_size
        self.window_wrap = window_wrap
        self.unified_decoding = unified_decoding
        self.dynamic_queries = dynamic_queries
        self.dynamic_query_source = dynamic_query_source
        self.unmask_all_false = unmask_all_false

        # Only initialize learned queries if not using dynamic queries
        if not dynamic_queries:
            self.initial_queries = nn.Parameter(torch.randn(self._num_queries, decoder_layer_config["dim"]))

        if self.local_strided_attn:
            assert self.attn_type in {"torch", "flex"}, (
                f"Invalid attention type when local_strided_attn is True: {self.attn_type}, must be 'torch' or 'flex'"
            )
        assert not (self.local_strided_attn and self.mask_attention), "local_strided_attn and mask_attention cannot both be True"

    def _extract_kmeans_logits(self, layer_outputs: dict[str, object], num_constituents: int) -> Tensor:
        """Extract the dense assignment logit tensor required by KMeansCrossAttention."""
        for task_outputs in layer_outputs.values():
            if not isinstance(task_outputs, dict):
                continue
            dense_logits = [
                v
                for k, v in task_outputs.items()
                if k.endswith("_logit") and isinstance(v, Tensor) and v.dim() == 3 and v.shape[-1] == num_constituents
            ]
            if dense_logits:
                return dense_logits[0]
        raise ValueError("cross_attn_mode='kmeans' requires a task output with 3D *_logit matching key length.")

    def num_queries(self, x) -> int:
        """Return the current number of queries given the input state dict."""
        if self.dynamic_queries:
            return x["query_embed"].shape[1]
        return self._num_queries

    def initialize_dynamic_queries(self, x: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Initialize queries dynamically using the `query_init` task.

        This selects hit embeddings whose predicted first-hit probability passes the task threshold,
        then keeps the top-k by probability where k is `self._num_queries`.

        Returns:
            Tuple of (query_embed, query_valid).
        """
        query_init_task = next((t for t in self.encoder_tasks if t.name == "query_init"), None)
        if query_init_task is None:
            raise ValueError("dynamic_queries=True requires a task named 'query_init' in encoder_tasks.")

        source_embed_key = f"{self.dynamic_query_source}_embed"
        source_valid_key = f"{self.dynamic_query_source}_valid"
        if source_embed_key not in x or source_valid_key not in x:
            raise ValueError(f"dynamic_queries=True requires '{source_embed_key}' and '{source_valid_key}' in decoder input dict")

        # Get predictions
        outputs = query_init_task.forward(x)
        preds = query_init_task.predict(outputs)

        prob_key = next((k for k in preds if k.endswith("_prob")), None)
        if prob_key is None:
            raise ValueError(f"query_init task '{query_init_task.name}' did not return a probability prediction.")

        source_prob = preds[prob_key]  # (B, N_source)
        if source_prob.shape[0] != 1:
            raise ValueError(f"dynamic_queries only supports batch_size=1, got {source_prob.shape[0]}")

        # Filter: probability >= threshold AND valid, then select top-k by probability
        valid_mask = (source_prob[0] >= query_init_task.threshold) & x[source_valid_key][0]
        selected_indices = torch.where(valid_mask)[0]

        if selected_indices.numel() == 0:
            selected_indices = source_prob[0].topk(self._num_queries).indices

        # If more candidates than needed, keep top-k by probability
        if selected_indices.numel() > self._num_queries:
            probs = source_prob[0, selected_indices]
            top_k_idx = probs.topk(self._num_queries).indices
            selected_indices = selected_indices[top_k_idx]

        # Sort to preserve original spatial ordering
        selected_indices = selected_indices.sort().values
        num_selected = selected_indices.numel()
        device = x[source_embed_key].device

        # get selected embeddings
        selected_constituent_embeds = x[source_embed_key][0, selected_indices].detach()

        # Pad to fixed num_queries size if fewer were selected to match the full target set
        if num_selected < self._num_queries:
            num_padding = self._num_queries - num_selected
            null_padding = torch.zeros(num_padding, self.dim, device=device, dtype=selected_constituent_embeds.dtype)
            query_embed = torch.cat([selected_constituent_embeds, null_padding], dim=0).unsqueeze(0)
            query_valid = torch.cat([
                torch.ones(num_selected, dtype=torch.bool, device=device),
                torch.zeros(num_padding, dtype=torch.bool, device=device),
            ]).unsqueeze(0)
        else:
            query_embed = selected_constituent_embeds.unsqueeze(0)
            query_valid = torch.ones(1, num_selected, dtype=torch.bool, device=device)

        return query_embed, query_valid

    def forward(self, x: dict[str, Tensor], input_names: list[str]) -> tuple[dict[str, Tensor], dict[str, dict]]:
        """Forward pass through decoder layers.

        Args:
            x: Dictionary containing embeddings and masks.
            input_names: List of input names for constructing attention masks.

        Returns:
            Tuple of (x, outputs) where outputs contains layer-wise task outputs.
        """
        batch_size = x["key_embed"].shape[0]
        num_constituents = x["key_embed"].shape[-2]

        # Generate or use pre-initialized queries
        outputs: dict[str, dict] = {"encoder": {}}
        if not self.dynamic_queries:
            x["query_embed"] = self.initial_queries.expand(batch_size, -1, -1)
        else:
            x["query_embed"], query_valid = self.initialize_dynamic_queries(x)
            x["query_mask"] = query_valid
            outputs["encoder"]["query_mask"] = query_valid

        if self.posenc:
            x["query_posenc"], x["key_posenc"] = self.generate_positional_encodings(x)

        attn_mask = None
        attn_mask_transpose = None
        if self.local_strided_attn:
            assert x["query_embed"].shape[0] == 1, "Local strided attention only supports batch size 1"
            if self.attn_type == "torch":
                attn_mask = auto_local_ca_mask(x["query_embed"], x["key_embed"], self.window_size, wrap=self.window_wrap)
            elif self.attn_type == "flex":
                device = x["query_embed"].device
                q_len = x["query_embed"].shape[1]
                kv_len = x["key_embed"].shape[1]
                dtype_float = x["query_embed"].dtype
                attn_mask = self.flex_local_ca_mask(q_len, kv_len, device, dtype_float)
                attn_mask_transpose = transpose_blockmask(attn_mask, q_tokens=q_len, kv_tokens=kv_len, dev=device)

        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            outputs[f"layer_{layer_index}"] = {}

            # if maskattention, PE should be added before generating the mask
            if self.posenc and self.mask_attention:
                x["query_embed"] = x["query_embed"] + x["query_posenc"]
                x["key_embed"] = x["key_embed"] + x["key_posenc"]

            attn_masks: dict[str, torch.Tensor] = {}

            for task in self.tasks:
                if not task.should_run_at_layer(layer_index):
                    continue

                task_outputs = task(x, outputs=outputs[f"layer_{layer_index}"])
                outputs[f"layer_{layer_index}"][task.name] = task_outputs

                # Collect attention masks from different tasks
                task_attn_masks = task.attn_mask(task_outputs)
                for input_name, task_attn_mask in task_attn_masks.items():
                    if input_name in attn_masks:
                        attn_masks[input_name] |= task_attn_mask
                    else:
                        attn_masks[input_name] = task_attn_mask

            # Construct the full attention mask for MaskAttention decoder
            if attn_masks and self.mask_attention:
                if self.unified_decoding:
                    if len(attn_masks) > 1:
                        raise ValueError(f"In merged input mode, expected only one attention mask, got {len(attn_masks)}")
                    attn_mask = next(iter(attn_masks.values()))
                    if attn_mask.dim() == 2:
                        attn_mask = attn_mask.unsqueeze(-1).expand(-1, -1, num_constituents)
                else:
                    attn_mask = torch.full((batch_size, self.num_queries(x), num_constituents), False, device=x["key_embed"].device)
                    for input_name, task_attn_mask in attn_masks.items():
                        attn_mask[x[f"key_is_{input_name}"].unsqueeze(1).expand_as(attn_mask)] = task_attn_mask.flatten()

                attn_mask = attn_mask.detach()
                if self.unmask_all_false:
                    attn_mask = torch.where(torch.all(~attn_mask, dim=-1, keepdim=True), True, attn_mask)

            if (attn_mask is not None) and self.attn_type != "flex":
                outputs[f"layer_{layer_index}"]["attn_mask"] = attn_mask

            logits = None
            if decoder_layer.cross_attn_mode == "kmeans":
                logits = self._extract_kmeans_logits(outputs[f"layer_{layer_index}"], num_constituents)

            # Update the keys and queries
            x["query_embed"], x["key_embed"] = decoder_layer(
                x["query_embed"],
                x["key_embed"],
                attn_mask=attn_mask,
                q_mask=x.get("query_mask"),
                kv_mask=x.get("key_valid"),
                query_posenc=x["query_posenc"] if self.posenc else None,
                key_posenc=x["key_posenc"] if self.posenc else None,
                attn_mask_transpose=attn_mask_transpose,
                logits=logits,
            )

            # update the individual input constituent representations only if not in merged input mode
            if not self.unified_decoding:
                x = unmerge_inputs(x, input_names)

        return x, outputs

    def flex_local_ca_mask(self, q_len: int, kv_len: int, device, dtype_float):
        """Build a flex-attention BlockMask for local strided cross-attention."""
        stride = kv_len / q_len
        window_mask_func = sliding_window_mask_strided_wrapped if self.window_wrap else sliding_window_mask_strided
        return window_mask_func(self.window_size, stride=stride, q_len=q_len, kv_len=kv_len, device=str(device))

    def generate_positional_encodings(self, x: dict):
        """Compute symmetric positional encodings for queries and keys."""
        idx = torch.arange(self.num_queries(x), device=x["query_embed"].device, dtype=x["query_embed"].dtype)
        x["query_phi"] = 2 * torch.pi * idx / self.num_queries(x)
        query_posenc = pos_enc(x["query_phi"], self.dim, self.posenc["alpha"], self.posenc["base"], symmetric=True)
        key_posenc = pos_enc(x["key_phi"], self.dim, self.posenc["alpha"], self.posenc["base"], symmetric=True)
        return query_posenc, key_posenc
