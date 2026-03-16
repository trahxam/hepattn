"""Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former.
"""

from functools import partial
from typing import Literal

import torch
from torch import Tensor, nn

from hepattn.flex.fast_local_ca import build_strided_sliding_window_blockmask
from hepattn.flex.local_ca import sliding_window_mask_strided, sliding_window_mask_strided_wrapped, transpose_blockmask
from hepattn.models.attention import Attention
from hepattn.models.dense import Dense
from hepattn.models.encoder import Residual
from hepattn.models.norm import get_hybrid_norm_config
from hepattn.models.posenc import pos_enc_symmetric
from hepattn.models.task import IncidenceRegressionTask, ObjectClassificationTask
from hepattn.utils.kmeans_ca import KMeansCrossAttention
from hepattn.utils.local_ca import auto_local_ca_mask
from hepattn.utils.model_utils import unmerge_inputs


class MaskFormerDecoder(nn.Module):
    def __init__(
        self,
        num_queries: int,
        decoder_layer_config: dict,
        num_decoder_layers: int,
        mask_attention: bool = True,
        use_query_masks: bool = False,
        posenc: dict[str, float] | None = None,
        local_strided_attn: bool = False,
        window_size: int = 512,
        window_wrap: bool = True,
        fast_local_ca: bool = False,
        block_size: int = 128,
        unified_decoding: bool = False,
        phi_shift: float = 0.0,
        unmask_all_false: bool = False,
        kmeans_affinity_task: str | list[str] | None = None,
        per_input_decoding: bool = False,
        num_input_types: int | None = None,
        query_dim_slices: dict[str, list[int]] | None = None,
        split_query_types: list[str] | None = None,
        cross_query_attn: bool = False,
        num_tracking_layers: int = 0,
        tracking_input_names: list[str] | None = None,
        query_slot_ranges: dict[str, list[int]] | None = None,
    ):
        """MaskFormer decoder that handles multiple decoder layers and task integration.

        Args:
            num_queries: The number of object-level queries.
            decoder_layer_config: Configuration dictionary used to initialize each MaskFormerDecoderLayer.
            num_decoder_layers: The number of decoder layers to stack.
            mask_attention: If True, attention masks will be used to control which input constituents are attended to.
            use_query_masks: If True, predicted query masks will be used to control which queries are valid.
            posenc: Optional module for positional encoding.
            local_strided_attn: If True, uses local strided window attention.
            window_size: The size of the window for local strided window attention.
            window_wrap: If True, wraps the window for local strided window attention.
            fast_local_ca: If True, uses fast local CA.
            block_size: The size of the block for fast local CA.
            unified_decoding: If True, inputs remain merged for task processing instead of being unmerged after each layer.
            phi_shift: The shift in the phi angle for positional encoding.
            unmask_all_false: If True, queries with all-false attention masks will be unmasked to attend everywhere.
            kmeans_affinity_task: If using cross_attn_mode="kmeans", optionally select which task name(s)
                provide affinity logits via affinity(). If None, all task affinities in that layer are combined.
            per_input_decoding: If True, each input type gets its own dedicated decoder layer per macro-layer.
                Queries are updated sequentially by attending to each hit type through its own decoder.
                Requires unified_decoding=False and num_input_types to be set.
            num_input_types: Number of input hit types. Required when per_input_decoding=True.
            query_dim_slices: Optional mapping from input type name to [start, end] index range (exclusive end)
                defining which slice of the query embedding is exclusively updated by that hit type.
                Requires per_input_decoding=True. All dims from max(end) onwards are treated as shared and
                are updated by every hit type. Hit types not listed here only update the shared dims.
                Example: {"vtxd": [0, 32], "trkr": [32, 64]} with dim=256 gives vtxd dims [0:32],
                trkr dims [32:64], and shared dims [64:256] updated by all types.
            split_query_types: Optional list of input type names that each receive their own independent
                learned initial query embedding (nn.Parameter). Requires per_input_decoding=True.
                Types not listed fall back to the shared initial_queries. With full gradient isolation:
                the tracker loss cannot contaminate calo initial query params and vice versa.
                Per-type queries are stored in x["query_embed_{name}"] for downstream task use.
                x["query_embed"] is set to the mean of all per-type queries for shared task compatibility.
            cross_query_attn: If True (and split_query_types is set), add one self-attention + FFN layer
                per macro-decoder-layer that operates over all per-type queries concatenated along the
                sequence dim. This allows information to flow between per-type query sets while keeping
                the initial query parameters gradient-isolated.
        """
        super().__init__()

        self.per_input_decoding = per_input_decoding
        self.num_tracking_layers = num_tracking_layers
        self.tracking_input_names = tracking_input_names or []
        self.query_slot_ranges = query_slot_ranges
        if per_input_decoding:
            if num_input_types is None:
                raise ValueError("num_input_types must be specified when per_input_decoding=True")
            if unified_decoding:
                raise ValueError("per_input_decoding is incompatible with unified_decoding")
            if local_strided_attn:
                raise ValueError("per_input_decoding is incompatible with local_strided_attn")
            num_tracking_types = len(self.tracking_input_names)
            depth_offset = num_tracking_layers * num_tracking_types
            # Tracking phase layers: num_tracking_layers x num_tracking_types
            if num_tracking_layers > 0:
                self.tracking_decoder_layers = nn.ModuleList([
                    nn.ModuleList([
                        MaskFormerDecoderLayer(depth=i * num_tracking_types + j, **decoder_layer_config)
                        for j in range(num_tracking_types)
                    ])
                    for i in range(num_tracking_layers)
                ])
            # Combined phase layers: num_decoder_layers x num_input_types
            # decoder_layers[i][j] = layer for macro-layer i, input type j
            self.decoder_layers = nn.ModuleList([
                nn.ModuleList([
                    MaskFormerDecoderLayer(depth=depth_offset + i * num_input_types + j, **decoder_layer_config)
                    for j in range(num_input_types)
                ])
                for i in range(num_decoder_layers)
            ])
        else:
            self.decoder_layers = nn.ModuleList([MaskFormerDecoderLayer(depth=i, **decoder_layer_config) for i in range(num_decoder_layers)])

        if query_dim_slices is not None and not per_input_decoding:
            raise ValueError("query_dim_slices requires per_input_decoding=True")
        self.query_dim_slices = query_dim_slices
        self.query_shared_start = max(s[1] for s in query_dim_slices.values()) if query_dim_slices else 0
        self.dim = decoder_layer_config["dim"]
        self.tasks: list | None = None  # Will be set by MaskFormer
        self.num_queries = num_queries

        # Split initial queries: separate learned nn.Parameter per input type
        self.split_query_types = split_query_types
        if split_query_types is not None:
            if not per_input_decoding:
                raise ValueError("split_query_types requires per_input_decoding=True")
            self.per_type_initial_queries = nn.ParameterDict({
                name: nn.Parameter(torch.randn(num_queries, decoder_layer_config["dim"]))
                for name in split_query_types
            })

        # Cross-query self-attention: one SA+FFN layer per macro-layer over concatenated per-type queries
        # Covers both tracking and combined phases: entries 0..num_tracking_layers-1 are tracking,
        # entries num_tracking_layers..num_tracking_layers+num_decoder_layers-1 are combined.
        self.cross_query_attn = cross_query_attn and split_query_types is not None
        if self.cross_query_attn:
            _dim = decoder_layer_config["dim"]
            _norm = decoder_layer_config.get("norm", "LayerNorm")
            _hybrid_norm = decoder_layer_config.get("hybrid_norm", False)
            _attn_kwargs = decoder_layer_config.get("attn_kwargs", {})
            _dense_kwargs = decoder_layer_config.get("dense_kwargs", {})
            _attn_norm, _dense_post_norm, _ = get_hybrid_norm_config(_norm, 0, _hybrid_norm, False)
            _residual = partial(Residual, dim=_dim)
            total_cqa_layers = num_tracking_layers + num_decoder_layers
            self.cross_query_sa_layers = nn.ModuleList([
                nn.ModuleList([
                    _residual(Attention(_dim, **_attn_kwargs), norm=_attn_norm),
                    _residual(Dense(_dim, **_dense_kwargs), norm=_norm, post_norm=_dense_post_norm),
                ])
                for _ in range(total_cqa_layers)
            ])
        self.mask_attention = mask_attention
        self.use_query_masks = use_query_masks
        self.posenc = posenc
        self.local_strided_attn = local_strided_attn
        self.attn_type = decoder_layer_config.get("attn_kwargs", {}).get("attn_type", "torch")
        self.window_size = window_size
        self.window_wrap = window_wrap
        self.unified_decoding = unified_decoding
        self.initial_queries = nn.Parameter(torch.randn(self.num_queries, decoder_layer_config["dim"]))
        self.fast_local_ca = fast_local_ca
        self.block_size = block_size
        self.phi_shift = phi_shift
        self.unmask_all_false = unmask_all_false
        self.kmeans_affinity_task = kmeans_affinity_task

        if self.local_strided_attn:
            assert self.attn_type in {"torch", "flex"}, (
                f"Invalid attention type when local_strided_attn is True: {self.attn_type}, must be 'torch' or 'flex'"
            )
        assert not (self.local_strided_attn and self.mask_attention), "local_strided_attn and mask_attention cannot both be True"

    def forward(self, x: dict[str, Tensor], input_names: list[str]) -> tuple[dict[str, Tensor], dict[str, dict]]:
        """Forward pass through decoder layers.

        Args:
            x: Dictionary containing embeddings and masks.
            input_names: List of input names for constructing attention masks.

        Returns:
            Tuple containing updated embeddings and outputs from each decoder layer and final outputs.

        Raises:
            ValueError: If in merged input mode and multiple attention masks are provided.
        """
        batch_size = x["key_embed"].shape[0]
        num_constituents = x["key_embed"].shape[-2]

        # Generate the queries that represent objects
        x["query_embed"] = self.initial_queries.expand(batch_size, -1, -1)
        x["query_valid"] = torch.full((batch_size, self.num_queries), True, device=x["query_embed"].device)

        # For split-query mode: initialise per-type query embeddings from separate parameters
        if self.split_query_types is not None:
            for name in input_names:
                if name in self.per_type_initial_queries:
                    x[f"query_embed_{name}"] = self.per_type_initial_queries[name].expand(batch_size, -1, -1)
                else:
                    x[f"query_embed_{name}"] = x["query_embed"]
            # Shared query = mean of per-type queries; used by tasks for class prediction etc.
            x["query_embed"] = torch.stack([x[f"query_embed_{n}"] for n in input_names], dim=0).mean(0)

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

        outputs: dict[str, dict] = {}

        if self.per_input_decoding:
            # Helper to run one macro-layer's worth of per-type decoder updates.
            # Returns the layer output dict (filled with task outputs for active tasks).
            def _run_per_input_macro_layer(
                layer_index: int,
                type_layers: nn.ModuleList,
                active_type_names: list[str],
                active_task_stages,
                cqa_index: int,
            ) -> dict:
                layer_out: dict = {}
                query_mask = None

                assert self.tasks is not None
                for task in self.tasks:
                    if not task.has_intermediate_loss:
                        continue
                    if layer_index == 0 and not task.has_first_layer_loss:
                        continue
                    # Only include tasks whose stage is in the allowed set
                    if getattr(task, "task_stage", 0) not in active_task_stages:
                        continue

                    task_outputs = task(x)

                    if isinstance(task, IncidenceRegressionTask):
                        x["incidence"] = task_outputs[task.incidence_key].detach()
                    if isinstance(task, ObjectClassificationTask):
                        x["class_probs"] = task_outputs[task.probs_key].detach()

                    layer_out[task.name] = task_outputs

                    if self.use_query_masks:
                        task_query_mask = task.query_mask(task_outputs)
                        if task_query_mask is not None:
                            query_mask = task_query_mask if query_mask is None else query_mask | task_query_mask
                            x["query_mask"] = query_mask

                # Collect per-type attention masks from tasks
                attn_masks: dict[str, torch.Tensor] = {}
                if self.mask_attention:
                    for task in self.tasks:
                        task_outputs_for_mask = layer_out.get(task.name)
                        if task_outputs_for_mask is None:
                            continue
                        for input_name, task_attn_mask in task.attn_mask(task_outputs_for_mask).items():
                            if input_name in attn_masks:
                                attn_masks[input_name] |= task_attn_mask
                            else:
                                attn_masks[input_name] = task_attn_mask

                # Compute merged affinity logits (B, N, M_total) then slice per type below
                affinity_logits = None
                if getattr(type_layers[0], "cross_attn_mode", "softmax") == "kmeans":
                    requested_names = None
                    if self.kmeans_affinity_task is not None:
                        if isinstance(self.kmeans_affinity_task, str):
                            requested_names = {self.kmeans_affinity_task}
                        else:
                            requested_names = set(self.kmeans_affinity_task)

                    for task in self.tasks:
                        if requested_names is not None and task.name not in requested_names:
                            continue
                        # Skip tasks not active in this phase
                        if getattr(task, "task_stage", 0) not in active_task_stages:
                            continue

                        task_out = layer_out.get(task.name)
                        if task_out is None:
                            task_out = task(x)

                        task_affinity = task.affinity(task_out, x, num_constituents)
                        if task_affinity is None:
                            continue

                        affinity_logits = task_affinity if affinity_logits is None else torch.maximum(affinity_logits, task_affinity)

                # Update queries sequentially, one hit type at a time
                for input_name, type_layer in zip(active_type_names, type_layers):
                    x_type = x[f"{input_name}_embed"]
                    kv_mask_type = x.get(f"{input_name}_valid")

                    # Slice affinity to just this type's hits: (B, N, M_type)
                    per_type_affinity = None
                    if affinity_logits is not None:
                        type_mask = x[f"key_is_{input_name}"][0]  # (M_total,) bool
                        per_type_affinity = affinity_logits[:, :, type_mask]

                        # Apply query slot range masking: restrict which query slots
                        # can be assigned to this hit type via kMaX argmax.
                        if self.query_slot_ranges is not None and input_name in self.query_slot_ranges:
                            start, end = self.query_slot_ranges[input_name]
                            N = per_type_affinity.shape[1]
                            slot_mask = torch.zeros(N, dtype=torch.bool, device=per_type_affinity.device)
                            slot_mask[start:end] = True
                            per_type_affinity = per_type_affinity.masked_fill(~slot_mask.view(1, N, 1), float("-inf"))

                    # Slice attention mask to this type's hits: (B, N, M_type)
                    per_type_attn_mask = None
                    if self.mask_attention and input_name in attn_masks:
                        per_type_attn_mask = attn_masks[input_name].detach()
                        if self.unmask_all_false:
                            per_type_attn_mask = torch.where(
                                torch.all(~per_type_attn_mask, dim=-1, keepdim=True),
                                True,
                                per_type_attn_mask,
                            )

                    # Use per-type query when in split-query mode, else the shared query
                    if self.split_query_types is not None:
                        old_query = x[f"query_embed_{input_name}"]
                    else:
                        old_query = x["query_embed"]

                    new_query, new_kv = type_layer(
                        old_query,
                        x_type,
                        attn_mask=per_type_attn_mask,
                        q_mask=x.get("query_mask"),
                        kv_mask=kv_mask_type,
                        query_posenc=None,
                        key_posenc=None,
                        affinity_logits=per_type_affinity,
                    )

                    if self.split_query_types is not None:
                        x[f"query_embed_{input_name}"] = new_query
                    elif self.query_dim_slices is not None:
                        merged = old_query.clone()
                        merged[..., self.query_shared_start:] = new_query[..., self.query_shared_start:]
                        if input_name in self.query_dim_slices:
                            s, e = self.query_dim_slices[input_name]
                            merged[..., s:e] = new_query[..., s:e]
                        x["query_embed"] = merged
                    else:
                        x["query_embed"] = new_query
                    x[f"{input_name}_embed"] = new_kv

                # Cross-query self-attention over the active per-type query sets
                if self.cross_query_attn:
                    N = self.num_queries
                    q_all = torch.cat([x[f"query_embed_{n}"] for n in active_type_names], dim=1)
                    sa_layer, dense_layer = self.cross_query_sa_layers[cqa_index]
                    q_all = sa_layer(q_all, k=q_all, v=q_all)
                    q_all = dense_layer(q_all)
                    for i, n in enumerate(active_type_names):
                        x[f"query_embed_{n}"] = q_all[:, i * N : (i + 1) * N]

                # Update shared query_embed to mean of ALL per-type queries for task calls
                if self.split_query_types is not None:
                    x["query_embed"] = torch.stack([x[f"query_embed_{n}"] for n in input_names], dim=0).mean(0)

                # Scatter updated per-type embeds back into key_embed for task consistency
                embed_dim = x["key_embed"].shape[-1]
                new_key_embed = torch.empty_like(x["key_embed"])
                for name in input_names:
                    type_mask = x[f"key_is_{name}"]  # (B, M_total)
                    new_key_embed[type_mask] = x[f"{name}_embed"].reshape(-1, embed_dim)
                x["key_embed"] = new_key_embed

                return layer_out

            # --- Tracking phase (silicon-only layers) ---
            if self.num_tracking_layers > 0:
                for layer_index, type_layers in enumerate(self.tracking_decoder_layers):
                    outputs[f"tracking_layer_{layer_index}"] = _run_per_input_macro_layer(
                        layer_index=layer_index,
                        type_layers=type_layers,
                        active_type_names=self.tracking_input_names,
                        active_task_stages={0, 1},
                        cqa_index=layer_index,
                    )

                # Detach tracking-type queries so calo gradients cannot flow back into
                # the tracking decoder parameters.
                for name in self.tracking_input_names:
                    x[f"query_embed_{name}"] = x[f"query_embed_{name}"].detach()
                if self.split_query_types is not None:
                    x["query_embed"] = torch.stack([x[f"query_embed_{n}"] for n in input_names], dim=0).mean(0)

            # --- Combined phase (all hit types) ---
            for layer_index, type_layers in enumerate(self.decoder_layers):
                outputs[f"layer_{layer_index}"] = _run_per_input_macro_layer(
                    layer_index=layer_index,
                    type_layers=type_layers,
                    active_type_names=input_names,
                    active_task_stages={0, 2} if self.num_tracking_layers > 0 else {0, 1, 2},
                    cqa_index=self.num_tracking_layers + layer_index,
                )

        else:
            for layer_index, decoder_layer in enumerate(self.decoder_layers):
                outputs[f"layer_{layer_index}"] = {}

                # if maskattention, PE should be added before generating the mask
                if self.posenc and self.mask_attention:
                    x["query_embed"] = x["query_embed"] + x["query_posenc"]
                    x["key_embed"] = x["key_embed"] + x["key_posenc"]

                attn_masks: dict[str, torch.Tensor] = {}
                query_mask = None

                assert self.tasks is not None
                for task in self.tasks:
                    if not task.has_intermediate_loss:
                        continue
                    if layer_index == 0 and not task.has_first_layer_loss:
                        continue

                    # Get the outputs of the task given the current embeddings
                    task_outputs = task(x)

                    # Update x with task outputs for downstream use
                    if isinstance(task, IncidenceRegressionTask):
                        x["incidence"] = task_outputs[task.incidence_key].detach()
                    if isinstance(task, ObjectClassificationTask):
                        x["class_probs"] = task_outputs[task.probs_key].detach()

                    outputs[f"layer_{layer_index}"][task.name] = task_outputs

                    # Collect attention masks from different tasks
                    task_attn_masks = task.attn_mask(task_outputs)
                    for input_name, task_attn_mask in task_attn_masks.items():
                        if input_name in attn_masks:
                            attn_masks[input_name] |= task_attn_mask
                        else:
                            attn_masks[input_name] = task_attn_mask

                    # Collect query masks
                    if self.use_query_masks:
                        task_query_mask = task.query_mask(task_outputs)
                        if task_query_mask is not None:
                            query_mask = task_query_mask if query_mask is None else query_mask | task_query_mask
                            x["query_mask"] = query_mask

                # Construct the full attention mask for MaskAttention decoder
                if attn_masks and self.mask_attention:
                    if self.unified_decoding:
                        if len(attn_masks) > 1:
                            raise ValueError(f"In merged input mode, expected only one attention mask, got {len(attn_masks)}")
                        attn_mask = next(iter(attn_masks.values()))
                        if attn_mask.dim() == 2:  # (batch, num_queries) -> (batch, num_queries, num_constituents)
                            attn_mask = attn_mask.unsqueeze(-1).expand(-1, -1, num_constituents)
                    else:
                        attn_mask = torch.full((batch_size, self.num_queries, num_constituents), False, device=x["key_embed"].device)
                        for input_name, task_attn_mask in attn_masks.items():
                            attn_mask[x[f"key_is_{input_name}"].unsqueeze(1).expand_as(attn_mask)] = task_attn_mask.flatten()

                    attn_mask = attn_mask.detach()
                    # If the attn mask is completely invalid for a given query, allow it to attend everywhere
                    if self.unmask_all_false:
                        attn_mask = torch.where(torch.all(~attn_mask, dim=-1, keepdim=True), True, attn_mask)

                if (attn_mask is not None) and self.attn_type != "flex":
                    outputs[f"layer_{layer_index}"]["attn_mask"] = attn_mask

                # If this decoder layer uses kmeans cross-attn, provide affinity logits (B, N, M_total)
                affinity_logits = None
                if getattr(decoder_layer, "cross_attn_mode", "softmax") == "kmeans":
                    requested_names = None
                    if self.kmeans_affinity_task is not None:
                        if isinstance(self.kmeans_affinity_task, str):
                            requested_names = {self.kmeans_affinity_task}
                        else:
                            requested_names = set(self.kmeans_affinity_task)

                    for task in self.tasks:
                        if requested_names is not None and task.name not in requested_names:
                            continue

                        task_outputs = outputs[f"layer_{layer_index}"].get(task.name)
                        if task_outputs is None:
                            task_outputs = task(x)

                        task_affinity = task.affinity(task_outputs, x, num_constituents)
                        if task_affinity is None:
                            continue

                        if affinity_logits is None:
                            affinity_logits = task_affinity
                        else:
                            affinity_logits = torch.maximum(affinity_logits, task_affinity)

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
                    affinity_logits=affinity_logits,
                )

                # update the individual input constituent representations only if not in merged input mode
                if not self.unified_decoding:
                    x = unmerge_inputs(x, input_names)

        return x, outputs

    def flex_local_ca_mask(self, q_len: int, kv_len: int, device, dtype_float):
        stride = kv_len / q_len
        if self.fast_local_ca:
            return build_strided_sliding_window_blockmask(
                window_size=self.window_size,
                block_size=self.block_size,
                stride=kv_len / q_len,
                q_len=q_len,
                kv_len=kv_len,
                device=device,
                wrap=self.window_wrap,
                dtype_float=dtype_float,
            )
        window_mask_func = sliding_window_mask_strided_wrapped if self.window_wrap else sliding_window_mask_strided
        return window_mask_func(self.window_size, stride=stride, q_len=q_len, kv_len=kv_len, device=str(device))

    def generate_positional_encodings(self, x: dict):
        idx = torch.arange(self.num_queries, device=x["query_embed"].device, dtype=x["query_embed"].dtype)
        x["query_phi"] = 2 * torch.pi * (idx / self.num_queries - self.phi_shift)
        query_posenc = pos_enc_symmetric(x["query_phi"], self.dim, self.posenc["alpha"], self.posenc["base"])
        key_posenc = pos_enc_symmetric(x["key_phi"], self.dim, self.posenc["alpha"], self.posenc["base"])
        return query_posenc, key_posenc


class MaskFormerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        norm: str = "LayerNorm",
        depth: int = 0,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        bidirectional_ca: bool = True,
        qkv_norm: bool = False,
        hybrid_norm: bool = False,
        cross_attn_mode: Literal["softmax", "kmeans"] = "softmax",
        kmeans_kwargs: dict | None = None,
    ) -> None:
        """Initialize a MaskFormer decoder layer.

        Args:
            dim: Embedding dimension.
            norm: Normalization type.
            depth: Layer depth index.
            dense_kwargs: Optional arguments for Dense layers.
            attn_kwargs: Optional arguments for Attention layers.
            bidirectional_ca: Enable bidirectional cross-attention.
            qkv_norm: Apply normalization to QKV in attention.
            hybrid_norm: Enable hybrid normalization from 2503.04598.
            cross_attn_mode: "softmax" (standard attention) or "kmeans" (kMaX-style hard assignment update).
            kmeans_kwargs: Optional kwargs passed to KMeansCrossAttention when cross_attn_mode="kmeans".
        """
        super().__init__()
        self.dim = dim
        self.bidirectional_ca = bidirectional_ca
        self.cross_attn_mode = cross_attn_mode

        attn_norm, dense_post_norm, qkv_norm = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm)

        attn_kwargs = attn_kwargs or {}
        self.attn_type = attn_kwargs.get("attn_type", "torch")
        dense_kwargs = dense_kwargs or {}

        residual = partial(Residual, dim=dim)

        if self.cross_attn_mode == "kmeans":
            kmeans_kwargs = kmeans_kwargs or {}
            self.q_ca = residual(KMeansCrossAttention(dim, **kmeans_kwargs), norm=attn_norm)
        else:
            self.q_ca = residual(Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)

        self.q_sa = residual(Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)
        self.q_dense = residual(Dense(dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)

        if self.bidirectional_ca:
            self.kv_ca = residual(Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)
            self.kv_dense = residual(Dense(dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        attn_mask: Tensor | None = None,
        q_mask: Tensor | None = None,
        kv_mask: Tensor | None = None,
        query_posenc: Tensor | None = None,
        key_posenc: Tensor | None = None,
        attn_mask_transpose: Tensor | None = None,
        affinity_logits: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass for the decoder layer.

        Args:
            q: Query embeddings.
            kv: Key/value embeddings.
            attn_mask: Optional attention mask (B, N, M) for q<-kv, or blockmask for flex.
            q_mask: Optional query mask (B, N).
            kv_mask: Optional key/value mask (B, M).
            query_posenc: Optional query positional encoding.
            key_posenc: Optional key positional encoding.
            attn_mask_transpose: Optional transposed attention mask for flex attention.
            affinity_logits: If cross_attn_mode="kmeans", logits (B, N, M) used for hard assignment.

        Returns:
            tuple[Tensor, Tensor]: Updated (q, kv).
        """
        q_pe = q if query_posenc is None else q + query_posenc
        kv_pe = kv if key_posenc is None else kv + key_posenc

        if self.cross_attn_mode == "kmeans":
            q = self.q_ca(
                q_pe,
                k=kv_pe,
                v=kv,
                attn_mask=attn_mask,
                q_mask=q_mask,
                kv_mask=kv_mask,
                affinity_logits=affinity_logits,
            )
        else:
            q = self.q_ca(q_pe, k=kv_pe, v=kv, attn_mask=attn_mask, q_mask=q_mask, kv_mask=kv_mask)

        q = self.q_dense(q)
        q = self.q_sa(q, k=q, v=q, q_mask=q_mask)

        # Update key/constituent embeddings with the query/object embeddings
        if self.bidirectional_ca:
            if attn_mask is not None:
                if self.attn_type == "flex":
                    assert attn_mask_transpose is not None, "attn_mask_transpose must be provided for flex attention"
                attn_mask = attn_mask_transpose if attn_mask_transpose is not None else attn_mask.transpose(-2, -1)

            q_pe = q if query_posenc is None else q + query_posenc
            kv_pe = kv if key_posenc is None else kv + key_posenc

            kv = self.kv_ca(kv_pe, k=q_pe, v=q, attn_mask=attn_mask, q_mask=kv_mask, kv_mask=q_mask)
            kv = self.kv_dense(kv)

        return q, kv

    def set_backend(self, attn_type: str) -> None:
        """Set the backend for the attention layers.

        Args:
            attn_type: Attention implementation type to use.
        """
        if hasattr(self.q_ca.fn, "set_backend"):
            self.q_ca.fn.set_backend(attn_type)
        if hasattr(self.q_sa.fn, "set_backend"):
            self.q_sa.fn.set_backend(attn_type)

        if self.bidirectional_ca and hasattr(self.kv_ca.fn, "set_backend"):
            self.kv_ca.fn.set_backend(attn_type)
