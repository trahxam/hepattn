
import torch
from torch import Tensor, nn
from hepattn.models.decoder import MaskFormerDecoderLayer
from hepattn.models.matcher import Matcher


class MaskFormer(nn.Module):
    def __init__(
            self,
            input_nets: nn.ModuleList,
            encoder: nn.Module,
            decoder_layer_config: dict,
            num_decoder_layers: int,
            tasks: nn.ModuleList,
            matcher: None | nn.Module,
            num_queries: int,
            embed_dim: int,
        ):
            super().__init__()

            self.input_nets = input_nets
            self.encoder = encoder #nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 8, 128, batch_first=True), 4)
            self.decoder_layers = nn.ModuleList([MaskFormerDecoderLayer(**decoder_layer_config) for _ in range(num_decoder_layers)])
            self.tasks = tasks
            self.matcher = matcher
            self.num_queries = num_queries
            self.query_initial = nn.Parameter(torch.randn(num_queries, embed_dim))
            
    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        input_features = [input_net.feature for input_net in self.input_nets]

        assert "key" not in input_features, "'key' input name is reserved."
        assert "query" not in input_features, "'query' input name is reserved."
    
        # Embed the input objects
        x = {}

        # Used for un-merging features later
        feature_slices = {}
        slice_start = 0

        for input_net in self.input_nets:
            feature = input_net.feature
            x[feature + "_embed"] = input_net(inputs)
            x[feature + "_valid"] = inputs[feature + "_valid"]

            slice_size = inputs[feature + "_valid"].shape[-1]
            feature_slices[feature] = slice(slice_start, slice_start + slice_size)
            slice_start += slice_size

        # Merge the input objects and he padding mask into a single set
        # TODO: Need to correctly account for token ordering if we have multiple input features
        x["key_embed"] = torch.concatenate([x[feature + "_embed"] for feature in input_features], dim=-2)
        x["key_valid"] = torch.concatenate([x[feature + "_valid"] for feature in input_features], dim=-1)

        # Pass merged input objects through the encoder
        if self.encoder is not None:
            # Note that a padded feature is a feature that is not valid
            x["key_embed"] = self.encoder(x["key_embed"])

        # Unmerge the updated features back into the separate input types
        for feature in input_features:
            x[feature + "_embed"] = x["key_embed"][...,feature_slices[feature],:]

        # Generate the queries / tracks
        batch_size = x["key_valid"].shape[0]
        x["query_embed"] = self.query_initial.expand(batch_size, -1, -1)
        x["query_valid"] = torch.full((batch_size, self.num_queries), True)

        # Pass encoded inputs through decoder to produce outputs
        outputs = {}
        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            outputs[f"layer_{layer_index}"] = {}

            # Here we check if each task has an attention mask to contribute, then after
            # we fill in any attention masks for any features that did not get an attention mask
            attn_masks = {}
            for task in self.tasks:
                # Get the outputs of the task given the current embeddings and record them
                task_outputs = task(x)
                outputs[f"layer_{layer_index}"][task.name] = task_outputs

                # If the task has attention masks to provide, record them
                if hasattr(task, "attn_mask"):
                    for feature, attn_mask in task.attn_mask(task_outputs).items():
                        # If a feature already has an attention mask from another task, need to update it
                        if feature in attn_masks:
                            attn_masks[feature] = attn_masks[feature] & attn_mask
                        else:
                            attn_masks[feature] = attn_mask

            # Fill in attention masks for features that did not get one specified by any task
            if attn_masks:
                for feature in input_features:
                    if feature not in attn_masks:
                        attn_masks[feature] = torch.full((batch_size, self.num_queries, x[feature + "_valid"].shape[-1]), False)
            
                attn_mask = torch.concatenate([attn_masks[feature] for feature in input_features], dim=-1)
            
            # If no attention masks were specified, set it to none to avoid redundant masking
            else:
                attn_mask = None

            # Update the keys and queries
            x["query_embed"], x["key_embed"] = decoder_layer(x["query_embed"], x["key_embed"], attn_mask=attn_mask)

            # Unmerge the updated features back into the separate input types
            for feature in input_features:
                x[feature + "_embed"] = x["key_embed"][...,feature_slices[feature],:]

        # Get the final outputs - we don't need to compute attention masks or update things here
        outputs["final"] = {}
        for task in self.tasks:
           outputs["final"][task.name] = task(x)

        return outputs

    def predict(self, outputs: dict) -> dict:
        """ Takes the raw model outputs and produces a set of actual inferences / predictions.
        For example will take output probabilies and apply threshold cuts to prduce boolean predictions.

        Parameters
        ----------
        outputs:
            The outputs produces the forward pass of the model.
        """
        preds = {}

        # Compute predictions for each task in each block
        for layer_name, layer_outputs in outputs.items():
            preds[layer_name] = {}

            for task in self.tasks:
                preds[layer_name][task.name] = task.predict(outputs[layer_name][task.name])
        
        return preds

    def loss(self, outputs: dict, targets: dict) -> dict:
        """ Computes the loss between the forward pass of the model and the data / targets.
        It first computes the cost / loss between each of the predicted and true tracks in each ROI
        and then uses the Hungarian algorihtm to perform an optimal bipartite matching. The model
        predictions are then permuted to match this optimal matching, after which the final loss
        between the model and target is computed.

        Parameters
        ----------
        outputs:
            The outputs produces the forward pass of the model.
        outputs:
            The data containing the targets.
        """
        # Will hold the costs between all pairs of objects - cost axes are (batch, pred, true)
        costs = {}
        for layer_name, layer_outputs in outputs.items():
            layer_costs = None
            # Get the cost contribution from each of the tasks
            for task in self.tasks:
                # If the task has no cost to contribute then skip it
                if not hasattr(task, "cost"): continue

                # Only use the cost from the final set of predictions
                task_costs = task.cost(outputs[layer_name][task.name], targets)
                
                # Add the cost on to our running cost total, otherwise initialise a running cost matrix
                for cost_name, cost in task_costs.items():
                    if layer_costs is not None:
                        layer_costs += cost
                    else:
                        layer_costs = cost

            costs[layer_name] = layer_costs.detach()

        # Permute the outputs for each output in each layer
        # Note that this permutes the outputs in place
        for layer_name in outputs.keys():
            # Get the indicies that can permute the predictions to yield their optimal matching
            pred_idxs = self.matcher(costs[layer_name])
            batch_idxs = torch.arange(costs[layer_name].shape[0]).unsqueeze(1).expand(-1, self.num_queries)

            for task in self.tasks:
                for output_feature in task.output_features:
                    outputs[layer_name][task.name][output_feature] = outputs[layer_name][task.name][output_feature][batch_idxs,pred_idxs]

        # Compute the losses for each task in each block
        losses = {}
        for layer_name in outputs.keys():
            losses[layer_name] = {}
            for task in self.tasks:
                losses[layer_name][task.name] = task.loss(outputs[layer_name][task.name], targets)

        return losses
