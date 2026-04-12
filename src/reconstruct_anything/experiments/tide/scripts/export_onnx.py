# ruff: noqa
"""Export a TIDE MaskFormer model to ONNX format.

Usage:
    pixi run python src/reconstruct_anything/experiments/tide/scripts/export_onnx.py \
        --ckpt_path logs/.../epoch=009-train_loss=0.63014.ckpt \
        --output_path tide_model.onnx \
        --num_pix 256
"""

import argparse
from pathlib import Path

import onnx
import torch
from torch import nn

from reconstruct_anything.experiments.tide.model import TIDEModel


class OnnxWrapper(nn.Module):
    """Wraps the MaskFormer to accept/return flat tensors for ONNX export.

    ONNX doesn't support dict inputs/outputs, so we flatten the interface:
    - Input: pix_valid (float 0/1), and each pixel feature as a separate tensor
    - Output: pred_valid_prob [B, Q], pred_pix_valid_prob [B, Q, N_pix]
    """

    def __init__(self, model, input_fields):
        super().__init__()
        self.model = model
        self.input_fields = input_fields

    def forward(self, pix_valid, *field_tensors):
        inputs = {"pix_valid": pix_valid > 0.5}
        for field_name, tensor in zip(self.input_fields, field_tensors, strict=False):
            inputs[f"pix_{field_name}"] = tensor

        outputs = self.model(inputs)

        final = outputs["final"]
        pred_valid_prob = final["pred_valid"]["pred_class_prob"][..., 0]
        pred_pix_logit = final["pred_pix_assignment"]["pred_pix_logit"]
        pred_pix_valid_prob = torch.sigmoid(pred_pix_logit)

        return pred_valid_prob, pred_pix_valid_prob


def get_input_fields(model):
    """Extract all pixel input field names needed (net fields + posenc fields)."""
    fields = []
    for input_net in model.model.input_nets:
        if input_net.input_name == "pix":
            fields.extend(input_net.fields)
            if hasattr(input_net, "posenc") and input_net.posenc is not None:
                for f in input_net.posenc.fields:
                    if f not in fields:
                        fields.append(f)
            return fields
    raise ValueError("No pix input net found in model")


def make_dummy_field(field_name, batch_size, num_pix):
    """Create a dummy tensor for a given field, with the correct shape."""
    if field_name == "charge_matrix":
        return torch.randn(batch_size, num_pix, 49)
    if field_name == "pitches":
        return torch.randn(batch_size, num_pix, 7)
    return torch.randn(batch_size, num_pix)


def main():
    parser = argparse.ArgumentParser(description="Export TIDE model to ONNX")
    parser.add_argument("--ckpt_path", required=True, help="Path to checkpoint")
    parser.add_argument("--output_path", default="tide_model.onnx", help="Output ONNX path")
    parser.add_argument("--num_pix", type=int, default=256, help="Number of pixel hits for dummy input")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for tracing (must be > 1)")
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    assert args.batch_size > 1, "batch_size must be > 1 to avoid constant-folding batch dim"

    # Load model
    print(f"Loading checkpoint: {args.ckpt_path}")
    lit_model = TIDEModel.load_from_checkpoint(args.ckpt_path)
    lit_model.eval()
    lit_model.cpu()

    input_fields = get_input_fields(lit_model)
    print(f"Input fields ({len(input_fields)}): {input_fields}")

    wrapper = OnnxWrapper(lit_model.model, input_fields)
    wrapper.eval()

    # Create dummy inputs (pix_valid as float for tracing compatibility)
    B, N = args.batch_size, args.num_pix
    pix_valid = torch.ones(B, N, dtype=torch.float32)

    field_tensors = []
    input_names = ["pix_valid"]
    dynamic_axes = {"pix_valid": {0: "batch", 1: "num_pix"}}

    for field in input_fields:
        name = f"pix_{field}"
        t = make_dummy_field(field, B, N)
        field_tensors.append(t)
        input_names.append(name)
        dynamic_axes[name] = {0: "batch", 1: "num_pix"}

    output_names = ["pred_valid_prob", "pred_pix_valid_prob"]
    dynamic_axes["pred_valid_prob"] = {0: "batch"}
    dynamic_axes["pred_pix_valid_prob"] = {0: "batch", 2: "num_pix"}

    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        pred_valid_prob, pred_pix_valid_prob = wrapper(pix_valid, *field_tensors)
    print(f"  pred_valid_prob:     {pred_valid_prob.shape}")
    print(f"  pred_pix_valid_prob: {pred_pix_valid_prob.shape}")

    # Export to ONNX via TorchScript tracing
    print(f"Exporting to ONNX (opset {args.opset_version})...")
    torch.onnx.export(
        wrapper,
        (pix_valid, *field_tensors),
        args.output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    # Verify
    onnx_model = onnx.load(args.output_path)
    onnx.checker.check_model(onnx_model)

    file_size_mb = Path(args.output_path).stat().st_size / 1e6
    print(f"\nExport successful: {args.output_path} ({file_size_mb:.1f} MB)")
    print(f"  Inputs:  {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")


if __name__ == "__main__":
    main()
