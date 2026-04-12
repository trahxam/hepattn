# ruff: noqa
"""Test an exported ONNX TIDE model against the PyTorch model on real ROI data.

Loads a few real ROIs via the TIDE dataloader, runs both the PyTorch and ONNX
models, and compares the outputs to verify the export is correct.

Usage:
    pixi run python src/reconstruct_anything/experiments/tide/scripts/test_onnx.py \
        --onnx_path tide_model.onnx \
        --ckpt_path logs/.../epoch=009-train_loss=0.63014.ckpt \
        --num_samples 5
"""

import argparse

import numpy as np
import onnxruntime as ort
import torch
import yaml

from reconstruct_anything.experiments.tide.data import ROIDataModule
from reconstruct_anything.experiments.tide.model import TIDEModel
from reconstruct_anything.experiments.tide.scripts.export_onnx import OnnxWrapper, get_input_fields


def main():
    parser = argparse.ArgumentParser(description="Test ONNX TIDE model against PyTorch")
    parser.add_argument("--onnx_path", required=True, help="Path to exported ONNX model")
    parser.add_argument("--ckpt_path", required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("-c", "--config", action="append", required=True, help="Config YAML file(s)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of ROIs to test")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    args = parser.parse_args()

    # Load merged config for datamodule
    merged_config = {}
    for cfg_path in args.config:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        _deep_merge(merged_config, cfg)

    # Load ONNX model
    print(f"Loading ONNX model: {args.onnx_path}")
    session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    onnx_input_names = [inp.name for inp in session.get_inputs()]
    onnx_output_names = [out.name for out in session.get_outputs()]
    print(f"  Inputs:  {onnx_input_names}")
    print(f"  Outputs: {onnx_output_names}")

    # Load PyTorch model
    print(f"Loading PyTorch model: {args.ckpt_path}")
    lit_model = TIDEModel.load_from_checkpoint(args.ckpt_path)
    lit_model.eval()
    lit_model.cpu()

    input_fields = get_input_fields(lit_model)

    # Create a datamodule to get real batches
    data_cfg = merged_config.get("data", {})
    data_cfg["num_test"] = args.num_samples
    data_cfg["batch_size"] = 1
    data_cfg["num_workers"] = 0
    dm = ROIDataModule(**data_cfg)
    dm.setup("test")
    test_loader = dm.test_dataloader()

    # Compare outputs
    all_pass = True
    wrapper = OnnxWrapper(lit_model.model, input_fields)
    wrapper.eval()

    for i, (inputs, targets) in enumerate(test_loader):
        if i >= args.num_samples:
            break

        num_pix = inputs["pix_valid"].shape[-1]
        print(f"\nROI {i}: {num_pix} pixel hits")

        # PyTorch forward (using the wrapper to get the same outputs)
        with torch.no_grad():
            pix_valid_float = inputs["pix_valid"].float()
            field_tensors = []
            for field in input_fields:
                field_tensors.append(inputs[f"pix_{field}"])
            pt_valid, pt_pix = wrapper(pix_valid_float, *field_tensors)
            pt_valid = pt_valid.numpy()
            pt_pix = pt_pix.numpy()

        # ONNX forward
        onnx_feeds = {}
        for name in onnx_input_names:
            if name == "pix_valid":
                onnx_feeds[name] = inputs["pix_valid"].float().numpy()
            elif name in inputs:
                arr = inputs[name].numpy()
                onnx_feeds[name] = arr.astype(np.float32) if arr.dtype != np.float32 else arr
            else:
                print(f"  WARNING: missing input {name}")
                all_pass = False

        if len(onnx_feeds) != len(onnx_input_names):
            print("  SKIP: missing inputs")
            continue

        onnx_results = session.run(onnx_output_names, onnx_feeds)
        onnx_valid = onnx_results[0]
        onnx_pix = onnx_results[1]

        # Compare
        valid_close = np.allclose(pt_valid, onnx_valid, atol=args.atol, rtol=args.rtol)
        pix_close = np.allclose(pt_pix, onnx_pix, atol=args.atol, rtol=args.rtol)

        valid_max_diff = np.max(np.abs(pt_valid - onnx_valid))
        pix_max_diff = np.max(np.abs(pt_pix - onnx_pix))

        status = "PASS" if (valid_close and pix_close) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  pred_valid_prob:     max_diff={valid_max_diff:.2e} {'OK' if valid_close else 'MISMATCH'}")
        print(f"  pred_pix_valid_prob: max_diff={pix_max_diff:.2e} {'OK' if pix_close else 'MISMATCH'}")
        print(f"  {status}")

    print(f"\n{'=' * 40}")
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


def _deep_merge(base, override):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


if __name__ == "__main__":
    exit(main())
