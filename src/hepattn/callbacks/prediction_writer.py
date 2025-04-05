from pathlib import Path

import awkward as ak
import numpy as np
import torch
import matplotlib.pyplot as plt
from lightning import Callback, LightningModule, Trainer
from collections import defaultdict


class TestEvalWriter(Callback):
    def __init__(self) -> None:
        super().__init__()
        
        self.data = defaultdict(list)   

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        super().setup(trainer=trainer, pl_module=module, stage=stage)
        self.trainer = trainer
        self.module = module

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):        
        # Get the final predictions only, and put them all into the same dictionary level
        preds, data = batch
        final_preds = {}
        for task_name, task_preds in preds["final"].items():
            for pred_name, pred_value in task_preds.items():
                final_preds[pred_name] = pred_value

        # Move the data and predictions onto the CPU and store them
        for k, v in (data | final_preds).items():
            if v.dtype == torch.bfloat16:
                v = v.float()
            v = v.detach().cpu().numpy()

            # Append the values from this epoch onto our running list
            self.data[k].append(v)

    def on_test_epoch_end(self, trainer, module):
        # Concatenate the values from the running lists together
        data = {k: np.concatenate(v, axis=0) for k, v in self.data.items()}

        # Get info for where to save the test set eval
        save_dir = Path(trainer.log_dir) / Path("testevals")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the data and predcitions together in a compressed numpy format
        np.savez_compressed(Path(f"{save_dir}/data_{module.current_epoch}.npz"), **data)
        print(f"Saved test set evaluation to {save_dir}")
