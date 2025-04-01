import torch
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


# TODO: Add back in class structure / dynamic lap / lap1015 support


def get_optimal_matching(costs):
    # Cost matrix indices are batch, pred, true
    # Have to detach and move the tensor from GPU to CPU then convert to numpy so we can use SciPy matcher
    device = costs.device
    costs = costs.detach().to(torch.float32).cpu().numpy()
    pred_idxs = np.zeros(shape=(costs.shape[0], costs.shape[1]), dtype=int)
    
    # Do the matching sequentially for each example in the batch
    for k in range(len(costs)):
        true_idx, pred_idx = linear_sum_assignment(costs[k].T)
        # These indicies can be used to permute the predictions so they now match the truth objects
        pred_idxs[k] = pred_idx

    # Convert back into a torch tensor and move it back onto the GPU
    pred_idxs = torch.from_numpy(pred_idxs).long().to(device)
    return pred_idxs