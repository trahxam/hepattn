# matcher.py
import time
import warnings
from multiprocessing.pool import ThreadPool as Pool

import numpy as np
import scipy
import torch
from torch import Tensor, nn

from hepattn.utils.import_utils import check_import_safe


def _prep_cost(cost: np.ndarray) -> np.ndarray:
    """Ensure a stable, contiguous dtype/layout for solvers."""
    return np.ascontiguousarray(cost, dtype=np.float64)


def solve_scipy(cost: np.ndarray) -> np.ndarray:
    cost = _prep_cost(cost)
    _, col_idx = scipy.optimize.linear_sum_assignment(cost)
    return np.asarray(col_idx, dtype=np.int64)


SOLVERS = {
    "scipy": solve_scipy,
}

# Some compiled extension can cause SIGKILL errors if compiled for the wrong arch
# So we have to check they won't kill everything when we import them
if check_import_safe("lap1015"):
    import lap1015

    def solve_1015_early(cost: np.ndarray) -> np.ndarray:
        cost = _prep_cost(cost)
        return np.asarray(lap1015.lap_early(cost), dtype=np.int64)

    def solve_1015_late(cost: np.ndarray) -> np.ndarray:
        cost = _prep_cost(cost)
        return np.asarray(lap1015.lap_late(cost), dtype=np.int64)

    SOLVERS["lap1015_late"] = solve_1015_late
    # SOLVERS["lap1015_early"] = solve_1015_early
else:
    warnings.warn(
        """Failed to import lap1015 solver. This could be because it is not installed,
or because it was built targeting a different architecture than supported on the current machine.
Rebuilding the package on the current machine may fix this.""",
        ImportWarning,
        stacklevel=2,
    )


def match_individual(solver_fn, cost: np.ndarray, default_idx: np.ndarray) -> np.ndarray:
    """
    cost: (T_valid, P)  rows=true, cols=pred
    returns: permutation of predictions length P (first T_valid are the matched ones)
    """
    cost = _prep_cost(cost)

    pred_idx = np.asarray(solver_fn(cost), dtype=np.int64)

    # Hard correctness guard: indices must be valid column indices (pred axis)
    n_pred = cost.shape[1]
    if pred_idx.ndim != 1:
        raise RuntimeError(f"Bad solver output shape: {pred_idx.shape}")
    if (pred_idx < 0).any() or (pred_idx >= n_pred).any():
        raise RuntimeError(
            f"Solver produced out-of-range indices. "
            f"min={pred_idx.min()} max={pred_idx.max()} n_pred={n_pred}. "
            f"First few: {pred_idx[:10]}"
        )

    # Extend to full permutation of predictions (length = P)
    # Only SciPy branch needs this extra step in your original design.
    if solver_fn is SOLVERS["scipy"]:
        missing = default_idx[~np.isin(default_idx, pred_idx)]
        pred_idx = np.concatenate([pred_idx, missing], axis=0)

    return pred_idx


def match_parallel(solver_fn, costs: np.ndarray, batch_obj_lengths: torch.Tensor, n_jobs: int = 8) -> torch.Tensor:
    """
    Parallel over batch using threads. Note: compiled solvers may not be thread-safe.
    costs: (B, P, T)
    batch_obj_lengths: (B, 1) number of valid true objects per batch item
    """
    batch_size = len(costs)
    chunk_size = (batch_size + n_jobs - 1) // n_jobs

    B, P, T = costs.shape
    default_idx = np.arange(P, dtype=np.int64)

    lengths_np = batch_obj_lengths.squeeze(-1).detach().cpu().numpy().astype(np.int64)

    args = []
    for i in range(batch_size):
        L = int(lengths_np[i])
        cost_i = costs[i, :, :L].T  # (L, P)
        args.append((solver_fn, cost_i, default_idx))

    with Pool(processes=n_jobs) as pool:
        results = pool.starmap(match_individual, args, chunksize=chunk_size)

    return torch.as_tensor(np.stack(results, axis=0), dtype=torch.long)


class Matcher(nn.Module):
    def __init__(
        self,
        default_solver: str = "scipy",
        adaptive_solver: bool = True,
        adaptive_check_interval: int = 1000,
        parallel_solver: bool = False,
        n_jobs: int = 8,
        verbose: bool = False,
    ):
        super().__init__()
        """Used to match predictions to targets based on a given cost matrix.

        Parameters
        ----------
        default_solver : str
            The default solving algorithm to use.
        adaptive_solver : bool
            If true, then after every adaptive_check_interval calls of the solver,
            each solver algorithm is timed and used to determine the fastest solver, which
            is then set as the current solver.
        adaptive_check_interval : int
            Interval for checking which solver is the fastest.
        parallel_solver : bool
            If true, then the solver will use a parallel implementation to speed up the matching.
        n_jobs : int
            Number of jobs to use for parallel matching. Only used if parallel_solver is True.
        verbose : bool
            If true, extra information on solver timing is printed.
        """
        if default_solver not in SOLVERS:
            raise ValueError(f"Unknown solver: {default_solver}. Available solvers: {list(SOLVERS.keys())}")
        self.solver = default_solver
        self.adaptive_solver = adaptive_solver
        self.adaptive_check_interval = adaptive_check_interval
        self.parallel_solver = parallel_solver
        self.n_jobs = n_jobs
        self.step = 0
        self.verbose = verbose

    def compute_matching(self, costs: np.ndarray, object_valid_mask=None) -> torch.Tensor:
        # costs: (B, P, T) numpy
        B, P, T = costs.shape

        if object_valid_mask is None:
            # Mask is over TRUE objects (T), not predictions (P)
            object_valid_mask = torch.ones((B, T), dtype=torch.bool, device="cpu")
        else:
            object_valid_mask = object_valid_mask.detach().bool().cpu()

        batch_obj_lengths = object_valid_mask.sum(dim=1, keepdim=True)  # (B, 1)

        if self.parallel_solver:
            return match_parallel(SOLVERS[self.solver], costs, batch_obj_lengths, n_jobs=self.n_jobs)

        idxs = []
        default_idx = np.arange(P, dtype=np.int64)

        lengths = batch_obj_lengths.squeeze(-1).to(torch.int64).tolist()
        for k, L in enumerate(lengths):
            L = int(L)
            cost = costs[k, :, :L].T  # (L, P)
            pred_idx = match_individual(SOLVERS[self.solver], cost, default_idx)
            idxs.append(pred_idx)

        return torch.as_tensor(np.stack(idxs, axis=0), dtype=torch.long)

    @torch.no_grad()
    def forward(self, costs: Tensor, object_valid_mask: Tensor | None = None) -> torch.Tensor:
        # costs: (B, P, T)
        B, P, T = costs.shape

        costs_np = costs.detach().to(torch.float64).cpu().numpy()

        if self.adaptive_solver and self.step % self.adaptive_check_interval == 0:
            self.adapt_solver(costs_np, object_valid_mask=object_valid_mask)

        pred_idxs = self.compute_matching(costs_np, object_valid_mask)
        self.step += 1

        # Global guard
        mn = int(pred_idxs.min().item())
        mx = int(pred_idxs.max().item())
        if mn < 0 or mx >= P:
            raise RuntimeError(f"Matcher produced out-of-range indices: min={mn} max={mx} n_pred={P}")

        return pred_idxs

    def adapt_solver(self, costs: np.ndarray, object_valid_mask: Tensor | None = None):
        solver_times = {}

        if self.verbose:
            print("\nAdaptive LAP Solver: Starting solver check...")

        # Time each solver on this batch (same mask)
        current_solver = self.solver
        for solver in SOLVERS:
            self.solver = solver
            start_time = time.time()
            _ = self.compute_matching(costs, object_valid_mask=object_valid_mask)
            solver_times[solver] = time.time() - start_time

            if self.verbose:
                print(f"Adaptive LAP Solver: Evaluated {solver}, took {solver_times[solver]:.4f}s")

        fastest_solver = min(solver_times, key=solver_times.get)

        if self.verbose:
            if fastest_solver != current_solver:
                print(f"Adaptive LAP Solver: Switching from {current_solver} solver to {fastest_solver} solver\n")
            else:
                print(f"Adaptive LAP Solver: Sticking with {current_solver} solver\n")

        self.solver = fastest_solver
