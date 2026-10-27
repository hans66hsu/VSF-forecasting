from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
from torch.utils.data import Dataset



def get_data_stats(data: np.ndarray) -> Dict[str, np.ndarray]:
    flat = data.reshape(-1, data.shape[-1])
    data_min = np.min(flat, axis=0)
    data_max = np.max(flat, axis=0)
    scale = np.where(data_max > data_min, data_max - data_min, 1.0)
    return {"min": data_min, "max": data_max, "scale": scale}


def normalize_data(data: np.ndarray, stats: Mapping[str, np.ndarray]) -> np.ndarray:
    normalized = (data - stats["min"]) / stats["scale"]
    return normalized * 2.0 - 1.0


def unnormalize_data(data: np.ndarray, stats: Mapping[str, np.ndarray]) -> np.ndarray:
    unscaled = (data + 1.0) / 2.0
    return unscaled * stats["scale"] + stats["min"]


def get_anchor_index_range(
    trajectory_length: int,
    obs_horizon: int,
    future_horizon: int,
    future_start_offset: int = 0,
) -> Tuple[int, int]:
    if obs_horizon <= 0:
        raise ValueError(f"obs_horizon must be positive, got {obs_horizon}.")
    if future_horizon <= 0:
        raise ValueError(f"future_horizon must be positive, got {future_horizon}.")
    if future_start_offset < 0:
        raise ValueError(
            f"future_start_offset must be nonnegative, got {future_start_offset}."
        )

    min_anchor_idx = obs_horizon - 1
    max_anchor_idx = trajectory_length - future_start_offset - future_horizon
    if max_anchor_idx < min_anchor_idx:
        required_length = obs_horizon + future_start_offset + future_horizon - 1
        raise ValueError(
            f"Trajectory length must be at least {required_length} "
            f"but got trajectory_length={trajectory_length}, "
            f"obs_horizon={obs_horizon}, future_horizon={future_horizon}, "
            f"future_start_offset={future_start_offset}."
        )
    return min_anchor_idx, max_anchor_idx


def _slice_trajectory_bank(
    bank: Mapping[str, np.ndarray],
    start_idx: int,
    end_idx: int,
) -> Dict[str, np.ndarray]:
    num_trajectories = bank["states"].shape[0]
    sliced = {}
    for key, value in bank.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (num_trajectories,):
            sliced[key] = value[start_idx:end_idx]
        else:
            sliced[key] = value
    return sliced


def split_trajectory_bank(
    bank: Mapping[str, np.ndarray],
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> Dict[str, Dict[str, np.ndarray]]:
    num_trajectories = bank["states"].shape[0]
    num_train = int(num_trajectories * train_fraction)
    num_val = int(num_trajectories * val_fraction)
    train_end = num_train
    val_end = min(num_trajectories, num_train + num_val)

    return {
        "train": _slice_trajectory_bank(bank, 0, train_end),
        "val": _slice_trajectory_bank(bank, train_end, val_end),
        "test": _slice_trajectory_bank(bank, val_end, num_trajectories),
    }


@dataclass(frozen=True)
class ForecastWindow:
    trajectory_index: int
    anchor_index: int


class ForecastDataset(Dataset):
    def __init__(
        self,
        trajectory_bank: Mapping[str, np.ndarray],
        obs_horizon: int,
        pred_horizon: int,
    ) -> None:
        self.states = np.asarray(trajectory_bank["states"], dtype=np.float32)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.state_dim = self.states.shape[-1]

        self.stats = get_data_stats(self.states)
        self.normalized_states = normalize_data(self.states, self.stats).astype(
            np.float32
        )

        min_anchor_idx, max_anchor_idx = get_anchor_index_range(
            trajectory_length=self.states.shape[1],
            obs_horizon=self.obs_horizon,
            future_horizon=self.pred_horizon,
        )

        self.windows = [
            ForecastWindow(traj_idx, anchor_idx)
            for traj_idx in range(self.states.shape[0])
            for anchor_idx in range(min_anchor_idx, max_anchor_idx + 1)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        window = self.windows[idx]
        trajectory = self.normalized_states[window.trajectory_index]

        obs_start = window.anchor_index - self.obs_horizon + 1
        obs_end = window.anchor_index + 1
        target_start = window.anchor_index
        target_end = window.anchor_index + self.pred_horizon

        obs = trajectory[obs_start:obs_end]
        target = trajectory[target_start:target_end]

        return {
            "obs": obs.astype(np.float32),
            "target": target.astype(np.float32),
            "trajectory_index": np.int64(window.trajectory_index),
            "anchor_index": np.int64(window.anchor_index),
        }
