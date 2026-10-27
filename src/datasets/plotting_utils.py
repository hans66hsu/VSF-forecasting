from typing import Optional

import numpy as np


def _flatten_grouped_plot_arrays(
    states: np.ndarray,
    initial_conditions: np.ndarray,
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if states.ndim == 4:
        n_contexts, n_samples = states.shape[:2]
        return (
            states.reshape(n_contexts * n_samples, *states.shape[2:]),
            np.repeat(initial_conditions, n_samples, axis=0),
            parameters.reshape(-1),
            np.repeat(np.arange(n_contexts), n_samples),
        )
    return states, initial_conditions, parameters.reshape(-1), np.arange(
        states.shape[0]
    )


def _select_grouped_plot_index_grid(
    states: np.ndarray,
    initial_condition_indices: Optional[np.ndarray],
    n_rows: int,
    n_cols: int,
    parameter_label: str,
) -> np.ndarray:
    n_plots = n_rows * n_cols
    if states.shape[0] < n_plots:
        raise ValueError(
            f"Need at least {n_plots} trajectories for a {n_rows}x{n_cols} plot, "
            f"got {states.shape[0]}."
        )

    if initial_condition_indices is None:
        return np.arange(n_plots, dtype=np.int64).reshape(n_rows, n_cols)

    initial_condition_indices = np.asarray(initial_condition_indices)
    unique_indices = []
    for init_idx in initial_condition_indices:
        if init_idx not in unique_indices:
            unique_indices.append(init_idx)
        if len(unique_indices) == n_rows:
            break
    if len(unique_indices) < n_rows:
        raise ValueError(
            f"Need at least {n_rows} unique initial conditions, "
            f"got {len(unique_indices)}."
        )

    grouped_plot_indices = []
    for init_idx in unique_indices:
        matches = np.flatnonzero(initial_condition_indices == init_idx)
        if matches.size < n_cols:
            raise ValueError(
                f"Initial condition {init_idx} has {matches.size} "
                f"{parameter_label} samples, but the plot needs {n_cols}."
            )
        grouped_plot_indices.append(matches[:n_cols])
    return np.asarray(grouped_plot_indices, dtype=np.int64)


def _as_context_samples(trajectories: np.ndarray) -> np.ndarray:
    array = np.asarray(trajectories, dtype=np.float32)
    if array.ndim == 2:
        return array[None, None]
    if array.ndim == 3:
        return array[:, None]
    if array.ndim == 4:
        return array
    raise ValueError(f"Unexpected trajectory shape {array.shape}.")


def _overview_colors(num_samples: int):
    import matplotlib.colors as mcolors

    rng = np.random.default_rng(17)
    hues = (np.linspace(0.0, 1.0, num_samples, endpoint=False) + rng.random()) % 1.0
    rng.shuffle(hues)
    return [mcolors.hsv_to_rgb((hue, 0.72, 0.82)) for hue in hues]
