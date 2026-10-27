import argparse
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.datasets.plotting_utils import (
    _as_context_samples,
    _flatten_grouped_plot_arrays,
    _overview_colors,
    _select_grouped_plot_index_grid,
)
from src.datasets.rossler.simulation import simulate_rossler


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset" / "rossler.yaml"


def _as_1d_float_array(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    return array


def _c_sampling_bounds(c_values: Sequence[float]) -> tuple[float, float]:
    values = _as_1d_float_array("c_values", c_values)
    if values.size < 1:
        raise ValueError("c_values must contain at least one value.")
    c_min = float(np.min(values))
    c_max = float(np.max(values))
    return c_min, c_max


def _signed_unit_centers() -> np.ndarray:
    return np.asarray(
        [
            [x, y, z]
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )


def load_dataset_config(config_path: Path) -> Dict[str, object]:
    loaded = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping config in {config_path}, got {type(loaded)}.")
    return loaded


def simulate_trajectory_bank(
    num_initial_conditions: Optional[int] = None,
    num_cs_per_initial_condition: int = 10,
    num_trajectories: Optional[int] = None,
    t_max: float = 30.0,
    dt: float = 0.02,
    c_values: Sequence[float] = (5.2, 5.7, 6.2),
    initial_noise_std: float = 0.01,
    a: float = 0.2,
    b: float = 0.2,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    if num_cs_per_initial_condition <= 0:
        raise ValueError(
            "num_cs_per_initial_condition must be positive, "
            f"got {num_cs_per_initial_condition}."
        )

    if num_initial_conditions is None:
        if num_trajectories is None:
            num_initial_conditions = 20
        else:
            if num_trajectories <= 0:
                raise ValueError(
                    f"num_trajectories must be positive, got {num_trajectories}."
                )
            if num_trajectories % num_cs_per_initial_condition != 0:
                raise ValueError(
                    "num_trajectories must be divisible by "
                    "num_cs_per_initial_condition when num_initial_conditions "
                    "is not provided."
                )
            num_initial_conditions = num_trajectories // num_cs_per_initial_condition
    if num_initial_conditions <= 0:
        raise ValueError(
            "num_initial_conditions must be positive, "
            f"got {num_initial_conditions}."
        )

    c_min, c_max = _c_sampling_bounds(c_values)
    rng = np.random.default_rng(seed)
    centers = _signed_unit_centers()
    center_indices = rng.integers(0, len(centers), size=num_initial_conditions)
    selected_centers = centers[center_indices]
    initial_conditions = selected_centers + rng.normal(
        loc=0.0,
        scale=initial_noise_std,
        size=(num_initial_conditions, 3),
    )
    cs = rng.uniform(
        c_min,
        c_max,
        size=(num_initial_conditions, num_cs_per_initial_condition),
    )

    states = []
    times = None

    for init_idx, y0 in enumerate(
        tqdm(initial_conditions, desc="Simulating Rossler initial conditions")
    ):
        state_samples = []
        for c in cs[init_idx]:
            sim = simulate_rossler(
                t_max=t_max,
                dt=dt,
                y0=tuple(float(value) for value in y0),
                a=a,
                b=b,
                c=float(c),
            )

            state_samples.append(sim["states"].astype(np.float32))
            times = sim["t"].astype(np.float32)
        states.append(np.stack(state_samples))

    return {
        "states": np.stack(states),
        "initial_conditions": initial_conditions.astype(np.float32),
        "initial_condition_centers": selected_centers.astype(np.float32),
        "initial_condition_center_indices": center_indices.astype(np.int64),
        "cs": cs.astype(np.float32),
        "times": times,
    }


def _set_equal_3d_axes(ax, trajectories: np.ndarray) -> None:
    points = trajectories.reshape(-1, trajectories.shape[-1])
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius <= 0.0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def overview_trajectory(
    real_trajs: np.ndarray,
    model_trajs: np.ndarray,
    state_names: Sequence[str],
    max_contexts: int = 1,
    title: str | None = None,
):
    del state_names
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = _as_context_samples(real_trajs)
    model = _as_context_samples(model_trajs)
    figures = {}
    for context_idx in range(min(max_contexts, real.shape[0], model.shape[0])):
        fig = plt.figure(figsize=(10, 4.5))
        for plot_idx, (samples, label) in enumerate(
            ((real[context_idx], "ground truth"), (model[context_idx], "generated")),
            start=1,
        ):
            ax = fig.add_subplot(1, 2, plot_idx, projection="3d")
            for trajectory, color in zip(samples, _overview_colors(len(samples))):
                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    trajectory[:, 2],
                    color=color,
                    linewidth=0.7,
                )
            _set_equal_3d_axes(ax, samples)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(label)
            ax.view_init(elev=24, azim=38)
        fig.suptitle(f"{title or 'Rossler 3D'} context {context_idx}")
        fig.tight_layout()
        figures[f"context_{context_idx}"] = fig
    return figures


def paired_trajectory_grid(
    real_trajs: np.ndarray,
    model_trajs: np.ndarray,
    state_names: Sequence[str],
    max_contexts: int = 3,
    max_trajectories: int = 20,
    title: str | None = None,
):
    del state_names
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = _as_context_samples(real_trajs)
    model = _as_context_samples(model_trajs)
    figures = {}
    for context_idx in range(min(max_contexts, real.shape[0], model.shape[0])):
        num_trajectories = min(max_trajectories, real.shape[1], model.shape[1])
        fig = plt.figure(figsize=(10, 3.2 * num_trajectories))
        for row_idx in range(num_trajectories):
            pair = np.stack(
                [real[context_idx, row_idx], model[context_idx, row_idx]]
            )
            for col_idx, (trajectory, label) in enumerate(
                ((pair[0], "ground truth"), (pair[1], "generated"))
            ):
                ax = fig.add_subplot(
                    num_trajectories,
                    2,
                    2 * row_idx + col_idx + 1,
                    projection="3d",
                )
                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    trajectory[:, 2],
                    linewidth=0.7,
                )
                _set_equal_3d_axes(ax, pair)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                if row_idx == 0:
                    ax.set_title(label)
                ax.view_init(elev=24, azim=38)
        fig.suptitle(f"{title or 'Rossler 3D pairs'} context {context_idx}", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        figures[f"context_{context_idx}"] = fig
    return figures


def save_trajectory_bank_projection_grid(
    bank: Mapping[str, np.ndarray],
    output_dir: Path,
    n_rows: int = 10,
    n_cols: int = 3,
    output_name: str = "rossler_dataset_xy_projection_grid.png",
    eval_anchor_index: Optional[int] = 150,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"n_rows and n_cols must be positive, got {n_rows}x{n_cols}.")

    states, initial_conditions, cs, initial_condition_indices = (
        _flatten_grouped_plot_arrays(
            np.asarray(bank["states"]),
            np.asarray(bank["initial_conditions"]),
            np.asarray(bank["cs"]),
        )
    )

    if states.ndim != 3 or states.shape[-1] != 3:
        raise ValueError(
            "bank['states'] must have shape "
            f"(num_trajectories, num_steps, 3), got {states.shape}."
        )

    index_grid = _select_grouped_plot_index_grid(
        states,
        initial_condition_indices,
        n_rows,
        n_cols,
        "c",
    )
    plot_indices = index_grid.reshape(-1)
    n_plots = plot_indices.size
    if (
        initial_conditions.shape[0] <= int(np.max(plot_indices))
        or cs.shape[0] <= int(np.max(plot_indices))
    ):
        raise ValueError(
            "initial_conditions and cs must contain one entry per plotted trajectory."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 2.15 * n_rows),
        sharex=False,
        sharey=False,
    )
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    for plot_idx in range(n_plots):
        row_idx, col_idx = divmod(plot_idx, n_cols)
        ax = axes[row_idx, col_idx]
        trajectory_idx = int(plot_indices[plot_idx])
        trajectory = states[trajectory_idx]
        initial_condition = initial_conditions[trajectory_idx]
        c = float(cs[trajectory_idx])

        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            linewidth=0.35,
            color="#1f77b4",
        )
        ax.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            s=28,
            color="#dc2626",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        anchor_label = ""
        if (
            eval_anchor_index is not None
            and 0 <= eval_anchor_index < trajectory.shape[0]
        ):
            anchor = trajectory[int(eval_anchor_index)]
            ax.scatter(
                anchor[0],
                anchor[1],
                s=38,
                marker="X",
                color="#111827",
                edgecolors="white",
                linewidths=0.65,
                zorder=6,
            )
            anchor_label = f"\nanchor={int(eval_anchor_index)}"
        ax.text(
            0.02,
            0.98,
            f"y0=({initial_condition[0]:.2f}, "
            f"{initial_condition[1]:.2f}, {initial_condition[2]:.2f})\n"
            f"c={c:.3f}{anchor_label}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
        )
        ax.set_title(f"trajectory {trajectory_idx}", fontsize=9)
        if row_idx == n_rows - 1:
            ax.set_xlabel("x")
        if col_idx == 0:
            ax.set_ylabel("y")
        ax.grid(alpha=0.18)

    fig.suptitle("Rossler Dataset x-y Projections: Rows Share Initial State", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_trajectory_bank_3d_overlap_plot(
    bank: Mapping[str, np.ndarray],
    output_dir: Path,
    n_rows: int = 10,
    n_cols: int = 3,
    output_name: str = "rossler_dataset_3d_overlap.png",
    eval_anchor_index: Optional[int] = 150,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"n_rows and n_cols must be positive, got {n_rows}x{n_cols}.")

    states, initial_conditions, cs, initial_condition_indices = (
        _flatten_grouped_plot_arrays(
            np.asarray(bank["states"]),
            np.asarray(bank["initial_conditions"]),
            np.asarray(bank["cs"]),
        )
    )

    if states.ndim != 3 or states.shape[-1] != 3:
        raise ValueError(
            "bank['states'] must have shape "
            f"(num_trajectories, num_steps, 3), got {states.shape}."
        )

    index_grid = _select_grouped_plot_index_grid(
        states,
        initial_condition_indices,
        n_rows,
        n_cols,
        "c",
    )
    if (
        initial_conditions.shape[0] <= int(np.max(index_grid))
        or cs.shape[0] <= int(np.max(index_grid))
    ):
        raise ValueError(
            "initial_conditions and cs must contain one entry per plotted trajectory."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    subplot_cols = 1 if n_rows == 1 else 2
    subplot_rows = int(np.ceil(n_rows / subplot_cols))
    base_colors = np.asarray(
        [
            "#2563eb",
            "#dc2626",
            "#16a34a",
            "#7c3aed",
            "#0891b2",
            "#db2777",
            "#111827",
            "#14b8a6",
        ]
    )
    colors = base_colors[np.arange(n_cols) % len(base_colors)]

    fig = plt.figure(figsize=(5.8 * subplot_cols, 4.6 * subplot_rows))
    for row_idx, trajectory_indices in enumerate(index_grid):
        ax = fig.add_subplot(subplot_rows, subplot_cols, row_idx + 1, projection="3d")
        selected = states[trajectory_indices]
        initial_condition = initial_conditions[int(trajectory_indices[0])]

        for color, trajectory_idx in zip(colors, trajectory_indices):
            trajectory = states[int(trajectory_idx)]
            c = float(cs[int(trajectory_idx)])
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                linewidth=0.45,
                alpha=0.7,
                color=color,
                label=f"c={c:.3f}",
            )
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                trajectory[0, 2],
                s=12,
                color=color,
                edgecolors="white",
                linewidths=0.35,
            )
            if (
                eval_anchor_index is not None
                and 0 <= eval_anchor_index < trajectory.shape[0]
            ):
                anchor = trajectory[int(eval_anchor_index)]
                ax.scatter(
                    anchor[0],
                    anchor[1],
                    anchor[2],
                    s=24,
                    marker="X",
                    color="#111827",
                    edgecolors="white",
                    linewidths=0.5,
                )

        _set_equal_3d_axes(ax, selected)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=24, azim=38)
        anchor_title = (
            "" if eval_anchor_index is None else f", anchor={int(eval_anchor_index)}"
        )
        ax.set_title(
            f"init {row_idx}: y0=({initial_condition[0]:.2f}, "
            f"{initial_condition[1]:.2f}, {initial_condition[2]:.2f})"
            f"{anchor_title}",
            fontsize=8,
        )
        ax.legend(fontsize=6, loc="upper left")

    fig.suptitle("Rossler 3D Overlap by Initial Condition", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def load_trajectory_bank(dataset_path: Path) -> Dict[str, np.ndarray]:
    loaded = np.load(dataset_path)
    return {key: loaded[key] for key in loaded.files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or load a Rossler trajectory bank and plot dataset trajectories."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Dataset YAML config to use.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Folder containing/saving the dataset .npz named by the config file_name.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="rossler_dataset_xy_projection_grid.png",
    )
    parser.add_argument(
        "--overlap-output-name",
        type=str,
        default="rossler_dataset_3d_overlap.png",
    )
    parser.add_argument("--n-rows", type=int, default=10)
    parser.add_argument("--n-cols", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_dataset_config(args.config_path)
    output_dir = Path("outputs") / str(cfg.get("name", "rossler"))
    is_deterministic = str(cfg.get("kind", "probabilistic")) == "deterministic"

    num_initial_conditions = int(cfg.get("num_initial_conditions", args.n_rows))
    num_cs_per_initial_condition = int(
        cfg.get("num_cs_per_initial_condition", args.n_cols)
    )
    dataset_path = args.data_dir / str(cfg["file_name"])

    bank = None
    if dataset_path.exists() and not bool(cfg.get("regenerate", False)):
        bank = load_trajectory_bank(dataset_path)
        expected_ndim = 3 if is_deterministic else 4
        if np.asarray(bank["states"]).ndim == expected_ndim:
            print(f"Loaded dataset bank from {dataset_path}")
        else:
            print(
                "Existing Rossler bank shape does not match the config; regenerating."
            )
            bank = None
    if bank is None:
        bank = simulate_trajectory_bank(
            num_initial_conditions=num_initial_conditions,
            num_cs_per_initial_condition=num_cs_per_initial_condition,
            t_max=float(cfg["t_max"]),
            dt=float(cfg["dt"]),
            c_values=tuple(cfg["c_values"]),
            initial_noise_std=float(cfg["initial_noise_std"]),
            a=float(cfg["a"]),
            b=float(cfg["b"]),
            seed=int(cfg["seed"]),
        )
        if is_deterministic:
            bank["states"] = bank["states"][:, 0]
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dataset_path, **bank)
        print(f"Saved dataset bank to {dataset_path}")

    eval_anchor_index = cfg.get("eval_anchor_index", None)
    plot_n_rows = min(args.n_rows, num_initial_conditions)
    plot_n_cols = min(args.n_cols, num_cs_per_initial_condition)

    output_path = save_trajectory_bank_projection_grid(
        bank=bank,
        output_dir=output_dir,
        n_rows=plot_n_rows,
        n_cols=plot_n_cols,
        output_name=args.output_name,
        eval_anchor_index=eval_anchor_index,
    )
    print(f"Saved {output_path}")
    overlap_output_path = save_trajectory_bank_3d_overlap_plot(
        bank=bank,
        output_dir=output_dir,
        n_rows=plot_n_rows,
        n_cols=plot_n_cols,
        output_name=args.overlap_output_name,
        eval_anchor_index=eval_anchor_index,
    )
    print(f"Saved {overlap_output_path}")


if __name__ == "__main__":
    main()
