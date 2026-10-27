import argparse
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.datasets.double_pendulum.simulation import angles_to_xy, simulate_double_pendulum
from src.datasets.plotting_utils import _as_context_samples, _overview_colors


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs" / "dataset" / "double-pendulum.yaml"
)


def load_dataset_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping config in {config_path}, got {type(loaded)}.")
    return loaded


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def simulate_trajectory_bank(
    num_trajectories: int = 192,
    t_max: float = 8.0,
    dt: float = 0.05,
    theta1_range: Tuple[float, float] = (-np.pi, np.pi),
    theta2_range: Tuple[float, float] = (-np.pi, np.pi),
    omega1_range: Tuple[float, float] = (-1.5, 1.5),
    omega2_range: Tuple[float, float] = (-1.5, 1.5),
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
    seed: int = 0,
    max_initial_angle_difference: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    states = []
    coordinates = []
    cartesian_states = []
    angles = []
    initial_conditions = []
    times = None

    for _ in range(num_trajectories):
        while True:
            theta1_0 = rng.uniform(*theta1_range)
            theta2_0 = rng.uniform(*theta2_range)
            if (
                max_initial_angle_difference is None
                or abs(theta1_0 - theta2_0) <= max_initial_angle_difference
            ):
                break
        omega1_0 = rng.uniform(*omega1_range)
        omega2_0 = rng.uniform(*omega2_range)

        sim = simulate_double_pendulum(
            t_max=t_max,
            dt=dt,
            theta1=theta1_0,
            omega1=omega1_0,
            theta2=theta2_0,
            omega2=omega2_0,
            m1=m1,
            m2=m2,
            l1=l1,
            l2=l2,
            g=g,
        )

        states.append(
            np.stack(
                [
                    sim["x1"],
                    sim["y1"],
                    sim["x2"],
                    sim["y2"],
                    sim["omega1"],
                    sim["omega2"],
                ],
                axis=-1,
            ).astype(np.float32)
        )
        coordinates.append(
            np.stack(
                [sim["x1"], sim["y1"], sim["x2"], sim["y2"]],
                axis=-1,
            ).astype(np.float32)
        )
        cartesian_states.append(
            np.stack(
                [
                    sim["x1"],
                    sim["y1"],
                    sim["x2"],
                    sim["y2"],
                    sim["vx1"],
                    sim["vy1"],
                    sim["vx2"],
                    sim["vy2"],
                ],
                axis=-1,
            ).astype(np.float32)
        )
        angles.append(
            np.stack(
                [
                    wrap_angle(sim["theta1"]),
                    sim["omega1"],
                    wrap_angle(sim["theta2"]),
                    sim["omega2"],
                ],
                axis=-1,
            ).astype(np.float32)
        )
        initial_conditions.append([theta1_0, omega1_0, theta2_0, omega2_0])
        times = sim["t"].astype(np.float32)

    return {
        "states": np.stack(states),
        "coordinates": np.stack(coordinates),
        "cartesian_states": np.stack(cartesian_states),
        "angles": np.stack(angles),
        "initial_conditions": np.asarray(initial_conditions, dtype=np.float32),
        "times": times,
    }


def load_trajectory_bank(dataset_path: Path) -> Dict[str, np.ndarray]:
    loaded = np.load(dataset_path)
    return {key: loaded[key] for key in loaded.files}


def _mass_xy(samples: np.ndarray, state_names: Sequence[str]) -> np.ndarray:
    names = tuple(state_names)
    if "theta1" in names and "theta2" in names:
        x1, y1, x2, y2 = angles_to_xy(
            samples[..., names.index("theta1")],
            samples[..., names.index("theta2")],
        )
        return np.stack([x1, y1, x2, y2], axis=-1)
    return samples[..., :4]


def overview_trajectory(
    real_trajs: np.ndarray,
    model_trajs: np.ndarray,
    state_names: Sequence[str],
    max_contexts: int = 1,
    title: str | None = None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = _as_context_samples(real_trajs)
    model = _as_context_samples(model_trajs)
    figures = {}
    for context_idx in range(min(max_contexts, real.shape[0], model.shape[0])):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), squeeze=False)
        for ax, samples, label in (
            (axes[0, 0], real[context_idx], "ground truth"),
            (axes[0, 1], model[context_idx], "generated"),
        ):
            coords = _mass_xy(samples, state_names)
            for trajectory, color in zip(coords, _overview_colors(len(coords))):
                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    "--",
                    color=color,
                    linewidth=0.8,
                )
                ax.plot(
                    trajectory[:, 2],
                    trajectory[:, 3],
                    "-",
                    color=color,
                    linewidth=0.9,
                )
            ax.scatter([0.0], [0.0], s=14, color="black")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(label)
            ax.grid(alpha=0.2)
        fig.suptitle(f"{title or 'double pendulum'} context {context_idx}")
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = _as_context_samples(real_trajs)
    model = _as_context_samples(model_trajs)
    figures = {}
    for context_idx in range(min(max_contexts, real.shape[0], model.shape[0])):
        num_trajectories = min(max_trajectories, real.shape[1], model.shape[1])
        fig, axes = plt.subplots(
            num_trajectories,
            2,
            figsize=(9, 2.4 * num_trajectories),
            squeeze=False,
        )
        for row_idx in range(num_trajectories):
            for col_idx, (samples, label) in enumerate(
                ((real[context_idx], "ground truth"), (model[context_idx], "generated"))
            ):
                ax = axes[row_idx, col_idx]
                trajectory = _mass_xy(samples[row_idx], state_names)
                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    "--",
                    linewidth=0.8,
                )
                ax.plot(
                    trajectory[:, 2],
                    trajectory[:, 3],
                    "-",
                    linewidth=0.9,
                )
                ax.scatter([0.0], [0.0], s=14, color="black")
                ax.set_aspect("equal", adjustable="box")
                if row_idx == 0:
                    ax.set_title(label)
                ax.grid(alpha=0.2)
        fig.suptitle(
            f"{title or 'double pendulum pairs'} context {context_idx}",
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        figures[f"context_{context_idx}"] = fig
    return figures


def save_mass_trajectory_grid(
    bank: Mapping[str, np.ndarray],
    output_dir: Path,
    output_name: str = "double_pendulum_mass_trajectories.png",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coordinates = np.asarray(bank["coordinates"])
    initial_conditions = np.asarray(bank["initial_conditions"])
    n_plots = coordinates.shape[0]
    n_cols = min(4, max(1, n_plots))
    n_rows = int(np.ceil(n_plots / n_cols))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.3 * n_cols, 3.3 * n_rows),
        squeeze=False,
    )

    for plot_idx, ax in enumerate(axes.ravel()):
        if plot_idx >= n_plots:
            ax.axis("off")
            continue
        trajectory = coordinates[plot_idx]
        theta1, omega1, theta2, omega2 = initial_conditions[plot_idx]
        ax.plot(trajectory[:, 0], trajectory[:, 1], linewidth=0.8, label="m1")
        ax.plot(trajectory[:, 2], trajectory[:, 3], linewidth=0.8, label="m2")
        ax.scatter([0.0], [0.0], s=12, color="black")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{plot_idx}: ({theta1:.2f}, {omega1:.2f}, {theta2:.2f}, {omega2:.2f})",
            fontsize=7,
        )
        ax.grid(alpha=0.2)
        if plot_idx == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle("Double Pendulum Mass Trajectories by Initial Condition", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or load a double-pendulum bank and plot mass trajectories."
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-name",
        type=str,
        default="double_pendulum_mass_trajectories.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_dataset_config(args.config_path)
    dataset_path = args.data_dir / str(cfg["file_name"])
    output_dir = Path("outputs") / str(cfg.get("name", "double-pendulum"))

    if dataset_path.exists() and not bool(cfg.get("regenerate", False)):
        bank = load_trajectory_bank(dataset_path)
        print(f"Loaded dataset bank from {dataset_path}")
    else:
        bank = simulate_trajectory_bank(
            num_trajectories=int(cfg["num_trajectories"]),
            t_max=float(cfg["t_max"]),
            dt=float(cfg["dt"]),
            theta1_range=tuple(cfg["theta1_range"]),
            theta2_range=tuple(cfg["theta2_range"]),
            max_initial_angle_difference=cfg.get(
                "max_initial_angle_difference",
            ),
            omega1_range=tuple(cfg["omega1_range"]),
            omega2_range=tuple(cfg["omega2_range"]),
            m1=float(cfg["m1"]),
            m2=float(cfg["m2"]),
            l1=float(cfg["l1"]),
            l2=float(cfg["l2"]),
            g=float(cfg["g"]),
            seed=int(cfg["seed"]),
        )
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dataset_path, **bank)
        print(f"Saved dataset bank to {dataset_path}")

    output_path = save_mass_trajectory_grid(
        bank,
        output_dir,
        output_name=args.output_name,
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
