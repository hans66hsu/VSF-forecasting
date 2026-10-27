from __future__ import annotations

import gc
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.experiments.train_forecasting import MODEL_LABELS, run_forecasting
from src.utils.value_utils import jsonable

MetricPath = tuple[str, ...]


def _is_numeric_scalar(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.number))


def _flatten_numeric_metrics(
    value: Any,
    prefix: MetricPath = (),
) -> dict[MetricPath, float]:
    if isinstance(value, Mapping):
        flattened: dict[MetricPath, float] = {}
        for key, nested_value in value.items():
            flattened.update(
                _flatten_numeric_metrics(nested_value, prefix + (str(key),))
            )
        return flattened

    if isinstance(value, np.ndarray):
        if value.ndim == 0 and np.issubdtype(value.dtype, np.number):
            return {prefix: float(value)}
        if np.issubdtype(value.dtype, np.number):
            return {
                prefix + tuple(str(index) for index in indices): float(item)
                for indices, item in np.ndenumerate(value)
            }
        return {}

    if isinstance(value, (list, tuple)):
        flattened = {}
        for index, nested_value in enumerate(value):
            flattened.update(
                _flatten_numeric_metrics(nested_value, prefix + (str(index),))
            )
        return flattened

    if prefix and _is_numeric_scalar(value):
        return {prefix: float(value)}
    return {}


def _insert_nested_metric(
    output: dict[str, Any],
    path: MetricPath,
    value: Mapping[str, float | int],
) -> None:
    current = output
    for name in path[:-1]:
        current = current.setdefault(name, {})
    current[path[-1]] = dict(value)


def aggregate_seed_metrics(
    seed_results: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    values_by_model: dict[str, dict[MetricPath, list[float]]] = {}
    for metrics_by_model in seed_results.values():
        for model_name, metrics in metrics_by_model.items():
            model_values = values_by_model.setdefault(model_name, {})
            for path, value in _flatten_numeric_metrics(metrics).items():
                model_values.setdefault(path, []).append(value)

    summary: dict[str, Any] = {}
    for model_name, metrics in sorted(values_by_model.items()):
        model_summary: dict[str, Any] = {}
        for path, values in sorted(metrics.items()):
            values_array = np.asarray(values, dtype=np.float64)
            _insert_nested_metric(
                model_summary,
                path,
                {
                    "mean": float(np.mean(values_array)),
                    "std": float(np.std(values_array)),
                    "n": int(values_array.size),
                },
            )
        summary[model_name] = model_summary
    return summary


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n")


def _print_summary_tree(tree: Mapping[str, Any], indent: int = 2) -> None:
    for key, value in tree.items():
        if isinstance(value, Mapping) and {"mean", "std", "n"}.issubset(value):
            print(
                f"{' ' * indent}{key}: "
                f"mean={value['mean']:.6g}, std={value['std']:.6g}, "
                f"n={value['n']}"
            )
        elif isinstance(value, Mapping):
            print(f"{' ' * indent}{key}:")
            _print_summary_tree(value, indent + 2)


def _print_summary(summary: Mapping[str, Any]) -> None:
    if not summary:
        print("\nNo metrics were returned. Is eval.enabled=false?")
        return

    print("\nMetric summary over seeds")
    for model_name, metrics in summary.items():
        label = MODEL_LABELS.get(model_name, model_name)
        print(f"\n{label} ({model_name})")
        _print_summary_tree(metrics)


def _configured_seeds(cfg: DictConfig) -> list[int]:
    return [int(seed) for seed in cfg.seeds]


def _run_pipeline(
    cfg: DictConfig,
    run_dir: Path,
    seed: int,
) -> Mapping[str, Mapping[str, Any]]:
    if str(cfg.dataset.name) in {"pusht", "cube", "tworoom", "reacher"}:
        from src.experiments.world_model.evaluate import run_world_model

        return run_world_model(cfg, run_dir=run_dir, seed=seed)
    return run_forecasting(cfg, run_dir=run_dir, seed=seed)


@hydra.main(version_base=None, config_path="configs", config_name="forecast")
def main(cfg: DictConfig) -> None:
    run_dir = Path.cwd()
    seeds = _configured_seeds(cfg)
    print(f"Running {len(seeds)} seed(s): {seeds}")
    print(f"Multi-seed run directory: {run_dir}")

    seed_results: dict[int, Mapping[str, Mapping[str, Any]]] = {}
    for seed in seeds:
        seed_run_dir = run_dir / f"seed_{seed}"
        seed_run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Seed {seed} ===")
        seed_results[seed] = _run_pipeline(cfg, seed_run_dir, seed)
        _save_json(run_dir / "seed_metrics.json", seed_results)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = aggregate_seed_metrics(seed_results)
    _save_json(run_dir / "metrics_summary.json", summary)
    _print_summary(summary)
    print(f"\nSaved seed metrics to {run_dir / 'seed_metrics.json'}")
    print(f"Saved metric summary to {run_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
