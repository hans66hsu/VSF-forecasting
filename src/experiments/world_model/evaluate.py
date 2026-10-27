from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from stable_worldmodel.solver import CEMSolver

from src.algo.common.jepa import FrozenJEPA
from src.experiments.world_model.embed import image_transform
from src.experiments.world_model.train import (
    FLOW_PREDICTOR_NAMES,
    SF_PREDICTOR_NAME,
    TRAINABLE_PREDICTOR_NAMES,
    VSF_PREDICTOR_NAME,
    build_sf_model,
    build_vsf_model,
    load_predictor_checkpoint,
    predictor_checkpoint_path,
    predictor_names,
    train_world_model,
    validate_world_model_config,
)
from src.utils.utils import get_default_device, set_seed
from src.utils.value_utils import jsonable

SEED = 100


def make_world(cfg: DictConfig, num_envs: int):
    return swm.World(
        **OmegaConf.to_container(cfg.dataset.world, resolve=True),
        num_envs=num_envs,
        max_episode_steps=2 * int(cfg.eval.eval_budget),
        image_shape=(int(cfg.dataset.image_size),) * 2,
    )


def _sample_tasks(
    dataset,
    count: int,
    goal_offset: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample valid (episode, start_step) evaluation tasks whose goal at
    start_step + goal_offset remains is in the same trajectory.
    """
    valid_counts = np.maximum(dataset.lengths - goal_offset, 0)
    cumulative = np.concatenate([[0], np.cumsum(valid_counts)])
    total = int(cumulative[-1])
    reference_total = total - 1  # Match LeWM's released evaluation sampler.
    if count > reference_total:
        raise ValueError(
            f"Requested {count} tasks, but only {reference_total} are valid."
        )

    # This is equivalent to the reference's sorted sample from valid HDF5 rows.
    chosen = np.sort(
        np.random.default_rng(seed).choice(reference_total, size=count, replace=False)
    )
    episodes = np.searchsorted(cumulative, chosen, side="right") - 1
    starts = chosen - cumulative[episodes]
    return episodes.astype(np.int64), starts.astype(np.int64)


def _evaluate_predictor(
    cfg: DictConfig,
    predictor_name: str,
    device: torch.device,
    dataset,
    process: dict,
    run_dir: Path,
    seed: int,
    epoch: int | None = None,
) -> dict:
    if predictor_name == SF_PREDICTOR_NAME:
        model = build_sf_model(cfg)
    elif predictor_name == VSF_PREDICTOR_NAME:
        model = build_vsf_model(cfg)
    else:
        model = FrozenJEPA.from_ar_checkpoint(to_absolute_path(cfg.jepa.checkpoint_dir))
    if predictor_name in TRAINABLE_PREDICTOR_NAMES:
        load_predictor_checkpoint(
            model,
            cfg,
            predictor_name,
            seed,
            use_ema=predictor_name in FLOW_PREDICTOR_NAMES,
            epoch=epoch,
        )
    model.to(device).eval().requires_grad_(False)

    solver_cfg = cfg.planning.solver
    solver = CEMSolver(
        model=model,
        batch_size=int(solver_cfg.batch_size),
        num_samples=int(solver_cfg.num_samples),
        var_scale=float(solver_cfg.var_scale),
        n_steps=int(solver_cfg.n_steps),
        topk=int(solver_cfg.topk),
        device=device,
        seed=seed,
    )
    policy_kwargs = dict(
        solver=solver,
        config=swm.PlanConfig(
            horizon=int(cfg.planning.horizon),
            receding_horizon=int(cfg.planning.receding_horizon),
            action_block=int(cfg.planning.action_block),
        ),
        process=process,
        transform={
            "pixels": image_transform(int(cfg.dataset.image_size)),
            "goal": image_transform(int(cfg.dataset.image_size)),
        },
    )
    policy = swm.policy.WorldModelPolicy(**policy_kwargs)
    world = make_world(cfg, int(cfg.eval.num_eval))
    world.set_policy(policy)

    episodes, starts = _sample_tasks(
        dataset,
        int(cfg.eval.num_eval),
        int(cfg.eval.goal_offset_steps),
        seed,
    )
    output_dir = run_dir / "evaluation" / predictor_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tasks.json").write_text(
        json.dumps(
            {"episode_idx": episodes.tolist(), "step_idx": starts.tolist()},
            indent=2,
        )
        + "\n"
    )

    started = time.perf_counter()
    callables = OmegaConf.to_container(cfg.dataset.eval_callables, resolve=True)
    try:
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=episodes.tolist(),
            start_steps=starts.tolist(),
            goal_offset=int(cfg.eval.goal_offset_steps),
            eval_budget=int(cfg.eval.eval_budget),
            callables=callables,
            video=output_dir / "videos" if bool(cfg.eval.video) else None,
        )
    finally:
        world.close()

    evaluation_seconds = time.perf_counter() - started
    result = {
        "predictor": predictor_name,
        "seed": seed,
        "metrics": metrics,
        "evaluation_seconds": evaluation_seconds,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (output_dir / "results.json").write_text(
        json.dumps(
            jsonable(result, non_finite_to_none=True, recurse_numpy=True),
            indent=2,
        )
        + "\n"
    )
    metrics_json = json.dumps(
        jsonable(metrics, non_finite_to_none=True, recurse_numpy=True),
        indent=2,
    )
    print(f"{predictor_name}:\n{metrics_json}")
    print(f"saved {output_dir / 'results.json'}")
    return metrics


def evaluate_world_model(
    cfg: DictConfig,
    run_dir: Path | None = None,
    seed: int = SEED,
    epoch: int | None = None,
) -> dict[str, dict]:
    validate_world_model_config(cfg, evaluation=True)
    names = predictor_names(cfg)
    seed = int(seed)
    device = get_default_device(cfg.device)
    set_seed(seed)
    run_dir = Path.cwd() if run_dir is None else Path(run_dir)
    print(f"Using seed: {seed}")
    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")

    data_path = Path(to_absolute_path(cfg.dataset.path))
    dataset = swm.data.HDF5Dataset(
        path=data_path,
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )
    process = {}
    for column in cfg.dataset.keys_to_cache:
        values = dataset.get_col_data(column)
        values = values[~np.isnan(values).any(axis=1)]
        process[column] = preprocessing.StandardScaler().fit(values)
        if column != "action":
            process[f"goal_{column}"] = process[column]

    return {
        name: _evaluate_predictor(
            cfg,
            name,
            device,
            dataset,
            process,
            run_dir,
            seed,
            epoch,
        )
        for name in names
    }


def run_world_model(
    cfg: DictConfig,
    run_dir: Path | None = None,
    seed: int = SEED,
) -> dict[str, dict]:
    target_epoch = int(cfg.eval.epoch)
    for predictor_name in predictor_names(cfg):
        if predictor_name not in TRAINABLE_PREDICTOR_NAMES:
            continue
        checkpoint_path = predictor_checkpoint_path(
            cfg, predictor_name, seed, epoch=target_epoch
        )
        if not checkpoint_path.is_file():
            print(f"Checkpoint missing: {checkpoint_path}; training model.")
            train_world_model(cfg, seed=seed, predictor_name=predictor_name)
    return evaluate_world_model(cfg, run_dir=run_dir, seed=seed, epoch=target_epoch)


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="world_model",
)
def run(cfg: DictConfig) -> dict[str, dict]:
    return run_world_model(cfg, seed=SEED)


if __name__ == "__main__":
    run()
