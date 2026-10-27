from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


def as_tuple(values: Iterable[Any]) -> tuple:
    """Resolve an OmegaConf collection and return its values as a tuple."""
    if OmegaConf.is_config(values):
        values = OmegaConf.to_container(values, resolve=True)
    return tuple(values)


def jsonable(
    value: Any,
    *,
    non_finite_to_none: bool = False,
    recurse_numpy: bool = False,
) -> Any:
    """Convert nested project values to structures accepted by ``json``."""
    if isinstance(value, Mapping):
        return {
            str(key): jsonable(
                item,
                non_finite_to_none=non_finite_to_none,
                recurse_numpy=recurse_numpy,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            jsonable(
                item,
                non_finite_to_none=non_finite_to_none,
                recurse_numpy=recurse_numpy,
            )
            for item in value
        ]
    if isinstance(value, np.ndarray):
        converted = value.tolist()
        if recurse_numpy:
            return jsonable(
                converted,
                non_finite_to_none=non_finite_to_none,
                recurse_numpy=True,
            )
        return converted
    if isinstance(value, np.generic):
        converted = value.item()
        if recurse_numpy:
            return jsonable(
                converted,
                non_finite_to_none=non_finite_to_none,
                recurse_numpy=True,
            )
        return converted
    if non_finite_to_none and isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value
