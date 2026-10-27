from typing import Any, Dict, Tuple

import numpy as np

try:
    from scipy.integrate import solve_ivp
except ImportError:
    solve_ivp = None


def rossler_rhs(
    t: float,
    state: np.ndarray,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
) -> list[float]:
    del t
    x, y, z = state
    return [
        -y - z,
        x + a * y,
        b + z * (x - c),
    ]


def simulate_rossler(
    t_max: float = 30.0,
    dt: float = 0.02,
    y0: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
    method: str = "DOP853",
) -> Dict[str, Any]:
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if t_max <= 0.0:
        raise ValueError(f"t_max must be positive, got {t_max}.")
    if len(y0) != 3:
        raise ValueError(f"y0 must contain three values, got {y0}.")

    n_steps = int(np.floor(t_max / dt + 1e-12)) + 1
    t_eval = np.linspace(0.0, dt * (n_steps - 1), n_steps)

    rhs = lambda t, state: np.asarray(
        rossler_rhs(t, state, a=a, b=b, c=c),
        dtype=np.float64,
    )

    if solve_ivp is not None:
        sol = solve_ivp(
            fun=rhs,
            t_span=(0.0, float(t_eval[-1])),
            y0=y0,
            t_eval=t_eval,
            method=method,
            rtol=1e-9,
            atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(f"Rossler solve failed for c={c}: {sol.message}")
        times = sol.t
        states = sol.y.T
    else:
        states = np.empty((n_steps, 3), dtype=np.float64)
        states[0] = np.asarray(y0, dtype=np.float64)
        for idx in range(1, n_steps):
            t = t_eval[idx - 1]
            state = states[idx - 1]
            k1 = rhs(t, state)
            k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
            k3 = rhs(t + 0.5 * dt, state + 0.5 * dt * k2)
            k4 = rhs(t + dt, state + dt * k3)
            states[idx] = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        times = t_eval

    return {
        "t": times,
        "x": states[:, 0],
        "y": states[:, 1],
        "z": states[:, 2],
        "states": states,
        "params": {"a": a, "b": b, "c": c},
        "init": {"x": y0[0], "y": y0[1], "z": y0[2]},
    }
