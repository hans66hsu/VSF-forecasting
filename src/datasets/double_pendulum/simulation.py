import numpy as np

try:
    from scipy.integrate import solve_ivp
except ImportError:
    raise ImportError("Please install scipy: pip install scipy")


# =========================================================
# 1) Double pendulum dynamics
# State = [theta1, omega1, theta2, omega2]
# =========================================================
def double_pendulum_rhs(t, state, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    theta1, omega1, theta2, omega2 = state

    delta = theta2 - theta1

    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta) ** 2
    den2 = (l2 / l1) * den1

    dtheta1 = omega1
    dtheta2 = omega2

    domega1 = (
        m2 * l1 * omega1**2 * np.sin(delta) * np.cos(delta)
        + m2 * g * np.sin(theta2) * np.cos(delta)
        + m2 * l2 * omega2**2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(theta1)
    ) / den1

    domega2 = (
        -m2 * l2 * omega2**2 * np.sin(delta) * np.cos(delta)
        + (m1 + m2)
        * (
            g * np.sin(theta1) * np.cos(delta)
            - l1 * omega1**2 * np.sin(delta)
            - g * np.sin(theta2)
        )
    ) / den2

    return [dtheta1, domega1, dtheta2, domega2]


# =========================================================
# 2) Convert angles to Cartesian coordinates
# Pivot is at (0, 0)
# =========================================================
def angles_to_xy(theta1, theta2, l1=1.0, l2=1.0):
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)

    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    return x1, y1, x2, y2


def angular_state_to_xy_velocity(theta1, omega1, theta2, omega2, l1=1.0, l2=1.0):
    vx1 = l1 * np.cos(theta1) * omega1
    vy1 = l1 * np.sin(theta1) * omega1

    vx2 = vx1 + l2 * np.cos(theta2) * omega2
    vy2 = vy1 + l2 * np.sin(theta2) * omega2

    return vx1, vy1, vx2, vy2


# =========================================================
# 3) Simulate one trajectory
# =========================================================
def simulate_double_pendulum(
    t_max=10.0,
    dt=0.01,
    theta1=np.pi / 2,
    omega1=0.0,
    theta2=np.pi / 2,
    omega2=0.0,
    m1=1.0,
    m2=1.0,
    l1=1.0,
    l2=1.0,
    g=9.81,
    method="DOP853",
):
    # Step 1: choose initial conditions.
    # The state vector is ordered as:
    # [theta1(0), omega1(0), theta2(0), omega2(0)].
    y0 = [theta1, omega1, theta2, omega2]

    # Step 2: define the time grid.
    # t_eval stores the exact times where we want the numerical solution.
    n_steps = int(np.floor(t_max / dt)) + 1
    t_eval = np.linspace(0.0, dt * (n_steps - 1), n_steps)

    # Step 3: numerically integrate the ODE system over time.
    # solve_ivp returns theta1(t), omega1(t), theta2(t), and omega2(t)
    # evaluated on the time grid above.
    sol = solve_ivp(
        fun=lambda t, y: double_pendulum_rhs(t, y, m1=m1, m2=m2, l1=l1, l2=l2, g=g),
        t_span=(0.0, float(t_eval[-1])),
        y0=y0,
        t_eval=t_eval,
        method=method,
        rtol=1e-9,
        atol=1e-9,
    )

    th1 = sol.y[0]
    om1 = sol.y[1]
    th2 = sol.y[2]
    om2 = sol.y[3]

    # Step 4: convert angles to Cartesian coordinates and velocities.
    # Once theta1(t) and theta2(t) are known, we compute:
    # x1(t), y1(t), x2(t), y2(t), and their time derivatives.
    # These Cartesian values are what we use later for plots and ML datasets.
    x1, y1, x2, y2 = angles_to_xy(th1, th2, l1=l1, l2=l2)
    vx1, vy1, vx2, vy2 = angular_state_to_xy_velocity(
        th1,
        om1,
        th2,
        om2,
        l1=l1,
        l2=l2,
    )

    return {
        "t": sol.t,
        "theta1": th1,
        "omega1": om1,
        "theta2": th2,
        "omega2": om2,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "vx1": vx1,
        "vy1": vy1,
        "vx2": vx2,
        "vy2": vy2,
        "params": {"m1": m1, "m2": m2, "l1": l1, "l2": l2, "g": g},
        "init": {
            "theta1": theta1,
            "omega1": omega1,
            "theta2": theta2,
            "omega2": omega2,
        },
    }
