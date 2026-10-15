from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .half_car import EffectivePitchParameters


@dataclass(frozen=True)
class ReferenceTrajectory:
    time_s: np.ndarray
    pitch_angle_rad: np.ndarray
    pitch_rate_radps: np.ndarray


def simulate_reference(
    time_s: np.ndarray,
    ax_mps2: np.ndarray,
    parameters: EffectivePitchParameters,
    initial_theta_rad: float = 0.0,
    initial_pitch_rate_radps: float = 0.0,
) -> ReferenceTrajectory:
    """Integrate the effective pitch model on the requested sample times."""

    time = np.asarray(time_s, dtype=float).reshape(-1)
    ax = np.asarray(ax_mps2, dtype=float).reshape(-1)
    if time.size != ax.size or time.size < 2:
        raise ValueError("time and Ax need equal lengths of at least two")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(ax)):
        raise ValueError("time and Ax must be finite")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("time must be strictly increasing")

    damping = float(parameters.damping_rate_s_inv)
    stiffness = float(parameters.stiffness_rate_s2)
    gain = float(parameters.ax_gain_rad_per_m)

    def ode(current_time: float, state: np.ndarray) -> list[float]:
        theta, pitch_rate = state
        current_ax = float(np.interp(current_time, time, ax))
        pitch_acceleration = gain * current_ax - damping * pitch_rate - stiffness * theta
        return [pitch_rate, pitch_acceleration]

    solution = solve_ivp(
        ode,
        (float(time[0]), float(time[-1])),
        [initial_theta_rad, initial_pitch_rate_radps],
        t_eval=time,
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return ReferenceTrajectory(time.copy(), solution.y[0], solution.y[1])
