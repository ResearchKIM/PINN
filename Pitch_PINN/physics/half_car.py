from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EffectivePitchParameters:
    """Pitch ODE coefficients normalized by pitch inertia."""

    damping_rate_s_inv: Any
    stiffness_rate_s2: Any
    ax_gain_rad_per_m: Any

    def __post_init__(self) -> None:
        for name in ("damping_rate_s_inv", "stiffness_rate_s2"):
            value = getattr(self, name)
            if isinstance(value, Real) and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        gain = self.ax_gain_rad_per_m
        if isinstance(gain, Real) and not np.isfinite(gain):
            raise ValueError("ax_gain_rad_per_m must be finite")


def pitch_ode_residual(
    theta: Any,
    theta_dot: Any,
    theta_ddot: Any,
    ax: Any,
    parameters: EffectivePitchParameters,
    *,
    skyhook_moment_value: Any | None = None,
    pitch_inertia_kgm2: float | None = None,
) -> Any:
    """Return the normalized forced-pitch ODE residual."""

    residual = (
        theta_ddot
        + parameters.damping_rate_s_inv * theta_dot
        + parameters.stiffness_rate_s2 * theta
        - parameters.ax_gain_rad_per_m * ax
    )
    if skyhook_moment_value is not None:
        if pitch_inertia_kgm2 is None or pitch_inertia_kgm2 <= 0.0:
            raise ValueError("positive pitch inertia is required for skyhook moment")
        residual = residual - skyhook_moment_value / pitch_inertia_kgm2
    return residual

