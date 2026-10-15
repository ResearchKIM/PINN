from __future__ import annotations

import numpy as np

from .scenario import ScenarioData, SparseScenario


def preprocess_scenario(
    data: ScenarioData, *, remove_bias_samples: int = 0, baseline_samples: int = 0,
    drop_last_samples: int = 0,
) -> ScenarioData:
    """Remove invalid rows, sort time, collapse duplicates, and optionally de-bias."""

    finite = np.isfinite(data.time_s) & np.isfinite(data.ax_mps2) & np.isfinite(data.pitch_rate_radps)
    finite &= np.all(np.isfinite(data.wheel_velocity_mps), axis=1)
    finite &= np.all(np.isfinite(data.wheel_acceleration_mps2), axis=1)
    finite &= np.all(np.isfinite(data.damper_length_m), axis=1)
    finite &= np.all(np.isfinite(data.damper_velocity_mps), axis=1)
    if not np.any(finite):
        raise ValueError("scenario contains no finite Ax/Pitch Rate samples")

    time = data.time_s[finite]
    ax = data.ax_mps2[finite]
    pitch_rate = data.pitch_rate_radps[finite]
    wheel_velocity = data.wheel_velocity_mps[finite]
    wheel_acceleration = data.wheel_acceleration_mps2[finite]
    damper_length = data.damper_length_m[finite]
    damper_velocity = data.damper_velocity_mps[finite]
    order = np.argsort(time, kind="stable")
    time = time[order]
    ax = ax[order]
    pitch_rate = pitch_rate[order]
    wheel_velocity = wheel_velocity[order]
    wheel_acceleration = wheel_acceleration[order]
    damper_length = damper_length[order]
    damper_velocity = damper_velocity[order]

    keep_last = np.r_[time[1:] != time[:-1], True]
    time = time[keep_last]
    ax = ax[keep_last]
    pitch_rate = pitch_rate[keep_last]
    wheel_velocity = wheel_velocity[keep_last]
    wheel_acceleration = wheel_acceleration[keep_last]
    damper_length = damper_length[keep_last]
    damper_velocity = damper_velocity[keep_last]
    if time.size < 2:
        raise ValueError("scenario needs at least two unique timestamps")
    if drop_last_samples < 0 or drop_last_samples > time.size - 2:
        raise ValueError("drop_last_samples must leave at least two timestamps")
    if drop_last_samples:
        keep = slice(None, -drop_last_samples)
        time, ax, pitch_rate = time[keep], ax[keep], pitch_rate[keep]
        wheel_velocity, wheel_acceleration, damper_length, damper_velocity = wheel_velocity[keep], wheel_acceleration[keep], damper_length[keep], damper_velocity[keep]

    if remove_bias_samples < 0:
        raise ValueError("remove_bias_samples must be non-negative")
    if remove_bias_samples:
        count = min(remove_bias_samples, time.size)
        ax = ax - float(np.mean(ax[:count]))
        pitch_rate = pitch_rate - float(np.mean(pitch_rate[:count]))

    if baseline_samples < 0:
        raise ValueError("baseline_samples must be non-negative")
    if baseline_samples:
        count = min(baseline_samples, time.size)
        damper_length = damper_length - np.mean(damper_length[:count], axis=0, keepdims=True)
    return ScenarioData(time, ax, pitch_rate, data.name, data.vehicle, wheel_velocity, wheel_acceleration, damper_length, damper_velocity)


def select_sparse_observations(data: ScenarioData, stride: int) -> SparseScenario:
    """Retain deterministic observations while always preserving both endpoints."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    indices = np.arange(0, len(data), stride, dtype=np.int64)
    indices = np.unique(np.r_[indices, len(data) - 1])
    return SparseScenario(data, indices)
