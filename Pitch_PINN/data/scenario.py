from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_vector(values: np.ndarray, field: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{field} must not be empty")
    return vector


@dataclass(frozen=True)
class ScenarioData:
    """Signals for one longitudinal pitch scenario in SI units."""

    time_s: np.ndarray
    ax_mps2: np.ndarray
    pitch_rate_radps: np.ndarray
    name: str
    vehicle: str
    wheel_velocity_mps: np.ndarray | None = None
    wheel_acceleration_mps2: np.ndarray | None = None
    damper_length_m: np.ndarray | None = None
    damper_velocity_mps: np.ndarray | None = None

    def __post_init__(self) -> None:
        time = _as_vector(self.time_s, "time_s")
        ax = _as_vector(self.ax_mps2, "ax_mps2")
        pitch_rate = _as_vector(self.pitch_rate_radps, "pitch_rate_radps")
        if not (time.size == ax.size == pitch_rate.size):
            raise ValueError("time, Ax, and Pitch Rate must have equal lengths")
        if not np.all(np.isfinite(time)):
            raise ValueError("time_s must contain only finite values")
        corner_values = []
        for field in ("wheel_velocity_mps", "wheel_acceleration_mps2", "damper_length_m", "damper_velocity_mps"):
            raw = getattr(self, field)
            value = np.zeros((time.size, 4), dtype=float) if raw is None else np.asarray(raw, dtype=float)
            if value.shape != (time.size, 4):
                raise ValueError(f"{field} must have shape ({time.size}, 4) in FL, FR, RL, RR order")
            corner_values.append((field, value))
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "ax_mps2", ax)
        object.__setattr__(self, "pitch_rate_radps", pitch_rate)
        for field, value in corner_values:
            object.__setattr__(self, field, value)

    def __len__(self) -> int:
        return self.time_s.size


@dataclass(frozen=True)
class SparseScenario:
    """A full scenario and deterministic indices exposed during training."""

    full: ScenarioData
    observation_indices: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.observation_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("observation_indices must not be empty")
        if np.any(indices < 0) or np.any(indices >= len(self.full)):
            raise ValueError("observation index is outside the scenario")
        if np.unique(indices).size != indices.size:
            raise ValueError("observation_indices must be unique")
        object.__setattr__(self, "observation_indices", np.sort(indices))

    @property
    def observation_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.full), dtype=bool)
        mask[self.observation_indices] = True
        return mask

    @property
    def time_s(self) -> np.ndarray:
        return self.full.time_s[self.observation_indices]

    @property
    def ax_mps2(self) -> np.ndarray:
        return self.full.ax_mps2[self.observation_indices]

    @property
    def pitch_rate_radps(self) -> np.ndarray:
        return self.full.pitch_rate_radps[self.observation_indices]
