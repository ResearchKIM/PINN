from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SkyhookConfig:
    enabled: bool = False
    gain_nms_per_rad: float = 0.0
    max_moment_nm: float = 0.0

    def __post_init__(self) -> None:
        if self.gain_nms_per_rad < 0.0 or self.max_moment_nm < 0.0:
            raise ValueError("skyhook gain and moment limit must be non-negative")


def skyhook_moment(
    pitch_rate: Any, gain_nms_per_rad: float, limit_nm: float
) -> Any:
    """Oppose body pitch velocity and saturate the commanded moment."""

    if gain_nms_per_rad < 0.0 or limit_nm < 0.0:
        raise ValueError("skyhook gain and limit must be non-negative")
    try:
        import torch

        if isinstance(pitch_rate, torch.Tensor):
            return torch.clamp(-gain_nms_per_rad * pitch_rate, -limit_nm, limit_nm)
    except ImportError:
        pass
    values = np.asarray(pitch_rate, dtype=float)
    return np.clip(-gain_nms_per_rad * values, -limit_nm, limit_nm)

