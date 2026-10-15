from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .scenario import ScenarioData


def load_mat_scenario(
    path: Path,
    keys: Mapping[str, str],
    *,
    name: str,
    vehicle: str,
) -> ScenarioData:
    """Load three configured MATLAB signals without guessing names or units."""

    mat_path = Path(path)
    if not mat_path.is_file():
        raise FileNotFoundError(mat_path)
    required = {"time", "ax", "pitch_rate", "wheel_velocity", "wheel_acceleration", "damper_length", "damper_velocity"}
    missing_mapping = required.difference(keys)
    if missing_mapping:
        raise KeyError(f"missing MAT key mappings: {sorted(missing_mapping)}")

    content = loadmat(mat_path)
    corner_names = ("fl", "fr", "rl", "rr")
    scalar_names = ("time", "ax", "pitch_rate")
    missing_signals = [keys[item] for item in scalar_names if keys[item] not in content]
    for group in ("wheel_velocity", "wheel_acceleration", "damper_length", "damper_velocity"):
        mapping = keys[group]
        if not isinstance(mapping, Mapping) or any(c not in mapping for c in corner_names):
            raise KeyError(f"{group} must map fl, fr, rl, rr")
        missing_signals.extend(mapping[c] for c in corner_names if mapping[c] not in content)
    if missing_signals:
        raise KeyError(f"MAT file is missing signals: {missing_signals}")

    return ScenarioData(
        time_s=np.asarray(content[keys["time"]], dtype=float).reshape(-1),
        ax_mps2=np.asarray(content[keys["ax"]], dtype=float).reshape(-1),
        pitch_rate_radps=np.asarray(
            content[keys["pitch_rate"]], dtype=float
        ).reshape(-1),
        name=name,
        vehicle=vehicle,
        wheel_velocity_mps=np.column_stack([np.asarray(content[keys["wheel_velocity"][c]]).reshape(-1) for c in corner_names]),
        wheel_acceleration_mps2=np.column_stack([np.asarray(content[keys["wheel_acceleration"][c]]).reshape(-1) for c in corner_names]),
        damper_length_m=np.column_stack([np.asarray(content[keys["damper_length"][c]]).reshape(-1) for c in corner_names]),
        damper_velocity_mps=np.column_stack([np.asarray(content[keys["damper_velocity"][c]]).reshape(-1) for c in corner_names]),
    )
