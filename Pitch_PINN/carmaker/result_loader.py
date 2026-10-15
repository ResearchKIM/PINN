from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from Pitch_PINN.data.preprocess import preprocess_scenario
from Pitch_PINN.data.scenario import ScenarioData


def load_carmaker_result(
    path: Path,
    keys: Mapping[str, str],
    *,
    scenario_name: str,
    vehicle: str,
) -> ScenarioData:
    content = loadmat(Path(path))
    required = ("time", "ax", "pitch_rate")
    missing = [keys[name] for name in required if keys.get(name) not in content]
    if missing:
        raise KeyError(f"CarMaker result is missing signals: {missing}")
    loaded = ScenarioData(
        np.asarray(content[keys["time"]]).reshape(-1),
        np.asarray(content[keys["ax"]]).reshape(-1),
        np.asarray(content[keys["pitch_rate"]]).reshape(-1),
        scenario_name, vehicle,
    )
    return preprocess_scenario(loaded)
