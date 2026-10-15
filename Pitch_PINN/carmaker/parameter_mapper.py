from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from Pitch_PINN.physics.full_car import SuspensionRates


@dataclass(frozen=True)
class ParameterUpdate:
    k_front_npm: float
    k_rear_npm: float
    c_front_nspm: float
    c_rear_nspm: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def build_parameter_update(parameters: SuspensionRates) -> ParameterUpdate:
    return ParameterUpdate(**{name: float(getattr(parameters, name)) for name in parameters.__dataclass_fields__})


def write_parameter_diff(update: ParameterUpdate, path: Path) -> Path:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(update.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return destination
