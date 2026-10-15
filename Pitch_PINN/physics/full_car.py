from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch


@dataclass(frozen=True)
class FullCarSpec:
    sprung_mass_kg: float
    roll_inertia_kgm2: float
    pitch_inertia_kgm2: float
    cg_to_front_m: float
    cg_to_rear_m: float
    front_track_m: float
    rear_track_m: float
    cg_height_m: float

    def __post_init__(self) -> None:
        if any(float(v) <= 0 for v in self.__dict__.values()):
            raise ValueError("all full-car specification values must be positive")

    @property
    def x(self) -> tuple[float, ...]:
        return (self.cg_to_front_m, self.cg_to_front_m, -self.cg_to_rear_m, -self.cg_to_rear_m)

    @property
    def y(self) -> tuple[float, ...]:
        return (self.front_track_m / 2, -self.front_track_m / 2, self.rear_track_m / 2, -self.rear_track_m / 2)


@dataclass(frozen=True)
class SuspensionRates:
    k_front_npm: Any
    k_rear_npm: Any
    c_front_nspm: Any
    c_rear_nspm: Any


@dataclass(frozen=True)
class FullCarResidual:
    heave: torch.Tensor
    roll: torch.Tensor
    pitch: torch.Tensor


def corner_kinematics(state: torch.Tensor, state_rate: torch.Tensor, wheel_z: torch.Tensor, wheel_velocity: torch.Tensor, spec: FullCarSpec) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(spec.x, dtype=state.dtype, device=state.device)
    y = torch.as_tensor(spec.y, dtype=state.dtype, device=state.device)
    body_z = state[:, 0:1] + state[:, 1:2] * y - state[:, 2:3] * x
    body_v = state_rate[:, 0:1] + state_rate[:, 1:2] * y - state_rate[:, 2:3] * x
    return wheel_z - body_z, wheel_velocity - body_v


def corner_forces(stroke: torch.Tensor, stroke_rate: torch.Tensor, rates: SuspensionRates) -> torch.Tensor:
    k = torch.stack([torch.as_tensor(rates.k_front_npm, dtype=stroke.dtype, device=stroke.device)] * 2 + [torch.as_tensor(rates.k_rear_npm, dtype=stroke.dtype, device=stroke.device)] * 2)
    c = torch.stack([torch.as_tensor(rates.c_front_nspm, dtype=stroke.dtype, device=stroke.device)] * 2 + [torch.as_tensor(rates.c_rear_nspm, dtype=stroke.dtype, device=stroke.device)] * 2)
    return stroke * k + stroke_rate * c


def full_car_residual(state_acceleration: torch.Tensor, state_rate: torch.Tensor, forces: torch.Tensor, skyhook: torch.Tensor, ax: torch.Tensor, spec: FullCarSpec) -> FullCarResidual:
    del state_rate, skyhook
    x = torch.as_tensor(spec.x, dtype=forces.dtype, device=forces.device)
    y = torch.as_tensor(spec.y, dtype=forces.dtype, device=forces.device)
    return FullCarResidual(
        spec.sprung_mass_kg * state_acceleration[:, 0:1] - forces.sum(1, keepdim=True),
        spec.roll_inertia_kgm2 * state_acceleration[:, 1:2] - (forces * y).sum(1, keepdim=True),
        spec.pitch_inertia_kgm2 * state_acceleration[:, 2:3] + (forces * x).sum(1, keepdim=True) + spec.sprung_mass_kg * spec.cg_height_m * ax,
    )
