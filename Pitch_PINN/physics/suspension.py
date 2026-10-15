from __future__ import annotations

from dataclasses import dataclass, fields

from .half_car import EffectivePitchParameters


@dataclass(frozen=True)
class VehiclePitchSpec:
    pitch_inertia_kgm2: float | None = None
    cg_to_front_axle_m: float | None = None
    cg_to_rear_axle_m: float | None = None
    baseline_k_front_npm: float | None = None
    baseline_k_rear_npm: float | None = None
    baseline_c_front_nspm: float | None = None
    baseline_c_rear_nspm: float | None = None

    @property
    def is_complete(self) -> bool:
        values = [getattr(self, item.name) for item in fields(self)]
        return all(value is not None and value > 0.0 for value in values)


@dataclass(frozen=True)
class SuspensionScales:
    rotational_stiffness_nm_per_rad: float
    rotational_damping_nms_per_rad: float
    alpha_k: float
    alpha_c: float


def effective_to_suspension_scales(
    parameters: EffectivePitchParameters, spec: VehiclePitchSpec
) -> SuspensionScales | None:
    """Convert normalized coefficients when every required vehicle value exists."""

    if not spec.is_complete:
        return None

    inertia = float(spec.pitch_inertia_kgm2)
    front_arm = float(spec.cg_to_front_axle_m)
    rear_arm = float(spec.cg_to_rear_axle_m)
    stiffness = inertia * float(parameters.stiffness_rate_s2)
    damping = inertia * float(parameters.damping_rate_s_inv)
    baseline_stiffness = (
        float(spec.baseline_k_front_npm) * front_arm**2
        + float(spec.baseline_k_rear_npm) * rear_arm**2
    )
    baseline_damping = (
        float(spec.baseline_c_front_nspm) * front_arm**2
        + float(spec.baseline_c_rear_nspm) * rear_arm**2
    )
    if baseline_stiffness <= 0.0 or baseline_damping <= 0.0:
        raise ValueError("baseline rotational stiffness and damping must be positive")
    return SuspensionScales(
        rotational_stiffness_nm_per_rad=stiffness,
        rotational_damping_nms_per_rad=damping,
        alpha_k=stiffness / baseline_stiffness,
        alpha_c=damping / baseline_damping,
    )

