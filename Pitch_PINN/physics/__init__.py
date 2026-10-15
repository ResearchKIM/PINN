"""Physics equations used by the pitch PINN and reference generator."""

from .half_car import EffectivePitchParameters, pitch_ode_residual
from .reference_model import ReferenceTrajectory, simulate_reference
from .skyhook import SkyhookConfig, skyhook_moment
from .suspension import (
    SuspensionScales,
    VehiclePitchSpec,
    effective_to_suspension_scales,
)

__all__ = [
    "EffectivePitchParameters",
    "ReferenceTrajectory",
    "SkyhookConfig",
    "SuspensionScales",
    "VehiclePitchSpec",
    "effective_to_suspension_scales",
    "pitch_ode_residual",
    "simulate_reference",
    "skyhook_moment",
]

