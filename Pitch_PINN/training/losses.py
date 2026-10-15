from __future__ import annotations

from dataclasses import dataclass

import torch

from Pitch_PINN.models.pitch_pinn import AxInterpolator, PitchPINN
from Pitch_PINN.physics.half_car import pitch_ode_residual
from Pitch_PINN.models.pitch_pinn import FullCarPitchPINN
from Pitch_PINN.physics.full_car import FullCarSpec, corner_forces, corner_kinematics, full_car_residual


@dataclass(frozen=True)
class ObservationBatch:
    time_s: torch.Tensor
    pitch_rate_radps: torch.Tensor

    def __post_init__(self) -> None:
        if self.time_s.shape != self.pitch_rate_radps.shape:
            raise ValueError("observation time and Pitch Rate shapes must match")


@dataclass(frozen=True)
class PitchLossWeights:
    data: float = 1.0
    physics: float = 1.0
    initial: float = 1.0
    regularization: float = 0.0


@dataclass(frozen=True)
class PitchLossResult:
    total: torch.Tensor
    data: torch.Tensor
    physics: torch.Tensor
    initial: torch.Tensor
    regularization: torch.Tensor


@dataclass(frozen=True)
class FullCarLossWeights:
    pitch_data: float
    physics_heave: float
    physics_roll: float
    physics_pitch: float
    damper_length: float
    damper_velocity: float
    initial: float
    regularization: float


@dataclass(frozen=True)
class FullCarLossResult:
    total: torch.Tensor
    pitch_data: torch.Tensor
    physics_heave: torch.Tensor
    physics_roll: torch.Tensor
    physics_pitch: torch.Tensor
    damper_length: torch.Tensor
    damper_velocity: torch.Tensor
    initial: torch.Tensor
    regularization: torch.Tensor


def compute_pitch_losses(
    model: PitchPINN,
    observations: ObservationBatch,
    collocation_time_s: torch.Tensor,
    ax_interpolator: AxInterpolator,
    weights: PitchLossWeights,
) -> PitchLossResult:
    """Combine sparse observations with the forced-pitch ODE residual."""

    observation_prediction = model.predict_with_derivatives(observations.time_s)
    data_loss = torch.mean(
        (observation_prediction.pitch_rate_radps - observations.pitch_rate_radps) ** 2
    )

    physics_prediction = model.predict_with_derivatives(collocation_time_s)
    ax = ax_interpolator(physics_prediction.time_s)
    residual = pitch_ode_residual(
        physics_prediction.theta_rad,
        physics_prediction.pitch_rate_radps,
        physics_prediction.pitch_acceleration_radps2,
        ax,
        model.physical_parameters.values(),
    )
    physics_loss = torch.mean(residual**2)
    initial_loss = (
        physics_prediction.theta_rad[0].square().mean()
        + physics_prediction.pitch_rate_radps[0].square().mean()
    )
    regularization_loss = model.physical_parameters.regularization_loss()
    total = (
        weights.data * data_loss
        + weights.physics * physics_loss
        + weights.initial * initial_loss
        + weights.regularization * regularization_loss
    )
    return PitchLossResult(
        total=total,
        data=data_loss,
        physics=physics_loss,
        initial=initial_loss,
        regularization=regularization_loss,
    )


def compute_full_car_losses(
    model: FullCarPitchPINN,
    observations: ObservationBatch,
    collocation_time_s: torch.Tensor,
    observed_damper_length_m: torch.Tensor,
    observed_damper_velocity_mps: torch.Tensor,
    spec: FullCarSpec,
    weights: FullCarLossWeights,
    *, front_junction_ratio: float, rear_junction_ratio: float,
    damper_polarity: float = 1.0,
) -> FullCarLossResult:
    observed_prediction = model.predict_with_derivatives(observations.time_s)
    data = torch.mean((observed_prediction.pitch_rate_radps - observations.pitch_rate_radps) ** 2)
    prediction = model.predict_with_derivatives(collocation_time_s)
    stroke, stroke_rate = corner_kinematics(
        prediction.state, prediction.state_rate, prediction.wheel_displacement,
        prediction.wheel_velocity, spec,
    )
    forces = corner_forces(stroke, stroke_rate, model.physical_parameters.values())
    residual = full_car_residual(
        prediction.state_acceleration, prediction.state_rate, forces,
        torch.zeros_like(forces), prediction.features[:, 1:2], spec,
    )
    junction = torch.as_tensor(
        [front_junction_ratio, front_junction_ratio, rear_junction_ratio, rear_junction_ratio],
        dtype=stroke.dtype, device=stroke.device,
    )
    length_loss = torch.mean((damper_polarity * stroke * junction - observed_damper_length_m) ** 2)
    velocity_loss = torch.mean((damper_polarity * stroke_rate * junction - observed_damper_velocity_mps) ** 2)
    heave = torch.mean((residual.heave / spec.sprung_mass_kg) ** 2)
    roll = torch.mean((residual.roll / spec.roll_inertia_kgm2) ** 2)
    pitch = torch.mean((residual.pitch / spec.pitch_inertia_kgm2) ** 2)
    initial = prediction.state[0].square().mean() + prediction.state_rate[0].square().mean()
    regularization = model.physical_parameters.regularization_loss()
    total = (weights.pitch_data * data + weights.physics_heave * heave +
             weights.physics_roll * roll + weights.physics_pitch * pitch +
             weights.damper_length * length_loss + weights.damper_velocity * velocity_loss +
             weights.initial * initial + weights.regularization * regularization)
    return FullCarLossResult(total, data, heave, roll, pitch, length_loss, velocity_loss, initial, regularization)
