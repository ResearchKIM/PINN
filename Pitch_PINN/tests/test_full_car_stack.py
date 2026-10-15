from __future__ import annotations

import numpy as np
import pytest
import torch

from Pitch_PINN.models.input_signals import FeatureNormalizer, MultiSignalInterpolator
from Pitch_PINN.models.pitch_pinn import FullCarPitchPINN
from Pitch_PINN.models.state_network import FullCarStateNetwork
from Pitch_PINN.models.trainable_parameters import BoundedSuspensionRates, SuspensionRateBounds
from Pitch_PINN.physics.full_car import (
    FullCarSpec,
    SuspensionRates,
    corner_forces,
    corner_kinematics,
    full_car_residual,
)


def ev6_spec() -> FullCarSpec:
    return FullCarSpec(
        sprung_mass_kg=1711.4,
        roll_inertia_kgm2=877.833,
        pitch_inertia_kgm2=2813.085,
        cg_to_front_m=1.184,
        cg_to_rear_m=1.476,
        front_track_m=1.570,
        rear_track_m=1.570,
        cg_height_m=0.625,
    )


def test_interpolator_midpoint_and_endpoint_clamping() -> None:
    interp = MultiSignalInterpolator(
        np.array([0.0, 1.0]), np.array([[0.0, 10.0], [2.0, 14.0]])
    )
    query = torch.tensor([[-1e-9], [0.5], [1.0 + 1e-9]], dtype=torch.float64)
    result = interp(query).detach().numpy()
    np.testing.assert_allclose(result, [[0.0, 10.0], [1.0, 12.0], [2.0, 14.0]])


def test_constant_feature_normalization_is_finite() -> None:
    normalizer = FeatureNormalizer.fit(np.ones((3, 2)))
    result = normalizer(torch.ones((3, 2), dtype=torch.float64))
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.zeros_like(result))


def test_corner_kinematics_pitch_uses_fl_fr_rl_rr_order() -> None:
    state = torch.tensor([[0.0, 0.0, 0.1]], dtype=torch.float64)
    rate = torch.zeros_like(state)
    wheel = torch.zeros((1, 4), dtype=torch.float64)
    stroke, stroke_rate = corner_kinematics(state, rate, wheel, wheel, ev6_spec())
    torch.testing.assert_close(
        stroke,
        torch.tensor([[0.1184, 0.1184, -0.1476, -0.1476]], dtype=torch.float64),
    )
    torch.testing.assert_close(stroke_rate, torch.zeros_like(stroke_rate))


def test_front_and_rear_forces_create_opposite_pitch_moments() -> None:
    spec = ev6_spec()
    zero = torch.zeros((1, 3), dtype=torch.float64)
    front = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    rear = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
    front_residual = full_car_residual(zero, zero, front, torch.zeros_like(front), torch.zeros((1, 1)), spec)
    rear_residual = full_car_residual(zero, zero, rear, torch.zeros_like(rear), torch.zeros((1, 1)), spec)
    assert front_residual.pitch.item() > 0.0
    assert rear_residual.pitch.item() < 0.0


def test_ax_pitch_moment_is_fixed_by_mass_height() -> None:
    spec = ev6_spec()
    zero = torch.zeros((1, 3), dtype=torch.float64)
    residual = full_car_residual(
        zero, zero, torch.zeros((1, 4), dtype=torch.float64),
        torch.zeros((1, 4), dtype=torch.float64), torch.ones((1, 1), dtype=torch.float64), spec
    )
    assert residual.pitch.item() == pytest.approx(spec.sprung_mass_kg * spec.cg_height_m)


def test_corner_forces_share_rates_by_axle() -> None:
    forces = corner_forces(
        torch.ones((1, 4), dtype=torch.float64),
        2.0 * torch.ones((1, 4), dtype=torch.float64),
        SuspensionRates(10.0, 20.0, 3.0, 4.0),
    )
    torch.testing.assert_close(
        forces, torch.tensor([[16.0, 16.0, 28.0, 28.0]], dtype=torch.float64)
    )


def test_model_features_exclude_pitch_truth() -> None:
    time = np.array([0.0, 1.0])
    wheel_velocity = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 4.0, 6.0, 8.0],
    ])
    wheel_acceleration = np.array([
        [10.0, 20.0, 30.0, 40.0],
        [12.0, 22.0, 32.0, 42.0],
    ])
    parameters = BoundedSuspensionRates(
        SuspensionRateBounds((1e3, 1e5), (1e3, 1e5), (100, 1e4), (100, 1e4)),
        SuspensionRates(35000.0, 45000.0, 4600.0, 4600.0),
    )
    model = FullCarPitchPINN(
        FullCarStateNetwork(hidden_width=4, hidden_layers=1), parameters,
        time, np.array([0.0, 1.0]), wheel_velocity, wheel_acceleration,
    )
    prediction = model.predict_with_derivatives(torch.tensor([[0.5]], dtype=torch.float64))
    assert prediction.features.shape == (1, 6)
    torch.testing.assert_close(
        prediction.features,
        torch.tensor([[0.5, 0.5, 11.0, 21.0, 31.0, 41.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        prediction.wheel_velocity,
        torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        prediction.wheel_displacement,
        torch.tensor([[0.5, 1.0, 1.5, 2.0]], dtype=torch.float64),
    )
    assert "feature_mean" in model.state_dict()
    assert "feature_scale" in model.state_dict()
