from __future__ import annotations

import numpy as np
import pytest
import torch

from Pitch_PINN.models.pitch_pinn import AxInterpolator, PitchPINN
from Pitch_PINN.models.state_network import PitchStateNetwork
from Pitch_PINN.models.trainable_parameters import (
    BoundedPitchParameters,
    PitchParameterBounds,
)
from Pitch_PINN.physics.half_car import EffectivePitchParameters
from Pitch_PINN.training.losses import (
    ObservationBatch,
    PitchLossWeights,
    compute_pitch_losses,
)


torch.set_default_dtype(torch.float64)


def make_parameters() -> BoundedPitchParameters:
    return BoundedPitchParameters(
        PitchParameterBounds(
            damping_rate_s_inv=(0.1, 10.0),
            stiffness_rate_s2=(1.0, 100.0),
            ax_gain_rad_per_m=(-1.0, 1.0),
        ),
        EffectivePitchParameters(2.0, 20.0, -0.2),
    )


def make_model() -> PitchPINN:
    network = PitchStateNetwork(0.0, 1.0, hidden_width=8, hidden_layers=2)
    return PitchPINN(network, make_parameters())


def zero_network(model: PitchPINN) -> None:
    with torch.no_grad():
        for parameter in model.state_network.parameters():
            parameter.zero_()


def test_bounded_parameters_stay_inside_limits() -> None:
    parameters = make_parameters()
    with torch.no_grad():
        parameters.raw_damping.fill_(1000.0)
        parameters.raw_stiffness.fill_(-1000.0)
        parameters.raw_ax_gain.fill_(1000.0)

    values = parameters.values()

    assert 0.1 <= values.damping_rate_s_inv.item() <= 10.0
    assert 1.0 <= values.stiffness_rate_s2.item() <= 100.0
    assert -1.0 <= values.ax_gain_rad_per_m.item() <= 1.0


def test_network_outputs_finite_values_for_constant_time_range() -> None:
    network = PitchStateNetwork(2.0, 2.0, hidden_width=8, hidden_layers=1)
    result = network(torch.tensor([[2.0], [2.0]]))

    assert result.shape == (2, 1)
    assert torch.isfinite(result).all()


def test_autograd_returns_angle_rate_and_acceleration_shapes() -> None:
    model = make_model()
    time = torch.linspace(0.0, 1.0, 7).reshape(-1, 1).requires_grad_(True)

    prediction = model.predict_with_derivatives(time)

    assert prediction.theta_rad.shape == (7, 1)
    assert prediction.pitch_rate_radps.shape == (7, 1)
    assert prediction.pitch_acceleration_radps2.shape == (7, 1)
    assert torch.isfinite(prediction.pitch_acceleration_radps2).all()


def test_zero_exact_solution_has_near_zero_physics_loss() -> None:
    model = make_model()
    zero_network(model)
    observation_time = torch.tensor([[0.0], [1.0]])
    observations = ObservationBatch(observation_time, torch.zeros((2, 1)))
    collocation = torch.linspace(0.0, 1.0, 9).reshape(-1, 1)
    interpolator = AxInterpolator(np.array([0.0, 1.0]), np.array([0.0, 0.0]))

    losses = compute_pitch_losses(
        model,
        observations,
        collocation,
        interpolator,
        PitchLossWeights(data=1.0, physics=1.0, initial=1.0, regularization=0.0),
    )

    assert losses.physics.item() == pytest.approx(0.0, abs=1e-14)
    assert losses.total.item() == pytest.approx(0.0, abs=1e-14)


def test_data_loss_uses_only_supplied_sparse_observations() -> None:
    model = make_model()
    zero_network(model)
    observations = ObservationBatch(
        torch.tensor([[0.0], [1.0]]), torch.tensor([[1.0], [3.0]])
    )
    collocation = torch.linspace(0.0, 1.0, 5).reshape(-1, 1)
    interpolator = AxInterpolator(np.array([0.0, 1.0]), np.array([0.0, 0.0]))

    losses = compute_pitch_losses(
        model,
        observations,
        collocation,
        interpolator,
        PitchLossWeights(data=1.0, physics=0.0, initial=0.0, regularization=0.0),
    )

    assert losses.data.item() == pytest.approx(5.0)
    assert losses.total.item() == pytest.approx(5.0)
