from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from Pitch_PINN.models.pitch_pinn import FullCarPitchPINN
from Pitch_PINN.models.state_network import FullCarStateNetwork
from Pitch_PINN.models.trainable_parameters import BoundedSuspensionRates, SuspensionRateBounds
from Pitch_PINN.physics.full_car import FullCarSpec, SuspensionRates
from Pitch_PINN.training.losses import FullCarLossWeights, ObservationBatch, compute_full_car_losses
from Pitch_PINN.training.trainer import TrainingConfig, train_full_car_pinn
from Pitch_PINN.training.losses import PitchLossWeights
from Pitch_PINN.data.scenario import ScenarioData
from Pitch_PINN.data.preprocess import select_sparse_observations


def test_zero_trajectory_has_zero_data_and_physics_loss() -> None:
    torch.set_default_dtype(torch.float64)
    time = np.array([0.0, 1.0])
    params = BoundedSuspensionRates(
        SuspensionRateBounds((1e3, 1e5), (1e3, 1e5), (100, 1e4), (100, 1e4)),
        SuspensionRates(35000, 45000, 4600, 4600),
    )
    model = FullCarPitchPINN(
        FullCarStateNetwork(hidden_width=4, hidden_layers=1), params, time,
        np.zeros(2), np.zeros((2, 4)), np.zeros((2, 4)),
    )
    with torch.no_grad():
        for parameter in model.state_network.parameters():
            parameter.zero_()
    spec = FullCarSpec(1711.4, 877.833, 2813.085, 1.184, 1.476, 1.57, 1.57, 0.625)
    losses = compute_full_car_losses(
        model,
        ObservationBatch(torch.tensor([[0.0], [1.0]]), torch.zeros((2, 1))),
        torch.tensor([[0.0], [1.0]]),
        torch.zeros((2, 4)), torch.zeros((2, 4)), spec,
        FullCarLossWeights(pitch_data=1, physics_heave=1, physics_roll=1, physics_pitch=1,
                           damper_length=1, damper_velocity=1, initial=1, regularization=0),
        front_junction_ratio=1.0, rear_junction_ratio=1.0,
    )
    assert losses.total.item() == 0.0


def test_full_car_trainer_runs_optional_lbfgs(tmp_path: Path) -> None:
    torch.set_default_dtype(torch.float64)
    time = np.linspace(0.0, 1.0, 5)
    scenario = ScenarioData(
        time, np.zeros(5), np.zeros(5), "test", "EV6",
        wheel_velocity_mps=np.zeros((5, 4)),
        wheel_acceleration_mps2=np.zeros((5, 4)),
        damper_length_m=np.zeros((5, 4)),
        damper_velocity_mps=np.zeros((5, 4)),
    )
    params = BoundedSuspensionRates(
        SuspensionRateBounds((1e3, 1e5), (1e3, 1e5), (100, 1e4), (100, 1e4)),
        SuspensionRates(35000, 45000, 4600, 4600),
    )
    model = FullCarPitchPINN(
        FullCarStateNetwork(hidden_width=4, hidden_layers=1), params, time,
        np.zeros(5), np.zeros((5, 4)), np.zeros((5, 4)),
    )
    spec = FullCarSpec(1711.4, 877.833, 2813.085, 1.184, 1.476, 1.57, 1.57, 0.625)
    weights = FullCarLossWeights(1, 1, 1, 1, 1, 1, 1, 0)
    result = train_full_car_pinn(
        model, select_sparse_observations(scenario, 2),
        TrainingConfig(0, 0.001, 5, 10, 1, PitchLossWeights(), use_lbfgs=True, lbfgs_max_iter=1),
        tmp_path, spec, weights, front_junction_ratio=1.0, rear_junction_ratio=1.0,
    )
    assert len(result.history) == 2
