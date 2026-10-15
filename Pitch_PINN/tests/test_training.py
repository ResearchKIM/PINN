from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from Pitch_PINN.data.preprocess import select_sparse_observations
from Pitch_PINN.data.scenario import ScenarioData
from Pitch_PINN.models.pitch_pinn import PitchPINN
from Pitch_PINN.models.state_network import PitchStateNetwork
from Pitch_PINN.models.trainable_parameters import (
    BoundedPitchParameters,
    PitchParameterBounds,
)
from Pitch_PINN.physics.half_car import EffectivePitchParameters
from Pitch_PINN.physics.reference_model import simulate_reference
from Pitch_PINN.training.evaluate import (
    _save_figure_with_fallback,
    evaluate_predictions,
    evaluate_reference,
    save_evaluation_artifacts,
)
import matplotlib.pyplot as plt
from Pitch_PINN.training.losses import PitchLossWeights
from Pitch_PINN.training.trainer import TrainingConfig, train_pitch_pinn
from Pitch_PINN.train_inverse_pinn import load_config


torch.set_default_dtype(torch.float64)


def synthetic_sparse_scenario():
    time = np.linspace(0.0, 1.0, 31)
    ax = np.where(time >= 0.2, 1.0, 0.0)
    truth = simulate_reference(
        time, ax, EffectivePitchParameters(2.0, 12.0, -0.08)
    )
    scenario = ScenarioData(time, ax, truth.pitch_rate_radps, "synthetic", "EV6")
    return select_sparse_observations(scenario, stride=5)


def make_model(seed: int = 7) -> PitchPINN:
    torch.manual_seed(seed)
    network = PitchStateNetwork(0.0, 1.0, hidden_width=12, hidden_layers=2)
    parameters = BoundedPitchParameters(
        PitchParameterBounds((0.1, 10.0), (1.0, 100.0), (-0.5, 0.5)),
        EffectivePitchParameters(2.0, 12.0, -0.08),
    )
    return PitchPINN(network, parameters)


def short_config(epochs: int) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs,
        learning_rate=0.01,
        collocation_points=21,
        log_every=100,
        seed=7,
        weights=PitchLossWeights(
            data=1000.0, physics=1.0, initial=10.0, regularization=0.0
        ),
    )


def test_short_training_reduces_total_loss(tmp_path: Path) -> None:
    result = train_pitch_pinn(
        make_model(), synthetic_sparse_scenario(), short_config(epochs=20), tmp_path
    )

    assert result.history[-1]["total"] < result.history[0]["total"]
    assert np.isfinite(result.best_loss)


def test_hidden_point_metrics_exclude_observations() -> None:
    metrics = evaluate_reference(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 1.0, 10.0, 3.0]),
        np.zeros(4),
        np.array([0, 2]),
    )

    assert metrics.observed_rmse_radps == pytest.approx(np.sqrt(50.0))
    assert metrics.hidden_rmse_radps == pytest.approx(np.sqrt(5.0))


def test_physics_integrated_reference_is_reported_separately() -> None:
    report = evaluate_predictions(
        time_s=np.array([0.0, 1.0, 2.0]),
        truth_pitch_rate=np.array([0.0, 1.0, 2.0]),
        pinn_pitch_rate=np.array([0.0, 1.0, 1.0]),
        physics_pitch_rate=np.array([0.0, 0.0, 0.0]),
        observation_indices=np.array([0, 2]),
    )

    assert report.pinn.full_rmse_radps != report.physics_model.full_rmse_radps
    assert report.pinn.hidden_rmse_radps == pytest.approx(0.0)
    assert report.physics_model.hidden_rmse_radps == pytest.approx(1.0)


def test_artifacts_include_checkpoint_metrics_csv_and_plot(tmp_path: Path) -> None:
    sparse = synthetic_sparse_scenario()
    model = make_model()
    result = train_pitch_pinn(model, sparse, short_config(epochs=1), tmp_path)
    full = sparse.full
    prediction = np.zeros(len(full))
    report = evaluate_predictions(
        full.time_s,
        full.pitch_rate_radps,
        prediction,
        prediction,
        sparse.observation_indices,
    )

    paths = save_evaluation_artifacts(
        tmp_path,
        model,
        result,
        full.time_s,
        full.pitch_rate_radps,
        prediction,
        prediction,
        sparse.observation_indices,
        report,
        {"project": {"seed": 7}},
    )

    assert {"checkpoint", "metrics", "reference_csv", "history_csv", "plot", "config"} <= set(paths)
    assert all(path.is_file() for path in paths.values())
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"pinn", "physics_model", "learned_parameters"}


def test_seed_reproduces_initial_and_final_loss(tmp_path: Path) -> None:
    first = train_pitch_pinn(
        make_model(), synthetic_sparse_scenario(), short_config(epochs=1), tmp_path / "a"
    )
    second = train_pitch_pinn(
        make_model(), synthetic_sparse_scenario(), short_config(epochs=1), tmp_path / "b"
    )

    assert first.history == second.history


def test_cli_file_path_help_works_from_project_root() -> None:
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "Pitch_PINN/train_inverse_pinn.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Train a sparse-data full-car pitch PINN" in completed.stdout


def test_default_config_model_dimensions_are_numeric() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "tesla_ev6.yaml"
    config = load_config(config_path)

    assert isinstance(config["model"]["hidden_width"], int)
    assert isinstance(config["model"]["hidden_layers"], int)


def test_plot_save_uses_unique_fallback_when_target_cannot_be_overwritten(
    tmp_path: Path,
) -> None:
    blocked_target = tmp_path / "reference_comparison.png"
    blocked_target.mkdir()
    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [0.0, 1.0])

    saved = _save_figure_with_fallback(figure, blocked_target, dpi=80)
    plt.close(figure)

    assert saved != blocked_target
    assert saved.is_file()
    assert saved.name.startswith("reference_comparison_")
