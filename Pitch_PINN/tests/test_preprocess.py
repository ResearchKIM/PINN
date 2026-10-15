from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from Pitch_PINN.data.mat_loader import load_mat_scenario
from Pitch_PINN.data.preprocess import preprocess_scenario, select_sparse_observations
from Pitch_PINN.data.scenario import ScenarioData


def make_scenario(time: np.ndarray) -> ScenarioData:
    return ScenarioData(
        time_s=time,
        ax_mps2=np.arange(time.size, dtype=float),
        pitch_rate_radps=np.arange(time.size, dtype=float) * 0.1,
        name="test",
        vehicle="EV6",
    )


def test_loads_column_vectors_from_mat(tmp_path: Path) -> None:
    path = tmp_path / "scenario.mat"
    savemat(
        path,
        {
            "Sim_Time": np.array([[0.0], [0.1], [0.2]]),
            "Ax": np.array([[0.0], [0.5], [1.0]]),
            "PITCH_RATE": np.array([[0.0], [-0.01], [-0.02]]),
            **{f"W_{c}": np.arange(3.0)[:, None] for c in ("FL", "FR", "RL", "RR")},
            **{f"A_{c}": np.arange(3.0)[:, None] + 5 for c in ("FL", "FR", "RL", "RR")},
            **{f"L_{c}": np.arange(3.0)[:, None] + 10 for c in ("FL", "FR", "RL", "RR")},
            **{f"V_{c}": np.arange(3.0)[:, None] + 20 for c in ("FL", "FR", "RL", "RR")},
        },
    )

    result = load_mat_scenario(
        path,
        {
            "time": "Sim_Time", "ax": "Ax", "pitch_rate": "PITCH_RATE",
            "wheel_velocity": {c.lower(): f"W_{c}" for c in ("FL", "FR", "RL", "RR")},
            "wheel_acceleration": {c.lower(): f"A_{c}" for c in ("FL", "FR", "RL", "RR")},
            "damper_length": {c.lower(): f"L_{c}" for c in ("FL", "FR", "RL", "RR")},
            "damper_velocity": {c.lower(): f"V_{c}" for c in ("FL", "FR", "RL", "RR")},
        },
        name="acc",
        vehicle="EV6",
    )

    assert result.time_s.shape == (3,)
    assert result.ax_mps2.shape == (3,)
    assert result.pitch_rate_radps.shape == (3,)
    np.testing.assert_allclose(result.pitch_rate_radps, [0.0, -0.01, -0.02])
    assert result.wheel_velocity_mps.shape == (3, 4)
    assert result.wheel_acceleration_mps2.shape == (3, 4)
    np.testing.assert_allclose(result.damper_length_m[0], [10.0] * 4)


def test_duplicate_times_keep_last_finite_sample() -> None:
    raw = ScenarioData(
        time_s=np.array([0.0, 1.0, 1.0, 2.0]),
        ax_mps2=np.array([0.0, 10.0, 20.0, 30.0]),
        pitch_rate_radps=np.array([0.0, 1.0, 2.0, 3.0]),
        name="duplicate",
        vehicle="EV6",
    )

    result = preprocess_scenario(raw)

    np.testing.assert_allclose(result.time_s, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(result.ax_mps2, [0.0, 20.0, 30.0])
    np.testing.assert_allclose(result.pitch_rate_radps, [0.0, 2.0, 3.0])
    assert np.all(np.diff(result.time_s) > 0.0)


def test_non_finite_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="time"):
        make_scenario(np.array([0.0, np.nan, 1.0]))


def test_sparse_stride_includes_endpoints() -> None:
    sparse = select_sparse_observations(make_scenario(np.arange(11.0)), stride=3)

    np.testing.assert_array_equal(sparse.observation_indices, [0, 3, 6, 9, 10])
    np.testing.assert_array_equal(
        np.flatnonzero(sparse.observation_mask), [0, 3, 6, 9, 10]
    )


def test_stride_larger_than_signal_has_unique_endpoints() -> None:
    sparse = select_sparse_observations(make_scenario(np.arange(3.0)), stride=10)

    np.testing.assert_array_equal(sparse.observation_indices, [0, 2])
    assert np.unique(sparse.observation_indices).size == 2


def test_sparse_stride_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        select_sparse_observations(make_scenario(np.arange(3.0)), stride=0)


def test_drop_last_samples_removes_terminal_logger_reset() -> None:
    result = preprocess_scenario(make_scenario(np.arange(4.0)), drop_last_samples=1)
    np.testing.assert_allclose(result.time_s, [0.0, 1.0, 2.0])
