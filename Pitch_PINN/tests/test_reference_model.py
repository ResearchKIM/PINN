from __future__ import annotations

import numpy as np

from Pitch_PINN.physics.half_car import EffectivePitchParameters
from Pitch_PINN.physics.reference_model import simulate_reference


def test_reference_zero_input_remains_at_rest() -> None:
    time = np.linspace(0.0, 1.0, 11)
    trajectory = simulate_reference(
        time,
        np.zeros_like(time),
        EffectivePitchParameters(2.0, 10.0, -0.1),
    )

    np.testing.assert_allclose(trajectory.pitch_angle_rad, 0.0, atol=1e-12)
    np.testing.assert_allclose(trajectory.pitch_rate_radps, 0.0, atol=1e-12)


def test_positive_ax_uses_configured_moment_sign() -> None:
    time = np.linspace(0.0, 0.2, 21)
    positive = simulate_reference(
        time,
        np.ones_like(time),
        EffectivePitchParameters(0.1, 0.1, 0.5),
    )
    negative = simulate_reference(
        time,
        np.ones_like(time),
        EffectivePitchParameters(0.1, 0.1, -0.5),
    )

    assert positive.pitch_rate_radps[-1] > 0.0
    assert negative.pitch_rate_radps[-1] < 0.0
    np.testing.assert_allclose(
        positive.pitch_rate_radps, -negative.pitch_rate_radps, rtol=1e-6, atol=1e-9
    )


def test_reference_preserves_requested_timestamps() -> None:
    time = np.array([0.0, 0.01, 0.03, 0.1])
    trajectory = simulate_reference(
        time,
        np.array([0.0, 1.0, 1.0, 0.0]),
        EffectivePitchParameters(1.0, 5.0, -0.2),
    )

    np.testing.assert_array_equal(trajectory.time_s, time)
    assert trajectory.pitch_rate_radps.shape == time.shape
