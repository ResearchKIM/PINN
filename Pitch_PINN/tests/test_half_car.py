from __future__ import annotations

import numpy as np
import pytest

from Pitch_PINN.physics.half_car import EffectivePitchParameters, pitch_ode_residual
from Pitch_PINN.physics.skyhook import skyhook_moment
from Pitch_PINN.physics.suspension import (
    VehiclePitchSpec,
    effective_to_suspension_scales,
)


def test_equilibrium_has_zero_residual() -> None:
    parameters = EffectivePitchParameters(2.0, 10.0, -0.1)
    residual = pitch_ode_residual(
        theta=np.array([0.0]),
        theta_dot=np.array([0.0]),
        theta_ddot=np.array([0.0]),
        ax=np.array([0.0]),
        parameters=parameters,
    )
    np.testing.assert_allclose(residual, [0.0])


@pytest.mark.parametrize(
    ("damping", "stiffness"),
    [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)],
)
def test_positive_coefficients_reject_invalid_values(
    damping: float, stiffness: float
) -> None:
    with pytest.raises(ValueError):
        EffectivePitchParameters(damping, stiffness, -0.1)


def test_zero_skyhook_gain_returns_zero() -> None:
    result = skyhook_moment(
        pitch_rate=np.array([-2.0, 0.0, 3.0]), gain_nms_per_rad=0.0, limit_nm=0.0
    )
    np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])


def test_effective_coefficients_convert_to_expected_scales() -> None:
    parameters = EffectivePitchParameters(1.0, 10.0, -0.1)
    spec = VehiclePitchSpec(
        pitch_inertia_kgm2=100.0,
        cg_to_front_axle_m=1.0,
        cg_to_rear_axle_m=1.0,
        baseline_k_front_npm=500.0,
        baseline_k_rear_npm=500.0,
        baseline_c_front_nspm=50.0,
        baseline_c_rear_nspm=50.0,
    )

    scales = effective_to_suspension_scales(parameters, spec)

    assert scales is not None
    assert scales.rotational_stiffness_nm_per_rad == pytest.approx(1000.0)
    assert scales.rotational_damping_nms_per_rad == pytest.approx(100.0)
    assert scales.alpha_k == pytest.approx(1.0)
    assert scales.alpha_c == pytest.approx(1.0)


def test_missing_vehicle_values_returns_no_scales() -> None:
    parameters = EffectivePitchParameters(1.0, 10.0, -0.1)
    spec = VehiclePitchSpec()

    assert effective_to_suspension_scales(parameters, spec) is None

