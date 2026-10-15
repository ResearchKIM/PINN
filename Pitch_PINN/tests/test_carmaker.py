from __future__ import annotations

from pathlib import Path

from scipy.io import savemat

from Pitch_PINN.carmaker.parameter_mapper import (
    build_parameter_update,
    write_parameter_diff,
)
from Pitch_PINN.carmaker.result_loader import load_carmaker_result
from Pitch_PINN.carmaker.runner import CarMakerRunConfig, DryRunResult, run_carmaker
from Pitch_PINN.physics.full_car import SuspensionRates


def test_full_car_rates_are_written_in_physical_units(tmp_path: Path) -> None:
    update = build_parameter_update(SuspensionRates(35000.0, 45000.0, 4600.0, 4700.0))
    path = write_parameter_diff(update, tmp_path / "update.json")
    text = path.read_text(encoding="utf-8")
    assert '"k_front_npm": 35000.0' in text
    assert '"c_rear_nspm": 4700.0' in text


def test_dry_run_does_not_modify_target(tmp_path: Path) -> None:
    project = tmp_path / "PINN_file"
    project.mkdir()
    target = project / "vehicle.par"
    target.write_text("original", encoding="utf-8")
    config = CarMakerRunConfig(
        dry_run=True,
        project_root=project,
        command=("does-not-run.exe", "-batch"),
        parameter_target=target,
    )

    result = run_carmaker(config)

    assert isinstance(result, DryRunResult)
    assert target.read_text(encoding="utf-8") == "original"


def test_live_run_rejects_incomplete_configuration_before_launch(
    tmp_path: Path,
) -> None:
    project = tmp_path / "PINN_file"
    project.mkdir()
    import pytest
    with pytest.raises(ValueError, match="command"):
        run_carmaker(
            CarMakerRunConfig(
                dry_run=False,
                project_root=project,
                command=None,
                parameter_target=project / "vehicle.par",
            )
        )


def test_target_outside_pinn_file_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "PINN_file"
    project.mkdir()
    outside = tmp_path / "installed" / "Demo_Tesla_Y"
    outside.parent.mkdir()
    outside.write_text("original", encoding="utf-8")

    import pytest
    with pytest.raises(ValueError, match="project root"):
        run_carmaker(
            CarMakerRunConfig(
                dry_run=True,
                project_root=project,
                command=("does-not-run.exe",),
                parameter_target=outside,
            )
        )


def test_result_loader_returns_scenario_data(tmp_path: Path) -> None:
    path = tmp_path / "result.mat"
    savemat(
        path,
        {
            "Sim_Time": [[0.0], [0.1]],
            "Ax": [[0.0], [1.0]],
            "PITCH_RATE": [[0.0], [-0.01]],
        },
    )

    result = load_carmaker_result(
        path,
        {"time": "Sim_Time", "ax": "Ax", "pitch_rate": "PITCH_RATE"},
        scenario_name="acc",
        vehicle="Tesla Y",
    )

    assert result.name == "acc"
    assert result.vehicle == "Tesla Y"
    assert result.pitch_rate_radps.tolist() == [0.0, -0.01]
