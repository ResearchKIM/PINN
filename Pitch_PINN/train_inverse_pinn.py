from __future__ import annotations

import argparse, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from Pitch_PINN.data.mat_loader import load_mat_scenario
from Pitch_PINN.data.preprocess import preprocess_scenario, select_sparse_observations
from Pitch_PINN.models.pitch_pinn import FullCarPitchPINN
from Pitch_PINN.models.state_network import FullCarStateNetwork
from Pitch_PINN.models.trainable_parameters import BoundedSuspensionRates, SuspensionRateBounds
from Pitch_PINN.physics.full_car import FullCarSpec, SuspensionRates, corner_forces, corner_kinematics, full_car_residual
from Pitch_PINN.training.evaluate import ReferenceMetrics, evaluate_reference, save_full_car_artifacts
from Pitch_PINN.training.losses import FullCarLossWeights, PitchLossWeights
from Pitch_PINN.training.trainer import FullCarTrainingResult, TrainingConfig, train_full_car_pinn

torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class PipelineResult:
    output_dir: Path
    training: FullCarTrainingResult
    evaluation: ReferenceMetrics


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def _parameter(config: dict[str, Any], name: str) -> tuple[tuple[float, float], float]:
    value = config["model"]["parameters"][name]
    return (float(value["min"]), float(value["max"])), float(value["initial"])


def run_training_from_config(config: dict[str, Any], *, epochs: int | None = None, output_dir: Path | None = None) -> PipelineResult:
    parameters = config.get("model", {}).get("parameters", {})
    legacy = {"damping_rate_s_inv", "stiffness_rate_s2", "ax_gain_rad_per_m"}.intersection(parameters)
    if legacy:
        raise ValueError("legacy normalized pitch parameters are unsupported; use k_front_npm, k_rear_npm, c_front_nspm, c_rear_nspm")
    dc = config["data"]
    scenario = preprocess_scenario(
        load_mat_scenario(Path(dc["mat_path"]), dc["keys"], name=dc["scenario_name"], vehicle=dc["vehicle"]),
        remove_bias_samples=int(dc.get("remove_bias_samples", 0)), baseline_samples=int(dc.get("baseline_samples", 0)),
        drop_last_samples=int(dc.get("drop_last_samples", 0)),
    )
    sparse = select_sparse_observations(scenario, int(dc["sparse_stride"]))
    vc = config["vehicle"]; spec = FullCarSpec(**{name: float(vc[name]) for name in FullCarSpec.__dataclass_fields__})
    names = ("k_front_npm", "k_rear_npm", "c_front_nspm", "c_rear_nspm")
    parsed = {name: _parameter(config, name) for name in names}
    bounds = SuspensionRateBounds(**{name: parsed[name][0] for name in names})
    initial = SuspensionRates(**{name: parsed[name][1] for name in names})
    mc = config["model"]; scales = mc["output_scales"]
    torch.manual_seed(int(config["project"]["seed"]))
    model = FullCarPitchPINN(
        FullCarStateNetwork(hidden_width=int(mc["hidden_width"]), hidden_layers=int(mc["hidden_layers"]),
                            heave_scale_m=float(scales["heave_m"]), roll_scale_rad=float(scales["roll_rad"]), pitch_scale_rad=float(scales["pitch_rad"])),
        BoundedSuspensionRates(bounds, initial), scenario.time_s, scenario.ax_mps2,
        scenario.wheel_velocity_mps, scenario.wheel_acceleration_mps2,
    )
    tc = config["training"]
    settings = TrainingConfig(int(tc["epochs"] if epochs is None else epochs), float(tc["learning_rate"]), int(tc["collocation_points"]), int(tc["log_every"]), int(config["project"]["seed"]), PitchLossWeights())
    weights = FullCarLossWeights(**{k: float(v) for k, v in tc["weights"].items()})
    aux = config["auxiliary"]; destination = Path(output_dir or config["project"]["output_dir"])
    trained = train_full_car_pinn(model, sparse, settings, destination, spec, weights,
                                  front_junction_ratio=float(aux["front_junction_ratio"]), rear_junction_ratio=float(aux["rear_junction_ratio"]), damper_polarity=float(aux.get("damper_polarity", 1.0)))
    chunks = []
    for start in range(0, len(scenario), 1000):
        t = torch.as_tensor(scenario.time_s[start:start+1000]).reshape(-1, 1)
        prediction = model.predict_with_derivatives(t)
        stroke, stroke_rate = corner_kinematics(prediction.state, prediction.state_rate, prediction.wheel_displacement, prediction.wheel_velocity, spec)
        forces = corner_forces(stroke, stroke_rate, model.physical_parameters.values())
        residual = full_car_residual(prediction.state_acceleration, prediction.state_rate, forces, torch.zeros_like(forces), prediction.features[:, 1:2], spec)
        chunks.append((prediction.pitch_rate_radps.detach().numpy().reshape(-1), residual.heave.detach().numpy().reshape(-1), residual.roll.detach().numpy().reshape(-1), residual.pitch.detach().numpy().reshape(-1)))
    predicted = np.concatenate([x[0] for x in chunks]); residuals = {n: np.concatenate([x[i] for x in chunks]) for i, n in enumerate(("heave", "roll", "pitch"), start=1)}
    metrics = evaluate_reference(scenario.time_s, scenario.pitch_rate_radps, predicted, sparse.observation_indices)
    save_full_car_artifacts(destination, model, trained, scenario.time_s, scenario.pitch_rate_radps, predicted, sparse.observation_indices, residuals, config)
    return PipelineResult(destination, trained, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sparse-data full-car pitch PINN")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config" / "tesla_ev6.yaml")
    parser.add_argument("--epochs", type=int); parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(); result = run_training_from_config(load_config(args.config), epochs=args.epochs, output_dir=args.output_dir)
    p = result.training.learned_parameters
    print(f"output_dir={result.output_dir}")
    print(f"learned: Kf={p.k_front_npm:.6g} N/m, Kr={p.k_rear_npm:.6g} N/m, Cf={p.c_front_nspm:.6g} N*s/m, Cr={p.c_rear_nspm:.6g} N*s/m")
    print(f"hidden_rmse={result.evaluation.hidden_rmse_radps:.6e} rad/s")


if __name__ == "__main__":
    main()
