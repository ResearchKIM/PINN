from __future__ import annotations

import csv
import json
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from Pitch_PINN.models.pitch_pinn import PitchPINN

from .trainer import TrainingResult
from Pitch_PINN.physics.full_car import SuspensionRates


@dataclass(frozen=True)
class ReferenceMetrics:
    full_rmse_radps: float
    observed_rmse_radps: float
    hidden_rmse_radps: float
    peak_magnitude_error_radps: float
    peak_time_error_s: float


@dataclass(frozen=True)
class EvaluationReport:
    pinn: ReferenceMetrics
    physics_model: ReferenceMetrics


def _save_figure_with_fallback(
    figure: plt.Figure, target: Path, *, dpi: int
) -> Path:
    """Save a plot, using a unique name when Windows locks the old PNG."""

    destination = Path(target)
    try:
        figure.savefig(destination, dpi=dpi)
        return destination
    except OSError as error:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fallback = destination.with_name(
            f"{destination.stem}_{timestamp}{destination.suffix}"
        )
        figure.savefig(fallback, dpi=dpi)
        print(
            f"warning: could not overwrite {destination!s} ({error}); "
            f"plot saved as {fallback!s}"
        )
        return fallback


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error)))) if error.size else float("nan")


def evaluate_reference(
    time_s: np.ndarray,
    truth_pitch_rate: np.ndarray,
    predicted_pitch_rate: np.ndarray,
    observation_indices: np.ndarray,
) -> ReferenceMetrics:
    time = np.asarray(time_s, dtype=float).reshape(-1)
    truth = np.asarray(truth_pitch_rate, dtype=float).reshape(-1)
    prediction = np.asarray(predicted_pitch_rate, dtype=float).reshape(-1)
    observed = np.asarray(observation_indices, dtype=np.int64).reshape(-1)
    if not (time.size == truth.size == prediction.size):
        raise ValueError("time, truth, and prediction lengths must match")
    mask = np.zeros(time.size, dtype=bool)
    mask[observed] = True
    error = prediction - truth
    truth_peak = int(np.argmax(np.abs(truth)))
    prediction_peak = int(np.argmax(np.abs(prediction)))
    return ReferenceMetrics(
        full_rmse_radps=_rmse(error),
        observed_rmse_radps=_rmse(error[mask]),
        hidden_rmse_radps=_rmse(error[~mask]),
        peak_magnitude_error_radps=float(
            abs(prediction[prediction_peak]) - abs(truth[truth_peak])
        ),
        peak_time_error_s=float(time[prediction_peak] - time[truth_peak]),
    )


def evaluate_predictions(
    time_s: np.ndarray,
    truth_pitch_rate: np.ndarray,
    pinn_pitch_rate: np.ndarray,
    physics_pitch_rate: np.ndarray,
    observation_indices: np.ndarray,
) -> EvaluationReport:
    return EvaluationReport(
        pinn=evaluate_reference(
            time_s, truth_pitch_rate, pinn_pitch_rate, observation_indices
        ),
        physics_model=evaluate_reference(
            time_s, truth_pitch_rate, physics_pitch_rate, observation_indices
        ),
    )


def save_evaluation_artifacts(
    output_dir: Path,
    model: PitchPINN,
    training_result: TrainingResult,
    time_s: np.ndarray,
    truth_pitch_rate: np.ndarray,
    pinn_pitch_rate: np.ndarray,
    physics_pitch_rate: np.ndarray,
    observation_indices: np.ndarray,
    report: EvaluationReport,
    resolved_config: dict[str, Any],
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "checkpoint": output / "model.pt",
        "metrics": output / "metrics.json",
        "reference_csv": output / "reference.csv",
        "history_csv": output / "loss_history.csv",
        "plot": output / "reference_comparison.png",
        "config": output / "resolved_config.yaml",
    }
    torch.save(model.state_dict(), paths["checkpoint"])

    metrics = {
        "pinn": asdict(report.pinn),
        "physics_model": asdict(report.physics_model),
        "learned_parameters": asdict(training_result.learned_parameters),
    }
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with paths["config"].open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved_config, stream, sort_keys=False, allow_unicode=True)

    observed_mask = np.zeros(len(time_s), dtype=bool)
    observed_mask[np.asarray(observation_indices, dtype=int)] = True
    with paths["reference_csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s",
                "truth_pitch_rate_radps",
                "pinn_pitch_rate_radps",
                "physics_pitch_rate_radps",
                "is_observed",
            ]
        )
        writer.writerows(
            zip(
                time_s,
                truth_pitch_rate,
                pinn_pitch_rate,
                physics_pitch_rate,
                observed_mask.astype(int),
                strict=True,
            )
        )

    if not paths["history_csv"].is_file():
        with paths["history_csv"].open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(training_result.history[0]))
            writer.writeheader()
            writer.writerows(training_result.history)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(time_s, truth_pitch_rate, label="EV6 full ground truth", color="black")
    axis.plot(time_s, pinn_pitch_rate, label="PINN reference", color="tab:blue")
    axis.plot(
        time_s,
        physics_pitch_rate,
        label="Integrated physics reference",
        color="tab:orange",
        linestyle="--",
    )
    axis.scatter(
        np.asarray(time_s)[observation_indices],
        np.asarray(truth_pitch_rate)[observation_indices],
        label="Sparse training observations",
        color="tab:red",
        s=14,
        zorder=3,
    )
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Pitch Rate [rad/s]")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    paths["plot"] = _save_figure_with_fallback(figure, paths["plot"], dpi=160)
    plt.close(figure)
    return paths


def save_full_car_artifacts(
    output_dir: Path, model: torch.nn.Module, training_result: Any,
    time_s: np.ndarray, truth_pitch_rate: np.ndarray, predicted_pitch_rate: np.ndarray,
    observation_indices: np.ndarray, residuals: dict[str, np.ndarray],
    resolved_config: dict[str, Any],
) -> dict[str, Path]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_reference(time_s, truth_pitch_rate, predicted_pitch_rate, observation_indices)
    paths = {name: output / filename for name, filename in {
        "checkpoint": "model.pt", "metrics": "metrics.json", "reference_csv": "reference.csv",
        "residuals_csv": "physics_residuals.csv", "history_csv": "loss_history.csv",
        "plot": "reference_comparison.png", "config": "resolved_config.yaml"}.items()}
    torch.save(model.state_dict(), paths["checkpoint"])
    learned = {name: float(getattr(training_result.learned_parameters, name)) for name in training_result.learned_parameters.__dataclass_fields__}
    paths["metrics"].write_text(json.dumps({"pinn": asdict(metrics), "learned_parameters": learned}, indent=2), encoding="utf-8")
    with paths["config"].open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved_config, stream, sort_keys=False, allow_unicode=True)
    mask = np.zeros(len(time_s), dtype=int); mask[np.asarray(observation_indices, dtype=int)] = 1
    with paths["reference_csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["time_s", "truth_pitch_rate_radps", "pinn_pitch_rate_radps", "is_observed"])
        writer.writerows(zip(time_s, truth_pitch_rate, predicted_pitch_rate, mask, strict=True))
    with paths["residuals_csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["time_s", "heave", "roll", "pitch"])
        writer.writerows(zip(time_s, residuals["heave"], residuals["roll"], residuals["pitch"], strict=True))
    figure, axis = plt.subplots(figsize=(10, 5)); axis.plot(time_s, truth_pitch_rate, color="black", label="EV6 ground truth")
    axis.plot(time_s, predicted_pitch_rate, color="tab:blue", label="Full-car PINN")
    axis.scatter(np.asarray(time_s)[observation_indices], np.asarray(truth_pitch_rate)[observation_indices], s=12, color="tab:red", label="Sparse observations")
    axis.set(xlabel="Time [s]", ylabel="Pitch Rate [rad/s]"); axis.grid(True, alpha=.3); axis.legend(); figure.tight_layout(); paths["plot"] = _save_figure_with_fallback(figure, paths["plot"], dpi=160); plt.close(figure)
    return paths
