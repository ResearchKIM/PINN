from __future__ import annotations

import copy
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from Pitch_PINN.data.scenario import SparseScenario
from Pitch_PINN.models.pitch_pinn import AxInterpolator, PitchPINN
from Pitch_PINN.physics.half_car import EffectivePitchParameters
from Pitch_PINN.models.pitch_pinn import FullCarPitchPINN
from Pitch_PINN.models.input_signals import MultiSignalInterpolator
from Pitch_PINN.physics.full_car import FullCarSpec, SuspensionRates

from .losses import ObservationBatch, PitchLossWeights, compute_pitch_losses, FullCarLossWeights, compute_full_car_losses


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    collocation_points: int
    log_every: int
    seed: int
    weights: PitchLossWeights
    use_lbfgs: bool = False
    lbfgs_max_iter: int = 100

    def __post_init__(self) -> None:
        if self.epochs < 0 or self.learning_rate <= 0.0:
            raise ValueError("epochs must be non-negative and learning rate positive")
        if self.collocation_points < 2 or self.log_every <= 0:
            raise ValueError("collocation_points must be >= 2 and log_every positive")


@dataclass(frozen=True)
class TrainingResult:
    history: list[dict[str, float]]
    best_loss: float
    learned_parameters: EffectivePitchParameters


@dataclass(frozen=True)
class FullCarTrainingResult:
    history: list[dict[str, float]]
    best_loss: float
    learned_parameters: SuspensionRates


def _loss_row(epoch: int, losses) -> dict[str, float]:
    return {
        "epoch": float(epoch),
        "total": float(losses.total.detach()),
        "data": float(losses.data.detach()),
        "physics": float(losses.physics.detach()),
        "initial": float(losses.initial.detach()),
        "regularization": float(losses.regularization.detach()),
    }


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_pitch_pinn(
    model: PitchPINN,
    sparse_scenario: SparseScenario,
    config: TrainingConfig,
    output_dir: Path,
) -> TrainingResult:
    """Train a pitch PINN deterministically and retain the lowest-loss state."""

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    observations = ObservationBatch(
        torch.as_tensor(sparse_scenario.time_s).reshape(-1, 1),
        torch.as_tensor(sparse_scenario.pitch_rate_radps).reshape(-1, 1),
    )
    full = sparse_scenario.full
    collocation = torch.linspace(
        float(full.time_s[0]),
        float(full.time_s[-1]),
        config.collocation_points,
    ).reshape(-1, 1)
    ax_interpolator = AxInterpolator(full.time_s, full.ax_mps2)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )

    def calculate_losses():
        return compute_pitch_losses(
            model, observations, collocation, ax_interpolator, config.weights
        )

    initial_losses = calculate_losses()
    history = [_loss_row(0, initial_losses)]
    best_loss = history[0]["total"]
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = calculate_losses()
        losses.total.backward()
        optimizer.step()

        evaluated = calculate_losses()
        row = _loss_row(epoch, evaluated)
        history.append(row)
        if row["total"] < best_loss:
            best_loss = row["total"]
            best_state = copy.deepcopy(model.state_dict())
        if epoch % config.log_every == 0 or epoch == config.epochs:
            print(
                f"epoch={epoch:5d} total={row['total']:.6e} "
                f"data={row['data']:.6e} physics={row['physics']:.6e}"
            )

    if config.use_lbfgs and config.lbfgs_max_iter > 0:
        lbfgs = torch.optim.LBFGS(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_iter=config.lbfgs_max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss = calculate_losses().total
            loss.backward()
            return loss

        lbfgs.step(closure)
        evaluated = calculate_losses()
        row = _loss_row(config.epochs + 1, evaluated)
        history.append(row)
        if row["total"] < best_loss:
            best_loss = row["total"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    values = model.physical_parameters.values()
    learned = EffectivePitchParameters(
        float(values.damping_rate_s_inv.detach()),
        float(values.stiffness_rate_s2.detach()),
        float(values.ax_gain_rad_per_m.detach()),
    )
    torch.save(model.state_dict(), output_path / "model.pt")
    _write_history(output_path / "loss_history.csv", history)
    return TrainingResult(history, best_loss, learned)


def train_full_car_pinn(
    model: FullCarPitchPINN, sparse_scenario: SparseScenario, config: TrainingConfig,
    output_dir: Path, spec: FullCarSpec, weights: FullCarLossWeights,
    *, front_junction_ratio: float, rear_junction_ratio: float,
    damper_polarity: float = 1.0,
) -> FullCarTrainingResult:
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    output_path = Path(output_dir); output_path.mkdir(parents=True, exist_ok=True)
    observations = ObservationBatch(
        torch.as_tensor(sparse_scenario.time_s).reshape(-1, 1),
        torch.as_tensor(sparse_scenario.pitch_rate_radps).reshape(-1, 1),
    )
    full = sparse_scenario.full
    collocation = torch.linspace(float(full.time_s[0]), float(full.time_s[-1]), config.collocation_points).reshape(-1, 1)
    length_interp = MultiSignalInterpolator(full.time_s, full.damper_length_m)
    velocity_interp = MultiSignalInterpolator(full.time_s, full.damper_velocity_mps)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)

    def calculate():
        return compute_full_car_losses(
            model, observations, collocation, length_interp(collocation), velocity_interp(collocation),
            spec, weights, front_junction_ratio=front_junction_ratio,
            rear_junction_ratio=rear_junction_ratio, damper_polarity=damper_polarity,
        )

    def row(epoch: int, loss) -> dict[str, float]:
        return {"epoch": float(epoch), **{name: float(getattr(loss, name).detach()) for name in loss.__dataclass_fields__}}

    initial = calculate(); history = [row(0, initial)]
    best_loss = history[0]["total"]; best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True); loss = calculate(); loss.total.backward(); optimizer.step()
        evaluated = calculate(); current = row(epoch, evaluated); history.append(current)
        if current["total"] < best_loss:
            best_loss, best_state = current["total"], copy.deepcopy(model.state_dict())
        if epoch % config.log_every == 0 or epoch == config.epochs:
            print(f"epoch={epoch:5d} total={current['total']:.6e} data={current['pitch_data']:.6e} physics_pitch={current['physics_pitch']:.6e}")
    if config.use_lbfgs and config.lbfgs_max_iter > 0:
        lbfgs = torch.optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            max_iter=config.lbfgs_max_iter, line_search_fn="strong_wolfe",
        )
        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss = calculate().total
            loss.backward()
            return loss
        lbfgs.step(closure)
        evaluated = calculate(); current = row(config.epochs + 1, evaluated); history.append(current)
        if current["total"] < best_loss:
            best_loss, best_state = current["total"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    _write_history(output_path / "loss_history.csv", history)
    torch.save(model.state_dict(), output_path / "model.pt")
    values = model.physical_parameters.values()
    learned = SuspensionRates(*(float(getattr(values, n).detach()) for n in values.__dataclass_fields__))
    return FullCarTrainingResult(history, best_loss, learned)
