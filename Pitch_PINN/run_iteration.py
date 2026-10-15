from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Pitch_PINN.carmaker.parameter_mapper import (
    build_parameter_update,
    write_parameter_diff,
)
from Pitch_PINN.carmaker.runner import (
    CarMakerRunConfig,
    DryRunResult,
    run_carmaker,
)
from Pitch_PINN.train_inverse_pinn import load_config, run_training_from_config


@dataclass(frozen=True)
class IterationResult:
    output_dir: Path
    carmaker_was_live: bool


def _command(raw: Any) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return tuple(raw)
    raise ValueError("carmaker.command must be null, a string, or a string list")


def run_iteration_from_config(
    config: dict[str, Any],
    *,
    epochs: int | None = None,
    output_root: Path | None = None,
    allow_carmaker: bool = False,
) -> IterationResult:
    root = Path(output_root or config["project"]["output_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = root / f"iteration_{timestamp}"
    pipeline = run_training_from_config(config, epochs=epochs, output_dir=output_dir)

    update = build_parameter_update(pipeline.training.learned_parameters)
    write_parameter_diff(update, output_dir / "parameter_update.json")

    carmaker = config.get("carmaker", {})
    configured_dry_run = bool(carmaker.get("dry_run", True))
    live_requested = allow_carmaker and not configured_dry_run
    target_value = carmaker.get("parameter_target")
    run_result = run_carmaker(
        CarMakerRunConfig(
            dry_run=not live_requested,
            project_root=Path(carmaker["project_root"]),
            command=_command(carmaker.get("command")),
            parameter_target=Path(target_value) if target_value else None,
        )
    )
    was_live = not isinstance(run_result, DryRunResult)
    return IterationResult(output_dir=output_dir, carmaker_was_live=was_live)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, map parameters, and optionally run CarMaker"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "tesla_ev6.yaml",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--allow-carmaker",
        action="store_true",
        help="Permit a live run only when YAML also sets dry_run: false",
    )
    arguments = parser.parse_args()
    result = run_iteration_from_config(
        load_config(arguments.config),
        epochs=arguments.epochs,
        output_root=arguments.output_root,
        allow_carmaker=arguments.allow_carmaker,
    )
    print(f"iteration_output={result.output_dir}")
    print(f"carmaker_live={result.carmaker_was_live}")


if __name__ == "__main__":
    main()
