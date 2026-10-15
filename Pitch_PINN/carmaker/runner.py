from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CarMakerRunConfig:
    dry_run: bool
    project_root: Path
    command: tuple[str, ...] | None
    parameter_target: Path | None


@dataclass(frozen=True)
class DryRunResult:
    project_root: Path
    command: tuple[str, ...] | None
    parameter_target: Path | None


def _validate_target(project_root: Path, target: Path | None) -> Path | None:
    if target is None:
        return None
    resolved = target.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError("parameter target must stay inside the CarMaker project root") from error
    return resolved


def run_carmaker(
    config: CarMakerRunConfig,
) -> subprocess.CompletedProcess[str] | DryRunResult:
    """Validate the execution boundary and run only after explicit live opt-in."""

    project_root = Path(config.project_root).resolve()
    if not project_root.is_dir():
        raise ValueError(f"CarMaker project root does not exist: {project_root}")
    target = _validate_target(project_root, config.parameter_target)
    if config.dry_run:
        return DryRunResult(project_root, config.command, target)

    if not config.command:
        raise ValueError("CarMaker command is required for a live run")
    if target is None:
        raise ValueError("parameter target is required for a live run")
    if not target.is_file():
        raise ValueError(f"parameter target does not exist: {target}")
    return subprocess.run(
        list(config.command),
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

