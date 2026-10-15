from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class ConfigChange(StrictModel):
    path: str
    value: StrictInt | StrictFloat


class FileChange(StrictModel):
    path: str
    content: str


class AssistantPlan(StrictModel):
    kind: Literal['config_change', 'code_change', 'clarification', 'unsupported']
    summary: str
    explanation: str
    questions: list[str]
    config_changes: list[ConfigChange]
    file_changes: list[FileChange]


def validate_plan(plan: AssistantPlan) -> None:
    if not plan.summary.strip():
        raise ValueError('summary must not be empty')
    if plan.kind == 'config_change' and (not plan.config_changes or plan.file_changes or plan.questions):
        raise ValueError('config_change requires only configuration changes')
    if plan.kind == 'code_change' and (not plan.file_changes or plan.questions):
        raise ValueError('code_change requires file changes and no questions')
    if plan.kind in ('clarification', 'unsupported') and (plan.config_changes or plan.file_changes):
        raise ValueError('non-executable plans cannot contain changes')
    if plan.kind == 'clarification' and not any(q.strip() for q in plan.questions):
        raise ValueError('clarification requires a question')


@dataclass(frozen=True)
class PlanResult:
    plan: AssistantPlan
    requested_model: str
    returned_model: str
    response_id: str


@dataclass(frozen=True)
class AssistantOptions:
    config_path: Path
    request: str = ''
    model: str | None = None
    train: bool = False
    code: bool = False
    run_candidate: bool = False
    resume_dir: Path | None = None
    backend: Literal['codex', 'api'] = 'codex'


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    state: str
    exit_code: int
