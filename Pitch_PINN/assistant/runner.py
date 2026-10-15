from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import yaml

from .artifacts import safe_text, write_json
from .candidates import build_candidate, copy_validated_candidate, run_candidate_tests, run_candidate_training, validate_candidate
from .configuration import apply_config_changes
from .llm import PlannerError, collect_source_context, request_plan
from .paths import reject_links
from .schemas import AssistantOptions, AssistantPlan, RunResult, validate_plan

DEFAULT_MODEL = 'gpt-6.1-sol'


def train_configuration(config: dict, output_dir: Path):
    from Pitch_PINN.train_inverse_pinn import run_training_from_config
    return run_training_from_config(config, output_dir=output_dir)


def _resolve_data_path(config: dict, root: Path) -> None:
    path = Path(config['data']['mat_path'])
    if not path.is_absolute():
        config['data']['mat_path'] = str((root/path).resolve())


def _resume_source(directory: Path, run_root: Path) -> Path:
    raw = directory.absolute()
    raw.relative_to(run_root)
    reject_links(raw, run_root)
    resolved = raw.resolve()
    resolved.relative_to(run_root)
    if resolved == run_root or not resolved.is_dir():
        raise ValueError('resume must select an existing run')
    validate_candidate(resolved/'candidate')
    return resolved


def run_request(options: AssistantOptions, *, project_root: Path, planner=request_plan) -> RunResult:
    root = project_root.resolve()
    run_root = root/'Pitch_PINN/assistant_runs'
    reject_links(run_root, root)
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:12])
    run_dir.mkdir()
    started = time.monotonic()
    status = {'stage':'initialization', 'state':'running', 'requested_model':options.model or os.environ.get('PINN_LLM_MODEL') or DEFAULT_MODEL}

    def finish(state: str, exit_code: int, error_code: str | None = None):
        status.update(state=state, error_code=error_code, elapsed_s=round(time.monotonic()-started, 3))
        write_json(run_dir/'status.json', status)
        return RunResult(run_dir, state, exit_code)

    write_json(run_dir/'status.json', status)
    (run_dir/'request.txt').write_text(safe_text(options.request), encoding='utf-8')
    try:
        candidate = None
        if options.resume_dir is not None:
            status['stage'] = 'resume_validation'
            previous = _resume_source(options.resume_dir, run_root)
            plan = AssistantPlan.model_validate_json((previous/'plan.json').read_text(encoding='utf-8'))
            validate_plan(plan)
            if plan.kind != 'code_change':
                raise ValueError('resume supports code candidates only')
            config = yaml.safe_load((previous/'resolved_config.yaml').read_text(encoding='utf-8'))
            apply_config_changes(config, [])
            _resolve_data_path(config, root)
            copy_validated_candidate(previous/'candidate', run_dir/'candidate')
            for name in ('source_hashes.json', 'changes.diff', 'config_diff.json'):
                shutil.copy2(previous/name, run_dir/name)
            candidate = run_dir/'candidate'
            previous_status = json.loads((previous/'status.json').read_text(encoding='utf-8'))
            status.update({k:previous_status.get(k) for k in ('requested_model','returned_model','response_id')})
            status['resumed_from'] = previous.name
            (run_dir/'request.txt').write_text((previous/'request.txt').read_text(encoding='utf-8'), encoding='utf-8')
        else:
            status['stage'] = 'planning'
            if not options.request.strip():
                raise ValueError('request cannot be empty')
            config = yaml.safe_load(options.config_path.read_text(encoding='utf-8-sig'))
            config, _ = apply_config_changes(config, [])
            _resolve_data_path(config, root)
            sources = collect_source_context(root) if options.code else {}
            result = planner(options.request, config, model=status['requested_model'], sources=sources)
            plan = result.plan
            validate_plan(plan)
            status.update(returned_model=result.returned_model, response_id=result.response_id)
        write_json(run_dir/'plan.json', plan.model_dump())
        if plan.kind in ('clarification','unsupported'):
            return finish('needs_clarification' if plan.kind=='clarification' else 'unsupported', 2)
        status['stage'] = 'configuration_validation'
        if options.resume_dir is None:
            config, diff = apply_config_changes(config, plan.config_changes)
            write_json(run_dir/'config_diff.json', diff)
        (run_dir/'resolved_config.yaml').write_text(safe_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False)), encoding='utf-8')
        if plan.kind == 'config_change':
            if options.run_candidate:
                raise ValueError('run-candidate requires a code plan')
            if not options.train:
                return finish('planned', 0)
            status['stage'] = 'training'
            log = run_dir/'training.log'
            try:
                with log.open('w',encoding='utf-8') as stream, redirect_stdout(stream), redirect_stderr(stream):
                    train_configuration(config,run_dir/'training')
            finally:
                if log.exists():
                    log.write_text(safe_text(log.read_text(encoding='utf-8')),encoding='utf-8')
            return finish('trained', 0)
        status['stage'] = 'candidate_creation'
        if candidate is None:
            if not options.code:
                raise ValueError('code changes require --code')
            candidate = build_candidate(root, run_dir, plan.file_changes)
        if not options.run_candidate:
            return finish('candidate_ready', 0)
        status['stage'] = 'candidate_tests'
        tests = run_candidate_tests(candidate)
        if tests.returncode != 0:
            return finish('failed', 1, 'candidate_tests_failed')
        if not options.train:
            return finish('tests_passed', 0)
        status['stage'] = 'candidate_training'
        training = run_candidate_training(candidate,run_dir/'resolved_config.yaml',run_dir/'training')
        if training.returncode != 0:
            return finish('failed', 1, 'candidate_training_failed')
        return finish('trained', 0)
    except PlannerError as error:
        return finish('failed',1,error.code)
    except subprocess.TimeoutExpired:
        return finish('failed',1,'process_timeout')
    except (ValueError, OSError, KeyError, TypeError):
        return finish('failed',1,'validation_or_io_error')
    except Exception:
        return finish('failed',1,'execution_error')
