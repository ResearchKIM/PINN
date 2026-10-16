"""Use a ChatGPT-authenticated Codex CLI without API-key credentials."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .llm import PlannerError, system_prompt
from .schemas import AssistantPlan, PlanResult, validate_plan


def resolve_codex() -> Path:
    explicit = os.environ.get('PINN_CODEX_PATH')
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file() and (os.name != 'nt' or candidate.suffix.lower() == '.exe'):
            return candidate.resolve()
        raise PlannerError('codex_executable_not_found')
    found = shutil.which('codex')
    if found and (os.name != 'nt' or Path(found).suffix.lower() == '.exe'):
        return Path(found)
    local = os.environ.get('LOCALAPPDATA')
    if local:
        candidates = list((Path(local)/'OpenAI/Codex/bin').glob('*/codex.exe'))
        if candidates:
            return max(candidates, key=lambda p:p.stat().st_mtime)
    raise PlannerError('codex_executable_not_found')


def codex_environment() -> dict[str, str]:
    excluded = {'OPENAI_API_KEY','CODEX_API_KEY','OPENAI_BASE_URL'}
    return {k:v for k,v in os.environ.items() if k.upper() not in excluded}


def request_codex_plan(request: str, base: dict, *, model: str, sources: dict[str, str], datasets: list[dict[str, str]] | None = None) -> PlanResult:
    executable = str(resolve_codex())
    env = codex_environment()
    try:
        authentication = subprocess.run([executable,'login','status'],env=env,shell=False,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=15,check=False)
        if authentication.returncode:
            raise PlannerError('codex_login_required')
        if 'chatgpt' not in (authentication.stdout+'\n'+authentication.stderr).lower():
            raise PlannerError('codex_chatgpt_login_required')
        prompt = system_prompt()+'\nDo not call tools, read files, or run commands. Use only the supplied input and return the final JSON plan.\n'
        prompt += json.dumps({'request':request,'configuration':base,'sources':sources,'available_datasets':datasets or []},ensure_ascii=False,allow_nan=False)
        with tempfile.TemporaryDirectory(prefix='pinn-codex-') as directory:
            work = Path(directory)
            schema, output = work/'plan-schema.json', work/'plan-output.json'
            schema.write_text(json.dumps(AssistantPlan.model_json_schema(),ensure_ascii=False),encoding='utf-8')
            args = [executable,'exec','--ignore-user-config','--ephemeral','--skip-git-repo-check',
                    '--sandbox','read-only','--color','never','--json','--model',model,
                    '--output-schema',str(schema),'--output-last-message',str(output),'-']
            completed = subprocess.run(args,input=prompt,cwd=work,env=env,shell=False,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180,check=False)
            if completed.returncode:
                raise PlannerError('codex_exec_failed')
            thread_id, returned_model, turn_completed = '', 'unreported_by_codex_cli', False
            for line in completed.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event,dict):
                    continue
                if event.get('type') in ('turn.failed','error'):
                    raise PlannerError('codex_exec_failed')
                if event.get('type') == 'thread.started':
                    thread_id = str(event.get('thread_id',''))
                if event.get('type') == 'turn.completed':
                    turn_completed = True
                if isinstance(event.get('model'),str):
                    returned_model = event['model']
            if not turn_completed or not output.is_file():
                raise PlannerError('codex_incomplete')
            plan = AssistantPlan.model_validate_json(output.read_text(encoding='utf-8-sig'))
            validate_plan(plan)
            return PlanResult(plan,model,returned_model,thread_id)
    except PlannerError:
        raise
    except subprocess.TimeoutExpired:
        raise PlannerError('codex_timeout') from None
    except (ValueError, UnicodeError):
        raise PlannerError('invalid_plan') from None
    except OSError:
        raise PlannerError('codex_launch_failed') from None
