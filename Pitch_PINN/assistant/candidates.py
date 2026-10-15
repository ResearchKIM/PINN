from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .artifacts import safe_text, write_json
from .paths import allowed_source, reject_links
from .schemas import FileChange

COPY_DIRS = {'models', 'physics', 'training', 'data', 'carmaker', 'config', 'tests', 'assistant'}


def _files(root: Path):
    package = root / 'Pitch_PINN'
    reject_links(package, root)
    for directory, dirs, files in os.walk(package, followlinks=False):
        parent = Path(directory)
        for name in dirs:
            reject_links(parent/name, root)
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.pytest_cache') and not d.startswith('.') and (parent != package or d in COPY_DIRS)]
        for name in sorted(files):
            p = parent/name
            if p.suffix in ('.py', '.yaml', '.yml'):
                reject_links(p, root)
                yield p


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_candidate(project_root: Path, run_dir: Path, changes: list[FileChange]) -> Path:
    project_root = project_root.resolve()
    candidate = run_dir / 'candidate'
    if candidate.exists() or not changes:
        raise ValueError('candidate already exists or no changes provided')
    targets, seen = [], set()
    for change in changes:
        path = allowed_source(project_root, change.path)
        canonical = path.relative_to(project_root).as_posix()
        if canonical.casefold() in seen:
            raise ValueError('duplicate source path')
        seen.add(canonical.casefold())
        targets.append((canonical, path, change.content))
    candidate.mkdir(parents=True)
    hashes, diffs = {}, []
    for path in _files(project_root):
        raw = path.relative_to(project_root).as_posix()
        dest = candidate/raw
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes[raw] = _digest(path)
    for raw, path, content in targets:
        before = path.read_text(encoding='utf-8-sig')
        diffs.extend(difflib.unified_diff(before.splitlines(keepends=True), content.splitlines(keepends=True), fromfile='original/'+raw, tofile='candidate/'+raw))
        (candidate/raw).write_text(content, encoding='utf-8')
    write_json(run_dir/'source_hashes.json', hashes)
    (run_dir/'changes.diff').write_text(safe_text(''.join(diffs)), encoding='utf-8')
    validate_candidate(candidate)
    return candidate


def validate_candidate(candidate_root: Path) -> None:
    root = candidate_root.resolve()
    reject_links(candidate_root, candidate_root.parent)
    hashes = json.loads((root.parent/'source_hashes.json').read_text(encoding='utf-8'))
    files = {p.relative_to(root).as_posix():p for p in _files(root)}
    if set(files) != set(hashes):
        raise ValueError('candidate file set changed')
    for raw, path in files.items():
        try:
            allowed_source(root, raw)
            editable = True
        except ValueError:
            editable = False
        if not editable and _digest(path) != hashes[raw]:
            raise ValueError('non-editable candidate file changed')
        if path.suffix == '.py':
            try:
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=raw)
            except (SyntaxError, UnicodeError):
                raise ValueError('candidate contains invalid Python syntax') from None


def child_environment() -> dict[str, str]:
    env = {k:v for k,v in os.environ.items() if k.upper() not in ('OPENAI_API_KEY', 'PYTHONPATH', 'PYTHONHOME')}
    env['PYTHONUTF8'] = '1'
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    env.pop('PYTEST_ADDOPTS', None)
    return env


def _execute(candidate: Path, args: list[str], log: Path, timeout_s: int):
    validate_candidate(candidate)
    try:
        result = subprocess.run([sys.executable, *args], cwd=candidate, env=child_environment(), shell=False, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ''
        if isinstance(output, bytes):
            output = output.decode('utf-8', errors='replace')
        log.write_text(safe_text(output)+'\nTIMEOUT\n', encoding='utf-8')
        raise
    log.write_text(safe_text(result.stdout+'\n'+result.stderr), encoding='utf-8')
    return result


def run_candidate_tests(candidate_root: Path, *, timeout_s: int = 180) -> subprocess.CompletedProcess[str]:
    return _execute(candidate_root, ['-m','pytest','Pitch_PINN/tests','-q','--import-mode=importlib'], candidate_root.parent/'tests.log', timeout_s)


def run_candidate_training(candidate_root: Path, config_path: Path, output_dir: Path, *, timeout_s: int = 3600) -> subprocess.CompletedProcess[str]:
    return _execute(candidate_root, ['-m','Pitch_PINN.train_inverse_pinn','--config',str(config_path.resolve()),'--output-dir',str(output_dir.resolve())], candidate_root.parent/'training.log', timeout_s)
