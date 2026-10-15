import os
from pathlib import Path
import subprocess

import pytest

from Pitch_PINN.assistant.candidates import build_candidate, run_candidate_tests, run_candidate_training, validate_candidate, child_environment
from Pitch_PINN.assistant.schemas import FileChange


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'original'
    for raw, content in [('Pitch_PINN/__init__.py',''), ('Pitch_PINN/models/a.py','VALUE = 1\n'), ('Pitch_PINN/tests/test_a.py','def test_value():\n    from Pitch_PINN.models.a import VALUE\n    assert VALUE == 2\n'), ('Pitch_PINN/config/test.yaml','value: 1'), ('Pitch_PINN/outputs/old.yaml','old'), ('Pitch_PINN/data/raw.mat','raw'), ('Pitch_PINN/assistant_runs/old/a.py','old')]:
        p = root / raw
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
    return root


def test_candidate_preserves_original_and_hashes_diff(source, tmp_path):
    run = tmp_path / 'run'
    candidate = build_candidate(source, run, [FileChange(path='Pitch_PINN/models/a.py',content='VALUE = 2\n')])
    assert (source/'Pitch_PINN/models/a.py').read_text() == 'VALUE = 1\n'
    assert (candidate/'Pitch_PINN/models/a.py').read_text() == 'VALUE = 2\n'
    assert '+VALUE = 2' in (run/'changes.diff').read_text()
    assert (run/'source_hashes.json').is_file()
    assert (candidate/'Pitch_PINN/config/test.yaml').is_file()
    assert not (candidate/'Pitch_PINN/outputs').exists()
    assert not (candidate/'Pitch_PINN/data/raw.mat').exists()
    assert not (candidate/'Pitch_PINN/assistant_runs').exists()
    validate_candidate(candidate)


@pytest.mark.parametrize('raw', ['/x.py','C:/x.py','//server/x.py','../a.py','Pitch_PINN/models/../a.py','Pitch_PINN\\models\\a.py','Pitch_PINN/tests/test_a.py','Pitch_PINN/models/new.py','Pitch_PINN/assistant/x.py'])
def test_reject_invalid_paths(source, tmp_path, raw):
    with pytest.raises(ValueError):
        build_candidate(source,tmp_path/'run',[FileChange(path=raw,content='VALUE = 2')])


def test_reject_case_duplicate(source, tmp_path):
    with pytest.raises(ValueError):
        build_candidate(source,tmp_path/'run',[FileChange(path='Pitch_PINN/models/a.py',content='x=1'),FileChange(path='Pitch_PINN/models/A.py',content='x=2')])


def test_syntax_error_produces_failed_candidate_without_execution(source, tmp_path):
    with pytest.raises(ValueError):
        build_candidate(source,tmp_path/'run',[FileChange(path='Pitch_PINN/models/a.py',content='def broken(')])
    assert (tmp_path/'run/candidate/Pitch_PINN/models/a.py').is_file()


def test_child_environment_removes_key_and_pythonpath(monkeypatch):
    for key in ('OPENAI_API_KEY','PYTHONPATH','PYTHONHOME'):
        monkeypatch.setenv(key,'secret')
    assert not any(k in child_environment() for k in ('OPENAI_API_KEY','PYTHONPATH','PYTHONHOME'))


def test_candidate_imports_from_candidate_not_original(source, tmp_path):
    candidate = build_candidate(source,tmp_path/'run',[FileChange(path='Pitch_PINN/models/a.py',content='VALUE = 2\n')])
    result = run_candidate_tests(candidate)
    assert result.returncode == 0, result.stdout + result.stderr


def test_candidate_test_tampering_is_rejected(source, tmp_path):
    candidate = build_candidate(source,tmp_path/'run',[FileChange(path='Pitch_PINN/models/a.py',content='VALUE = 2\n')])
    (candidate/'Pitch_PINN/tests/test_a.py').write_text('def test_fake(): pass')
    with pytest.raises(ValueError):
        validate_candidate(candidate)


def test_link_escape_is_rejected(source, tmp_path):
    link = source/'Pitch_PINN/models/link.py'
    try:
        link.symlink_to(tmp_path/'outside.py')
    except OSError:
        pytest.skip('symlink permission unavailable')
    with pytest.raises(ValueError):
        build_candidate(source,tmp_path/'run',[FileChange(path='Pitch_PINN/models/link.py',content='x=1')])


def test_process_timeout_and_fixed_training_command(source,tmp_path,monkeypatch):
    candidate = build_candidate(source,tmp_path/'run',[FileChange(path='Pitch_PINN/models/a.py',content='VALUE=2')])
    calls = []
    def timeout(args, **kwargs):
        calls.append((args,kwargs))
        raise subprocess.TimeoutExpired(args, 1, output='partial')
    monkeypatch.setattr(subprocess,'run',timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        run_candidate_tests(candidate,timeout_s=1)
    assert (tmp_path/'run/tests.log').exists()
    with pytest.raises(subprocess.TimeoutExpired):
        run_candidate_training(candidate,tmp_path/'config.yaml',tmp_path/'training',timeout_s=1)
    assert '--output-dir' in calls[-1][0]
    assert calls[-1][1]['shell'] is False
