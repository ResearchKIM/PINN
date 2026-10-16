from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
import yaml

from Pitch_PINN.assistant import runner
from Pitch_PINN.assistant.schemas import AssistantOptions, AssistantPlan, ConfigChange, PlanResult


@pytest.fixture
def dataset_project(tmp_path):
    config = yaml.safe_load((Path(__file__).parents[1]/'config/tesla_ev6.yaml').read_text(encoding='utf-8'))
    source = tmp_path/'Pitch_PINN/config/tesla_ev6.yaml'
    source.parent.mkdir(parents=True)
    source.write_text(yaml.safe_dump(config), encoding='utf-8')
    signals = {name: np.arange(6, dtype=float) for name in ('Sim_Time', 'Ax', 'PITCH_RATE')}
    for group in ('wheel_velocity', 'wheel_acceleration', 'damper_length', 'damper_velocity'):
        signals.update({name: np.zeros(6) for name in config['data']['keys'][group].values()})
    folder = tmp_path/'PINN_file/src_cm4sl/Data'
    for scenario in ('3.ACC', '4.DEC'):
        target = folder/scenario/'0.2g_EV6.mat'
        target.parent.mkdir(parents=True)
        savemat(target, signals)
    savemat(folder/'3.ACC/0.2g_NX300h.mat', signals)
    return tmp_path, source, signals


def run_change(project, source, value, **kwargs):
    def planner(*args, **context):
        plan = AssistantPlan(kind='config_change', summary='데이터 변경', explanation='가속 데이터 선택',
                             questions=[], config_changes=[ConfigChange(path='data.mat_path', value=value)], file_changes=[])
        return PlanResult(plan, context['model'], 'test', 'test')
    return runner.run_request(AssistantOptions(config_path=source, request='0.2g EV6 가속 데이터로 변경', **kwargs), project_root=project, planner=planner)


def test_dataset_catalog_is_sent_without_mat_contents_and_does_not_guess_ambiguity(dataset_project):
    project, source, _ = dataset_project
    before = source.read_bytes()
    def planner(*args, **context):
        entries = context['datasets']
        paths = [entry['path'] for entry in entries]
        assert any('3.ACC/0.2g_EV6.mat' in p for p in paths)
        assert any('4.DEC/0.2g_EV6.mat' in p for p in paths)
        assert not any('NX300h' in p for p in paths)
        assert all(set(e) == {'path', 'scenario_name', 'vehicle'} for e in entries)
        plan = AssistantPlan(kind='clarification', summary='주행 조건 확인', explanation='가속과 감속 파일 존재', questions=['가속인가요, 감속인가요?'], config_changes=[], file_changes=[])
        return PlanResult(plan, context['model'], 'test', 'test')
    result = runner.run_request(AssistantOptions(config_path=source, request='0.2g EV6 데이터로 변경'), project_root=project, planner=planner)
    assert result.state == 'needs_clarification'
    assert source.read_bytes() == before
    assert (result.run_dir/'datasets.json').is_file()


def test_selected_dataset_updates_original_metadata_and_backup(dataset_project):
    project, source, _ = dataset_project
    before = source.read_bytes()
    old = yaml.safe_load(before)
    target = 'PINN_file/src_cm4sl/Data/3.ACC/0.2g_EV6.mat'
    result = run_change(project, source, target)
    assert result.state == 'config_applied'
    updated = yaml.safe_load(source.read_text(encoding='utf-8'))
    assert updated['data']['mat_path'] == target
    assert updated['data']['scenario_name'] == '3.ACC/0.2g_EV6'
    assert updated['training'] == old['training']
    assert (result.run_dir/'source_config_before.yaml').read_bytes() == before
    resolved = yaml.safe_load((result.run_dir/'resolved_config.yaml').read_text(encoding='utf-8'))
    assert Path(resolved['data']['mat_path']) == project/target
    diff = json.loads((result.run_dir/'config_diff.json').read_text(encoding='utf-8'))
    assert {d['path'] for d in diff} == {'data.mat_path', 'data.scenario_name'}
    assert json.loads((result.run_dir/'status.json').read_text())['dataset_validation']['samples'] == 5


def test_dataset_preview_checks_file_but_preserves_original(dataset_project):
    project, source, _ = dataset_project
    before = source.read_bytes()
    result = run_change(project, source, str(project/'PINN_file/src_cm4sl/Data/4.DEC/0.2g_EV6.mat'), preview=True)
    assert result.state == 'planned'
    assert source.read_bytes() == before


def test_training_receives_verified_dataset_and_metadata(dataset_project, monkeypatch):
    project, source, _ = dataset_project
    calls = []
    monkeypatch.setattr(runner, 'train_configuration', lambda config, out: calls.append((config, out)))
    target = 'PINN_file/src_cm4sl/Data/4.DEC/0.2g_EV6.mat'
    result = run_change(project, source, target, train=True)
    assert result.state == 'trained'
    assert len(calls) == 1
    assert Path(calls[0][0]['data']['mat_path']) == project/target
    assert calls[0][0]['data']['scenario_name'] == '4.DEC/0.2g_EV6'


@pytest.mark.parametrize('target', ['PINN_file/src_cm4sl/Data/3.ACC/missing.mat', '../outside.mat', 'https://example.com/data.mat', 'PINN_file/src_cm4sl/Data/3.ACC/0.2g_NX300h.mat'])
def test_unlisted_missing_external_or_other_vehicle_rejected(dataset_project, target):
    project, source, _ = dataset_project
    before = source.read_bytes()
    result = run_change(project, source, target)
    assert result.state == 'failed'
    assert source.read_bytes() == before
    assert not (result.run_dir/'source_config_before.yaml').exists()


@pytest.mark.parametrize('failure', ['missing_signal', 'invalid_mat', 'too_short', 'nonfinite'])
def test_incompatible_mat_rejected_before_source_write(dataset_project, failure):
    project, source, signals = dataset_project
    target = project/'PINN_file/src_cm4sl/Data/3.ACC/0.2g_EV6.mat'
    if failure == 'invalid_mat':
        target.write_text('not a MAT file')
    else:
        content = deepcopy(signals)
        if failure == 'missing_signal':
            del content['Ax']
        elif failure == 'too_short':
            content = {k:v[:2] for k,v in content.items()}
        else:
            content['Ax'][:] = np.nan
        savemat(target, content)
    before = source.read_bytes()
    result = run_change(project, source, str(target), train=True)
    assert result.state == 'failed'
    assert source.read_bytes() == before
    status = json.loads((result.run_dir/'status.json').read_text())
    assert status['error_code'] == 'dataset_invalid'
    assert not (result.run_dir/'training').exists()
    assert (result.run_dir/'plan.json').is_file()


def test_strings_remain_invalid_for_numeric_settings(dataset_project):
    from Pitch_PINN.assistant.configuration import apply_config_changes
    _, source, _ = dataset_project
    with pytest.raises(ValueError):
        apply_config_changes(yaml.safe_load(source.read_text()), [ConfigChange(path='training.epochs', value='30')])
