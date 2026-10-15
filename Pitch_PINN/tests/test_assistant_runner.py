from pathlib import Path
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import yaml

from Pitch_PINN.assistant import runner
from Pitch_PINN.assistant.__main__ import main
from Pitch_PINN.assistant.schemas import AssistantOptions, AssistantPlan, PlanResult


@pytest.fixture
def project(tmp_path):
    p = tmp_path/'Pitch_PINN/config/tesla_ev6.yaml'
    p.parent.mkdir(parents=True)
    p.write_text((Path(__file__).parents[1]/'config/tesla_ev6.yaml').read_text(encoding='utf-8'), encoding='utf-8')
    return tmp_path


def planner(kind='config_change', *, files=None, questions=None):
    plan = AssistantPlan(kind=kind, summary='학습 변경', explanation='설명', questions=questions or [], config_changes=[{'path':'training.epochs','value':2}] if kind=='config_change' else [], file_changes=files or [])
    return lambda *a,**k: PlanResult(plan,k['model'],'returned-model','resp_test')


def options(project, **kwargs):
    return AssistantOptions(config_path=project/'Pitch_PINN/config/tesla_ev6.yaml', request='2 epoch 학습', **kwargs)


def test_cli_preview_does_not_train(project,monkeypatch):
    train = Mock()
    monkeypatch.setattr(runner,'train_configuration',train)
    result = runner.run_request(options(project,preview=True),project_root=project,planner=planner())
    assert result.state == 'planned' and result.exit_code == 0
    train.assert_not_called()
    assert yaml.safe_load((result.run_dir/'resolved_config.yaml').read_text(encoding='utf-8'))['training']['epochs'] == 2
    assert json.loads((result.run_dir/'status.json').read_text())['returned_model']=='returned-model'


def test_train_uses_resolved_copy_and_unique_output(project,monkeypatch):
    calls = []
    monkeypatch.setattr(runner,'train_configuration',lambda config,out: calls.append((config,out)))
    before = options(project).config_path.read_bytes()
    first = runner.run_request(options(project,train=True),project_root=project,planner=planner())
    second = runner.run_request(options(project,train=True),project_root=project,planner=planner())
    assert first.state == second.state == 'trained'
    assert first.run_dir != second.run_dir
    assert calls[0][1].is_relative_to(first.run_dir)
    assert yaml.safe_load(options(project).config_path.read_text(encoding='utf-8'))['training']['epochs'] == 2
    assert (first.run_dir/'source_config_before.yaml').read_bytes()==before


@pytest.mark.parametrize('kind,questions,state',[('clarification',['몇 epoch인가요?'],'needs_clarification'),('unsupported',[],'unsupported')])
def test_clarification_and_unsupported_never_execute(project,monkeypatch,kind,questions,state):
    train=Mock()
    monkeypatch.setattr(runner,'train_configuration',train)
    result=runner.run_request(options(project,train=True),project_root=project,planner=planner(kind,questions=questions))
    assert result.state==state and result.exit_code==2
    train.assert_not_called()


def test_failure_status_written_without_secret(project, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','secret-canary')
    def fail(*a,**k):
        raise RuntimeError('secret-canary')
    result=runner.run_request(options(project),project_root=project,planner=fail)
    assert result.state=='failed' and result.exit_code==1
    assert 'secret-canary' not in (result.run_dir/'status.json').read_text()


def test_code_requires_flag_and_execution_requires_run_candidate(project,monkeypatch):
    p=project/'Pitch_PINN/models/a.py'
    p.parent.mkdir(); p.write_text('VALUE=1')
    fake=planner('code_change',files=[{'path':'Pitch_PINN/models/a.py','content':'VALUE=2'}])
    tests=Mock(return_value=NS(returncode=0))
    monkeypatch.setattr(runner,'run_candidate_tests',tests)
    result=runner.run_request(options(project),project_root=project,planner=fake)
    assert result.state=='failed'
    ready=runner.run_request(options(project,code=True,train=True),project_root=project,planner=fake)
    assert ready.state=='candidate_ready'
    tests.assert_not_called()
    assert p.read_text()=='VALUE=1'


@pytest.mark.parametrize('failure',[NS(returncode=1),'timeout'])
def test_tests_failed_or_timed_out_prevents_training(project,monkeypatch,failure):
    import subprocess
    p=project/'Pitch_PINN/models/a.py'; p.parent.mkdir(); p.write_text('VALUE=1')
    fake=planner('code_change',files=[{'path':'Pitch_PINN/models/a.py','content':'VALUE=2'}])
    def test(*a,**k):
        if failure=='timeout':
            raise subprocess.TimeoutExpired('test',1)
        return failure
    monkeypatch.setattr(runner,'run_candidate_tests',test)
    train=Mock(); monkeypatch.setattr(runner,'run_candidate_training',train)
    result=runner.run_request(options(project,code=True,run_candidate=True,train=True),project_root=project,planner=fake)
    assert result.state=='failed'
    train.assert_not_called()


def test_resume_without_api_and_with_preserved_source(project,monkeypatch):
    p=project/'Pitch_PINN/models/a.py'; p.parent.mkdir(); p.write_text('VALUE=1')
    ready=runner.run_request(options(project,code=True),project_root=project,planner=planner('code_change',files=[{'path':'Pitch_PINN/models/a.py','content':'VALUE=2'}]))
    saved_path=ready.run_dir/'resolved_config.yaml'
    saved=yaml.safe_load(saved_path.read_text(encoding='utf-8'))
    saved['data']['mat_path']='local_data/scenario.mat'
    saved_path.write_text(yaml.safe_dump(saved),encoding='utf-8')
    monkeypatch.setattr(runner,'run_candidate_tests',lambda *a,**k:NS(returncode=0))
    never=Mock(side_effect=AssertionError('no API call on resume'))
    resumed=runner.run_request(AssistantOptions(config_path=options(project).config_path,resume_dir=ready.run_dir,run_candidate=True),project_root=project,planner=never)
    assert resumed.state=='tests_passed'
    assert resumed.run_dir != ready.run_dir
    assert Path(yaml.safe_load((resumed.run_dir/'resolved_config.yaml').read_text(encoding='utf-8'))['data']['mat_path'])==(project/'local_data/scenario.mat').resolve()
    never.assert_not_called()


def test_missing_api_key_fails_without_training(project,monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    result=runner.run_request(options(project,train=True,backend='api'),project_root=project)
    status=json.loads((result.run_dir/'status.json').read_text())
    assert status['error_code']=='missing_key'


def test_relative_data_path_is_preserved_when_candidate_cwd_changes(project,monkeypatch):
    config_path=options(project).config_path
    base=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    base['data']['mat_path']='local_data/scenario.mat'
    config_path.write_text(yaml.safe_dump(base),encoding='utf-8')
    p=project/'Pitch_PINN/models/a.py'; p.parent.mkdir(); p.write_text('VALUE=1')
    monkeypatch.setattr(runner,'run_candidate_tests',lambda *a,**k:NS(returncode=0))
    seen=[]
    def train(candidate,config_path,out):
        seen.append(yaml.safe_load(config_path.read_text(encoding='utf-8'))['data']['mat_path'])
        return NS(returncode=0)
    monkeypatch.setattr(runner,'run_candidate_training',train)
    result=runner.run_request(options(project,code=True,run_candidate=True,train=True),project_root=project,planner=planner('code_change',files=[{'path':'Pitch_PINN/models/a.py','content':'VALUE=2'}]))
    assert result.state=='trained'
    assert Path(seen[0])==(project/'local_data/scenario.mat').resolve()


def test_cli_help_and_request_requirement(capsys):
    with pytest.raises(SystemExit) as e:
        main(['--help'])
    assert e.value.code==0 and '--resume' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([])


def test_cli_help_on_non_unicode_console():
    import os
    import subprocess
    import sys
    env = dict(os.environ, PYTHONIOENCODING='ascii')
    result = subprocess.run([sys.executable,'-m','Pitch_PINN.assistant','--help'],env=env,capture_output=True)
    assert result.returncode == 0, result.stderr.decode('ascii',errors='replace')
    assert '자연어' in result.stdout.decode('utf-8')


def test_default_request_updates_selected_yaml_and_keeps_exact_backup(project):
    path=options(project).config_path
    before=path.read_bytes()
    result=runner.run_request(AssistantOptions(config_path=path,request='2 epoch로 변경'),project_root=project,planner=planner())
    assert result.state=='config_applied'
    assert yaml.safe_load(path.read_text(encoding='utf-8'))['training']['epochs']==2
    assert (result.run_dir/'source_config_before.yaml').read_bytes()==before
    assert json.loads((result.run_dir/'status.json').read_text())['source_config_updated'] is True
    from Pitch_PINN.train_inverse_pinn import load_config
    assert load_config(path)['training']['epochs']==2


def test_apply_changes_are_cumulative(project):
    path=options(project).config_path
    runner.run_request(AssistantOptions(config_path=path,request='2 epoch로 변경'),project_root=project,planner=planner())
    def next_plan(*a,**k):
        assert a[1]['training']['epochs']==2
        plan=AssistantPlan(kind='config_change',summary='관측 간격 변경',explanation='설명',questions=[],config_changes=[{'path':'data.sparse_stride','value':10}],file_changes=[])
        return PlanResult(plan,k['model'],'mock','mock')
    result=runner.run_request(AssistantOptions(config_path=path,request='관측간격10'),project_root=project,planner=next_plan)
    current=yaml.safe_load(path.read_text(encoding='utf-8'))
    assert result.state=='config_applied' and current['training']['epochs']==2 and current['data']['sparse_stride']==10


def test_apply_preserves_comments_line_endings_and_relative_paths(project):
    path=options(project).config_path
    original_text=path.read_text(encoding='utf-8')
    epoch=yaml.safe_load(original_text)['training']['epochs']
    text=original_text.replace(f'epochs: {epoch}','epochs: 5000 # keep this comment')
    text=text.replace('mat_path: E:/PINN/PINN_file/src_cm4sl/Data/3.ACC/0.1g_EV6.mat','mat_path: local_data/test.mat')
    before=('# user header\n'+text).replace('\n','\r\n').encode('utf-8')
    path.write_bytes(before)
    result=runner.run_request(AssistantOptions(config_path=path,request='2 epoch'),project_root=project,planner=planner())
    assert result.state=='config_applied'
    assert path.read_bytes()==before.replace(b'epochs: 5000',b'epochs: 2')


def test_preview_leaves_selected_yaml_unchanged(project):
    path=options(project).config_path
    before=path.read_bytes()
    result=runner.run_request(AssistantOptions(config_path=path,request='2 epoch',preview=True),project_root=project,planner=planner())
    assert result.state=='planned' and path.read_bytes()==before


def test_source_edited_during_planning_is_not_overwritten(project):
    path=options(project).config_path
    before=path.read_bytes()
    def concurrent(*a,**k):
        path.write_bytes(before+b'\n# concurrent user edit\n')
        return planner()(*a,**k)
    result=runner.run_request(AssistantOptions(config_path=path,request='2 epoch'),project_root=project,planner=concurrent)
    assert result.state=='failed'
    assert path.read_bytes()==before+b'\n# concurrent user edit\n'


def test_preview_cannot_start_training(project,monkeypatch):
    train=Mock(); monkeypatch.setattr(runner,'train_configuration',train)
    result=runner.run_request(AssistantOptions(config_path=options(project).config_path,request='2 epoch',preview=True,train=True),project_root=project,planner=planner())
    assert result.state=='failed'
    train.assert_not_called()


def test_update_preserves_utf8_bom(project):
    path=options(project).config_path
    path.write_bytes(b'\xef\xbb\xbf'+path.read_bytes())
    result=runner.run_request(AssistantOptions(config_path=path,request='2 epoch'),project_root=project,planner=planner())
    assert result.state=='config_applied' and path.read_bytes().startswith(b'\xef\xbb\xbf')


def test_custom_config_is_updated_without_changing_default(project):
    default=options(project).config_path
    before=default.read_bytes()
    selected=default.with_name('custom.yaml')
    selected.write_bytes(before)
    result=runner.run_request(AssistantOptions(config_path=selected,request='2 epoch'),project_root=project,planner=planner())
    assert result.state=='config_applied' and default.read_bytes()==before
    assert yaml.safe_load(selected.read_text(encoding='utf-8'))['training']['epochs']==2


def test_invalid_changes_leave_source_untouched(project):
    path=options(project).config_path
    before=path.read_bytes()
    def invalid(*a,**k):
        plan=AssistantPlan(kind='config_change',summary='실패',explanation='음수',questions=[],config_changes=[{'path':'training.epochs','value':-1}],file_changes=[])
        return PlanResult(plan,k['model'],'mock','mock')
    result=runner.run_request(AssistantOptions(config_path=path,request='음수'),project_root=project,planner=invalid)
    assert result.state=='failed' and path.read_bytes()==before
    assert not (result.run_dir/'source_config_before.yaml').exists()
