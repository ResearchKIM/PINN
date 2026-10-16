import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import pytest

from Pitch_PINN.assistant.codex_planner import request_codex_plan, resolve_codex
from Pitch_PINN.assistant.llm import PlannerError


def payload():
    return {'kind':'config_change','summary':'학습 설정 변경','explanation':'설명','questions':[], 'config_changes':[{'path':'training.epochs','value':20}], 'file_changes':[]}


def process(monkeypatch, *, auth='Logged in using ChatGPT', auth_exit=0, exec_exit=0, text=None, event=None):
    calls=[]
    monkeypatch.setattr('Pitch_PINN.assistant.codex_planner.resolve_codex',lambda:Path('codex.exe'))
    def run(args,**kwargs):
        calls.append((args,kwargs))
        if args[1:]==['login','status']:
            return NS(returncode=auth_exit,stdout='',stderr=auth)
        target=Path(args[args.index('--output-last-message')+1])
        target.write_text(json.dumps(payload(),ensure_ascii=False) if text is None else text,encoding='utf-8')
        events=event if event is not None else [{'type':'thread.started','thread_id':'thread-test'},{'type':'turn.completed','usage':{'input_tokens':100,'output_tokens':20}}]
        return NS(returncode=exec_exit,stdout='\n'.join(json.dumps(e) for e in events),stderr='')
    monkeypatch.setattr(subprocess,'run',run)
    return calls


def test_codex_login_produces_validated_plan_without_api_key(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    calls=process(monkeypatch)
    result=request_codex_plan('20 epoch로',{'training':{'epochs':5000}},model='chosen',sources={})
    assert result.plan.config_changes[0].value==20
    assert result.requested_model=='chosen'
    assert result.response_id=='thread-test'
    assert result.returned_model=='unreported_by_codex_cli'
    args,kwargs=calls[1]
    assert args[args.index('--model')+1]=='chosen'
    assert args[args.index('--sandbox')+1]=='read-only'
    assert '--ignore-user-config' in args and '--ephemeral' in args
    assert kwargs['shell'] is False and kwargs['encoding']=='utf-8'
    assert '20 epoch' in kwargs['input']


def test_api_credentials_are_not_passed_to_codex(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','api-canary')
    monkeypatch.setenv('CODEX_API_KEY','codex-api-canary')
    calls=process(monkeypatch)
    request_codex_plan('요청',{},model='chosen',sources={})
    assert all('OPENAI_API_KEY' not in k['env'] and 'CODEX_API_KEY' not in k['env'] for _,k in calls)


def test_dataset_catalog_is_in_codex_prompt(monkeypatch):
    calls = process(monkeypatch)
    datasets = [{'path':'PINN_file/src_cm4sl/Data/3.ACC/0.2g_EV6.mat','scenario_name':'3.ACC/0.2g_EV6','vehicle':'EV6'}]
    request_codex_plan('0.2g EV6 가속 데이터', {}, model='chosen', sources={}, datasets=datasets)
    assert json.dumps(datasets, ensure_ascii=False) in calls[1][1]['input']
    assert 'return clarification' in calls[1][1]['input']


@pytest.mark.parametrize('auth,exit_code,error', [('Not logged in',1,'codex_login_required'),('Logged in using API key',0,'codex_chatgpt_login_required')])
def test_missing_or_api_login_stops_before_cloud(monkeypatch,auth,exit_code,error):
    calls=process(monkeypatch,auth=auth,auth_exit=exit_code)
    with pytest.raises(PlannerError,match=error):
        request_codex_plan('요청',{},model='chosen',sources={})
    assert len(calls)==1


@pytest.mark.parametrize('text', ['not JSON', '{}', json.dumps(dict(payload(),file_changes=[{'path':'x.py','content':'x=1'}]))])
def test_invalid_cli_output_never_becomes_plan(monkeypatch,text):
    process(monkeypatch,text=text)
    with pytest.raises(PlannerError,match='invalid_plan'):
        request_codex_plan('요청',{},model='chosen',sources={})


def test_cli_failure_and_timeout_are_sanitized(monkeypatch):
    process(monkeypatch,exec_exit=1)
    with pytest.raises(PlannerError,match='codex_exec_failed'):
        request_codex_plan('요청',{},model='chosen',sources={})
    def timeout(*a,**k):
        raise subprocess.TimeoutExpired('secret-canary',1)
    monkeypatch.setattr(subprocess,'run',timeout)
    with pytest.raises(PlannerError) as e:
        request_codex_plan('요청',{},model='chosen',sources={})
    assert e.value.code=='codex_timeout' and 'secret-canary' not in str(e.value)


def test_turn_failure_rejects_even_valid_final_json(monkeypatch):
    process(monkeypatch,event=[{'type':'turn.failed','error':{'message':'private error'}}])
    with pytest.raises(PlannerError,match='codex_exec_failed'):
        request_codex_plan('요청',{},model='chosen',sources={})


def test_no_turn_completion_rejects_final_json(monkeypatch):
    process(monkeypatch,event=[])
    with pytest.raises(PlannerError,match='codex_incomplete'):
        request_codex_plan('요청',{},model='chosen',sources={})


def test_app_executable_is_discovered_without_terminal_path(tmp_path,monkeypatch):
    monkeypatch.delenv('PINN_CODEX_PATH',raising=False)
    monkeypatch.setenv('LOCALAPPDATA',str(tmp_path))
    monkeypatch.setattr('shutil.which',lambda name:None)
    path=tmp_path/'OpenAI/Codex/bin/version/codex.exe'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'placeholder-not-executed')
    assert resolve_codex()==path


def test_missing_executable_has_actionable_error(tmp_path,monkeypatch):
    monkeypatch.setenv('PINN_CODEX_PATH',str(tmp_path/'missing.exe'))
    with pytest.raises(PlannerError,match='codex_executable_not_found'):
        resolve_codex()
