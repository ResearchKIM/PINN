import json
from types import SimpleNamespace as NS

import pytest

from Pitch_PINN.assistant.llm import PlannerError, collect_source_context, request_plan
from Pitch_PINN.assistant.schemas import AssistantPlan


def plan():
    return AssistantPlan(kind='config_change', summary='변경', explanation='설명', questions=[], config_changes=[{'path':'training.weights.physics_pitch','value':6.0}], file_changes=[])


def fake(response=None, error=None):
    calls = []
    def parse(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return response
    return NS(responses=NS(parse=parse)), calls


def test_responses_parse_receives_request_config_and_schema():
    client, calls = fake(NS(status='completed', output=[], output_parsed=plan(), model='returned', id='resp_1'))
    result = request_plan('물리 손실 3배', {'training':{'weights':{'physics_pitch':2.0}}}, model='chosen', sources={}, client=client)
    assert result.returned_model == 'returned'
    assert result.response_id == 'resp_1'
    assert len(calls) == 1 and calls[0]['model'] == 'chosen'
    assert calls[0]['text_format'] is AssistantPlan
    assert calls[0]['store'] is False
    assert '2.0' in json.dumps(calls[0]['input'])


@pytest.mark.parametrize('status,output,parsed,code', [('incomplete',[],None,'incomplete'), ('completed',[NS(content=[NS(type='refusal')])],None,'refused'), ('completed',[],None,'invalid_plan')])
def test_missing_key_refusal_incomplete_invalid_plan(status, output, parsed, code):
    client, _ = fake(NS(status=status, output=output, output_parsed=parsed))
    with pytest.raises(PlannerError) as e:
        request_plan('요청', {}, model='chosen', sources={}, client=client)
    assert e.value.code == code


def test_missing_key(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(PlannerError, match='missing_key'):
        request_plan('요청', {}, model='chosen', sources={})


def test_auth_timeout_model_error_redacts_secrets():
    import openai
    for exc, code in [(TimeoutError('secret'), 'timeout'), (ValueError('secret'),'invalid_plan'), (RuntimeError('secret'),'api_error')]:
        client, calls = fake(error=exc)
        with pytest.raises(PlannerError) as e:
            request_plan('요청', {}, model='chosen', sources={}, client=client)
        assert e.value.code == code and 'secret' not in str(e.value)
        assert len(calls) == 1


@pytest.mark.parametrize('name,status,expected', [('AuthenticationError',401,'authentication'),('PermissionDeniedError',403,'model_unavailable'),('NotFoundError',404,'model_unavailable')])
def test_sdk_auth_and_model_errors_are_classified_without_leaking(name,status,expected):
    import openai
    error=getattr(openai,name)('secret-canary',response=NS(request=None,status_code=status,headers={}),body=None)
    client,calls=fake(error=error)
    with pytest.raises(PlannerError) as e:
        request_plan('요청',{},model='chosen',sources={},client=client)
    assert e.value.code==expected
    assert 'secret-canary' not in str(e.value)
    assert len(calls)==1


def test_context_excludes_data_tests_and_secrets(tmp_path):
    root = tmp_path / 'Pitch_PINN'
    for name in ['models/a.py','tests/a.py','outputs/a.py','assistant/secret.py','physics/sample.mat']:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('x=1', encoding='utf-8')
    assert collect_source_context(tmp_path) == {'Pitch_PINN/models/a.py':'x=1'}
    with pytest.raises(PlannerError, match='context_too_large'):
        collect_source_context(tmp_path, max_bytes=1)
