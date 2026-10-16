from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from Pitch_PINN.assistant.configuration import apply_config_changes
from Pitch_PINN.assistant.schemas import AssistantPlan, ConfigChange, validate_plan


@pytest.fixture
def base():
    return yaml.safe_load((Path(__file__).parents[1] / 'config/tesla_ev6.yaml').read_text(encoding='utf-8'))


def test_config_copy_preserves_original(base):
    original = deepcopy(base)
    result, diff = apply_config_changes(base, [ConfigChange(path='training.epochs', value=20), ConfigChange(path='data.sparse_stride', value=10)])
    assert base == original
    assert result['training']['epochs'] == 20
    assert result['data']['sparse_stride'] == 10
    assert len(diff) == 2


def test_relative_request_uses_absolute_plan_value(base):
    base['training']['weights']['physics_pitch'] = 2.0
    result, diff = apply_config_changes(base, [ConfigChange(path='training.weights.physics_pitch', value=6.0)])
    assert diff == [{'path': 'training.weights.physics_pitch', 'before': 2.0, 'after': 6.0}]
    assert result['training']['weights']['physics_pitch'] == 6.0


@pytest.mark.parametrize('path,value', [('data.mat_path', 1), ('project.output_dir', 1), ('carmaker.command', 1), ('training.use_lbfgs', 1), ('training.epochs', 1.5), ('training.collocation_points', 1), ('training.learning_rate', 0), ('model.hidden_width', 0), ('project.seed', -1), ('training.weights.physics_pitch', -1), ('training.learning_rate', float('nan')), ('training.learning_rate', float('inf'))])
def test_reject_unknown_duplicate_and_nonfinite_changes(base, path, value):
    with pytest.raises(ValueError):
        apply_config_changes(base, [ConfigChange(path=path, value=value)])


def test_reject_bool_and_duplicate(base):
    with pytest.raises(ValidationError):
        ConfigChange(path='training.epochs', value=True)
    c = ConfigChange(path='training.epochs', value=10)
    with pytest.raises(ValueError):
        apply_config_changes(base, [c, c])


def test_validate_parameter_bounds_after_all_changes(base):
    root = 'model.parameters.k_front_npm.'
    result, _ = apply_config_changes(base, [ConfigChange(path=root+'min', value=200000), ConfigChange(path=root+'max', value=400000), ConfigChange(path=root+'initial', value=300000)])
    assert result['model']['parameters']['k_front_npm']['initial'] == 300000
    with pytest.raises(ValueError):
        apply_config_changes(base, [ConfigChange(path=root+'min', value=200000)])


def test_missing_base_and_zero_weights(base):
    with pytest.raises(ValueError):
        apply_config_changes({}, [])
    with pytest.raises(ValueError):
        apply_config_changes(base, [ConfigChange(path='training.weights.'+k, value=0) for k in base['training']['weights']])


@pytest.mark.parametrize('kind,changes,files,questions', [('config_change', [], [], []), ('code_change', [], [], []), ('clarification', [], [], []), ('unsupported', [{'path':'training.epochs','value':20}], [], []), ('config_change', [], [{'path':'x.py','content':'x=1'}], [])])
def test_plan_kind_payload_consistency(kind, changes, files, questions):
    plan = AssistantPlan(kind=kind, summary='요약', explanation='설명', questions=questions, config_changes=changes, file_changes=files)
    with pytest.raises(ValueError):
        validate_plan(plan)
