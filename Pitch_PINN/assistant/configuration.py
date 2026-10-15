from __future__ import annotations

from copy import deepcopy
import math

from .schemas import ConfigChange

PARAMETERS = ('k_front_npm', 'k_rear_npm', 'c_front_nspm', 'c_rear_nspm')
WEIGHTS = ('pitch_data', 'physics_heave', 'physics_roll', 'physics_pitch', 'damper_length', 'damper_velocity', 'initial', 'regularization')
INTEGER_MINIMUMS = {
    'project.seed': 0, 'training.epochs': 0, 'training.collocation_points': 2,
    'training.log_every': 1, 'data.sparse_stride': 1,
    'data.remove_bias_samples': 0, 'data.baseline_samples': 0,
    'data.drop_last_samples': 0, 'model.hidden_width': 1, 'model.hidden_layers': 1,
}
NUMERIC_PATHS = {'training.learning_rate'} | {'training.weights.'+w for w in WEIGHTS} | {
    f'model.parameters.{p}.{field}' for p in PARAMETERS for field in ('initial', 'min', 'max')
}


def _location(config: dict, path: str):
    parts = path.split('.')
    parent = config
    for part in parts[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            raise ValueError('missing configuration section')
        parent = parent[part]
    if not isinstance(parent, dict) or parts[-1] not in parent:
        raise ValueError('missing configuration field')
    return parent, parts[-1]


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def apply_config_changes(base: dict, changes: list[ConfigChange]) -> tuple[dict, list[dict]]:
    config, diff, seen = deepcopy(base), [], set()
    for change in changes:
        path = change.path
        if path in seen or path not in INTEGER_MINIMUMS and path not in NUMERIC_PATHS:
            raise ValueError('duplicate or unsupported configuration path')
        seen.add(path)
        parent, key = _location(config, path)
        diff.append({'path': path, 'before': parent[key], 'after': change.value})
        parent[key] = change.value
    for path, minimum in INTEGER_MINIMUMS.items():
        parent, key = _location(config, path)
        if type(parent[key]) is not int or parent[key] < minimum:
            raise ValueError(f'{path} must be an integer >= {minimum}')
    for path in NUMERIC_PATHS:
        parent, key = _location(config, path)
        if not _number(parent[key]):
            raise ValueError('numeric settings must be finite numbers')
    if config['training']['learning_rate'] <= 0:
        raise ValueError('learning rate must be positive')
    weights = [config['training']['weights'][w] for w in WEIGHTS]
    if any(w < 0 for w in weights) or not any(w > 0 for w in weights):
        raise ValueError('loss weights must be non-negative and not all zero')
    for p in PARAMETERS:
        item = config['model']['parameters'][p]
        if not 0 < item['min'] < item['max'] or not item['min'] <= item['initial'] <= item['max']:
            raise ValueError('invalid suspension parameter bounds or initial value')
    return config, diff
