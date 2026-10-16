"""Discover local scenario names and validate selections before changing YAML."""
from __future__ import annotations

from pathlib import Path

from .configuration import STRING_PATHS, apply_config_changes
from .paths import reject_links
from .schemas import ConfigChange


class DatasetError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def collect_datasets(root: Path, vehicle: str) -> list[dict[str, str]]:
    """Send names only; selecting a dataset does not change vehicle physics."""
    directory = root/'PINN_file/src_cm4sl/Data'
    reject_links(directory, root)
    entries = []
    for path in sorted(directory.rglob('*.mat')):
        if path.stem.casefold() != vehicle.casefold() and not path.stem.casefold().endswith('_'+vehicle.casefold()):
            continue
        try:
            reject_links(path, root)
            path.resolve().relative_to(directory.resolve())
        except ValueError:
            continue
        if path.is_file():
            entries.append({'path': path.relative_to(root).as_posix(),
                            'scenario_name': path.relative_to(directory).with_suffix('').as_posix(),
                            'vehicle': vehicle})
    return entries


def validate_dataset_changes(root: Path, base: dict, changes: list[ConfigChange], datasets: list[dict[str, str]]) -> tuple[list[ConfigChange], dict | None]:
    related = [change for change in changes if change.path in STRING_PATHS]
    if not related:
        return changes, None
    paths = [change for change in related if change.path == 'data.mat_path']
    if len(paths) != 1 or not isinstance(paths[0].value, str):
        raise DatasetError('dataset_not_allowed')
    selected = Path(paths[0].value)
    if not selected.is_absolute():
        selected = root/selected
    try:
        reject_links(selected, root)
        selected = selected.resolve()
        entry = next((item for item in datasets if (root/item['path']).resolve() == selected), None)
        if entry is None or entry['vehicle'] != base['data']['vehicle'] or not selected.is_file():
            raise ValueError('unlisted dataset')
    except (ValueError, OSError):
        raise DatasetError('dataset_not_allowed') from None
    names = [change for change in related if change.path == 'data.scenario_name']
    if len(names) > 1 or names and names[0].value != entry['scenario_name']:
        raise DatasetError('dataset_not_allowed')
    updated = list(changes)
    if not names:
        updated.append(ConfigChange(path='data.scenario_name', value=entry['scenario_name']))
    config, _ = apply_config_changes(base, updated)
    try:
        from Pitch_PINN.data.mat_loader import load_mat_scenario
        from Pitch_PINN.data.preprocess import preprocess_scenario
        dc = config['data']
        scenario = preprocess_scenario(
            load_mat_scenario(selected, dc['keys'], name=dc['scenario_name'], vehicle=dc['vehicle']),
            remove_bias_samples=dc['remove_bias_samples'], baseline_samples=dc['baseline_samples'],
            drop_last_samples=dc['drop_last_samples'])
    except Exception:
        raise DatasetError('dataset_invalid') from None
    return updated, {'path': str(selected), 'scenario_name': entry['scenario_name'], 'samples': len(scenario)}
