"""Shared source boundary for cloud context and candidate edits."""
from pathlib import Path, PurePosixPath


def reject_links(path: Path, root: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink() or getattr(part, 'is_junction', lambda: False)():
            raise ValueError('links are not allowed in source paths')
        if part == root:
            break


def allowed_source(root: Path, raw: str) -> Path:
    parts = raw.split('/')
    if '\\' in raw or ':' in raw or '\x00' in raw or any(p in ('', '.', '..') for p in parts):
        raise ValueError('invalid source path')
    path = PurePosixPath(raw)
    if path.is_absolute() or len(parts) < 2 or parts[0] != 'Pitch_PINN':
        raise ValueError('source must be inside Pitch_PINN')
    allowed = raw == 'Pitch_PINN/train_inverse_pinn.py' or len(parts) >= 3 and parts[1] in ('models', 'physics', 'training')
    if not allowed or path.suffix != '.py':
        raise ValueError('source is outside editable modules')
    root = root.resolve()
    target = root.joinpath(*parts)
    reject_links(target, root)
    target.resolve().relative_to(root)
    if not target.is_file():
        raise ValueError('only existing Python sources may be changed')
    return target


def source_paths(root: Path):
    package = root / 'Pitch_PINN'
    paths = [package / 'train_inverse_pinn.py']
    for folder in ('models', 'physics', 'training'):
        directory = package / folder
        reject_links(directory, root.resolve())
        paths.extend(directory.rglob('*.py'))
    for path in sorted(paths):
        if path.is_file():
            raw = path.relative_to(root).as_posix()
            yield raw, allowed_source(root, raw)
