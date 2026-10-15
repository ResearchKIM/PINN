from __future__ import annotations

import json
import os
from pathlib import Path


def safe_text(value: str) -> str:
    key = os.environ.get('OPENAI_API_KEY', '')
    return value.replace(key, '[REDACTED]') if key else value


def write_json(path: Path, data) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    key = os.environ.get('OPENAI_API_KEY', '')
    if key:
        text = text.replace(json.dumps(key)[1:-1], '[REDACTED]')
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)
