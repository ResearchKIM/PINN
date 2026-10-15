from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_request
from .schemas import AssistantOptions


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='자연어로 PINN 학습 설정과 코드 후보를 생성합니다.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--request', help='한국어 학습 변경 요청')
    source.add_argument('--resume', type=Path, help='검토한 코드 후보의 실행 폴더')
    parser.add_argument('--config', type=Path, default=root/'Pitch_PINN/config/tesla_ev6.yaml')
    parser.add_argument('--model', help='클라우드 모델 ID; PINN_LLM_MODEL보다 우선')
    parser.add_argument('--code', action='store_true', help='허용된 PINN 소스 문맥과 코드 후보 생성 사용')
    parser.add_argument('--train', action='store_true', help='설정 또는 검증된 후보로 로컬 학습')
    parser.add_argument('--run-candidate', action='store_true', help='후보 Python 코드를 로컬 권한으로 테스트 실행')
    args = parser.parse_args(argv)
    result = run_request(AssistantOptions(config_path=args.config, request=args.request or '', model=args.model, train=args.train, code=args.code, run_candidate=args.run_candidate, resume_dir=args.resume),project_root=root)
    print(f'state={result.state}\nrun_dir={result.run_dir}')
    status = result.run_dir/'status.json'
    if result.exit_code:
        print(f'요청 설명 또는 오류: {status}')
    return result.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
