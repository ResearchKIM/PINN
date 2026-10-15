# API 키 없는 Codex 로그인 방식으로 전환

- 날짜: 2026-10-15 KST
- 사용자 배경: 연구실 Business 계정을 사용하여 API 키를 사용하기 어려움. '로그인 방식으로 변경 될까?'로 기존 연결 교체 요청.
- 기존 기능 브랜치 feature/natural-laguage-pinn에서 작업. 시작 시 working tree clean.

## 변경

- 신규 assistant/codex_planner.py: 실행 파일 검색, ChatGPT 로그인 검사, API 키 환경변수 제거, 임시 작업 폴더에서 read-only/ephemeral/ignore-user-config codex exec 구조화 출력.
- llm.py: 공통 프롬프트, 기존 API 함수 분리, 기본 codex/선택적 api 라우팅. 자동 모델/인증 fallback 없음.
- schemas.py, runner.py, __main__.py: backend=codex 기본값과 --backend api 옵션, 상태 메타데이터 기록.
- test_assistant_codex.py 신규 12개 테스트. 기존 API 테스트는 API 함수를 명시하고 missing-key 테스트는 backend=api로 변경. 일반 suite는 실제 Codex 호출을 하지 않음.
- README, 사용 안내, 설계 상태, 공유 기억 갱신. 기존 설정 검증/코드 후보/학습 경로 유지.

## 실동작 확인

- 현재 PC Codex CLI 0.162.0-alpha.17.2 설치. codex login status: Logged in using ChatGPT.
- 실제 명령: .venv/Scripts/python.exe -m Pitch_PINN.assistant --request '학습 epoch를 20으로, 관측점 간격 sparse_stride를 10으로, pitch 물리 손실 가중치 physics_pitch를 6으로 설정해줘'
- exit0, state=planned, backend=codex, elapsed6.55s.
- 실행: Pitch_PINN/assistant_runs/20261015T110348Z_ab196a2baea8.
- config_diff: epochs5000→20, sparse_stride1→10, physics_pitch2.0→6.
- resolved_config.yaml을 Python YAML로 읽고 실제 세 값 일치 assert 통과. 원본 tesla_ev6.yaml Git diff 없음.
- API 키 없이 실제 클라우드 요청 성공. 요청 모델은 gpt-6.1-sol. CLI JSON 이벤트에 실제 모델 ID가 없어 returned_model=unreported_by_codex_cli로 기록 (모델 실행 메타데이터 확인으로 주장하지 않음).
- 이번 실제 요청은 설정 미리보기만 수행. --code 실LLM 출력/실제 Codex 계획의 학습까지는 이번 전환 테스트에서 수행하지 않음. 기존 모의 기반 code/train 통합 검증은 이전 인계 참고.

## 검증

- TDD: 새 모듈 부재 실패 후 구현. 새 backend 테스트12passed.
- 최종 전체: .venv/Scripts/python.exe -m pytest Pitch_PINN/tests -q →123passed,1skipped in27.25s.
- skip1은 이전과 동일한 Windows symlink 권한.
- git diff --check 통과.
- 독립 읽기전용 리뷰1회: Important/Critical 없음. 리뷰어의 실제 네트워크 재호출은 없었으며 부모의 실동작 근거와 코드/테스트를 검토.

## 다음과 제한

- 이제 기본 CLI에는 API 키가 필요하지 않음. 사용자 계정의 Codex 한도/조직 권한과 유효한 로그인이 필요.
- Windows 사용자 셸 PATH에 codex가 없어도 앱 bin의 실행 파일을 자동 탐색. 필요 시 PINN_CODEX_PATH에 .exe 전체 경로 지정.
- CLI 플래그 호환성은 현재 설치 버전에서 검증. 종료/잘린 응답/잘못된 JSON/비로그인/API키 로그인은 실패 처리.
- 현재 기본 동작은 원본 YAML을 덮어쓰지 않고 실행별 resolved_config.yaml 생성.
- GitHub의 기존 기능 브랜치에 커밋/push 후 동기화 상태 확인. main 병합/PR 없음.
- 공식 문서: https://learn.chatgpt.com/docs/non-interactive-mode 및 https://learn.chatgpt.com/docs/cli/reference 검색 후 열람.
