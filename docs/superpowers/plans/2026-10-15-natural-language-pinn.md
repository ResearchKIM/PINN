# Natural Language PINN Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 한국어 학습 요구를 클라우드 LLM으로 해석하여 PINN 설정 또는 코드 후보를 생성하고 로컬 검증·학습 결과를 실행별로 저장한다.

**Architecture:** OpenAI Responses API의 구조화 계획을 로컬 검증기가 처리한다. 설정 변경은 기존 학습 함수에 연결하며 코드 변경은 원본과 분리된 후보 트리에서 검증한다. 모델 선택·실행 플래그·실패 단계·산출물을 기록하고 임의 셸 명령은 모델로부터 받지 않는다.

**Tech Stack:** 기존 Python/PyTorch/SciPy/PyYAML/pytest, OpenAI Python SDK, Pydantic v2. Windows PowerShell, 로컬 Git.

**Spec:** `docs/superpowers/specs/2026-10-15-natural-language-pinn-design.md` (2026-10-15 사용자 승인).

## Global Constraints

- 첫 버전 인터페이스는 PowerShell에서 실행하는 Python CLI이다.
- 응답 종류: config_change, code_change, clarification, unsupported.
- 기본 호출 모델은 gpt-6.1-sol이며 --model 또는 PINN_LLM_MODEL로 변경한다. 자동 모델 대체 없음.
- OPENAI_API_KEY는 환경변수로만 받는다. 비밀 값이 포함될 수 있는 원문 오류는 그대로 저장하지 않는다.
- 실행별 출력은 Pitch_PINN/assistant_runs/<unique-id>/에 저장한다. 기존 outputs와 원본 설정은 덮어쓰지 않는다.
- 첫 버전 변경 후보는 Pitch_PINN/models/, physics/, training/ 및 train_inverse_pinn.py의 기존 Python 소스로 제한한다.
- 원시 MAT 데이터와 체크포인트는 전송하지 않는다. 테스트는 모델 변경 대상에서 제외한다.
- 코드 후보 실행은 별도 --run-candidate 플래그가 있어야 한다. 복사본은 OS 보안 샌드박스가 아니다.
- 후보 테스트가 통과하고 --train도 지정되면 후보 학습 진입점을 별도 프로세스로 실행한다.
- CarMaker 실제 실행, 새 순방향 full-car 모델, 목표 응답 최적화, 웹 UI, LLM 추가 학습은 첫 버전 범위 밖이다.
- 기존 제품 코드를 리팩터링하지 않는다. use_lbfgs 옵션은 현 CLI에 연결되지 않았으므로 첫 버전 설정 허용 목록에서 제외한다.
- 구현은 main이 아닌 기능 브랜치에서 진행하며 사용자 변경을 보존한다. 구현 실행 시 using-git-worktrees 절차에 따라 작업 위치를 확인한다.

## Review Focus

1. Windows 대소문자·역슬래시·드라이브·UNC·junction 경로로 변경 허용 범위를 벗어나거나 같은 파일을 중복 지정할 수 없어야 한다 (Task 3).
2. YAML 숫자 변환으로 bool/실수 epoch/NaN이 통과하거나 계수 bounds가 변경 순서에 따라 잘못 검증되지 않아야 한다 (Task 1).
3. API 거절·응답 잘림·모델 접근 오류·예외에 포함된 비밀이 실행 또는 로그 유출로 이어지지 않아야 한다 (Task 2/4).
4. 후보 pytest가 원본 모듈을 import하여 거짓 통과하지 않아야 하며 실패·시간 초과 뒤 학습이 시작되면 안 된다 (Task 3/4).
5. API 없는 모의 검증을 실제 클라우드 성공으로 보고하지 않고, 두 실행이 산출물을 공유하거나 원본 데이터를 수정하지 않아야 한다 (Task 4/5).

## 파일과 인터페이스

새 패키지: `Pitch_PINN/assistant/{__init__,__main__,schemas,configuration,llm,candidates,runner}.py`.
테스트: `Pitch_PINN/tests/test_assistant_{configuration,llm,candidates,runner}.py`.
문서/의존성: `Pitch_PINN/requirements-assistant.txt`, `Pitch_PINN/docs/natural_language_assistant.md`, 기존 README의 안내 링크.

공통 자료형은 schemas.py에 정의한다. 모든 Pydantic 모델은 extra='forbid'이며 union 숫자 타입은 강제 형변환 없이 보존한다.

- `ConfigChange(path: str, value: StrictInt | StrictFloat)`; value는 최종 절대값이다.
- `FileChange(path: str, content: str)`; 기존 UTF-8 Python 파일 전체 교체 내용이다.
- `AssistantPlan(kind: Literal[...], summary: str, explanation: str, questions: list[str], config_changes: list[ConfigChange], file_changes: list[FileChange])`.
- `PlanResult(plan: AssistantPlan, requested_model: str, returned_model: str, response_id: str)`.
- `AssistantOptions(config_path: Path, request: str, model: str | None, train: bool, code: bool, run_candidate: bool, resume_dir: Path | None)`는 CLI와 runner의 내부 dataclass이다.
- `RunResult(run_dir: Path, state: str, exit_code: int)`; 상태와 종료 코드 의미는 Task 4에서 고정한다.

### Task 1: 구조화 계획과 설정 검증

**Files:** Create `assistant/__init__.py`, `assistant/schemas.py`, `assistant/configuration.py`, `requirements-assistant.txt`, `tests/test_assistant_configuration.py` (위 경로는 Pitch_PINN/ 기준).

**Interfaces:** `apply_config_changes(base: dict, changes: list[ConfigChange]) -> tuple[dict, list[dict]]`; 원본을 수정하지 않고 변경된 설정과 path/before/after diff 반환. `validate_plan(plan: AssistantPlan) -> None`.

- [x] **1. 실패 테스트 작성:** `test_config_copy_preserves_original`은 epoch=20/stride=10 반영과 원본 불변을 검사한다. `test_relative_request_uses_absolute_plan_value`는 physics_pitch=2.0→6.0 diff를 검사한다. `test_plan_kind_payload_consistency`는 config 종류에 파일 변경, code 종류에 빈 파일, clarification/unsupported에 변경 내용이 있으면 거부하는지 검사한다.
- [x] **2. 경계 테스트 작성:** `test_reject_unknown_duplicate_and_nonfinite_changes`는 data.mat_path/project.output_dir/carmaker.command/중복/NaN/Infinity/bool/실수 epoch를 거부한다. `test_validate_parameter_bounds_after_all_changes`는 min/max/initial을 모두 반영한 후 유효성 판단한다. collocation>=2, epoch>=0, stride/width/layers/log_every>=1, seed 및 전처리 개수>=0, lr>0, loss>=0이고 최소 하나의 loss>0을 검사한다.
- [x] **3. RED 확인:** `python -m pytest Pitch_PINN/tests/test_assistant_configuration.py -q` 실행, 새 모듈 부재 실패 확인.
- [x] **4. 최소 구현:** 공통 모델과 deep-copy allowlist 적용기를 구현한다. 계수 initial/min/max 변경 후 네 계수 전체 bounds를 검증한다. base 설정에 키·필요 섹션이 없으면 명확히 실패한다. code 계획은 설정 변경을 함께 가질 수 있다. clarification은 질문을 최소 하나 요구한다.
- [x] **5. 의존성 준비:** 별도 `requirements-assistant.txt`에 openai와 pydantic>=2,<3를 명시한다. 기존 requirements를 바꾸지 않는다. 구현 환경에서 설치 후 실제 동작한 버전을 확인·기록하며 임의의 미검증 버전을 호환 보장하지 않는다.
- [x] **6. GREEN 및 커밋:** 위 테스트 통과 후 해당 파일만 `git add`하고 `feat: validate natural language PINN plans` 커밋.

### Task 2: 클라우드 요청과 제한된 코드 문맥

**Files:** Create `assistant/llm.py`, `tests/test_assistant_llm.py`.

**Interfaces:** `collect_source_context(project_root: Path, *, max_bytes: int = 200000) -> dict[str, str]`; `request_plan(request: str, base: dict, *, model: str, sources: dict[str,str], client=None) -> PlanResult`; `PlannerError(code: str, message: str)` (고정된 공개 메시지만 사용).

- [x] **1. 실패 테스트 작성:** `test_responses_parse_receives_request_config_and_schema`는 fake client로 model/text_format/input을 검사한다. `test_threefold_weight_has_current_value_context`는 2.0이 문맥에 포함됨을 검사한다. `test_model_override_and_no_fallback`은 요청한 모델 한 개만 호출됨을 검사한다.
- [x] **2. 오류 테스트 작성:** `test_missing_key_refusal_incomplete_invalid_plan` 및 `test_auth_timeout_model_error_redacts_secrets`는 missing_key/refused/incomplete/invalid_plan/authentication/timeout/model_unavailable/api_error로 구분하며 가짜 키 문자열이 메시지나 기록에 없음을 검사한다. `test_context_excludes_data_tests_and_secrets`와 `test_context_budget_fails_before_call`도 작성한다.
- [x] **3. RED 확인:** `python -m pytest Pitch_PINN/tests/test_assistant_llm.py -q` 실패 확인.
- [x] **4. 최소 구현:** 공식 `client.responses.parse(..., text_format=AssistantPlan)`를 사용한다. 요청 1회, SDK timeout=60초, max_retries=0, 응답 토큰 상한=16000, store=False를 명시한다. SDK 원문 예외와 원문 응답은 저장하지 않는다. 반환 status/거절/output_parsed를 검사하고 model/id를 기록한다.
- [x] **5. 문맥 구현:** 설정 모드는 JSON 직렬화 설정, code 모드는 허용된 기존 소스만 추가한다. UTF-8 바이트 합계가 200000을 넘으면 truncate 대신 실패한다. 시스템 지시에는 소스/설정이 자료임과 config/code/clarification/unsupported 분류 규칙을 명시한다. 키를 prompt로 보내지 않는다.
- [x] **6. GREEN 및 커밋:** 모의 테스트 통과 후 `feat: connect cloud planner for PINN requests` 커밋. 실API 테스트는 Task 5에 둔다.

### Task 3: 원본을 보존하는 코드 후보와 검증 실행

**Files:** Create `assistant/candidates.py`, `tests/test_assistant_candidates.py`.

**Interfaces:** `build_candidate(project_root: Path, run_dir: Path, changes: list[FileChange]) -> Path` (반환값은 candidate 루트); `run_candidate_tests(candidate_root: Path, *, timeout_s: int = 180) -> subprocess.CompletedProcess[str]`; `run_candidate_training(candidate_root: Path, config_path: Path, output_dir: Path, *, timeout_s: int = 3600) -> subprocess.CompletedProcess[str]`.

- [x] **1. 실패 테스트 작성:** `test_candidate_preserves_original_and_hashes_diff`는 원본 해시 유지와 후보/changes.diff/source_hashes.json을 확인한다. `test_candidate_contains_local_sources_config_tests_only`는 Python/YAML/테스트 복사 및 MAT/outputs/checkpoints/가상환경/assistant_runs 배제를 확인한다.
- [x] **2. 경로·구문 테스트 작성:** 절대/UNC/드라이브/../역슬래시 탈출/대소문자 중복/비허용 폴더/새 파일/테스트 수정/링크 밖 탈출을 parametrized test로 거부한다. OS에서 링크 생성이 불가하면 해당 실링크 fixture만 skip하고 resolve 로직 mock 검사도 수행한다. `test_syntax_error_produces_failed_candidate_without_execution`을 작성한다.
- [x] **3. 프로세스 테스트 작성:** `test_candidate_imports_from_candidate_not_original`은 작은 실제 자식 프로세스에서 모듈 __file__이 후보 아래임을 검증한다. `test_child_environment_removes_key_and_pythonpath`는 OPENAI_API_KEY/PYTHONPATH/PYTHONHOME 제거를 검사한다. `test_test_failure_and_timeout_stop_before_training`은 제어 흐름 실패를 검사한다.
- [x] **4. RED 확인:** `python -m pytest Pitch_PINN/tests/test_assistant_candidates.py -q` 실패 확인.
- [x] **5. 최소 구현:** symlink/junction 없는 허용 소스만 명시적 복사하고 모든 변경 경로/내용/구문을 검증한다. 경로는 실제 resolve된 기존 파일 기준으로 정규화·중복 검사한다. 후보 트리는 데이터 없이 구성한다. 구문 오류 후보는 보존하되 실행 불가 상태로 처리한다.
- [x] **6. 실행 구현:** sys.executable과 고정된 argv list, shell=False, cwd=candidate_root를 사용한다. 테스트는 `-m pytest Pitch_PINN/tests -q --import-mode=importlib`, 학습은 `-m Pitch_PINN.train_inverse_pinn --config <absolute-resolved-config> --output-dir <run-training-dir>`로 실행한다. 후보 로그는 성공/실패/시간 초과 모두 기록하고 키 문자열을 redact한다. 코드 실행에 OS sandbox 격리가 없음을 명시한다.
- [x] **7. GREEN 및 커밋:** 후보 검증 테스트 통과 후 `feat: stage and verify isolated PINN code candidates` 커밋.

### Task 4: CLI·실행 이력·로컬 학습 통합

**Files:** Create `assistant/runner.py`, `assistant/__main__.py`, `tests/test_assistant_runner.py`.

**Interfaces:** `run_request(options: AssistantOptions, *, project_root: Path, planner=request_plan) -> RunResult`; `main(argv: list[str] | None = None) -> int`.

- [x] **1. CLI 실패 테스트 작성:** `test_cli_preview_does_not_train`, `test_train_uses_resolved_copy_and_unique_output`, `test_code_requires_flag_and_execution_requires_run_candidate`, `test_clarification_and_unsupported_never_execute`를 작성한다. fake planner를 사용하고 학습 호출을 spy로 검증한다.
- [x] **2. 이력 테스트 작성:** `test_failure_status_written_without_secret`, `test_two_runs_have_distinct_directories`, `test_tests_failed_or_timed_out_prevents_training`, `test_original_tree_and_outputs_unchanged`를 작성한다. resume은 아래 절차를 검증한다.
- [x] **3. RED 확인:** `python -m pytest Pitch_PINN/tests/test_assistant_runner.py -q` 실패 확인.
- [x] **4. CLI 구현:** --request/--resume는 mutually exclusive 필수 입력, --config 기본값은 프로젝트 YAML, --model > PINN_LLM_MODEL > gpt-6.1-sol, --code/--train/--run-candidate를 구현한다. --run-candidate는 code 계획에서만 유효하고 --train만으로 후보 코드는 실행하지 않는다. --resume는 기존 실행 폴더의 코드 후보/계획을 재호출 없이 재검증·실행하여 검토 후 실행할 수 있게 한다. resume 경로는 assistant_runs 아래 기존 디렉터리만 허용한다.
- [x] **5. 오케스트레이션 구현:** UTC timestamp+UUID로 mkdir(exist_ok=False)한 실행 폴더를 만들고 request.txt/status.json을 먼저 쓴다. key 사전 검사 후 단일 LLM 호출→계획 검증→설정 적용→선택적 후보 생성→플래그에 따른 검증/학습 순서로 실행한다. 상태 파일은 tmp+replace로 기록하고 JSON allow_nan=False를 사용한다.
- [x] **6. 상태 구현:** planned/candidate_ready/tests_passed/trained는 exit=0, needs_clarification/unsupported는 exit=2, failed는 exit=1. status에는 stage, state, requested/returned model, response ID, 공개 오류 코드와 산출물 경로를 담는다. 대화 clarification/unsupported에는 plan/status만 보존하고 설정/코드를 적용하지 않는다. 무관측 hidden RMSE=NaN은 요약에서 null로 표현한다.
- [x] **7. 실행 통합:** 설정 학습은 기존 run_training_from_config에 명시적인 새 output_dir를 전달한다. code 학습은 Task 3 자식 프로세스만 사용한다. stdout/stderr와 elapsed time을 별도 로그에 저장한다. 원본 출력 경로 및 CarMaker runner는 호출하지 않는다.
- [x] **8. GREEN 및 커밋:** 네 assistant 테스트 파일 통과 후 `feat: run natural language PINN workflows from CLI` 커밋.

### Task 5: 실제 학습 smoke·사용 안내·최종 검증

**Files:** Create `Pitch_PINN/docs/natural_language_assistant.md`; Modify `Pitch_PINN/README.md` (assistant 안내 링크만 추가); Update implementation checklist and project memory.

- [x] **1. 문서 작성:** 별도 의존성 설치, API 키 환경변수 입력(실제 키는 문서/셸 이력에 넣지 않는 PowerShell 보안 입력 예), 모델 선택, 기본 preview와 --train, --code 생성 후 --resume로 후보 검토/실행, 출력 해석/오류 대응을 문서화한다. LLM 코드가 로컬 사용자 권한으로 실행됨, 후보 트리가 보안 sandbox가 아님, API/데이터 지역 구분을 설명한다.
- [x] **2. 전체 회귀:** `python -m pytest Pitch_PINN/tests -q`에서 기존 및 신규 테스트 통과 확인. 실API 테스트는 기본 suite에서 네트워크 호출하지 않는다.
- [x] **3. 실제 EV6 학습 smoke:** fake planner가 epochs=2, collocation_points=32, hidden_width=16, hidden_layers=2, sparse_stride=10을 반환하도록 주입하여 run_request(...train=True)를 실행한다. 새 run 폴더의 checkpoint/metrics/reference CSV/plot/config/status와 유한한 주요 지표를 확인한다. 작은 epoch 결과를 모델 정확도 입증으로 표현하지 않는다.
- [x] **4. 실API 조건부 검증:** OPENAI_API_KEY 존재 여부만 확인한다. 있으면 한국어 요청 1건을 먼저 preview로 호출하고 실제 returned model/id와 epoch/stride 해석을 확인한다. 없으면 나머지 로컬 검증을 완료하고 실API 단계만 미검증으로 기록한다. 인증 실패/계정 모델 접근 불가도 그대로 기록하며 다른 모델로 대체하지 않는다.
- [x] **5. 원본 보존·완료 검토:** 데이터/원본 YAML/기존 outputs 해시와 Git diff를 확인하고 `git diff --check`를 실행한다. 새 파일의 미완성 표식과 테스트·문서 일치 여부를 검토한다. 구현 중 새 실패가 없으면 전체 회귀를 불필요하게 반복하지 않는다.
- [x] **6. 최종 커밋과 보고:** 문서·메모리·계획 체크리스트를 갱신하고 `docs: explain and verify natural language PINN assistant` 커밋. 확인된 테스트/실API 여부/사용 명령을 보고하며 GitHub push·PR은 해당 시점 승인 범위와 작업 맥락을 확인한다.

## 계획 자체 검토

- 설정과 코드 변경 두 경로, 모호/비지원 요청, 오류 기록, 원본 보존, 실제 학습, 실API 제한이 각각 위 Task에 연결됨.
- 인터페이스 명칭과 Task 간 소비/생산 자료형을 일치시킴. Task 4의 --resume는 검토한 후보를 API 재호출 없이 실행하기 위한 CLI 세부 결정이다.
- 실제 API 키/SDK 설치는 아직 수행하지 않음. 현 Python 3.14.5 환경에서 의존성 호환성은 구현 단계에서 확인한다.
- 본 계획 승인 후 실행 방법을 선택한다. 권장: 직접 실행(Native). 공유 자료형과 순차 검증 의존성이 크므로 한 구현자가 이어서 작업하는 편이 적절하다.

## 2026-10-15 실행 결과

- 사용자 승인으로 직접 실행. Task 1–4 구현 및 테스트, Task 5 문서/회귀/실제 데이터 smoke 완료.
- API 키 존재 여부 확인: 미설정. Task 5 실API 호출 부분은 미검증으로 남긴다. API 계정과 기본 모델 접근성은 실제 요청 후 확인해야 한다.
- 프로젝트 전용 .venv에 OpenAI SDK 3.31.0/Pydantic 2.14.1 설치. 기존 수치 계산 환경 재사용. pip check 통과.
- 독립 리뷰의 Important 2건을 실패 테스트로 재현 후 수정: 상대 MAT 경로 정규화, 후보 전체 트리 검증 및 검증된 파일만 재개 복사. pytest 설정 고정과 package import origin 검사 추가.
- 후보 복사본 전체 테스트: 111 passed, 1 skipped (심볼릭 링크 권한). 실제 EV6 2 epoch 설정 학습과 강성 고정 코드 후보의 테스트→학습 성공. 원본 코드/데이터/기존 결과 보존 확인.
- 구현 세부 결정: 공통 paths.py/artifacts.py 추가, --resume마다 새 결과 폴더 생성, 기존 사용자 기능 브랜치에서 작업. 후보는 OS 보안 sandbox가 아니다.
- 제품 코드 커밋: ddb1a03, 9caece2, 50091f3, 36e2ed8, 347ca28, 1ff4d37.
