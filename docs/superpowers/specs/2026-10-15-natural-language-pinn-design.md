# 클라우드 LLM 기반 자연어 PINN 학습 제어 설계

날짜: 2026-10-15 KST
상태: 2026-10-15 사용자가 설계 승인. 구현 계획 검토 전이며 제품 구현 전.

## 1. 목적과 성공 기준

사용자가 이 채팅을 계속 열어두지 않고 로컬 프로그램에 한국어로 학습 요구를 입력하면, 클라우드 LLM이 현재 PINN 설정과 필요한 코드 문맥을 읽고 변경안을 생성한다. 로컬 프로그램은 변경 내용을 검증하고 기존 PINN 학습을 실행하여 결과를 저장한다.

핵심 요구는 자연어로 학습 방식을 지정하는 것이며, 단순 차량 응답 최적화와 구분한다. 설정 변경뿐 아니라 PINN 코드 변경이 필요한 요청도 구분하고 코드 변경안을 산출해야 한다. 첫 버전 인터페이스는 PowerShell에서 실행하는 Python CLI이다.

## 2. 확인된 현재 환경

- 프로젝트: E:/PINN. 설계 초안 당시에는 Git 저장소가 아니었으나 이후 GitHub origin 연결을 완료했다. 현재 개발 브랜치는 feature/natural-laguage-pinn이다. 후보 복사본 보존 방식은 유지한다.
- 기존 진입점: Pitch_PINN.train_inverse_pinn.run_training_from_config(config, epochs=None, output_dir=None).
- 설정: Pitch_PINN/config/tesla_ev6.yaml. 실제 기본 차량은 EV6.
- 기존 파라미터: k_front_npm, k_rear_npm, c_front_nspm, c_rear_nspm.
- 현재 셸 환경에서 openai 및 pydantic 패키지 없음, OPENAI_API_KEY 없음을 값 노출 없이 확인했다.
- 현재 기본 풀카 CLI는 TrainingConfig에 YAML의 use_lbfgs/lbfgs_max_iter를 전달하지 않는다. 이 옵션을 지원한다고 표시하려면 실행 경로 연결과 검증을 먼저 추가해야 한다.
- 앞선 검토에서 기존 46개 테스트 통과를 확인했으며, 이번 설계 단계에서는 재실행하지 않았다.

## 3. 선택한 접근과 대안

선택: Python OpenAI SDK의 Responses API와 Pydantic 구조화 출력으로 요청을 계획하고, 로컬 실행기가 설정/파일 변경을 수행한다. 기존 Python 프로젝트를 직접 활용할 수 있고 클라우드에 실행 도구나 차량 데이터를 전달하지 않아도 된다.

대안: Codex SDK 기반 범용 코드 에이전트는 다단계 코드 수정에 유리하지만 별도 실행 도구와 권한 모델까지 함께 정해야 한다. 첫 버전에서는 제한된 파일 변경 제안으로 시작하고 이후 에이전트 통합 여부를 평가한다.

기본 호출 모델은 이전 제안인 gpt-6.1-sol로 하며 --model 또는 PINN_LLM_MODEL로 명시적으로 바꿀 수 있다. 사용 불가능한 모델을 다른 모델로 자동 대체하지 않는다. 실제 계정 접근성 및 품질은 첫 실호출에서 확인한다.

## 4. 사용자 흐름

예정 CLI 형식이며 아직 실행 가능한 명령이 아니다.

```powershell
python -m Pitch_PINN.assistant --request "관측점은 10개마다 하나씩 쓰고 20 epoch 학습해줘"
python -m Pitch_PINN.assistant --request "관측점은 10개마다 하나씩 쓰고 20 epoch 학습해줘" --train
python -m Pitch_PINN.assistant --request "앞뒤 강성은 고정하고 감쇠만 학습하도록 코드를 수정해줘" --code
```

- 기본 실행은 계획과 변경된 설정/코드 후보 및 차이를 저장한다.
- --train은 검증된 설정 변경을 기존 학습 함수에 전달한다.
- --code는 허용된 PINN 파일의 코드 변경안을 실행별 복사본에 만든다. 원본에 자동 덮어쓰지 않는다.
- 코드 후보 실행은 별도 --run-candidate 플래그가 있어야 한다. 후보 파일을 검토하고 로컬 Python 코드 실행을 요청하는 의미이며, 복사본은 OS 보안 샌드박스가 아니다.
- 모호하거나 현재 지원되지 않는 요청은 이유와 질문을 저장하고 학습을 시작하지 않는다.

## 5. 요청 분류와 설정 제어

응답 종류: config_change, code_change, clarification, unsupported.

구조화 계획에는 요청 요약, 설명, 명확화 질문, 설정 변경 목록, 코드 파일 변경 목록을 포함한다. 변경 종류와 내용이 모순되면 거부한다.

설정 변경 허용 항목:

- project.seed
- training.epochs, learning_rate, collocation_points, log_every
- training.weights의 기존 손실 항목
- data.sparse_stride, remove_bias_samples, baseline_samples, drop_last_samples
- model.hidden_width, hidden_layers
- model.parameters의 기존 네 계수 initial/min/max

값의 자료형, 유한성, 양수/음이 아닌 정수 조건, min < max와 initial 범위를 검사한다. bool을 정수로 허용하지 않는다. 모르는 설정 경로는 오류로 처리한다. 데이터 파일, CarMaker 실행 명령, 출력 경로는 LLM 설정 변경 대상으로 두지 않는다.

예: "pitch 물리 손실 가중치를 현재의 3배로"는 현재 physics_pitch=2.0을 문맥으로 전달해 6.0으로 제안하고 변경 차이를 기록한다. "강성을 고정"은 현재 설정만으로 구현되지 않으므로 코드 변경으로 분류한다.

## 6. 코드 변경 경로

문맥은 기존 설정과 허용된 모델/물리/학습 Python 소스 및 해당 인터페이스로 제한하고 바이트 예산을 둔다. 원시 MAT 데이터와 체크포인트는 전송하지 않는다.

첫 버전 변경 후보는 Pitch_PINN/models/, physics/, training/ 및 train_inverse_pinn.py의 기존 Python 소스로 제한한다. 상대 경로 탈출, 절대 경로, 중복 경로, 링크를 통한 범위 이탈 및 비허용 파일 변경을 거부한다.

변경안은 파일 경로와 전체 교체 내용으로 반환받고 실행별 candidate/Pitch_PINN 아래에 작성한다. 후보를 만들 때 원본 Python 소스/설정/테스트를 복사하되 outputs, 원시 데이터, 모델 체크포인트는 복사하지 않는다. 원본 파일 해시와 unified diff를 기록한다. LLM이 변경하는 테스트로 검증을 우회하지 않도록 기존 테스트는 변경 대상에서 제외한다.

실행 전 Python 구문을 검사한다. --run-candidate가 있으면 후보 루트에서 고정된 pytest 명령을 실행하며, LLM이 셸 명령을 지정하지 못한다. 자식 프로세스에는 OPENAI_API_KEY를 전달하지 않는다. 시간 제한과 로그를 기록한다. 테스트 통과는 물리적 타당성의 증명이 아니므로 상태를 tests_passed로만 표시한다.

후보 테스트가 통과하고 --train도 지정되면 후보 학습 진입점을 별도 프로세스로 실행한다. 관련 원시 데이터는 기존 설정의 로컬 경로로 접근한다. 실행은 한 번씩 직렬 수행하며 무제한 자동 수정 루프는 두지 않는다.

## 7. API·실행·저장

- OpenAI SDK Responses 구조화 출력으로 계획을 받는다. 모델 ID와 응답 ID를 기록한다.
- OPENAI_API_KEY는 환경변수로만 받는다. 누락/인증 오류/접근 불가/시간 초과/거절/불완전 응답을 구분하고 실행 실패로 반환한다. 비밀 값이 포함될 수 있는 원문 오류는 그대로 저장하지 않는다.
- API 입력은 자연어 요청, 설정, 코드 변경 모드일 때 허용된 소스이다. 문서/코드 안의 문자열을 사용자 지시로 취급하지 않는다.
- 설정 모드는 기준 설정의 복사본에 변경을 적용해 기존 학습 함수를 호출한다.
- 실행별 출력은 Pitch_PINN/assistant_runs/<unique-id>/에 저장한다. 기존 outputs와 원본 설정은 덮어쓰지 않는다.
- 산출물: request.txt, plan.json, config_diff.json, resolved_config.yaml, status.json. 코드 모드는 source_hashes.json, changes.diff, candidate/를 추가한다. 실행 시 테스트/학습 로그와 기존 학습 산출물을 추가한다.
- 실패 시에도 실패 단계와 조건을 status에 남긴다. 성공하지 않은 작업을 완료로 보고하지 않는다.
- CarMaker 실제 실행, 새 순방향 full-car 모델, 목표 응답 최적화, 웹 UI, LLM 추가 학습은 이 첫 버전의 구현 범위에 포함하지 않는다.

## 8. 검증

- API 모의 응답으로 정상 설정 변경, 상대 변경, 명확화, 비지원 요청 검증.
- 범위 초과, NaN/Infinity, 자료형 오류, 잘못된 경로/중복 변경, 범위 불일치 검증.
- API 키 누락·인증/모델 오류·불완전 응답에서 학습 및 후보 실행이 시작되지 않는지 확인.
- 후보 구문 오류/테스트 실패/시간 초과에서 학습이 시작되지 않는지 확인.
- API 키가 후보 실행 환경 및 저장 산출물에 포함되지 않는지 확인.
- 원본 설정·소스·기존 산출물 보존 및 실행별 결과 분리 확인.
- 기존 테스트 전체 실행 및 실제 EV6 데이터의 작은 epoch 설정 변경 smoke 학습.
- 키 준비 후 한국어 요청 한 건의 실제 API 호출과 반환 모델 ID·변경 결과 확인. 키가 없으면 API 실검증 미완료로 명시.

## 9. 예상 파일 구성

- Pitch_PINN/assistant/: __main__.py, schemas.py, llm.py, configuration.py, candidates.py, runner.py
- Pitch_PINN/tests/test_assistant_*.py
- Pitch_PINN/requirements-assistant.txt: 기존 의존성과 분리된 OpenAI SDK/Pydantic 의존성
- Pitch_PINN/docs/natural_language_assistant.md: 환경변수, 설치, 예제, 실행 범위, 실패 대응
- Pitch_PINN/README.md: 실행 안내 링크

## 10. 근거

- 현재 대화에서 사용자가 클라우드 LLM + 로컬 실행 방식을 선택하고 제작 요청.
- 현재 PINN 소스, YAML, requirements 및 환경 확인 결과.
- https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses
- https://developers.openai.com/api/docs/guides/code-generation

## 검토 후 다음 단계

이 문서 검토 후 구체 구현 계획을 작성하고 사용자 검토를 거쳐 구현한다. 설치·학습과 후보 실행은 해당 단계에서 기록한다. 현재 문서는 구현 또는 실호출 완료를 의미하지 않는다.
