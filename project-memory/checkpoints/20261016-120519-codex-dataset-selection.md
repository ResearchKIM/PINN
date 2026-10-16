# 자연어 학습 데이터 선택 허용 범위 확장

- 사용자 요청: data.mat_path가 허용 범위 밖이라 거절되는 문제 해결. 앞서 제시한 목록 검색·선택·학습 입력 검증·YAML 반영 흐름을 승인하여 기존 브랜치 feature/natural-laguage-pinn에서 확장.
- 시작 시 사용자 YAML만 dirty: epochs637, pitch_data50, physics_heave10, physics_pitch100. 이번 작업 내내 보존하고 커밋에서 제외. 원시 MAT/outputs도 변경하지 않음.

## 구현

- assistant/datasets.py 추가: PINN_file/src_cm4sl/Data 아래 같은 차량명 MAT 목록을 검색. 링크/범위 밖 경로/타 차량 차단. 목록은 path/scenario_name/vehicle만 포함.
- codex_planner.py/llm.py: available_datasets를 두 backend 입력에 전달. ACC/DEC 모호성은 clarification, 경로 발명 금지, 자동 차량 변경 금지. MAT 신호 값은 전달하지 않음.
- schemas.py/configuration.py: 엄격한 문자열 설정 값 지원(data.mat_path/scenario_name만), 숫자 문자열은 기존 정수/숫자 검증에서 거부. 단일행 문자열/YAML 치환 유지.
- runner.py: datasets.json 저장, 제안 경로 목록 소속 확인, 기존 load_mat_scenario/preprocess_scenario로 검증 후 시나리오 이름 자동 추가. dataset_invalid/dataset_not_allowed 상태 처리. 원본 백업/preview/후보 및 절대 데이터 경로 snapshot 유지. 실패한 데이터 계획도 plan.json에 기록.
- README/사용 안내/설계 갱신. 새 test_assistant_datasets.py 및 backend 입력 테스트. 기존 physics_pitch 테스트는 fixture값2를 명시하여 사용자가 YAML 가중치를 바꿔도 독립적으로 실행.

## 검증

- TDD 신규 테스트7failed/5passed 확인 후 구현. 최종 전체 pytest147passed/1skip in16.29s (Windows 링크 생성 권한 skip). git diff --check 통과.
- 독립 리뷰 Important/Critical 없음. 리뷰에서 실제0.2g 신호 누락을 발견하고 scipy.io.whosmat로 재확인.
- 실제 Codex 로그인 호출, 모델요청gpt-6.1-sol/반환모델ID 미제공. 기본 원본 YAML이 아닌 임시 복사본으로 검증:
  - 20261016T030342Z_316acb08eb4f: 모호한0.2gEV6→needs_clarification, ACC/DEC 질문 확인.
  - 20261016T030420Z_8b94b2eb859a: 명확한0.2gACC→파일 선택 후 dataset_invalid, 원본 미변경.
  - 20261016T030430Z_1e8d9fa07a0f: 호환0.1gACC→config_applied, 시나리오 이름3.ACC/0.1g_EV6, 전처리8000샘플, 백업과 기존 load_config의 변경 값 읽기 확인. 다른 학습 설정 보존.
- 최초 smoke 스크립트 출력은 호스트stdout gbk로 UnicodeEncodeError 발생했지만 첫 실제 요청은 성공. 저장된 plan/status를 확인하고 이후 출력만UTF8로 설정. 모델 호출을 숨겨 재시도하지 않음.
- 원본 YAML 바이트 동일 확인, 실학습 미실행.

## 남은 조건

- 실제0.2g EV6 가속·감속 파일에는 AccZ_FL_WC/FR/RL/RR 네 신호가 없다. 따라서 이 파일은 기능 허용 이후에도 학습 입력 검증에서 차단된다. 기존0.1gACC에는 네 신호 존재.
- 해당 데이터를 실제 학습하려면 신호를 준비하거나 속도 미분 전처리의 타당성을 별도 설계·검증해야 한다. 임의 신호 생성/추정은 이번 변경에서 하지 않았다.
- 외부 데이터 디렉터리와 차량 물리 파라미터 자동 전환은 범위 밖. 새 같은 차량 파일을 프로젝트 Data 폴더에 넣으면 다음 요청 목록에 포함.
- 출처: 사용자 요청, 로컬 코드/데이터 메타데이터, pytest, 실제 Codex plan/status, 독립 리뷰.
