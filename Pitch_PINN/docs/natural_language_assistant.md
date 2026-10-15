# 자연어 PINN 학습 도우미

한국어 요청을 클라우드 LLM으로 해석하고 PINN 설정 또는 코드 후보를 생성하는 로컬 CLI입니다. 실제 학습은 로컬 Python/PyTorch에서 실행합니다. 첫 버전에는 CarMaker 실제 실행이나 목표 응답 자동 최적화가 포함되지 않습니다.

## 설치

PowerShell에서 프로젝트 루트로 이동합니다. 아래 .venv는 이미 설치된 수치 계산 패키지를 재사용하며 새 API 의존성은 프로젝트 안에 설치합니다.

```powershell
cd E:\PINN
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r Pitch_PINN\requirements.txt -r Pitch_PINN\requirements-assistant.txt
.\.venv\Scripts\python.exe -m Pitch_PINN.assistant --help
```

검증 환경: Python 3.14.5, OpenAI SDK 3.31.0, Pydantic 2.14.1. GitHub에서 코드만 받은 경우 YAML이 지정하는 EV6 MAT 파일은 별도로 같은 경로에 준비해야 합니다. 데이터와 모델 파일은 Git에 포함되지 않습니다.

## API 키

키는 현재 PowerShell 프로세스의 환경변수로 설정합니다. 다음 방식은 실제 키를 명령 이력에 적지 않습니다. 키는 클라우드 API 계정에서 발급받아 사용하며 이 문서나 채팅, Git에 저장하지 않습니다.

```powershell
$pinnApiSecret = Read-Host "OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $pinnApiSecret).Password
Remove-Variable pinnApiSecret
```

새 PowerShell 창에는 다시 설정해야 합니다. 도우미는 .env 파일을 자동 로드하지 않습니다. 기본 모델은 `gpt-6.1-sol`입니다. `--model` > `PINN_LLM_MODEL` > 기본값 순으로 선택하며 접근 불가능한 모델을 자동 대체하지 않습니다.

## 설정 변경 미리보기와 학습

```powershell
.\.venv\Scripts\python.exe -m Pitch_PINN.assistant --request "관측점은 10개마다 하나씩 쓰고 20 epoch 학습하도록 설정해줘"
.\.venv\Scripts\python.exe -m Pitch_PINN.assistant --request "관측점은 10개마다 하나씩 쓰고 20 epoch 학습하도록 설정해줘" --train
```

첫 명령은 설정과 변경 차이만 저장합니다. 두 번째 명령은 새 API 요청으로 해석한 설정을 검증한 뒤 학습합니다. 두 호출의 해석 결과가 같다고 보장하지 않으므로 각각의 plan.json/config_diff.json을 확인하세요.

조정할 수 있는 항목은 epoch, learning rate, collocation 수, 로그 주기, seed, 관측 stride와 전처리 개수, 신경망 폭/층수, 기존 손실 가중치, 기존 K/C 초기값·범위입니다. 데이터 파일·출력 경로·CarMaker 명령은 자연어로 변경하지 않습니다. 현재 학습 진입점에 연결되지 않은 use_lbfgs 옵션은 지원하지 않습니다.

사용자 지정 YAML의 data.mat_path가 상대 경로이면 프로젝트 루트를 기준으로 절대 경로로 해석해 저장합니다. 후보 폴더에서 실행해도 같은 원본 데이터를 사용합니다.

"pitch 물리 잔차 가중치를 현재의 3배로"는 현재 가중치가 2라면 최종값 6을 제안합니다. 자연어 해석의 의미적 정확성은 모델 응답 검토가 필요하며 스키마 검증만으로 증명되지 않습니다.

## 코드 후보 생성·검토·실행

```powershell
.\.venv\Scripts\python.exe -m Pitch_PINN.assistant --request "앞뒤 강성은 고정하고 감쇠만 학습하도록 코드를 수정해줘" --code
```

출력된 run_dir에서 plan.json과 changes.diff, candidate/의 실제 파일을 확인합니다. 허용 범위는 기존 models/, physics/, training/와 train_inverse_pinn.py입니다. 테스트와 도우미 자체는 모델 변경 대상이 아닙니다.

검토한 후보를 API 재호출 없이 테스트하려면 출력된 **실제 실행 폴더**를 지정합니다.

```powershell
.\.venv\Scripts\python.exe -m Pitch_PINN.assistant --resume "E:\PINN\Pitch_PINN\assistant_runs\실제_실행_ID" --run-candidate
.\.venv\Scripts\python.exe -m Pitch_PINN.assistant --resume "E:\PINN\Pitch_PINN\assistant_runs\실제_실행_ID" --run-candidate --train
```

재개 시에도 새 실행 폴더로 복사하므로 이전 결과를 덮어쓰지 않습니다. --train만으로 생성 코드를 실행하지 않습니다. --run-candidate는 테스트가 import하는 후보 Python 코드를 로컬 사용자 권한으로 실행하도록 요청하는 옵션입니다. 후보 복사본은 OS 보안 샌드박스가 아니며 코드가 임의 파일/네트워크에 접근하는 것을 차단하지 않습니다. 자식 프로세스의 OPENAI_API_KEY는 제거합니다. 테스트 통과는 물리적 타당성의 증명이 아닙니다.

## 출력과 상태

실행별 위치: `Pitch_PINN/assistant_runs/<UTC시각_UUID>/`.

| 파일 | 의미 |
|---|---|
| request.txt / plan.json | 요청과 구조화된 해석 |
| config_diff.json / resolved_config.yaml | 변경 전후 값과 최종 설정 |
| status.json | 단계·상태·소요 시간·모델 ID·오류 코드 |
| changes.diff / source_hashes.json / candidate/ | 코드 모드 후보와 원본 해시 |
| tests.log / training.log | 실행한 검증·학습 로그 |
| training/ | 기존 PINN checkpoint·metrics·CSV·그래프 |

planned는 설정 미리보기, candidate_ready는 실행 전 코드 후보, tests_passed는 후보 테스트 통과, trained는 학습 완료입니다. 종료 코드 0을 사용합니다. needs_clarification/unsupported는 종료 코드 2, failed는 종료 코드 1입니다.

실패하면 status.json의 stage/error_code를 확인합니다. missing_key는 환경변수 누락, authentication은 인증 실패, model_unavailable은 모델 접근 문제, incomplete/refused/invalid_plan은 처리할 수 없는 응답, process_timeout은 실행 시간 초과입니다. validation_or_io_error는 입력/후보 경로·구문·파일 접근 문제이며 candidate_tests_failed는 tests.log에서 원인을 확인합니다. 임의 예외 원문은 키 노출을 막기 위해 저장하지 않습니다.

API에는 요청과 설정, --code일 때 허용된 Python 소스가 전달됩니다. MAT 데이터와 체크포인트를 요청에 넣지 않습니다. 원본 설정/소스와 기존 outputs는 보존하며 생성 코드의 원본 반영은 자동으로 하지 않습니다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m pytest Pitch_PINN\tests -q
```

기본 테스트는 API 모의 응답으로 실행하므로 API 키와 네트워크가 필요하지 않습니다. 실제 API 동작은 별도의 요청으로 확인해야 합니다. 소수 epoch smoke 학습은 실행 연결을 확인하는 것이며 식별 정확도를 입증하지 않습니다.
