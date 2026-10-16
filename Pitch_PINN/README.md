# EV6 Full-Car Pitch PINN

EV6 0.1 g ACC 데이터의 `Ax`와 네 휠 `VelZ_*_WC`를 입력으로 사용하고, 희소 `PITCH_RATE`와 full-car 물리식을 결합해 EV6 pitch rate를 복원하는 inverse PINN입니다. `Pitch`와 `PITCH_RATE`는 입력 feature에 포함되지 않습니다.

## 실행

```powershell
cd E:\PINN
python -m pip install -r Pitch_PINN\requirements.txt
python Pitch_PINN\train_inverse_pinn.py --config Pitch_PINN\config\tesla_ev6.yaml
python Pitch_PINN\run_iteration.py --config Pitch_PINN\config\tesla_ev6.yaml
```

두 번째 명령은 기본적으로 CarMaker dry-run입니다.

## 출력

- `reference_comparison.png`: EV6 정답, 희소 관측점, full-car PINN 예측
- `reference.csv`: 전체 pitch-rate 예측 및 관측 마스크
- `physics_residuals.csv`: heave, roll, pitch 물리 잔차
- `metrics.json`: 전체/관측/숨긴 점 오차 및 학습된 K/C
- `loss_history.csv`, `model.pt`, `resolved_config.yaml`
- 반복 실행 시 `parameter_update.json`: `Kf`, `Kr`, `Cf`, `Cr`

## 테스트

```powershell
python -m pytest Pitch_PINN\tests -v
```

상세 신호 역할, 물리식 및 제한사항은 [docs/full_car_model.md](docs/full_car_model.md)를 참고하세요.

## 자연어로 학습 제어

Codex의 ChatGPT 로그인을 사용하는 설정 변경·코드 후보 생성·로컬 학습 CLI의 설치 및 사용법은 [자연어 PINN 학습 도우미](docs/natural_language_assistant.md)를 참고하세요. 기본 방식에는 API 키가 필요하지 않으며 Codex 로그인과 `requirements-assistant.txt`의 의존성이 필요합니다.

자연어 설정 변경은 기본 YAML에 백업 후 바로 반영되므로 기존 학습 명령이 새 값을 읽습니다. 파일을 바꾸지 않는 미리보기는 `--preview`를 사용하세요.

학습 데이터도 `PINN_file/src_cm4sl/Data/` 아래 같은 차량의 MAT 파일 중에서 자연어로 선택할 수 있습니다. 예: `--request "학습 데이터를 EV6 0.2g 가속 ACC 데이터로 변경해줘"`. 필수 신호와 전처리를 검증한 후 경로·시나리오 이름을 갱신합니다. 0.2g EV6에는 가속과 감속 파일이 있으므로 조건을 명시하세요.

현재 0.2g EV6 MAT 파일에는 필수 `AccZ_*_WC` 신호가 없어 `dataset_invalid`로 차단됩니다. 데이터 경로 변경 기능과 해당 파일의 학습 입력 호환성은 구별해야 하며 상세 내용은 위 사용 안내에 기록했습니다.

```powershell
.\.venv\Scripts\python.exe -m Pitch_PINN.assistant --request "학습 epoch를 30으로 변경해줘"
python Pitch_PINN\train_inverse_pinn.py --config Pitch_PINN\config\tesla_ev6.yaml
```
