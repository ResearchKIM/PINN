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
