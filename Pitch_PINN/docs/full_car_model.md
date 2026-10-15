# EV6 Full-Car Pitch PINN

기본 데이터는 `E:/PINN/PINN_file/src_cm4sl/Data/3.ACC/0.1g_EV6.mat`이다.

## 신호 역할

- 신경망 입력: `Sim_Time`, `Ax`, `VelZ_FL_WC`, `VelZ_FR_WC`, `VelZ_RL_WC`, `VelZ_RR_WC`
- 보조 관측값: `DampLength_*`, `DampVel_*`
- 희소 정답 및 평가값: `PITCH_RATE`
- 입력에서 제외: `Pitch`, `PITCH_RATE`

신경망은 sprung-body의 heave, roll, pitch를 출력한다. 시간 미분으로 각 속도와 가속도를 계산하고, 네 코너의 상대 변위·속도와 서스펜션 힘을 구성한다. 학습 결과는 유효 wheel rate인 `k_front_npm`, `k_rear_npm`, `c_front_nspm`, `c_rear_nspm`이다.

## 실행

```powershell
python Pitch_PINN/train_inverse_pinn.py --config Pitch_PINN/config/tesla_ev6.yaml
python Pitch_PINN/run_iteration.py --config Pitch_PINN/config/tesla_ev6.yaml
```

`run_iteration.py`는 기본적으로 CarMaker dry-run이며 `parameter_update.json`만 생성한다. 실제 실행은 YAML의 `dry_run: false`와 CLI의 `--allow-carmaker`가 모두 설정되어야 한다.

## 제한사항

현재 결과는 EV6 0.1 g 단일 궤적의 유효 선형 계수를 식별한다. Tesla Model Y 파라미터 식별 결과가 아니며 실제 부품 단위의 비선형 damper map을 유일하게 결정하지도 않는다.
