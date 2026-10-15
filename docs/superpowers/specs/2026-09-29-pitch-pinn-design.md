# Tesla Model Y Pitch PINN 설계

- 작성일: 2026-09-29
- 프로젝트 경로: `E:\PINN\Pitch_PINN`
- CarMaker 프로젝트 경로: `E:\PINN\PINN_file`

## 1. 목표

핵심 데모 목표는 적은 수의 EV6 Pitch Rate 관측점과 피치 운동방정식을 함께 사용해, 전체 시간 구간의 EV6 reference Pitch Rate를 PINN으로 복원할 수 있음을 보이는 것이다. 동시에 Tesla Model Y의 기존 질량, 피치 관성, 축거, 무게중심 위치 같은 고정 제원을 유지하면서 서스펜션 강성 `K`, 감쇠 `C`를 식별하여 이후 CarMaker 반복 검증에 넘길 수 있게 한다.

최종 정답 데이터는 동일 시나리오에서 EV6로부터 직접 취득한 Pitch Rate이다. 기존 Simulink reference model은 Tesla Y에 입력되는 `Ax` 또는 노면 입력으로 추종 목표를 생성하는 용도로 유지하되, 직접 측정한 EV6 Pitch Rate와 먼저 검증해야 한다.

## 2. 1차 구현 범위

첫 구현은 다음 조건으로 제한한다.

- 시나리오: `ACC 0.1 g`
- 타겟 차량: Tesla Model Y
- 레퍼런스 차량: EV6
- 비교 출력: Pitch Rate 하나
- 학습 파라미터: 등가 피치 계수 `k_eff`, `c_eff`, `g_ax`; 정확한 Tesla 제원이 제공되면 `alpha_K`, `alpha_C`로 변환
- skyhook gain: 기본값 `0`; YAML에서 학습 활성화 가능
- 차량 제원: 질량, 피치 관성, 축거, 전후 무게중심 거리는 고정
- 검증: PINN 내부 예측과 CarMaker 반복 시뮬레이션 결과를 모두 EV6 Pitch Rate와 비교

`DEC`, `Bump`, NX300h 레퍼런스, skyhook gain 학습은 1차 파이프라인이 검증된 뒤 같은 인터페이스에 추가한다. 이 순서는 Pitch Rate 하나만으로 너무 많은 파라미터를 동시에 식별하여 해가 불안정해지는 것을 피하기 위한 것이다. 1차 결과에는 전체 EV6 신호 중 설정된 소수 관측점만 학습에 사용하고, 나머지 점을 검증용으로 숨긴 결과를 반드시 포함한다.

## 3. 디렉터리 구조

```text
E:\PINN\
├─ PINN_file\                  # 기존 CarMaker 프로젝트
├─ Pitch_PINN\                 # Python/PINN 프로젝트
│  ├─ config\
│  │  └─ tesla_ev6.yaml
│  ├─ data\
│  │  ├─ __init__.py
│  │  ├─ mat_loader.py
│  │  ├─ preprocess.py
│  │  └─ scenario.py
│  ├─ physics\
│  │  ├─ __init__.py
│  │  ├─ half_car.py
│  │  ├─ suspension.py
│  │  ├─ skyhook.py
│  │  └─ reference_model.py
│  ├─ models\
│  │  ├─ __init__.py
│  │  ├─ state_network.py
│  │  ├─ trainable_parameters.py
│  │  └─ pitch_pinn.py
│  ├─ training\
│  │  ├─ __init__.py
│  │  ├─ losses.py
│  │  ├─ trainer.py
│  │  └─ evaluate.py
│  ├─ carmaker\
│  │  ├─ __init__.py
│  │  ├─ parameter_mapper.py
│  │  ├─ runner.py
│  │  └─ result_loader.py
│  ├─ tests\
│  │  ├─ test_half_car.py
│  │  ├─ test_preprocess.py
│  │  └─ test_reference_model.py
│  ├─ outputs\                 # 실행 시 생성, 버전 관리 제외
│  ├─ requirements.txt
│  ├─ train_inverse_pinn.py
│  └─ run_iteration.py
└─ docs\superpowers\specs\
```

`Pitch_PINN`과 `PINN_file`을 형제 폴더로 두어 Python 코드와 CarMaker 프로젝트를 분리한다. CarMaker 설치 폴더의 원본 차량 파일은 수정하지 않는다. 파라미터 적용은 `PINN_file` 내부의 작업용 차량 데이터에만 수행한다.

## 4. 데이터 계약

모든 로더의 출력은 다음 공통 형식을 사용한다.

```python
ScenarioData(
    time_s,             # [N], strictly increasing
    ax_mps2,            # [N]
    pitch_rate_radps,   # [N]
    road_front_m=None,  # Bump에서 사용
    road_rear_m=None,
    name="acc_0p1g",
    vehicle="EV6",
)
```

전처리 단계는 다음을 보장한다.

1. 중복 time stamp 제거 및 시간 오름차순 정렬
2. 모든 신호를 공통 시간축으로 보간
3. Pitch Rate를 `rad/s`, `Ax`를 `m/s^2`로 통일
4. 이벤트 시작점을 기준으로 Tesla와 EV6 데이터 정렬
5. 정지 구간 bias 제거와 필요한 경우 저역통과 필터 적용
6. 학습 구간과 검증 구간을 시나리오 단위로 분리

MAT 파일의 실제 key 이름은 YAML에서 지정한다. 코드 안에 신호명을 고정하지 않는다.

## 5. 물리 모델

### 5.1 ACC/DEC 피치 모델

1차 구현은 종방향 가속도에 의한 피치 운동을 사용한다.

```text
theta_ddot
  = g_ax * Ax
  - c_eff * theta_dot
  - k_eff * theta
```

여기서 `g_ax`는 부호를 포함한 종가속도-피치 결합 계수이고, `k_eff`, `c_eff`는 관성으로 정규화된 등가 피치 강성·감쇠 계수이다. 네트워크가 예측한 `theta(t)`를 자동미분하여 `theta_dot`, `theta_ddot`을 계산하고 방정식 잔차를 PINN loss에 포함한다. 실제 Tesla 제원이 제공되면 등가 계수를 물리 K/C 배율로 후처리한다.

### 5.2 Bump 확장

Bump 단계에서는 전후 노면 입력 시간차와 sprung mass의 heave/pitch를 포함하는 half-car 모델로 교체한다. 1차 ACC 인터페이스와 데이터 형식을 유지하여 동일 학습기에서 사용할 수 있게 한다.

### 5.3 Skyhook

기본 설정에서는 gain을 0으로 고정한다. `skyhook.py`는 gain이 0일 때 힘 0을 반환하고, YAML에서 학습을 활성화하면 제한 범위 안의 gain을 사용하는 연속 skyhook moment 항을 제공한다. 이후 실제 CarMaker 제어기의 force law와 saturation이 확정되면 같은 인터페이스의 구현만 교체한다.

## 6. 학습 파라미터와 제약

정확한 Tesla 피치 관성과 기준 서스펜션 값이 없는 초기 데모에서는 양수와 범위를 보장하는 내부 변수에서 등가 피치 계수를 생성한다.

```text
theta_ddot + c_eff * theta_dot + k_eff * theta = g_ax * Ax
```

정확한 `I_y`, 전후 CG 거리 `a`, `b`, 기준 `K_f`, `K_r`, `C_f`, `C_r`가 제공되면 `K_theta = I_y*k_eff`, `C_theta = I_y*c_eff`로 계산하고, `K_theta0 = K_f*a^2 + K_r*b^2` 및 같은 형태의 감쇠식으로 `alpha_K`, `alpha_C`를 산출한다. 초기에는 전후 비율과 compression/rebound map 형상을 유지한다. 실제 제원이 없으면 배율을 출력하지 않으며 임의의 값을 만들지 않는다.

## 7. Loss 구성

```text
L_total = w_data * L_pitch_rate
        + w_phys * L_ode_residual
        + w_ic   * L_initial_condition
        + w_reg  * L_parameter_regularization
```

- `L_pitch_rate`: 전체 신호 중 YAML의 sparse observation 비율로 선택한 EV6 Pitch Rate 관측점과 예측값의 차이
- `L_ode_residual`: 피치 운동방정식 잔차
- `L_initial_condition`: 이벤트 시작 시 pitch angle/rate 조건
- `L_parameter_regularization`: Tesla 기준값에서 과도하게 멀어지는 해 억제

평가 지표는 RMSE, peak error, peak time error, 정상상태 bias를 기록한다. 학습 loss와 CarMaker 재시뮬레이션 지표를 구분해서 저장한다.

## 8. 모듈 책임

- `config/tesla_ev6.yaml`: 경로, MAT key, 차량 고정 제원, 기준 K/C, 학습 범위, loss weight, CarMaker 실행 명령
- `data/*`: MAT 로딩, 단위 통일, 정렬, scenario 객체 생성
- `physics/*`: 피치/half-car 잔차, 서스펜션 환산, skyhook 인터페이스, Simulink reference model과 같은 reference 계산
- `models/*`: 상태 네트워크, 제한된 학습 파라미터, 전체 PINN 조립
- `training/*`: loss, optimizer loop, checkpoint, metric/plot 출력
- `carmaker/parameter_mapper.py`: 학습값을 작업용 차량 파라미터로 변환하고 변경 diff 생성
- `carmaker/runner.py`: YAML에 설정된 배치 명령 실행; 기본값은 dry-run
- `carmaker/result_loader.py`: CarMaker 출력 데이터를 공통 `ScenarioData` 형식으로 변환
- `train_inverse_pinn.py`: 단일 PINN 학습 진입점
- `run_iteration.py`: 학습 → 파라미터 매핑 → CarMaker 실행 → 평가의 반복 진입점

## 9. 실행 흐름

```text
EV6 직접 계측 데이터 ─> 희소 관측/검증 분리 ┐
                                          ├─> 전처리/시간 정렬 ─> data loss
Tesla 입력 Ax ─> reference model ──┘                 │
                                                     v
Ax + 등가 피치 ODE ─────────> PINN 물리 잔차 ─> k_eff, c_eff, g_ax
                                                     │
                                                     v
작업용 CarMaker 차량 파라미터 생성 ─> 재시뮬레이션 ─> EV6와 평가
                                                     │
                                       기준 미달이면 다음 iteration
```

reference model 출력은 EV6 직접 데이터와의 오차가 허용 수준인지 먼저 확인한다. 검증되지 않은 reference model 출력을 최종 정답으로 사용하지 않는다.

## 10. 안전한 CarMaker 연동

- 설치 경로의 `Demo_Tesla_Y`는 읽기 전용 원본으로 취급한다.
- 실제 변경 대상은 `E:\PINN\PINN_file` 안의 명시된 작업용 차량 파일 하나로 제한한다.
- `parameter_mapper.py`는 적용 전 원본과 변경 예정 값의 diff를 출력한다.
- `runner.py`는 기본 dry-run이며 YAML에서 실행 명령과 대상 경로가 모두 명시된 경우에만 실행한다.
- 각 iteration의 입력 파라미터, CarMaker 결과, metric을 별도 폴더에 남겨 재현 가능하게 한다.

## 11. 검증

다음 자동 테스트를 포함한다.

- `test_preprocess.py`: 중복 시간 제거, 단위 변환, 보간, 이벤트 정렬
- `test_half_car.py`: 평형 상태 잔차와 알려진 합성 응답
- `test_reference_model.py`: `Ax` 부호에 따른 피치 방향, 초기조건, 샘플 시간 처리
- 학습 파라미터가 설정한 범위와 양수 조건을 지키는지 확인
- 합성 데이터에서 data/physics loss가 감소하는지 smoke test
- dry-run에서 CarMaker 원본이나 작업 파일이 변경되지 않는지 확인

## 12. 1차 완료 조건

1. 전체 Python 모듈이 import되고 자동 테스트가 통과한다.
2. EV6 `ACC 0.1 g` MAT 데이터를 읽고 전체 점 중 설정된 소수 관측점만 사용해 학습할 수 있다.
3. 학습 결과로 전체 시간축 reference Pitch Rate, `k_eff`, `c_eff`, `g_ax`, checkpoint, metric JSON, 비교 plot을 생성한다. Tesla 실제 제원이 모두 제공된 경우에만 `alpha_K`, `alpha_C`도 생성한다.
4. CarMaker parameter mapping은 실제 적용 전에 diff를 생성하고 dry-run 검증을 통과한다.
5. 학습에 사용하지 않은 EV6 Pitch Rate 점에 대한 PINN reference RMSE와 reference model 비교 결과가 저장된다.
6. 실제 CarMaker 배치 명령과 작업용 차량 파라미터 경로가 YAML에 입력되기 전에는 외부 파일을 변경하지 않는다.

## 13. 이후 확장 순서

1. ACC 여러 가속도 크기로 공통 `alpha_K`, `alpha_C` 검증
2. DEC 추가 및 ACC/DEC 공동 학습
3. Bump용 4-DOF half-car residual 추가
4. ACC/DEC/Bump 공동 loss로 식별성 강화
5. 실제 skyhook force law 확정 후 gain 학습 활성화
6. EV6/NX300h reference profile 선택 기능 추가
