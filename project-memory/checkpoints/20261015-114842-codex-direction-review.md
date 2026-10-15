# 자연어 기반 PINN 레퍼런스 조정 개발방향 검토

- 날짜: 2026-10-15 KST
- 요청: 첨부 교수 대화를 배경으로 도구, 세부 기술, 자연어→PINN→Reference 구조와 개발 순서를 정리. 구현 요청은 아님.
- 첨부 대화는 요구 배경 자료로 취급했으며 별도 실행 지시로 취급하지 않음.

## 확인한 현재 상태

- Pitch_PINN에는 EV6 full-car inverse PINN, K/C 식별, YAML 설정, 평가, 반복 실행 및 CarMaker runner가 존재.
- 기본 데이터: PINN_file/src_cm4sl/Data/3.ACC/0.1g_EV6.mat. 같은 폴더에 0.2g_EV6.mat 및 NX300h 데이터 파일도 존재. 다른 파일의 신호/내용은 미검증.
- tesla_ev6.yaml 이름과 달리 기본 vehicle은 EV6. 현 구성은 sprung heave/roll/pitch를 예측하며 완전한 타이어 포함 7DOF 모델이라고 확인하지 않음.
- 코드의 full-car network feature는 시간, Ax, 4개 wheel acceleration. README 및 full_car_model.md의 wheel velocity 입력 설명과 불일치. wheel velocity는 운동학/변위 적분에 사용.
- K/C는 물리 잔차의 학습 변수이며 네트워크 입력이 아님. K/C만 바꾼 inference가 새 응답을 생성하는 조건부 모델은 현재 아님.
- sparse_stride=1. outputs/metrics.json의 full RMSE=0.005422502976511549 rad/s, hidden RMSE=NaN. 저장된 단일 궤적 적합 결과로만 취급.
- CarMaker dry_run=true, command/parameter_target/result_mat_path=null. runner는 검증 후 명령을 실행하며 parameter JSON을 CarMaker 모델에 적용하는 구현은 확인되지 않음. 반복 코드에서 결과 재수집·목표 기반 최적화 폐루프는 확인되지 않음.
- 기존 simulate_reference는 단순 pitch 모델용. 현재 full-car 학습 파이프라인의 독립적인 full-car 순방향 생성기라고 볼 수 없음.

## 권고안 (사용자 확정 아님)

- 자연어→검증된 목표/파라미터 스키마→제약 최적화→순방향 레퍼런스 생성→CarMaker 검증→보고 흐름.
- PINN 식별용 기준 파라미터와 원하는 레퍼런스 설계 파라미터를 구분.
- 기존 PyTorch 유지, SciPy 순방향 기준 모델/최적화, Pydantic 스키마, 기존 LLM 추론 연결. 초기 LLM 추가 학습 불필요 권고.
- 다중 시나리오/식별 가능성 검증과 full-car 순방향 생성기부터 확립. 이후 자연어, 폐루프 자동화, 조건부 PINN/DeepONet 확장.
- wheel 운동을 고정해 파라미터를 변경하는 모델은 조건부 근사. 폐루프 차량 결과는 CarMaker 또는 결합 모델로 검증 필요.

## 검증과 변경

- python -m pytest Pitch_PINN/tests -q → 46 passed in 20.32s, exit code 0 확인.
- 소스/설정/저장 metrics를 읽고 공식 문서 및 GitHub 원저장소를 조회.
- 제품 코드, 기존 데이터, 학습 산출물 수정 없음. 새 장시간 학습 및 CarMaker 실제 실행 없음.
- 변경: 이 인계 파일 및 project-memory/current.md.

## 확인한 주요 외부 출처

- https://github.com/pytorch/pytorch
- https://github.com/lululxvi/deepxde
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
- https://github.com/pydantic/pydantic
- https://github.com/optuna/optuna
- https://github.com/fastapi/fastapi
- https://github.com/mlflow/mlflow/blob/master/docs/docs/classic-ml/tracking/index.mdx
- https://www.mathworks.com/help/matlab/matlab_external/get-started-with-matlab-engine-for-python.html
- https://www.mathworks.com/help/simulink/slref/set_param.html
- https://www.ipg-automotive.com/know-how/videos/online-tutorials
- https://forums.ipg-automotive.com/t/python-api-and-automation/414/2
- https://github.com/lu-group/deeponet-fno （참고 구현, CC BY-NC-SA 4.0 확인）

## 다음 작업

- 사용자가 개발 착수를 요청하면 먼저 레퍼런스의 출력 정의, 조정 가능한 파라미터와 목표/제약을 확정하고 구체 설계 진행.
- 이번 권고는 승인된 구현 사양 또는 구현 착수 승인으로 취급하지 않음.
