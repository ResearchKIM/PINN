# 클라우드 자연어 PINN 제작 요청 — 설계 단계

- 날짜: 2026-10-15 KST
- 사용자 요청: 클라우드 LLM API + 로컬 PINN 실행 방식으로 제작.
- 제품 구현 착수 전에 적용된 brainstorming 스킬의 written-spec 검토 절차를 진행.

## 변경

- docs/superpowers/specs/2026-10-15-natural-language-pinn-design.md 생성.
- project-memory/current.md에 이번 상태 추가.
- 제품 코드/데이터/학습 산출물 수정 및 의존성 설치 없음.

## 설계 요약

- Python CLI에서 한국어 요청을 Responses API로 구조화.
- 설정 변경은 검증된 복사본으로 기존 학습 함수에 전달.
- 코드 변경은 허용 소스를 실행별 후보 복사본으로 생성하고 diff/구문/기존 테스트로 확인. 실행은 명시적 플래그로 분리. 복사본을 OS sandbox라고 주장하지 않음.
- 키는 OPENAI_API_KEY 환경변수, 기본 모델 제안 gpt-6.1-sol. 실제 모델 접근성 미확인.
- 첫 버전은 CarMaker 실제 폐루프 및 웹 UI를 포함하지 않음.

## 확인과 제한

- 현재 git rev-parse 결과 Git 저장소 아님.
- 현재 Python 환경에서 openai, pydantic 설치 없음.
- 현재 프로세스의 OPENAI_API_KEY 미설정 (값은 읽어 출력하지 않음).
- 기존 run_training_from_config, BoundedSuspensionRates, YAML, requirements 및 README 읽음.
- 현 풀카 CLI에 use_lbfgs/lbfgs_max_iter 전달이 연결되지 않은 점 확인.
- 설계 파일 존재/내용 및 TODO/TBD/FIXME 부재 확인. 테스트 재실행/실API 호출 없음.
- 공식 structured outputs 및 code generation 문서 조회.

## 다음 단계

- 사용자에게 작성된 설계 검토 요청. 승인 후 writing-plans로 구현 계획 작성 및 검토 절차 준수.
- 새 기능 구현 완료 또는 API 실호출 성공으로 설명하지 말 것.
