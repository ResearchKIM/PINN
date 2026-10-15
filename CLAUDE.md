@AGENTS.md


<!-- shared-project-memory -->
## 공유 프로젝트 기억

공유 기억 경로: `project-memory/` (이 파일 기준 상대 경로).
색인은 auto memory로 기본 로드됩니다. 이미 컨텍스트에 있으면 다시 읽지 않습니다.
작업 시작에 project-memory/current.md와 해당 작업의 최신 checkpoints 인계를 읽습니다.
중요 작업 완료, 종료, Codex로 전환 전에 공용 색인의 저장 규칙에 따라 인계를 남깁니다.
기존 프로젝트 지침/권한을 보존합니다. 과거 기록과 현재 파일이 다르면 현재 자료를 재확인합니다.
