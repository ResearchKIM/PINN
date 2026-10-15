# 로컬 Git 기준 버전 준비

- 사용자가 직접 git init -b main 실행 후 다음 작업을 요청.
- 현재 E:/PINN의 main 저장소는 커밋 전 상태이며 원격 저장소 없음 확인.
- 루트 .gitignore 생성: 인증정보/로컬 설정, Python 캐시/환경, MAT/모델 파일, outputs*, assistant_runs, tmp, PINN_file, 참고논문 제외.
- PINN_file 전체는 첫 기준 버전에서 로컬 보존. 향후 필요한 자체 작성 CarMaker 소스는 별도 검토 후 추적 가능.
- 기존 Pitch_PINN/.gitignore 및 project-memory/.gitignore 보존.
- git check-ignore로 로컬 설정, MAT 데이터, 학습 metrics, 기억 초기화 마커, tmp 제외 확인.
- 코드/문서/공유 기억을 선택적으로 스테이징. 제품 코드/데이터 수정, 삭제, 업로드 없음.
- git user.name / user.email 미설정. 사용자 작성자 정보를 받기 전 커밋하지 않음.
- 다음 단계: 실제 이름/이메일을 이 저장소에만 설정하고 스테이징 검토 후 첫 커밋. 정보 추측 금지.
