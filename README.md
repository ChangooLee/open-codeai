# Open CodeAI 🚀

**대형 프로젝트/폐쇄망 환경을 위한 오픈소스 AI 코드 어시스턴트**

---

## 🖥️ OS별 설치/실행 가이드

### macOS

1. **필수 도구 설치**
   - [Homebrew](https://brew.sh/) 설치
   - Python 3.10+ 설치: `brew install python`
   - Docker Desktop 설치: [공식 다운로드](https://www.docker.com/products/docker-desktop/)
2. **오프라인 패키지/모델 복사**
   - `offline_packages/`, `data/models/` 폴더 복사
3. **설치 및 실행**
   ```bash
   chmod +x install.sh
   ./install.sh --offline
   ./start.sh
   ./index.sh /Users/yourname/Workspace/yourproject
   ```
4. **권한/보안 이슈**
   - 실행 권한 부여: `chmod +x *.sh`
   - 터미널에서 실행 권장 (zsh, bash)
   - Docker Desktop 실행 상태 확인

### Windows

1. **필수 도구 설치**
   - [Python 3.10+](https://www.python.org/downloads/windows/) 설치 (Add to PATH 체크)
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치
2. **오프라인 패키지/모델 복사**
   - `offline_packages\`, `data\models\` 폴더 복사
3. **설치 및 실행**
   - PowerShell에서:
     ```powershell
     .\install.bat
     .\start.bat
     .\index.bat C:\Users\yourname\Workspace\yourproject
     ```
   - 또는 WSL(권장): Ubuntu 환경에서 Linux 설치법과 동일하게 실행
4. **경로/권한/한글 경로 주의**
   - 경로에 한글/공백이 없도록
   - 관리자 권한으로 실행 필요 시 PowerShell "Run as Administrator"
   - Docker Desktop 실행 상태 확인

### Linux (Ubuntu 등)

1. **필수 도구 설치**
   - Python 3.10+ 설치: `sudo apt install python3 python3-venv python3-pip`
   - Docker 설치: `curl -fsSL https://get.docker.com | sh`
2. **오프라인 패키지/모델 복사**
   - `offline_packages/`, `data/models/` 폴더 복사
3. **설치 및 실행**
   ```bash
   chmod +x install.sh
   ./install.sh --offline
   ./start.sh
   ./index.sh /home/yourname/yourproject
   ```
4. **권한/보안 이슈**
   - 실행 권한 부여: `chmod +x *.sh`
   - Docker 그룹 추가: `sudo usermod -aG docker $USER`
   - Docker Desktop/엔진 실행 상태 확인

### 공통 주의사항

- 오프라인 설치 시, 모든 패키지/모델/도커 이미지를 미리 복사해야 함
- Docker Desktop/엔진이 반드시 실행 중이어야 함
- 환경변수/경로 문제 발생 시, OS별 안내 메시지 참고
- 경로에 한글/공백/특수문자 사용 금지 권장

---

## ✨ 주요 특징
- **완전 오프라인/에어갭 설치**: 인터넷 없이도 모든 기능 사용 가능
- **Qwen2.5-Coder 기반 LLM**: Cursor AI 수준의 코드 생성/보완 능력
- **FAISS + Graph DB(Neo4j/NetworkX)**: 대규모 코드베이스 의미/관계 검색
- **Continue.dev 연동**: VSCode/JetBrains에서 바로 사용
- **설치/설정 자동화**: config.yaml → .env, 오프라인 패키지/모델 자동 인식
- **컨테이너리스/샤딩/양자화 등 다양한 모드 지원**

---

## 🏁 빠른 시작 (오프라인/에어갭 환경)

### 1. 의존성/모델 준비
- `offline_packages/` 폴더에 Python wheel 파일(.whl) 사전 복사
- `data/models/` 폴더에 LLM/임베딩/그래프 모델 파일 사전 복사
- (옵션) `docker-images/` 폴더에 Docker 이미지 tar 파일 복사

### 2. 설치 및 초기화
```bash
# 1. 압축 해제 및 이동
$ tar -xzf open-codeai-*.tar.gz && cd open-codeai
# 2. 오프라인 설치
$ ./install.sh --offline
# 3. (모델/패키지 미리 복사 시 자동 인식)
```

### 3. 서버 실행 및 인덱싱
```bash
# 서버 실행
$ ./start.sh
# 프로젝트 인덱싱 (최초 1회)
$ ./index.sh /path/to/your/project
```

### 4. VSCode에서 Continue 확장 설치 후 바로 사용

---

## ⚙️ 설정 자동화 및 주요 옵션

- **config.yaml** 하나로 모든 설정 관리 (모델/DB/성능/모드 등)
- `scripts/generate_env.py`로 .env 자동 생성 (config.yaml 변경 시 실행)
- 주요 옵션 예시:
```yaml
llm:
  main_model:
    name: "qwen2.5-coder-32b"
    path: "./data/models/qwen2.5-coder-32b"
    use_vllm: true
    quantize: "4bit"   # none, 4bit, 8bit
    device: "auto"
database:
  graph:
    type: "networkx"   # neo4j or networkx
    auto_select: true
  vector:
    sharding: true
performance:
  gpu:
    enable: true
    mixed_precision: true
```

---

## 🧩 다양한 실행 모드/확장성

- **오프라인 설치**: offline_packages/, data/models/ 폴더만 있으면 인터넷 불필요
- **컨테이너리스**: config.yaml에서 `database.graph.type: networkx` 설정, Neo4j/Redis 미사용
- **최소 모드**: `--minimal` 플래그, 모니터링/Continue.dev 비활성화
- **양자화/샤딩**: `llm.main_model.quantize: 4bit`, `database.vector.sharding: true`
- **Neo4j 미사용**: `./install.sh --no-neo4j` 또는 config에서 networkx 지정

---

## 🛠️ 주요 스크립트/자동화 도구

- `install.sh` : 오프라인/온라인 자동 설치, 모델/패키지/도커 이미지 자동 인식
- `scripts/generate_env.py` : config.yaml → .env 자동 변환
- `start.sh` : 서버/도커/가상환경 통합 실행
- `index.sh` : 프로젝트 인덱싱 자동화
- `scripts/verify_installation.py` : 설치/환경 검증

---

## 🧑‍💻 개발/확장 가이드

- **모델 교체/추가**: config.yaml에서 main_model/embedding_model 경로만 변경
- **DB/인덱스 구조 확장**: config.yaml에서 graph/vector 옵션 조정, 샤딩/컨테이너리스 지원
- **Continue.dev 커스텀 명령/프롬프트**: `~/.continue/config.json` 자동 생성, 직접 수정 가능
- **코드/설정 리팩토링**: src/, configs/, scripts/ 구조 참고

---

## ❓ FAQ & 문제해결

- **Q. 오프라인 설치가 안 돼요!**
  - offline_packages/, data/models/ 폴더에 파일이 있는지 확인
  - install.sh 실행 시 로그/에러 메시지 확인
- **Q. Neo4j 없이 실행하고 싶어요**
  - `./install.sh --no-neo4j` 또는 config.yaml에서 `database.graph.type: networkx` 지정
- **Q. 모델/패키지 버전이 맞지 않아요**
  - config.yaml, requirements.txt, offline_packages/ 버전 일치 확인
- **Q. 인덱싱이 느려요/메모리 부족**
  - config.yaml에서 parallel_workers, memory_limit_gb, chunk_size 등 조정

---

## 📸 스크린샷/아키텍처
(아키텍처 다이어그램, VSCode 연동, 인덱싱/검색 예시 등 추가)

---

## 📝 참고/기여
- [공식 문서/위키](https://github.com/ChangooLee/open-codeai/wiki)
- [이슈/기여 가이드](https://github.com/ChangooLee/open-codeai/CONTRIBUTING.md)
- [Continue.dev 공식](https://continue.dev/)

---

**Open CodeAI는 대형 엔터프라이즈/공공기관/폐쇄망 환경에서도 최고의 코드 AI 경험을 제공합니다!**