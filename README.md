한국어 | [English](README_en.md)

# Open CodeAI

![License](https://img.shields.io/github/license/ChangooLee/open-codeai)

Open CodeAI는 대형 프로젝트/폐쇄망 환경을 위한 오픈소스 AI 코드 어시스턴트입니다. 데이터 프라이버시와 보안을 유지하면서 안전하고 맥락적인 AI 코드 지원을 제공합니다.

이 프로젝트는 [MIT 라이선스](LICENSE)를 따릅니다.

---

[목차](#목차)

## 🖥️ OS별 설치/실행 가이드

### macOS

1. **필수 도구 설치**
   - [Homebrew](https://brew.sh/) 설치
   - Python 3.10+ 설치: `brew install python`
   - Docker Desktop 설치: [공식 다운로드](https://www.docker.com/products/d-desktop/)
2. **프로젝트 클론**
   - 아래 명령어로 본 저장소를 클론하면 필요한 offline_packages/ 폴더가 함께 받아집니다.
   ```bash
   git clone https://github.com/ChangooLee/open-codeai.git
   cd open-codeai
   ```
3. **설치 및 빌드(완전 자동화)**
   - **아래 한 줄로 오프라인 빌드/설치/컨테이너 준비가 모두 자동화됩니다.**
   ```bash
   cd open-codeai   # 반드시 프로젝트 루트로 이동!
   chmod +x scripts/install.sh
   ./scripts/install.sh --offline
   ```
   - **별도의 docker-compose build 명령이 필요 없습니다.**
   - 모든 Docker 이미지는 install.sh에서 자동으로 build --no-cache로 새로 빌드됩니다.
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
- **Qwen2.5-Coder 기반 LLM**: Cursor AI 수준의 코드 생성/보완 능력 (중앙 서버의 vLLM 엔드포인트 연동)
- **FAISS + Graph DB(Neo4j/NetworkX)**: 대규모 코드베이스 의미/관계 검색
- **Continue.dev 연동**: VSCode/JetBrains에서 바로 사용
- **설치/설정 자동화**: config.yaml → .env, 오프라인 패키지/모델 자동 인식
- **컨테이너리스/샤딩/양자화 등 다양한 모드 지원**
- **vLLM 엔진**: 로컬 설치 없이 중앙 서버(외부 엔드포인트)만 연동합니다. requirements.txt에서 vllm 패키지는 설치하지 않습니다.

---

## 🏁 빠른 시작 (오프라인/에어갭 환경)

### 1. 의존성/모델 준비
- `offline_packages/` 폴더에 Python wheel 파일(.whl) 사전 복사
- `data/models/` 폴더에 LLM/임베딩/그래프 모델 파일 사전 복사
- (옵션) `docker-images/` 폴더에 Docker 이미지 tar 파일 복사

### 2. 설치 및 빌드(완전 자동화)
```bash
# 1. 압축 해제 및 이동
$ tar -xzf open-codeai-*.tar.gz && cd open-codeai
# 2. 오프라인 설치 및 빌드 (한 줄로 자동화)
$ ./scripts/install.sh --offline
# 3. (모델/패키지 미리 복사 시 자동 인식)
```
- **별도의 docker-compose build 명령이 필요 없습니다. install.sh에서 자동으로 빌드됩니다.**

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

- `install.sh` : 오프라인/온라인 자동 설치, **Docker 이미지 자동 빌드/컨테이너 준비까지 한 번에 처리**
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
- **Q. vLLM 환경에서 인증 오류가 발생해요**
  - 중앙 서버의 vLLM 엔드포인트 주소와 API Key가 올바른지 `.env` 파일에서 확인
  - 반드시 Authorization 헤더(`Bearer ...`)로 인증해야 하며, body에 api_key를 넣지 않아야 함
  - configs/ 등 하위 폴더에 환경설정 파일이 남아있지 않은지 확인
- **Q. docker-compose build를 따로 해야 하나요?**
  - **아니요! install.sh만 실행하면 자동으로 build --no-cache까지 모두 처리됩니다.**

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

---

## ⚙️ 환경설정(.env) 및 DB 안내

- 프로젝트 루트에 `.env` 또는 `.env.example` 파일을 사용하세요.
- **Vector DB/Graph DB**는 별도 설치/설정 없이 내부적으로 자동 관리됩니다. 사용자는 포트(VECTOR_DB_PORT, GRAPH_DB_PORT)만 필요시 지정하면 됩니다.
- 데이터 경로 등은 모두 자동 관리되며, 별도 지정이 필요 없습니다.

### .env.example 예시
```env
# Open CodeAI .env 예시 파일
# 이 파일을 복사하여 .env로 사용하세요 (cp .env.example .env)

# === LLM/vLLM 엔진 ===
VLLM_ENDPOINT=http://localhost:8800/v1
VLLM_API_KEY=your-vllm-api-key
VLLM_MODEL_ID=Qwen2.5-Coder-32B-Instruct

# === 서버 설정 ===
HOST=0.0.0.0
PORT=8800
LOG_LEVEL=INFO

# === 프로젝트 정보 ===
PROJECT_NAME=open-codeai
VERSION=1.0.0
ENVIRONMENT=development
DEBUG=True

# === 벡터 DB (FAISS 등) ===
# 별도 설치/설정 불필요, 내부적으로 자동 관리됨
VECTOR_DB_PORT=9000  # (필요시 포트만 지정, 기본값 9000)

# === 그래프 DB (NetworkX/Neo4j) ===
# 별도 설치/설정 불필요, 내부적으로 자동 관리됨
GRAPH_DB_PORT=7687  # (필요시 포트만 지정, 기본값 7687)

# === 기타 ===
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
GPU_MEMORY_FRACTION=0.7
USE_MIXED_PRECISION=True
API_KEY=open-codeai-local-key
```

## 🧩 Continue.dev 확장과 연동하기 (VSCode/IntelliJ)

Open CodeAI는 [Continue.dev](https://continue.dev/) 확장(Extension)과 완벽하게 연동됩니다.

### 1. VSCode/IntelliJ에 Continue 확장 설치
- [VSCode 마켓플레이스](https://marketplace.visualstudio.com/items?itemName=Continue.continue)
- [JetBrains Plugin Marketplace](https://plugins.jetbrains.com/plugin/21086-continue)

### 2. Open CodeAI API 서버 주소 설정
- 확장 설정에서 `API_BASE_URL`을 아래와 같이 입력하세요:
  - `http://localhost:8800` (로컬에서 docker-compose로 띄운 경우)
- 인증 토큰 등 추가 설정은 `.env` 파일 및 내부 정책에 맞게 적용

### 3. 연동 확인
- IDE 내에서 Continue 패널을 열고, 정상적으로 Open CodeAI와 대화/코드 생성이 되는지 확인

---

> **참고:**
> 별도의 continue-adapter 컨테이너는 필요하지 않습니다. IDE 확장만 설치하면 됩니다.

## 주요 기능
- 코드베이스 인덱싱 및 검색
- LLM 기반 코드 분석/질문/자동화
- **코드 임베딩 모델 커스텀 지원**
- 인덱싱 상태 실시간 확인 및 후속 안내
- `README.md`, `README_en.md` 자동 인덱싱

---

## 코드 임베딩 모델 설정법

### 추천 모델
- **BAAI/bge-code-v1.5** (권장, 다양한 언어 지원, dimension=1024)
- microsoft/codebert-base
- Salesforce/codet5-base

### 설정 예시
- `.env` 파일에 아래와 같이 추가:
  ```
  EMBEDDING_MODEL_NAME=BAAI/bge-code-v1.5
  EMBEDDING_MODEL_DIM=1024
  ```
- 또는 `src/config.py`에서 직접 경로/이름 지정

---

## 인덱싱 및 상태 확인

- 인덱싱 요청: `/admin/index-project` (POST)
- 인덱싱 상태 확인: `/status` (GET)
  - 인덱싱 중: `"indexing": true`, 진행률/메시지/후속 안내 제공
  - 인덱싱 완료: `"indexing": false`, "이제 코드 검색/질문을 해보세요" 안내

---

## README 자동 인덱싱 안내
- 프로젝트 인덱싱 시 `README.md`, `README_en.md` 파일이 자동으로 포함됩니다.
- LLM이 프로젝트 설명/사용법을 이해할 수 있도록 **README 파일을 최신 상태로 유지**하세요.

---

## 예시 API 응답
```json
{
  "indexing": true,
  "indexing_progress": 120,
  "indexing_total": 500,
  "message": "인덱싱 중입니다... (120/500)",
  "next_actions": [
    {"action": "wait", "description": "인덱싱이 끝날 때까지 기다려주세요."}
  ]
}
```

---

## 문의/기여
- [GitHub Issues](https://github.com/your-repo)

## 🧩 임베딩 모델(microsoft/codebert-base) 다운로드 안내

- 기본 임베딩 모델로 **microsoft/codebert-base**를 사용합니다. 이 모델은 공개(Public) 모델로, 별도의 인증 없이 다운로드할 수 있습니다.
- 아래 절차에 따라 모델을 다운로드한 뒤, 반드시 `offline_packages/codebert-base` 폴더에 복사해 주세요.

### 다운로드 절차
1. 터미널에서 아래 명령 실행
   ```bash
   python3 -m pip install huggingface_hub
   python3 -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/codebert-base', local_dir='offline_packages/codebert-base', local_dir_use_symlinks=False)"
   ```
2. 다운로드가 완료되면, `offline_packages/codebert-base` 폴더가 정상적으로 생성되어야 합니다.

---