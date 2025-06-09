한국어 | [English](README_en.md)

# Open CodeAI

![License](https://img.shields.io/github/license/ChangooLee/open-codeai)

Open CodeAI는 대형 프로젝트/폐쇄망 환경을 위한 오픈소스 AI 코드 어시스턴트입니다. 데이터 프라이버시와 보안을 유지하면서 안전하고 맥락적인 AI 코드 지원을 제공합니다.

이 프로젝트는 [MIT 라이선스](LICENSE)를 따릅니다.

---

[목차](#목차)

## 🖥️ OS별 설치/실행 가이드

### 설치 옵션

Open CodeAI는 온라인과 오프라인 환경 모두에서 설치할 수 있습니다.

#### 가상환경 설정
설치 전에 Python 가상환경을 생성하고 활성화하는 것을 권장합니다:

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Windows (Command Prompt)
.\.venv\Scripts\activate.bat
# macOS/Linux
source .venv/bin/activate
```

##### Windows 사용자를 위한 추가 안내
1. **PowerShell 실행 정책 설정**
   - PowerShell에서 스크립트 실행이 차단될 경우, 관리자 권한으로 다음 명령어 실행:
   ```powershell
   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **Visual Studio Build Tools**
   - 일부 패키지 설치 시 Visual Studio Build Tools가 필요할 수 있습니다
   - [Visual Studio Build Tools](https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/) 설치
   - 설치 시 "C++ 빌드 도구" 워크로드 선택

3. **Python 경로 설정**
   - 시스템 환경 변수에 Python과 pip가 추가되어 있는지 확인
   - 명령 프롬프트에서 다음 명령어로 확인:
   ```cmd
   python --version
   pip --version
   ```

4. **가상환경 문제 해결**
   - 가상환경 활성화가 안 되는 경우:
     - PowerShell: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`
     - Command Prompt: 관리자 권한으로 실행
   - 경로에 한글이 포함된 경우 영문 경로로 이동 후 시도

#### 온라인 설치 (기본)
온라인 환경에서는 다음 명령어로 설치할 수 있습니다:
```bash
./scripts/install.sh
```

#### 오프라인 설치
오프라인 환경에서 설치하기 위해서는 먼저 필요한 패키지들을 다운로드해야 합니다:

1. 온라인 환경에서 패키지 다운로드:
```bash
./scripts/download_offline_packages.sh
```

2. 다운로드된 패키지와 함께 설치:
```bash
./scripts/install.sh --offline
```

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

### 설치 문제 해결

1. **환경 변수 설정**
   - 프로젝트 루트에 `.env` 파일이 필요합니다:
   ```env
   PROJECT_PATH=/path/to/your/workspace
   PROJECT_BASENAME=open-codeai
   ```
   - 또는 직접 환경 변수 설정:
   ```bash
   export PROJECT_PATH=/path/to/your/workspace
   export PROJECT_BASENAME=open-codeai
   ```

2. **권한 문제**
   - 스크립트 실행 권한 확인: `chmod +x *.sh`
   - Docker 그룹 권한 확인: `sudo usermod -aG docker $USER`

3. **Docker Desktop 상태**
   - Docker Desktop이 실행 중인지 확인
   - Docker 데몬 상태 확인: `docker info`

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

- **.env** 파일로 모든 설정 관리 (모델/DB/성능/모드 등)
- `scripts/generate_env.py`로 .env 자동 생성
- 주요 옵션 예시:
```env
# LLM 설정
LLM_MODEL_NAME=qwen2.5-coder-32b
LLM_MODEL_PATH=./data/models/qwen2.5-coder-32b
USE_VLLM=true
QUANTIZE=4bit
DEVICE=auto

# 데이터베이스 설정
GRAPH_DB_TYPE=networkx
GRAPH_DB_AUTO_SELECT=true
VECTOR_DB_SHARDING=true

# 성능 설정
GPU_ENABLE=true
MIXED_PRECISION=true
```

---

## 🧩 다양한 실행 모드/확장성

- **오프라인 설치**: offline_packages/, data/models/ 폴더만 있으면 인터넷 불필요
- **컨테이너리스**: .env에서 `GRAPH_DB_TYPE=networkx` 설정, Neo4j/Redis 미사용
- **최소 모드**: `--minimal` 플래그, 모니터링/Continue.dev 비활성화
- **양자화/샤딩**: `QUANTIZE=4bit`, `VECTOR_DB_SHARDING=true`
- **Neo4j 미사용**: `./install.sh --no-neo4j` 또는 .env에서 networkx 지정

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
  - `./install.sh --no-neo4j` 또는 .env에서 networkx 지정
- **Q. 모델/패키지 버전이 맞지 않아요**
  - config.yaml, requirements.txt, offline_packages/ 버전 일치 확인
- **Q. 인덱싱이 느려요/메모리 부족**
  - config.yaml에서 parallel_workers, memory_limit_gb, chunk_size 등 조정
- **Q. vLLM 환경에서 인증 오류가 발생해요**
  - 중앙 서버의 vLLM 엔드포인트 주소와 API Key가 올바른지 .env 파일에서 확인
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
- 인증 토큰 등 추가 설정은 .env 파일 및 내부 정책에 맞게 적용

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

## 프로젝트별 코드 임베딩/분석 (Docker)

Open CodeAI는 Docker 컨테이너로 실행되며, 분석할 프로젝트의 루트 디렉토리와 명령을 지정하여 실행할 수 있습니다.

### 사용법

```bash
# [프로젝트_경로] [명령] 형식 (명령: start/stop/restart/status/logs/help)
./scripts/start.sh /분석할/프로젝트/경로 start
./scripts/start.sh /분석할/프로젝트/경로
./scripts/start.sh /분석할/프로젝트/경로 restart  # 특정 프로젝트 경로로 재시작
./scripts/start.sh start  # 현재 디렉토리를 프로젝트로 사용
./scripts/start.sh restart  # 현재 디렉토리로 재시작
```
- 첫 번째 인자가 존재하는 디렉토리면 프로젝트 경로로 인식, 두 번째 인자가 명령입니다.
- 첫 번째 인자가 디렉토리가 아니면 명령으로 인식, 프로젝트 경로는 현재 디렉토리로 사용합니다.
- 명령이 없으면 기본값은 `start`입니다.
- `restart` 명령은 다음과 같이 동작합니다:
  1. 실행 중인 모든 서비스(API 서버, Neo4j 등)를 중지
  2. 깔끔한 종료를 위해 2초 대기
  3. 현재 설정으로 모든 서비스 재시작
  4. 개발 모드에서는 uvicorn --reload로 핫 리로딩 지원

- 지정한 경로가 Docker 컨테이너의 `/workspace`로 마운트됩니다.
- FastAPI 서버는 `/workspace` 내의 코드를 임베딩/분석합니다.
- 여러 프로젝트를 분석하려면 start.sh를 다른 경로로 반복 실행하면 됩니다.

### 예시

```bash
./scripts/start.sh ~/projects/my-awesome-project start
./scripts/start.sh ~/projects/my-awesome-project
./scripts/start.sh ~/projects/my-awesome-project restart  # 특정 프로젝트로 재시작
./scripts/start.sh restart  # 현재 디렉토리로 재시작
```

> **참고:**
> - 컨테이너 실행 중에는 마운트 경로를 변경할 수 없습니다. 프로젝트를 바꾸려면 컨테이너를 중지 후 다시 실행하세요.
> - 운영 환경에서는 보안을 위해 꼭 필요한 경로만 마운트하세요.
> - restart 명령은 설정 변경 후 재시작이 필요하거나 서비스가 응답하지 않을 때 유용합니다.

---

## 📝 고급 로깅 시스템

- **Loguru 기반 고급 로깅**: 모든 요청/응답/에러/성능이 자동 기록됩니다.
- 로그 파일: `logs/opencodeai.log` (일반), `logs/error.log` (에러), `logs/performance.log` (성능)
- 로그는 자동 분할/압축/보관(최대 30/90/7일)
- 로그 레벨, 경로, 보관 주기는 config에서 조정 가능

## 🗃️ Vector DB/Graph DB/임베딩 시스템

- **Vector DB (FAISS)**: 코드 임베딩은 FAISS(HNSW) 인덱스에 저장/검색되며, 인덱스 파일은 `data/vector_index/프로젝트명/`에 자동 저장됩니다.
- **Graph DB (Neo4j/NetworkX)**: 코드 구조(파일, 함수, 클래스, 의존성, 호출관계 등)는 그래프 DB에 저장됩니다. Neo4j(기본) 또는 NetworkX(컨테이너리스)가 자동 선택되며, 그래프 파일은 `data/graph_db/프로젝트명/`에 저장됩니다.
- **임베딩 서버/모델**: huggingface 기반 임베딩 모델(`microsoft/codebert-base` 등)을 사용하며, .env/config.yaml에서 임베딩 모델명, 차원, 경로를 지정할 수 있습니다. 임베딩 API(`/embedding`)를 통해 텍스트/코드 임베딩을 직접 생성할 수 있습니다.

## 🛠️ start.sh, install.sh 주요 기능

- `start.sh`: 프로젝트별 데이터 디렉토리, 벡터/그래프 DB 경로 자동 지정, 다양한 명령(`start`, `stop`, `status`, `logs`) 지원, 로그/상태 확인 기능 내장
- `install.sh`: 오프라인 패키지 설치, Docker 이미지 빌드, Neo4j/Redis 컨테이너 준비, .env 자동 생성, 데이터 디렉토리 자동 생성, GPU/CPU 환경 자동 감지 등 완전 자동화

---