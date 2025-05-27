# Open CodeAI 🚀

**폐쇄망 환경을 위한 오픈소스 AI 코드 어시스턴트**

Continue.dev 플러그인과 완벽 호환되는 Function Calling 기반의 자율적인 AI 코딩 파트너입니다.

## 📋 프로젝트 개요

Open CodeAI는 Cursor AI의 핵심 기능을 폐쇄망 환경에서 완전히 재현하는 오픈소스 프로젝트입니다. 기존의 Continue.dev 플러그인을 재활용하여 빠른 개발과 안정성을 확보하면서, Function Calling 기반의 고급 AI 에이전트 기능을 제공합니다.

### 🎯 핵심 특징

- 🤖 **Function Calling 기반 AI 에이전트**: 능동적 코드 분석, 수정, 테스트 자동화
- 🔌 **Continue.dev 호환**: 기존 VS Code/JetBrains 플러그인 그대로 사용
- 🔒 **완전 폐쇄망**: 외부 인터넷 연결 불필요, 100% 로컬 동작
- 🌍 **OpenAI 호환 API**: 표준 API 스펙으로 다양한 클라이언트 지원
- 🧠 **지능형 RAG**: LlamaIndex + FAISS 기반 코드베이스 이해
- 🌳 **정확한 파싱**: Tree-sitter 다중 언어 코드 분석
- 📡 **실시간 모니터링**: 파일 변경 자동 감지 및 재인덱싱

### 🆚 기존 솔루션과의 차이점

| 기능 | Cursor AI | Continue.dev | Open CodeAI |
|------|-----------|--------------|-------------|
| 폐쇄망 지원 | ❌ | ❌ | ✅ |
| Function Calling | ✅ | ❌ | ✅ |
| 자율적 프로젝트 분석 | ✅ | ❌ | ✅ |
| 실시간 인덱싱 | ✅ | ❌ | ✅ |
| 완전 무료 | ❌ | ✅ | ✅ |
| IDE 통합 | ✅ | ✅ | ✅ |

## 🏗️ 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Continue.dev    │───►│ OpenAI 호환 API  │───►│ Function Calling│
│ Plugin          │    │ (FastAPI)       │    │ Agent           │
│ (VS Code/IDEA)  │    └─────────────────┘    └─────────────────┘
└─────────────────┘              │                       │
                                 ▼                       ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    Core AI Backend                                    │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┤
│   LLM Server    │   RAG System    │  File Watcher   │  Code Parser    │
│ (Qwen2.5-Coder) │ (LlamaIndex)    │  (Watchdog)     │ (Tree-sitter)   │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│                      Storage Layer                                    │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┤
│  Vector Store   │   Code Index    │   File Cache    │   Metadata      │
│    (FAISS)      │   (SQLite)      │   (Local FS)    │   (JSON)        │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

## 🛠️ 설치 및 설정

### 1. 시스템 요구사항

#### 최소 요구사항
- **Python**: 3.10+
- **메모리**: 16GB+ (권장: 32GB)
- **저장공간**: 50GB+
- **CPU**: 8코어+

#### 권장 요구사항
- **메모리**: 64GB+
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **저장공간**: NVMe SSD 100GB+

### 2. 설치 단계

#### Step 1: 저장소 클론
```bash
git clone https://github.com/ChangooLee/open-codeai.git
cd open-codeai
```

#### Step 2: 환경 설정
```bash
# Python 가상환경 생성
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt

# 설치 검증
python scripts/verify_installation.py
```

#### Step 3: 모델 다운로드
```bash
# LLM 모델 다운로드 (약 20GB)
python scripts/download_model.py

# 벡터 스토어 초기화
python scripts/setup_vector_store.py
```

#### Step 4: 서버 시작
```bash
# 환경변수 설정
cp .env.example .env
# .env 파일 수정 후

# API 서버 시작
python src/main.py
```

### 3. Continue.dev 플러그인 설정

#### VS Code에서 Continue.dev 설치
1. VS Code Extensions에서 "Continue" 검색 후 설치
2. 설정 파일 (`~/.continue/config.json`) 수정:

```json
{
  "models": [
    {
      "title": "Open CodeAI",
      "provider": "openai",
      "model": "open-codeai",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "open-codeai-local-key"
    }
  ]
}
```

#### JetBrains IDE에서 Continue.dev 설치
1. Plugins에서 "Continue" 검색 후 설치
2. 설정에서 동일한 API 엔드포인트 구성

## 🚀 사용법

### 기본 사용법 (Continue.dev 플러그인을 통해)

1. **코드 채팅**: 
   - `Ctrl+Shift+L` (VS Code) 또는 사이드바의 Continue 패널
   - 코드에 대해 자연어로 질문

2. **코드 자동완성**:
   - 코딩 중 자동으로 제안
   - `Tab`으로 수락

3. **코드 편집**:
   - 코드 선택 후 `Ctrl+Shift+M`
   - 편집 지시사항 입력

### 고급 기능 (Function Calling)

```python
# API 직접 호출로 고급 기능 사용
import requests

# 자율적 코드 리뷰 요청
response = requests.post("http://localhost:8000/api/agent/autonomous", {
    "task_type": "code_review",
    "project_path": "/path/to/project"
})

# 프로젝트 통계 조회
stats = requests.get("http://localhost:8000/api/project/stats/my-project")
```

## 📚 API 문서

서버 시작 후 다음 URL에서 API 문서 확인:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 주요 엔드포인트

#### OpenAI 호환 API
- `POST /v1/chat/completions` - 채팅 및 코드 생성
- `POST /v1/completions` - 자동완성
- `GET /v1/models` - 사용 가능한 모델 목록

#### 확장 API
- `POST /api/agent/chat` - AI 에이전트와 대화
- `POST /api/agent/autonomous` - 자율적 작업 수행
- `GET /api/project/stats/{project_path}` - 프로젝트 통계
- `POST /api/project/reindex/{project_path}` - 프로젝트 재인덱싱

## 🔧 개발 가이드

### 프로젝트 구조
```
open-codeai/
├── src/
│   ├── main.py                 # FastAPI 메인 서버
│   ├── config.py               # 설정 관리
│   ├── api/                    # API 라우터
│   │   ├── openai_compatible.py # OpenAI 호환 API
│   │   ├── agent.py            # AI 에이전트 API
│   │   └── project.py          # 프로젝트 관리 API
│   ├── core/                   # 핵심 로직
│   │   ├── agent.py            # Function Calling 에이전트
│   │   ├── llm_server.py       # LLM 서버
│   │   ├── rag_system.py       # RAG 시스템
│   │   ├── code_parser.py      # 코드 파서
│   │   └── file_watcher.py     # 파일 모니터링
│   └── utils/                  # 유틸리티
├── data/                       # 데이터 저장소
│   ├── models/                 # LLM 모델
│   ├── index/                  # 벡터 인덱스
│   └── logs/                   # 로그 파일
├── scripts/                    # 설치/관리 스크립트
├── tests/                      # 테스트 코드
└── requirements.txt            # Python 의존성
```

### 핵심 컴포넌트

#### 1. Function Calling Agent
```python
# AI 에이전트가 사용할 수 있는 도구들
tools = [
    "read_file",           # 파일 읽기
    "write_file",          # 파일 쓰기
    "list_files",          # 파일 목록
    "analyze_code_structure", # 코드 구조 분석
    "search_code",         # 코드 검색
    "run_tests",           # 테스트 실행
    "lint_code",           # 코드 린팅
    "get_git_diff"         # Git 변경사항
]
```

#### 2. RAG System
- **임베딩 모델**: BAAI/bge-large-en-v1.5
- **벡터 스토어**: FAISS (로컬)
- **청크 전략**: 코드 블록 기반 분할

#### 3. LLM Server
- **기본 모델**: Qwen2.5-Coder-32B
- **추론 엔진**: vLLM (고속 추론)
- **최적화**: GPU 메모리 효율적 사용

## 🧪 테스트

```bash
# 단위 테스트 실행
pytest tests/

# 통합 테스트 실행
pytest tests/integration/

# API 테스트 실행
pytest tests/api/

# 커버리지 리포트
pytest --cov=src tests/
```

## 📊 성능 최적화

### GPU 최적화
```bash
# GPU 메모리 사용률 조정
export GPU_MEMORY_UTILIZATION=0.8

# vLLM 최적화
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

### CPU 최적화
```bash
# PyTorch 스레드 수 조정
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

## 🚨 문제 해결

### 일반적인 문제들

#### 1. 모델 로딩 오류
```bash
# GPU 메모리 부족 시
export CUDA_VISIBLE_DEVICES=0
# .env 파일에서 GPU_MEMORY_UTILIZATION 값을 0.6으로 낮추기
```

#### 2. Continue.dev 연결 오류
```bash
# API 서버 상태 확인
curl http://localhost:8000/v1/models

# Continue.dev 설정 파일 위치
# ~/.continue/config.json (Linux/macOS)
# %USERPROFILE%\.continue\config.json (Windows)
```

#### 3. 성능 문제
```bash
# 인덱스 재구축
python scripts/reindex_project.py /path/to/project

# 로그 확인
tail -f data/logs/api_server.log
```

## 🤝 기여하기

### 개발 환경 설정
```bash
# 개발용 의존성 설치
pip install -r requirements-dev.txt

# pre-commit 훅 설치
pre-commit install

# 코드 포맷팅
black src/
isort src/

# 린팅
flake8 src/
```

### 기여 가이드라인
1. Fork 후 feature 브랜치 생성
2. 코드 변경 및 테스트 추가
3. Pre-commit 훅 통과 확인
4. Pull Request 생성

## 📄 라이선스

이 프로젝트는 **Apache 2.0 라이선스** 하에 배포됩니다.

- **상업적 사용**: ✅ 허용
- **수정 및 재배포**: ✅ 허용
- **특허 사용**: ✅ 허용
- **상표 사용**: ❌ 제한적

자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트들의 영감을 받았습니다:

- [Continue.dev](https://github.com/continuedev/continue) - IDE 통합 플러그인
- [LlamaIndex](https://github.com/run-llama/llama_index) - RAG 프레임워크
- [Tree-sitter](https://github.com/tree-sitter/tree-sitter) - 코드 파싱
- [FAISS](https://github.com/facebookresearch/faiss) - 벡터 검색
- [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) - 코딩 특화 LLM

## 📞 지원 및 커뮤니티

- **GitHub Issues**: 버그 리포트 및 기능 요청
- **GitHub Discussions**: 일반적인 질문 및 토론
- **Wiki**: 상세한 문서 및 튜토리얼

---

**Open CodeAI**로 폐쇄망에서도 최고 수준의 AI 코딩 경험을 시작하세요! 🚀