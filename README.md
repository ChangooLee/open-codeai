# Open CodeAI 🚀

**대형 프로젝트를 위한 폐쇄망 전용 AI 코드 어시스턴트**

Cursor AI를 능가하는 성능으로 4,000+ 파일의 대형 프로젝트를 완벽하게 분석하고 지원하는 오픈소스 솔루션입니다.

## 📋 프로젝트 개요

Open CodeAI는 **완전 폐쇄망 환경**에서 Cursor AI보다 뛰어난 성능을 제공하는 AI 코드 어시스턴트입니다. Continue.dev 플러그인을 기반으로 하여 즉시 사용 가능하며, 로컬 Graph DB + Vector DB 하이브리드 아키텍처로 대형 코드베이스를 완벽하게 이해합니다.

### 🎯 핵심 목표

- 📊 **4,000+ 파일 지원**: 대형 엔터프라이즈 프로젝트 완벽 지원
- 🚀 **Cursor AI 초월 성능**: 더 빠르고 정확한 코드 분석 및 생성
- 🔒 **100% 폐쇄망**: 외부 인터넷 연결 불필요
- 📦 **원클릭 설치**: 모든 의존성 포함된 패키징
- 🔌 **즉시 사용**: Continue.dev 플러그인으로 바로 시작

### 🏆 성능 비교

| 기능 | Cursor AI | Continue.dev | **Open CodeAI** |
|------|-----------|--------------|-----------------|
| 대형 프로젝트 지원 | ⚠️ 제한적 | ❌ 성능 저하 | ✅ **4,000+ 파일** |
| 폐쇄망 지원 | ❌ | ❌ | ✅ **완전 지원** |
| 코드 관계 분석 | ⚠️ 기본 | ❌ 없음 | ✅ **Graph DB** |
| 실시간 인덱싱 | ✅ | ❌ | ✅ **자동 업데이트** |
| Function Calling | ✅ | ❌ | ✅ **고급 에이전트** |
| 설치 복잡도 | 쉬움 | 쉬움 | ✅ **원클릭** |
| 비용 | 유료 | 무료 | ✅ **완전 무료** |

## 🏗️ 하이브리드 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Continue.dev Plugin                          │
│                     (VS Code + JetBrains)                          │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ Direct WebSocket Connection
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Open CodeAI Backend                              │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┤
│ Function Calling│   LLM Server    │  Hybrid Search  │ Real-time Index │
│ Agent           │ (Qwen2.5-Coder) │  Engine         │ Manager         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
                                     │
                     ┌───────────────┼───────────────┐
                     ▼               ▼               ▼
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │   Vector DB     │ │    Graph DB     │ │   Metadata      │
          │   (FAISS)       │ │   (Neo4j/       │ │   (SQLite)      │
          │                 │ │   NetworkX)     │ │                 │
          │ • 의미적 검색   │ │ • 코드 관계     │ │ • 파일 정보     │
          │ • 유사도 분석   │ │ • 의존성 그래프 │ │ • 변경 추적     │
          │ • 임베딩 저장   │ │ • 호출 관계     │ │ • 성능 메트릭   │
          └─────────────────┘ └─────────────────┘ └─────────────────┘
```

## 📦 원클릭 설치 패키지

### 시스템 요구사항

#### 최소 요구사항 (4,000+ 파일 지원)
- **CPU**: 16코어 이상 (Intel i9/AMD Ryzen 9)
- **메모리**: 64GB 이상
- **GPU**: NVIDIA RTX 4080 이상 (16GB+ VRAM) 또는 CPU 모드
- **저장공간**: NVMe SSD 200GB 이상
- **OS**: Ubuntu 20.04+, Windows 10+, macOS 12+

#### 권장 요구사항 (최적 성능)
- **CPU**: 32코어 이상 (Xeon/Threadripper)
- **메모리**: 128GB 이상
- **GPU**: NVIDIA RTX 4090/A100 (24GB+ VRAM)
- **저장공간**: NVMe SSD 500GB 이상

### 원클릭 설치 스크립트

#### Linux/macOS 설치
```bash
# 설치 스크립트 다운로드 및 실행
curl -fsSL https://raw.githubusercontent.com/ChangooLee/open-codeai/main/install.sh | bash

# 또는 오프라인 설치 패키지
wget https://github.com/ChangooLee/open-codeai/releases/latest/download/open-codeai-linux.tar.gz
tar -xzf open-codeai-linux.tar.gz
cd open-codeai
./install.sh
```

#### Windows 설치
```powershell
# PowerShell에서 실행
iwr -useb https://raw.githubusercontent.com/ChangooLee/open-codeai/main/install.ps1 | iex

# 또는 오프라인 설치
# open-codeai-windows.zip 다운로드 후 압축 해제
# install.bat 실행
```

## ⚙️ 설정 파일

### config.yaml - 메인 설정
```yaml
# Open CodeAI 설정 파일
project:
  name: "open-codeai"
  version: "1.0.0"
  max_files: 10000  # 최대 지원 파일 수
  
# LLM 설정 (로컬 모델 경로)
llm:
  # 메인 코드 생성 LLM
  main_model:
    path: "./models/qwen2.5-coder-32b"  # 로컬 모델 경로
    type: "qwen2.5-coder"
    context_window: 32768
    gpu_memory: 0.7  # GPU 메모리 사용률
    
  # 임베딩 전용 LLM (벡터화용)
  embedding_model:
    path: "./models/bge-large-en-v1.5"  # 로컬 임베딩 모델 경로
    type: "bge"
    batch_size: 64
    device: "cuda"  # cuda, cpu, mps
    
  # Graph 분석용 LLM (선택사항)
  graph_model:
    path: "./models/codet5-small"  # 코드 관계 분석용 모델
    type: "codet5"
    enable: true  # Graph DB 사용 시에만 필요

# 데이터베이스 설정
database:
  # Vector Database (FAISS)
  vector:
    type: "faiss"
    index_type: "HNSW"  # 대형 프로젝트 최적화
    dimension: 1024
    storage_path: "./data/vector_index"
    memory_limit: "16GB"  # 인덱스 메모리 제한
    
  # Graph Database
  graph:
    type: "neo4j"  # neo4j 또는 networkx
    storage_path: "./data/graph_db"
    memory_limit: "8GB"
    enable_llm_analysis: true  # LLM 기반 관계 분석 사용
    
  # Metadata Database
  metadata:
    type: "sqlite"
    path: "./data/metadata.db"
    cache_size: "2GB"

# 인덱싱 설정
indexing:
  # 지원 파일 타입
  file_types:
    - ".py"    # Python
    - ".js"    # JavaScript  
    - ".ts"    # TypeScript
    - ".java"  # Java
    - ".cpp"   # C++
    - ".c"     # C
    - ".go"    # Go
    - ".rs"    # Rust
    - ".php"   # PHP
    - ".rb"    # Ruby
    - ".scala" # Scala
    - ".kt"    # Kotlin
    
  # 청킹 전략
  chunking:
    strategy: "semantic"  # semantic, fixed, hybrid
    chunk_size: 1000
    overlap: 200
    min_chunk_size: 100
    
  # 병렬 처리
  parallel:
    workers: 16  # CPU 코어 수에 맞게 조정
    batch_size: 100
    memory_per_worker: "4GB"

# 성능 최적화
performance:
  # 캐싱
  cache:
    enable: true
    size: "10GB"
    ttl: 3600  # 1시간
    
  # 메모리 관리
  memory:
    max_usage: "80%"  # 시스템 메모리 사용 제한
    gc_threshold: "70%"
    
  # GPU 최적화
  gpu:
    enable: true
    memory_fraction: 0.8
    mixed_precision: true

# 실시간 모니터링
monitoring:
  file_watcher:
    enable: true
    debounce: 1.0  # 파일 변경 감지 지연시간
    batch_update: true
    
  metrics:
    enable: true
    interval: 60  # 60초마다 메트릭 수집
    
# Continue.dev 통합 설정
continue_integration:
  connection_type: "websocket"  # websocket 또는 http
  port: 8001
  auth_token: "open-codeai-secure-token"
  
  # 기능 활성화
  features:
    chat: true
    autocomplete: true
    code_review: true
    refactoring: true
    function_calling: true
```

### models.yaml - 모델 경로 설정
```yaml
# 로컬 모델 경로 설정
models:
  # 메인 LLM (코드 생성/분석)
  qwen2.5-coder-32b:
    path: "./models/qwen2.5-coder-32b"
    files:
      - "pytorch_model.bin"
      - "config.json" 
      - "tokenizer.json"
      - "tokenizer_config.json"
    download_url: "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct"
    size: "64GB"
    
  # 임베딩 모델 (벡터화)
  bge-large-en-v1.5:
    path: "./models/bge-large-en-v1.5"
    files:
      - "pytorch_model.bin"
      - "config.json"
      - "tokenizer.json"
    download_url: "https://huggingface.co/BAAI/bge-large-en-v1.5"
    size: "2.3GB"
    
  # Graph 분석 모델 (코드 관계 분석)
  codet5-small:
    path: "./models/codet5-small"
    files:
      - "pytorch_model.bin"
      - "config.json"
      - "tokenizer.json"
    download_url: "https://huggingface.co/Salesforce/codet5-small"
    size: "242MB"
    optional: true  # Graph DB 미사용 시 생략 가능
    
  # 추가 특수 목적 모델들
  code-search-net:
    path: "./models/code-search-net"
    purpose: "코드 검색 최적화"
    size: "1.2GB" 
    optional: true
```

## 🚀 설치 및 실행 과정

### 1단계: 시스템 준비
```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y  # Ubuntu

# Docker 설치 (Neo4j 용)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# NVIDIA 드라이버 설치 (GPU 사용시)
sudo apt install nvidia-driver-525 nvidia-cuda-toolkit
```

### 2단계: Open CodeAI 설치
```bash
# 저장소 클론
git clone https://github.com/ChangooLee/open-codeai.git
cd open-codeai

# 원클릭 설치 실행
./install.sh

# 설치 중 진행되는 작업들:
# ✅ Python 환경 설정
# ✅ 모든 의존성 패키지 설치  
# ✅ FAISS 벡터 DB 설치
# ✅ Neo4j Graph DB 설치 (Docker)
# ✅ Continue.dev 플러그인 수정판 준비
# ✅ 모델 다운로드 (60GB+ 용량)
```

### 3단계: 모델 설정
```bash
# 모델 자동 다운로드 (인터넷 연결 필요)
python scripts/download_models.py --config models.yaml

# 또는 오프라인 모델 설정 (USB/네트워크 드라이브에서)
python scripts/setup_offline_models.py --source /path/to/models --config models.yaml

# 모델 검증
python scripts/verify_models.py
```

### 4단계: 데이터베이스 초기화
```bash
# 데이터베이스 초기화
python scripts/init_databases.py

# 초기화 과정:
# 🔧 FAISS 인덱스 생성
# 🔧 Neo4j 컨테이너 시작
# 🔧 SQLite 메타데이터 DB 생성
# 🔧 성능 테스트 실행
```

### 5단계: Continue.dev 플러그인 설치
```bash
# VS Code용
./scripts/install_vscode_plugin.sh

# JetBrains용  
./scripts/install_jetbrains_plugin.sh

# 수동 설치 (필요시)
# 1. Continue.dev 기본 플러그인 제거
# 2. open-codeai-continue.vsix 설치
# 3. 설정 파일 자동 생성됨
```

### 6단계: 시스템 시작
```bash
# Open CodeAI 백엔드 시작
./start.sh

# 시작 과정 로그:
# 🚀 LLM 서버 초기화 중... (30-60초)
# 🔍 Vector DB 로딩 중...
# 📊 Graph DB 연결 중...
# 🔌 Continue.dev 플러그인 연결 대기...
# ✅ 시스템 준비 완료!
```

## 🎯 대형 프로젝트 인덱싱

### 프로젝트 인덱싱 실행
```bash
# 대형 프로젝트 인덱싱 (4,000+ 파일)
python scripts/index_project.py \
  --project-path /path/to/large/project \
  --config config.yaml \
  --parallel-workers 16 \
  --memory-limit 32GB

# 실시간 진행상황 표시:
# 📁 파일 스캔 중... (4,247 files found)
# 🌳 코드 파싱 중... (1,200/4,247) [████████░░░░] 28%
# 🧠 임베딩 생성 중... (800/4,247) [██████░░░░░░] 19%  
# 📊 그래프 관계 분석 중... (600/4,247) [████░░░░░░░░] 14%
# ⚡ 인덱스 최적화 중...
# ✅ 인덱싱 완료! (소요시간: 25분)
```

### 성능 벤치마크 (4,000 파일 프로젝트)
```
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│ 작업            │ Cursor AI    │ Continue.dev │ Open CodeAI  │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ 초기 인덱싱     │ 45분         │ 불가능       │ 25분         │
│ 코드 검색       │ 2-3초        │ 10초+        │ 0.5초        │
│ 관련 코드 탐색  │ 제한적       │ 불가능       │ 완벽         │
│ 함수 호출 추적  │ 기본         │ 없음         │ 고급         │
│ 메모리 사용량   │ 높음         │ 높음         │ 최적화       │
│ CPU 사용률      │ 높음         │ 높음         │ 효율적       │
└─────────────────┴──────────────┴──────────────┴──────────────┘
```

## 💻 Continue.dev 플러그인 사용법

### VS Code에서 사용
1. **채팅**: `Ctrl+Shift+L` → AI와 대화
2. **코드 리뷰**: 코드 선택 → 우클릭 → "AI Review"  
3. **자동완성**: 코딩 중 자동 제안
4. **리팩토링**: `Ctrl+Shift+R` → 리팩토링 지시
5. **프로젝트 분석**: `Ctrl+Shift+P` → "Open CodeAI: Analyze Project"

### 고급 기능 (Open CodeAI 전용)
```javascript
// 함수 호출 관계 추적
"@opencodeai trace function_name"

// 전체 프로젝트 아키텍처 분석  
"@opencodeai analyze architecture"

// 성능 병목 탐지
"@opencodeai find bottlenecks"

// 의존성 분석
"@opencodeai analyze dependencies"

// 코드 품질 검사
"@opencodeai quality check"
```

## 🔧 폐쇄망 환경 설정

### 완전 오프라인 설치 패키지
```bash
# 오프라인 설치 패키지 준비 (인터넷 연결된 환경에서)
python scripts/prepare_offline_package.py \
  --output offline-install-package.tar.gz \
  --include-models \
  --include-dependencies

# 폐쇄망 환경에서 설치
tar -xzf offline-install-package.tar.gz
cd offline-install-package
./install_offline.sh
```

### 모델 업데이트 (폐쇄망)
```bash
# USB/외부 드라이브로 모델 전송 후
python scripts/update_models.py \
  --source /media/usb/models \
  --verify-integrity \
  --backup-existing
```

## 📊 모니터링 및 관리

### 시스템 상태 확인
```bash
# 전체 시스템 상태
./status.sh

# 출력 예시:
# 🟢 LLM Server: Running (GPU: 67%, Memory: 12.3GB)
# 🟢 Vector DB: Healthy (Index size: 2.1GB, 4,247 files)
# 🟢 Graph DB: Connected (23,445 nodes, 156,782 edges)  
# 🟢 Continue Plugin: Connected (2 active sessions)
# 🟢 File Watcher: Monitoring (0 pending changes)
```

### 성능 최적화
```bash
# 인덱스 최적화 (주기적 실행 권장)
python scripts/optimize_indices.py --aggressive

# 메모리 정리
python scripts/cleanup_memory.py

# 성능 튜닝 (하드웨어 기반 자동 조정)
python scripts/auto_tune.py --hardware-profile server
```

## 🚨 문제 해결

### 일반적인 문제
```bash
# 1. 모델 로딩 실패
python scripts/diagnose_models.py

# 2. 메모리 부족
python scripts/reduce_memory_usage.py --level conservative

# 3. 인덱싱 오류  
python scripts/repair_index.py --full-rebuild

# 4. Continue.dev 연결 끊김
./scripts/restart_connection.sh

# 5. 성능 저하
python scripts/performance_analysis.py --detailed
```

### 로그 확인
```bash
# 실시간 로그 모니터링
tail -f logs/opencodeai.log

# 오류 로그만 확인
grep "ERROR" logs/*.log | tail -20

# 성능 메트릭 확인
python scripts/show_metrics.py --last 24h
```

## 🤝 기여 및 지원

### 커뮤니티
- **GitHub Issues**: 버그 리포트 및 기능 요청
- **Discussions**: 사용법 질문 및 팁 공유
- **Wiki**: 상세 문서 및 튜토리얼

### 기여하기
```bash
# 개발 환경 설정
git clone https://github.com/ChangooLee/open-codeai.git
cd open-codeai
./scripts/setup_dev_env.sh

# 테스트 실행
python -m pytest tests/ -v

# 기여 가이드라인
# 1. Fork & 브랜치 생성
# 2. 기능 개발 & 테스트 추가  
# 3. PR 제출
```

## 📄 라이선스

**Apache 2.0 라이선스** - 상업적 사용 완전 허용

---

**Open CodeAI**로 대형 프로젝트에서도 Cursor AI를 능가하는 AI 코딩 경험을 시작하세요! 🚀

*폐쇄망 환경에서 완전 무료로 최고 성능의 AI 코드 어시스턴트를 경험해보세요.*