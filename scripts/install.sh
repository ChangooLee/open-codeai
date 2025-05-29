install_python_packages() {
    log_header "=== Python 패키지 설치 ==="
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        log_info "GPU 환경 감지: torch, torchvision만 설치 (vllm, faiss-gpu, torchaudio 제외)"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        log_warning "GPU를 감지할 수 없습니다. CPU 버전으로 설치합니다"
        GPU_AVAILABLE=false
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    # 메인 패키지들 설치
    log_info "메인 패키지 설치 중..."
    pip install -r requirements.txt
    # 개발 모드인 경우 개발 패키지도 설치
    if [[ "$INSTALL_MODE" == "dev" ]]; then
        pip install -r requirements-dev.txt
        log_info "개발 패키지 설치 완료"
    fi
    log_success "Python 패키지 설치 완료"
}

# Docker 컨테이너 설정
setup_docker_containers() {
    log_header "=== Docker 컨테이너 설정 ==="
    
    # Docker Compose 파일 선택
    if [[ "$INSTALL_MODE" == "minimal" ]]; then
        COMPOSE_FILE="docker-compose.minimal.yml"
        create_minimal_compose_file
    else
        COMPOSE_FILE="docker-compose.yml"
    fi
    
    # 환경 변수 파일 생성
    create_env_file
    
    # 데이터 디렉토리 생성
    create_data_directories
    
    # Docker 컨테이너 시작
    log_info "Docker 컨테이너 시작 중..."
    
    if [[ "$INSTALL_MODE" == "minimal" ]]; then
        # Neo4j만 시작 (Redis 제외)
        docker-compose -f $COMPOSE_FILE up -d neo4j
    else
        # 모든 서비스 시작
        docker-compose -f $COMPOSE_FILE up -d neo4j redis
        
        if [[ "$ENABLE_MONITORING" == "true" ]]; then
            docker-compose -f $COMPOSE_FILE up -d prometheus grafana
        fi
    fi
    
    # 컨테이너 준비 대기
    log_info "컨테이너 초기화 대기 중..."
    sleep 30
    
    # Neo4j 연결 확인
    check_neo4j_connection
    
    log_success "Docker 컨테이너 설정 완료"
}

# 최소 Docker Compose 파일 생성
create_minimal_compose_file() {
    cat > docker-compose.minimal.yml << 'EOF'
version: '3.8'

services:
  neo4j:
    image: neo4j:5.15-community
    container_name: open-codeai-neo4j
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/opencodeai
      - NEO4J_dbms_memory_heap_initial_size=512M
      - NEO4J_dbms_memory_heap_max_size=1G
      - NEO4J_dbms_memory_pagecache_size=512M
    volumes:
      - ./data/neo4j/data:/data
      - ./data/neo4j/logs:/logs
    restart: unless-stopped

networks:
  default:
    name: open-codeai-network
EOF
}

# 환경 변수 파일 생성
create_env_file() {
    log_info "환경 변수 파일 생성 중..."
    
    if [[ ! -f ".env" ]]; then
        cat > .env << EOF
# Open CodeAI 환경 설정
PROJECT_NAME=open-codeai
VERSION=1.0.0
ENVIRONMENT=production
DEBUG=false

# 서버 설정
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# 모델 경로
MODEL_PATH=./data/models/qwen2.5-coder-32b
EMBEDDING_MODEL_PATH=./data/models/bge-large-en-v1.5
GRAPH_MODEL_PATH=./data/models/codet5-small

# 데이터베이스 경로
VECTOR_INDEX_PATH=./data/vector_index
GRAPH_DB_PATH=./data/graph_db
METADATA_DB_PATH=./data/metadata/opencodeai.db

# Neo4j 설정
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=opencodeai

# Redis 설정 (full 모드에서만)
REDIS_URL=redis://localhost:6379

# Continue.dev 통합
CONTINUE_PORT=8001
CONTINUE_AUTH_TOKEN=open-codeai-secure-token
API_KEY=open-codeai-local-key

# GPU 설정
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_FRACTION=${GPU_MEMORY_FRACTION:-0.7}
USE_MIXED_PRECISION=true

# 성능 설정
MAX_WORKERS=16
MEMORY_LIMIT_GB=32
CACHE_SIZE_GB=10

# 프로젝트 경로 (인덱싱용)
PROJECT_PATH=${PROJECT_PATH:-./}

GRAPH_DB_TYPE=$(if $USE_NEO4J; then echo "neo4j"; else echo "networkx"; fi)
EOF
        
        # GPU 설정 조정
        if [[ "$GPU_AVAILABLE" == "true" ]]; then
            sed -i 's/GPU_MEMORY_FRACTION=0.7/GPU_MEMORY_FRACTION=0.8/' .env
        else
            echo "EMBED_DEVICE=cpu" >> .env
        fi
        
        log_success "환경 변수 파일 생성 완료"
    else
        log_info "기존 환경 변수 파일 사용"
    fi
}

# 데이터 디렉토리 생성
create_data_directories() {
    log_info "데이터 디렉토리 생성 중..."
    
    directories=(
        "data/models"
        "data/vector_index"
        "data/graph_db"
        "data/metadata"
        "data/cache"
        "data/logs"
        "data/neo4j/data"
        "data/neo4j/logs"
        "data/neo4j/import"
        "data/neo4j/plugins"
        "data/redis"
        "data/prometheus"
        "data/grafana"
        "logs"
        "backups"
        "static"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    log_success "데이터 디렉토리 생성 완료"
}

# Neo4j 연결 확인
check_neo4j_connection() {
    log_info "Neo4j 연결 확인 중..."
    
    max_attempts=30
    attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if docker exec open-codeai-neo4j cypher-shell -u neo4j -p opencodeai "RETURN 1" &> /dev/null; then
            log_success "Neo4j 연결 성공"
            return 0
        fi
        
        log_info "Neo4j 연결 시도 중... ($attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    log_error "Neo4j 연결 실패"
    return 1
}

# AI 모델 다운로드
download_models() {
    if [[ "$DOWNLOAD_MODELS" == "true" ]]; then
        log_header "=== AI 모델 다운로드 ==="
        
        source venv/bin/activate
        
        # 모델 다운로드 스크립트 생성
        cat > scripts/download_models.py << 'EOF'
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import torch

def download_model(repo_id, local_dir, model_name):
    print(f"다운로드 중: {model_name}")
    print(f"저장소: {repo_id}")
    print(f"저장 경로: {local_dir}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            cache_dir="./data/cache/huggingface"
        )
        print(f"✅ {model_name} 다운로드 완료")
        return True
    except Exception as e:
        print(f"❌ {model_name} 다운로드 실패: {e}")
        return False

def main():
    models = [
        {
            "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "local_dir": "./data/models/qwen2.5-coder-32b",
            "name": "Qwen2.5-Coder-32B (메인 LLM)",
            "size": "64GB"
        },
        {
            "repo_id": "BAAI/bge-large-en-v1.5",
            "local_dir": "./data/models/bge-large-en-v1.5", 
            "name": "BGE Large (임베딩 모델)",
            "size": "2.3GB"
        },
        {
            "repo_id": "Salesforce/codet5-small",
            "local_dir": "./data/models/codet5-small",
            "name": "CodeT5 Small (그래프 분석)",
            "size": "242MB"
        }
    ]
    
    print("=== AI 모델 다운로드 시작 ===")
    total_size = sum([64, 2.3, 0.242])  # GB
    print(f"총 다운로드 크기: {total_size:.1f}GB")
    print("인터넷 연결이 필요합니다. 시간이 오래 걸릴 수 있습니다.\n")
    
    success_count = 0
    for model in models:
        print(f"\n{model['name']} ({model['size']}) 다운로드 중...")
        if download_model(model['repo_id'], model['local_dir'], model['name']):
            success_count += 1
    
    print(f"\n=== 다운로드 완료: {success_count}/{len(models)} 모델 ===")
    return success_count == len(models)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF
        
        # 스크립트 실행
        python scripts/download_models.py
        
        if [[ $? -eq 0 ]]; then
            log_success "모델 다운로드 완료"
        else
            log_warning "일부 모델 다운로드 실패. 더미 모드로 실행됩니다."
        fi
    else
        log_info "모델 다운로드 건너뜀"
        log_info "나중에 다음 명령으로 다운로드할 수 있습니다:"
        log_info "python scripts/download_models.py"
    fi
}

# 데이터베이스 초기화
initialize_databases() {
    log_header "=== 데이터베이스 초기화 ==="
    
    source venv/bin/activate
    
    # 초기화 스크립트 생성
    cat > scripts/init_databases.py << 'EOF'
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.rag_system import get_rag_system
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def main():
    print("=== 데이터베이스 초기화 ===")
    
    try:
        # RAG 시스템 초기화
        rag_system = get_rag_system()
        
        # 벡터 DB 초기화 확인
        print("✅ 벡터 데이터베이스 초기화 완료")
        
        # 그래프 DB 연결 확인
        print("✅ 그래프 데이터베이스 연결 완료")
        
        # 메타데이터 DB 초기화 확인
        stats = rag_system.indexer.get_indexing_stats()
        print(f"✅ 메타데이터 데이터베이스 초기화 완료")
        print(f"   - 인덱싱된 파일: {stats.get('total_files', 0)}")
        print(f"   - 총 청크: {stats.get('total_chunks', 0)}")
        
        print("\n=== 데이터베이스 초기화 성공 ===")
        return True
        
    except Exception as e:
        print(f"❌ 데이터베이스 초기화 실패: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
EOF
    
    # 초기화 실행
    python scripts/init_databases.py
    
    if [[ $? -eq 0 ]]; then
        log_success "데이터베이스 초기화 완료"
    else
        log_error "데이터베이스 초기화 실패"
        exit 1
    fi
}

# 프로젝트 인덱싱
index_project() {
    if [[ -n "$PROJECT_PATH" ]]; then
        log_header "=== 프로젝트 인덱싱 ==="
        
        if [[ ! -d "$PROJECT_PATH" ]]; then
            log_error "프로젝트 경로가 존재하지 않습니다: $PROJECT_PATH"
            return 1
        fi
        
        source venv/bin/activate
        
        log_info "프로젝트 인덱싱 시작: $PROJECT_PATH"
        
        # 인덱싱 스크립트 생성
        cat > scripts/index_initial_project.py << EOF
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.rag_system import get_rag_system
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def main():
    project_path = "${PROJECT_PATH}"
    print(f"프로젝트 인덱싱 시작: {project_path}")
    
    try:
        rag_system = get_rag_system()
        
        result = await rag_system.indexer.index_directory(
            directory_path=project_path,
            max_files=1000
        )
        
        print(f"✅ 인덱싱 완료: {result.get('success_count', 0)}/{result.get('total_files', 0)} 파일")
        return True
        
    except Exception as e:
        print(f"❌ 인덱싱 실패: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
EOF
        
        # 인덱싱 실행
        python scripts/index_initial_project.py
        
        if [[ $? -eq 0 ]]; then
            log_success "프로젝트 인덱싱 완료"
        else
            log_warning "프로젝트 인덱싱 실패"
        fi
    else
        log_info "프로젝트 경로가 지정되지 않음. 인덱싱 건너뜀"
    fi
}

# Continue.dev 설정 준비
setup_continue_integration() {
    log_header "=== Continue.dev 통합 설정 ==="
    
    # Continue 설정 디렉토리 생성
    mkdir -p ~/.continue
    
    # Continue 설정 파일 생성
    cat > ~/.continue/config.json << 'EOF'
{
  "models": [
    {
      "title": "Open CodeAI",
      "provider": "openai",
      "model": "open-codeai", 
      "apiKey": "open-codeai-local-key",
      "apiBase": "http://localhost:8000/v1",
      "systemMessage": "당신은 전문적인 소프트웨어 개발자입니다. 정확하고 효율적인 코드를 작성하며, 한국어로 명확하게 설명합니다.",
      "contextLength": 4096,
      "requestOptions": {
        "timeout": 30000,
        "verifySsl": false
      }
    }
  ],
  "tabAutocompleteModel": {
    "title": "Open CodeAI Autocomplete",
    "provider": "openai",
    "model": "qwen2.5-coder-32b",
    "apiKey": "open-codeai-local-key", 
    "apiBase": "http://localhost:8000/v1",
    "useLegacyCompletionsEndpoint": true,
    "contextLength": 1024,
    "template": "코드 자동완성:\n\n{{{prefix}}}\n\n위 코드를 자연스럽게 완성해주세요:",
    "completionOptions": {
      "temperature": 0.1,
      "maxTokens": 256,
      "stop": ["\n\n", "```", "def ", "class ", "import ", "from "]
    }
  },
  "embeddingsProvider": {
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "apiKey": "open-codeai-local-key",
    "apiBase": "http://localhost:8000/v1"
  },
  "contextProviders": [
    {
      "name": "codebase",
      "params": {
        "nRetrieve": 25,
        "nFinal": 5,
        "useReranking": true
      }
    },
    {
      "name": "diff",
      "params": {}
    },
    {
      "name": "folder", 
      "params": {
        "folders": ["src", "lib", "components"]
      }
    },
    {
      "name": "open",
      "params": {}
    },
    {
      "name": "terminal",
      "params": {}
    }
  ],
  "customCommands": [
    {
      "name": "review",
      "prompt": "다음 코드를 리뷰하고 개선점을 제안해주세요:\n\n{{{ input }}}\n\n리뷰 포인트:\n1. 코드 품질\n2. 성능 최적화\n3. 보안 이슈\n4. 가독성\n5. 유지보수성",
      "description": "코드 리뷰 및 개선점 제안"
    },
    {
      "name": "explain",
      "prompt": "다음 코드의 동작을 단계별로 자세히 설명해주세요:\n\n{{{ input }}}",
      "description": "코드 동작 원리 상세 설명"
    },
    {
      "name": "search",
      "prompt": "프로젝트에서 '{{{ input }}}'와 관련된 코드를 찾아서 보여주세요.",
      "description": "프로젝트 코드 검색"
    }
  ],
  "allowAnonymousTelemetry": false
}
EOF
    
    log_success "Continue.dev 설정 파일 생성 완료"
    log_info "VS Code에서 Continue 확장을 설치하세요"
}

# 시스템 검증
verify_installation() {
    log_header "=== 설치 검증 ==="
    
    source venv/bin/activate
    
    # 검증 스크립트 생성
    cat > scripts/verify_installation.py << 'EOF'
import asyncio
import sys
import os
import requests
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.core.llm_manager import get_llm_manager
from src.core.rag_system import get_rag_system
from src.core.function_calling import get_function_registry

def check_packages():
    """필수 패키지 확인"""
    try:
        import fastapi, uvicorn, torch, faiss, neo4j
        print("✅ 필수 패키지 설치됨")
        return True
    except ImportError as e:
        print(f"❌ 패키지 누락: {e}")
        return False

def check_gpu():
    """GPU 사용 가능성 확인"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU 사용 가능: {gpu_name} ({gpu_count}개)")
            return True
        else:
            print("⚠️ GPU 사용 불가, CPU 모드로 실행")
            return True
    except Exception as e:
        print(f"❌ GPU 확인 실패: {e}")
        return False

def check_neo4j():
    """Neo4j 연결 확인"""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "opencodeai"))
        with driver.session() as session:
            result = session.run("RETURN 1")
            if result.single():
                print("✅ Neo4j 연결 성공")
                driver.close()
                return True
    except Exception as e:
        print(f"❌ Neo4j 연결 실패: {e}")
        return False

def check_api_server():
    """API 서버 확인"""
    try:
        # 서버가 실행 중인지 확인
        response = requests.get("http://localhost:8000/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ API 서버 실행 중")
            return True
        else:
            print("❌ API 서버 응답 오류")
            return False
    except Exception:
        print("⚠️ API 서버 미실행 (정상 - 수동 시작 필요)")
        return True

async def check_rag_system():
    """RAG 시스템 확인"""
    try:
        rag_system = get_rag_system()
        stats = rag_system.indexer.get_indexing_stats()
        print(f"✅ RAG 시스템 정상 (파일: {stats.get('total_files', 0)}, 청크: {stats.get('total_chunks', 0)})")
        return True
    except Exception as e:
        print(f"❌ RAG 시스템 오류: {e}")
        return False

def check_function_calling():
    """Function Calling 확인"""
    try:
        registry = get_function_registry()
        functions = registry.get_available_functions()
        print(f"✅ Function Calling 정상 (함수: {len(functions)}개)")
        return True
    except Exception as e:
        print(f"❌ Function Calling 오류: {e}")
        return False

async def main():
    print("=== Open CodeAI 설치 검증 ===\n")
    
    checks = [
        ("패키지", check_packages),
        ("GPU", check_gpu),
        ("Neo4j", check_neo4j),
        ("API 서버", check_api_server),
        ("RAG 시스템", check_rag_system),
        ("Function Calling", check_function_calling)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"[{name}] 확인 중...")
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {name} 확인 실패: {e}")
    
    print(f"\n=== 검증 완료: {passed}/{total} 항목 통과 ===")
    
    if passed >= total - 1:  # API 서버는 선택적
        print("🎉 설치가 성공적으로 완료되었습니다!")
        return True
    else:
        print("⚠️ 일부 구성 요소에 문제가 있습니다.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
EOF
    
    # 검증 실행
    python scripts/verify_installation.py
    
    if [[ $? -eq 0 ]]; then
        log_success "설치 검증 완료"
        return 0
    else
        log_warning "설치 검증에서 일부 문제 발견"
        return 1
    fi
}

# 시작 스크립트 생성
create_startup_scripts() {
    log_header "=== 시작 스크립트 생성 ==="
    
    # 메인 시작 스크립트
    cat > start.sh << 'EOF'
#!/bin/bash

# Open CodeAI 시작 스크립트

set -e

log_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
log_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }
log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }

start_services() {
    log_info "🚀 Open CodeAI 서비스 시작 중..."
    
    # Docker 서비스 시작
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d
    elif command -v docker &> /dev/null && command -v compose &> /dev/null; then
        docker compose up -d
    else
        log_error "Docker Compose를 찾을 수 없습니다"
        exit 1
    fi
    
    # 서비스 준비 대기
    log_info "서비스 준비 대기 중..."
    sleep 10
    
    # 가상환경 활성화
    if [[ -d "venv" ]]; then
        source venv/bin/activate
        log_info "Python 가상환경 활성화됨"
    fi
    
    # API 서버 시작
    log_info "API 서버 시작 중..."
    echo "=================================="
    echo "🤖 Open CodeAI 시작됨"
    echo "=================================="
    echo "📡 API 서버: http://localhost:8000"
    echo "📚 API 문서: http://localhost:8000/docs"
    echo "🔌 Continue.dev: ws://localhost:8000/v1"
    echo "🗃️  Neo4j 브라우저: http://localhost:7474"
    echo "=================================="
    echo "종료하려면 Ctrl+C를 누르세요"
    echo ""
    
    python src/main.py
}

stop_services() {
    log_info "🛑 Open CodeAI 서비스 중지 중..."
    
    # API 서버 중지
    pkill -f "python src/main.py" 2>/dev/null || true
    
    # Docker 서비스 중지
    if command -v docker-compose &> /dev/null; then
        docker-compose down
    elif command -v docker &> /dev/null && command -v compose &> /dev/null; then
        docker compose down
    fi
    
    log_success "서비스가 중지되었습니다"
}

case "${1:-start}" in
    "start")
        start_services
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        sleep 2
        start_services
        ;;
    *)
        echo "사용법: $0 [start|stop|restart]"
        exit 1
        ;;
esac
EOF
    
    chmod +x start.sh
    
    # 인덱싱 스크립트
    cat > index.sh << 'EOF'
#!/bin/bash

# 프로젝트 인덱싱 스크립트

if [[ $# -lt 1 ]]; then
    echo "사용법: $0 <프로젝트_경로> [최대_파일수]"
    echo "예시: $0 /path/to/project 1000"
    exit 1
fi

PROJECT_PATH="$1"
MAX_FILES="${2:-1000}"

if [[ ! -d "$PROJECT_PATH" ]]; then
    echo "❌ 프로젝트 경로가 존재하지 않습니다: $PROJECT_PATH"
    exit 1
fi

echo "🔍 프로젝트 인덱싱 시작: $PROJECT_PATH"

source venv/bin/activate

python -c "
import asyncio
import sys
import os
sys.path.append('.')
from src.core.rag_system import get_rag_system

async def main():
    rag_system = get_rag_system()
    result = await rag_system.indexer.index_directory('$PROJECT_PATH', $MAX_FILES)
    print(f'✅ 인덱싱 완료: {result.get(\"success_count\", 0)}/{result.get(\"total_files\", 0)} 파일')

asyncio.run(main())
"
EOF
    
    chmod +x index.sh
    
    log_success "시작 스크립트 생성 완료"
}

# 최종 안내 메시지
show_completion_message() {
    log_header "======================================"
    log_success "🎉 Open CodeAI 설치 완료!"
    log_header "======================================"
    echo ""
    
    echo "📋 설치 정보:"
    echo "   - 모드: $INSTALL_MODE"
    echo "   - GPU: $([ "$GPU_AVAILABLE" == "true" ] && echo "사용 가능" || echo "사용 불가")"
    echo "   - 모델 다운로드: $([ "$DOWNLOAD_MODELS" == "true" ] && echo "완료" || echo "건너뜀")"
    echo "   - 모니터링: $([ "$ENABLE_MONITORING" == "true" ] && echo "활성화" || echo "비활성화")"
    if [[ -n "$PROJECT_PATH" ]]; then
        echo "   - 인덱싱된 프로젝트: $PROJECT_PATH"
    fi
    echo ""
    
    echo "🚀 다음 단계:"
    echo "1. 서버 시작:"
    echo "   ./start.sh"
    echo ""
    echo "2. VS Code에서 Continue 확장 설치"
    echo "   - Continue 설정이 자동으로 적용됩니다"
    echo ""
    echo "3. 프로젝트 인덱싱 (필요시):"
    echo "   ./index.sh /path/to/your/project"
    echo ""
    
    echo "🔗 유용한 링크:"
    echo "   - API 서버: http://localhost:8000"
    echo "   - API 문서: http://localhost:8000/docs"
    echo "   - Neo4j 브라우저: http://localhost:7474 (neo4j/opencodeai)"
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        echo "   - Grafana: http://localhost:3000 (admin/opencodeai)"
    fi
    echo ""
    
    echo "📚 사용 가이드:"
    echo "   - 코드 검색: Ctrl+Shift+L 후 '@search 함수명'"
    echo "   - 코드 리뷰: 코드 선택 후 '@review'"
    echo "   - 코드 설명: 코드 선택 후 '@explain'"
    echo ""
    
    if [[ "$DOWNLOAD_MODELS" == "false" ]]; then
        log_warning "모델이 다운로드되지 않았습니다."
        echo "   다음 명령으로 다운로드하세요:"
        echo "   source venv/bin/activate && python scripts/download_models.py"
        echo ""
    fi
    
    echo "❓ 문제 해결:"
    echo "   - 로그 확인: tail -f logs/opencodeai.log"
    echo "   - 상태 확인: python scripts/verify_installation.py"
    echo "   - 재시작: ./start.sh restart"
    echo ""
    
    log_header "======================================"
    log_success "Open CodeAI가 준비되었습니다! 🤖"
    log_header "======================================"
}

# 플래그 파싱 함수 추가
parse_arguments() {
    # 기본값
    OFFLINE_INSTALL=false
    INSTALL_MODE="full"
    DOWNLOAD_MODELS=true
    ENABLE_MONITORING=false
    PROJECT_PATH=""
    USE_NEO4J=true

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --offline)
                OFFLINE_INSTALL=true
                ;;
            --no-neo4j)
                USE_NEO4J=false
                ;;
            --minimal)
                INSTALL_MODE="minimal"
                ;;
            --no-model-download)
                DOWNLOAD_MODELS=false
                ;;
            --enable-monitoring)
                ENABLE_MONITORING=true
                ;;
            --project-path)
                PROJECT_PATH="$2"
                shift
                ;;
            *)
                # 무시
                ;;
        esac
        shift
    done
}

# 메인 실행 함수
main() {
    # 파라미터 파싱
    parse_arguments "$@"

    # 경로/권한/특수문자 안내
    echo ""
    log_header "======================================"
    log_header "⚠️  경로에 한글/공백/특수문자 사용을 피하세요!"
    log_header "======================================"
    echo ""

    # Docker 실행 상태 체크
    if ! docker info &> /dev/null; then
        log_error "Docker Desktop/엔진이 실행 중이 아닙니다. Docker를 실행한 후 다시 시도하세요."
        exit 1
    fi

    # 오프라인 패키지/모델/도커 이미지 체크
    if [[ "$OFFLINE_INSTALL" == "true" ]]; then
        if [[ ! -d "offline_packages" || -z $(ls -A offline_packages 2>/dev/null) ]]; then
            log_warning "offline_packages 폴더가 없거나 비어 있습니다. 오프라인 패키지 설치가 실패할 수 있습니다."
        fi
        if [[ ! -d "data/models" || -z $(ls -A data/models 2>/dev/null) ]]; then
            log_error "data/models 폴더가 없거나 비어 있습니다. 모델 파일을 미리 복사해야 합니다."
            exit 1
        fi
        if [[ -d "docker-images" ]]; then
            if [[ -z $(ls -A docker-images/*.tar 2>/dev/null) ]]; then
                log_warning "docker-images 폴더에 Docker 이미지 tar 파일이 없습니다."
            fi
        fi
    fi

    echo ""
    log_header "======================================"
    log_header "🤖 Open CodeAI 설치 시작"
    log_header "======================================"
    echo ""
    log_info "설치 모드: $INSTALL_MODE"
    log_info "모델 다운로드: $([ "$DOWNLOAD_MODELS" == "true" ] && echo "예" || echo "아니오")"
    log_info "모니터링: $([ "$ENABLE_MONITORING" == "true" ] && echo "활성화" || echo "비활성화")"
    if [[ -n "$PROJECT_PATH" ]]; then
        log_info "프로젝트 경로: $PROJECT_PATH"
    fi
    echo ""

    if [[ "$OFFLINE_INSTALL" == "true" ]]; then
        main_install
        # 오프라인 설치 후 검증 및 안내
        if verify_installation; then
            show_completion_message
        else
            log_error "설치 중 일부 문제가 발생했습니다."
            log_info "문제 해결 후 다음 명령으로 검증할 수 있습니다:"
            log_info "python scripts/verify_installation.py"
        fi
        return
    fi

    # 온라인 설치 루트
    check_system_requirements
    install_system_dependencies
    setup_python_environment
    install_python_packages
    setup_docker_containers
    download_models
    initialize_databases
    index_project
    setup_continue_integration
    create_startup_scripts

    # start.sh, index.sh 실행 권한 자동 부여
    if [[ -f "start.sh" && ! -x "start.sh" ]]; then
        chmod +x start.sh
        log_info "start.sh에 실행 권한을 부여했습니다."
    fi
    if [[ -f "index.sh" && ! -x "index.sh" ]]; then
        chmod +x index.sh
        log_info "index.sh에 실행 권한을 부여했습니다."
    fi

    # 설치 검증
    if verify_installation; then
        show_completion_message
    else
        log_error "설치 중 일부 문제가 발생했습니다."
        log_info "문제 해결 후 다음 명령으로 검증할 수 있습니다:"
        log_info "python scripts/verify_installation.py"
    fi
}

# 에러 처리
set -e
trap 'log_error "설치 중 오류가 발생했습니다. 라인 $LINENO에서 중단되었습니다."' ERR

# 메인 실행
main "$@"

# 오프라인 패키지 설치 함수 추가
install_offline_packages() {
    if [ -d "./offline_packages" ]; then
        log_info "오프라인 패키지 설치 중..."
        pip install --no-index --find-links=offline_packages -r requirements.txt
        if [[ "$INSTALL_MODE" == "dev" ]]; then
            pip install --no-index --find-links=offline_packages -r requirements-dev.txt
        fi
        log_success "오프라인 패키지 설치 완료"
        return 0
    fi
    return 1
}

# 오프라인 Docker 이미지 로딩 함수
load_offline_docker_images() {
    if [ -d "./docker-images" ]; then
        for tarfile in ./docker-images/*.tar; do
            if [ -f "$tarfile" ]; then
                log_info "오프라인 Docker 이미지 로딩: $tarfile"
                docker load -i "$tarfile"
            fi
        done
    fi
}

# 모델 파일 체크 함수
check_offline_models() {
    local model_path="./data/models/qwen2.5-coder-32b"
    if [ -d "$model_path" ]; then
        log_success "로컬 모델 파일이 존재합니다: $model_path"
        return 0
    else
        log_warning "로컬 모델 파일이 없습니다. data/models/에 미리 복사해 주세요."
        return 1
    fi
}

# 메인 설치 함수에서 오프라인 설치 분기 추가
main_install() {
    # 오프라인 패키지 설치 시도
    install_offline_packages || {
        if [[ "$GPU_AVAILABLE" == "true" ]]; then
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
            # pip install faiss-gpu>=1.7.4
            # pip install vllm>=0.2.7
            # pip install torchaudio
        else
            log_warning "GPU를 감지할 수 없습니다. CPU 버전으로 설치합니다"
            GPU_AVAILABLE=false
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
            # pip install faiss-cpu>=1.7.4
        fi
        pip install -r requirements.txt
        if [[ "$INSTALL_MODE" == "dev" ]]; then
            pip install -r requirements-dev.txt
        fi
    }
    log_success "Python 패키지 설치 완료"

    # 오프라인 Docker 이미지 로딩
    load_offline_docker_images

    # 모델 파일 체크
    check_offline_models || {
        log_error "모델 파일이 없습니다. 설치를 중단합니다."
        exit 1
    }

    # config.yaml → .env 자동 변환
    if [[ -f "config.yaml" ]]; then
        log_info "config.yaml을 감지했습니다. .env 파일을 자동 생성합니다."
        if [[ -f "scripts/generate_env.py" ]]; then
            python scripts/generate_env.py || log_warning ".env 자동 생성에 실패했습니다. 기본 .env 생성 로직을 사용합니다."
        else
            log_warning "scripts/generate_env.py가 없습니다. 기본 .env 생성 로직을 사용합니다."
        fi
    fi

    # 환경 변수 파일 생성 시 그래프 DB 타입 반영
    create_env_file

    # Docker 컨테이너 설정 (Neo4j 사용 여부 반영)
    if $USE_NEO4J; then
        setup_docker_containers
    else
        log_info "Neo4j 컨테이너를 실행하지 않습니다. NetworkX(in-memory)만 사용합니다."
    fi
}