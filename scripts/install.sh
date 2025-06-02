#!/bin/bash
set -e

# venv가 있으면 자동 활성화
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Deprecated Neo4j config cleanup (Neo4j 5.x 이상에서 필요)
cleanup_neo4j_conf() {
    for conf_file in data/neo4j/data/neo4j.conf data/neo4j/conf/neo4j.conf; do
        if [[ -f "$conf_file" ]]; then
            sed -i.bak '/dbms\.memory\.pagecache_size/d' "$conf_file"
        fi
    done
}

log_header() {
    echo -e "\033[1;36m$1\033[0m"
}
log_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}
log_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}
log_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}
log_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

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
    pip install --no-index --find-links=offline_packages -r requirements.txt
    log_info "메인 패키지 설치 완료"
}

# Docker 컨테이너 설정
setup_docker_containers() {
    log_header "=== Docker 컨테이너 설정 ==="
    cleanup_neo4j_conf
    create_env_file
    create_data_directories
    log_info "Docker 컨테이너 시작 중..."
    docker-compose -f docker-compose.yml up -d neo4j redis
    log_info "컨테이너 초기화 대기 중..."
    sleep 30
    check_neo4j_connection
    log_success "Docker 컨테이너 설정 완료"
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
PORT=8800
LOG_LEVEL=INFO
# === LLM/vLLM 엔진 ===
VLLM_ENDPOINT=http://localhost:8800/v1
VLLM_API_KEY=open-codeai-local-key
VLLM_MODEL_ID=Qwen2.5-Coder-32B-Instruct
# 데이터베이스 경로
VECTOR_INDEX_PATH=./data/vector_index
GRAPH_DB_PATH=./data/graph_db
METADATA_DB_PATH=./data/metadata/opencodeai.db
# Neo4j 설정
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=opencodeai
# Redis 설정
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
GRAPH_DB_TYPE=neo4j
EOF
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
        "logs"
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

# 오프라인 패키지 설치 함수
install_offline_packages() {
    if [ -d "./offline_packages" ]; then
        log_info "오프라인 패키지 설치 중..."
        pip install --no-index --find-links=offline_packages -r requirements.txt
        log_success "오프라인 패키지 설치 완료"
        return 0
    fi
    return 1
}

# 오프라인 Docker 이미지 로딩 함수
load_offline_docker_images() {
    if [ -d "./offline_packages/docker-images" ]; then
        for tarfile in ./offline_packages/docker-images/*.tar; do
            if [ -f "$tarfile" ]; then
                log_info "오프라인 Docker 이미지 로딩: $tarfile"
                docker load -i "$tarfile"
            fi
        done
    fi
}

# 메인 설치 함수
main_install() {
    log_header "=== Docker 이미지 빌드 (캐시 무시) ==="
    docker-compose -f docker-compose.yml build --no-cache
    install_offline_packages || install_python_packages
    load_offline_docker_images
    create_env_file
    setup_docker_containers
}

# 메인 실행 함수
main() {
    echo ""
    log_header "======================================"
    log_header "⚠️  경로에 한글/공백/특수문자 사용을 피하세요!"
    log_header "======================================"
    echo ""
    if ! docker info &> /dev/null; then
        log_error "Docker Desktop/엔진이 실행 중이 아닙니다. Docker를 실행한 후 다시 시도하세요."
        exit 1
    fi
    main_install
    if python scripts/verify_installation.py; then
        log_success "설치가 성공적으로 완료되었습니다!"
    else
        log_error "설치 중 일부 문제가 발생했습니다."
        log_info "문제 해결 후 다음 명령으로 검증할 수 있습니다:"
        log_info "python scripts/verify_installation.py"
    fi
}

set -e
trap 'log_error "설치 중 오류가 발생했습니다. 라인 $LINENO에서 중단되었습니다."' ERR

main