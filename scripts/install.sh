#!/bin/bash

# Open CodeAI 원클릭 설치 스크립트
# Ubuntu/Debian/CentOS/macOS 지원

set -e  # 오류 발생 시 스크립트 중단

# 색깔 출력을 위한 변수들
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수들
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 시스템 정보 감지
detect_system() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            SYSTEM="debian"
            PACKAGE_MANAGER="apt"
        elif [ -f /etc/redhat-release ]; then
            SYSTEM="redhat"
            PACKAGE_MANAGER="yum"
        else
            SYSTEM="linux"
            PACKAGE_MANAGER="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        SYSTEM="macos"
        PACKAGE_MANAGER="brew"
    else
        SYSTEM="unknown"
        PACKAGE_MANAGER="unknown"
    fi
    
    log_info "감지된 시스템: $SYSTEM"
}

# 필수 도구 설치
install_system_dependencies() {
    log_info "시스템 의존성 설치 중..."
    
    case $PACKAGE_MANAGER in
        "apt")
            sudo apt update
            sudo apt install -y \
                python3.10 python3.10-dev python3.10-venv \
                build-essential cmake git curl wget \
                pkg-config libssl-dev \
                nodejs npm \
                htop nvtop \
                docker.io docker-compose
            ;;
        "yum")
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                python3.10 python3.10-devel \
                cmake git curl wget \
                nodejs npm \
                docker docker-compose
            ;;
        "brew")
            brew install python@3.10 cmake git nodejs npm
            brew install --cask docker
            ;;
        *)
            log_error "지원하지 않는 패키지 매니저입니다: $PACKAGE_MANAGER"
            exit 1
            ;;
    esac
    
    log_success "시스템 의존성 설치 완료"
}

# Python 가상환경 설정
setup_python_environment() {
    log_info "Python 가상환경 설정 중..."
    
    # Python 버전 확인
    PYTHON_VERSION=$(python3.10 --version 2>/dev/null || python3 --version 2>/dev/null || python --version)
    log_info "Python 버전: $PYTHON_VERSION"
    
    # 가상환경 생성
    if [ ! -d "venv" ]; then
        python3.10 -m venv venv 2>/dev/null || python3 -m venv venv
        log_success "Python 가상환경 생성 완료"
    else
        log_info "기존 가상환경 발견, 재사용합니다"
    fi
    
    # 가상환경 활성화
    source venv/bin/activate
    
    # pip 업그레이드
    pip install --upgrade pip setuptools wheel
    
    log_success "Python 환경 설정 완료"
}

# Python 패키지 설치
install_python_packages() {
    log_info "Python 패키지 설치 중..."
    
    # 가상환경 활성화 확인
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        source venv/bin/activate
    fi
    
    # GPU 사용 가능 여부 확인
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU 감지됨, GPU 버전 패키지 설치"
        # GPU 버전 FAISS 설치
        pip install faiss-gpu>=1.7.4
        pip install vllm>=0.2.7
        GPU_AVAILABLE=true
    else
        log_warning "GPU를 감지할 수 없습니다, CPU 버전으로 설치"
        pip install faiss-cpu>=1.7.4
        GPU_AVAILABLE=false
    fi
    
    # 기본 패키지 설치
    pip install -r requirements.txt
    
    # 개발 패키지 설치 (선택적)
    if [ "$INSTALL_DEV" = "true" ]; then
        pip install -r requirements-dev.txt
    fi
    
    log_success "Python 패키지 설치 완료"
}

# Tree-sitter 언어 파서 설정
setup_tree_sitter() {
    log_info "Tree-sitter 언어 파서 설정 중..."
    
    # Tree-sitter CLI 설치
    if ! command -v tree-sitter &> /dev/null; then
        npm install -g tree-sitter-cli
    fi
    
    # Python으로 언어 파서 설정
    source venv/bin/activate
    python scripts/setup_tree_sitter.py
    
    log_success "Tree-sitter 설정 완료"
}

# Docker 및 Neo4j 설정
setup_databases() {
    log_info "데이터베이스 설정 중..."
    
    # Docker 서비스 시작
    if command -v systemctl &> /dev/null; then
        sudo systemctl start docker
        sudo systemctl enable docker
    elif command -v service &> /dev/null; then
        sudo service docker start
    fi
    
    # 현재 사용자를 docker 그룹에 추가
    if ! groups $USER | grep -q '\bdocker\b'; then
        sudo usermod -aG docker $USER
        log_warning "Docker 그룹에 추가되었습니다. 로그아웃 후 다시 로그인하세요."
    fi
    
    # Neo4j 컨테이너 시작
    if [ "$ENABLE_GRAPH_DB" = "true" ]; then
        docker-compose up -d neo4j
        log_info "Neo4j 컨테이너 시작됨"
        
        # Neo4j 준비 대기
        log_info "Neo4j 초기화 대기 중..."
        sleep 30
    fi
    
    # 벡터 및 메타데이터 DB 초기화
    source venv/bin/activate
    python scripts/init_databases.py
    
    log_success "데이터베이스 설정 완료"
}

# 모델 다운로드 (선택적)
download_models() {
    if [ "$DOWNLOAD_MODELS" = "true" ]; then
        log_info "AI 모델 다운로드 중... (시간이 오래 걸릴 수 있습니다)"
        
        source venv/bin/activate
        python scripts/download_models.py --config configs/models.yaml
        
        log_success "모델 다운로드 완료"
    else
        log_info "모델 다운로드 건너뜀 (나중에 수동으로 설정하세요)"
    fi
}

# 환경 변수 파일 생성
setup_environment_file() {
    log_info "환경 변수 파일 설정 중..."
    
    if [ ! -f ".env" ]; then
        cp .env.example .env
        
        # GPU 설정 자동 조정
        if [ "$GPU_AVAILABLE" = "true" ]; then
            sed -i 's/EMBED_DEVICE=cpu/EMBED_DEVICE=cuda/' .env
            sed -i 's/GPU_MEMORY_FRACTION=0.7/GPU_MEMORY_FRACTION=0.8/' .env
        fi
        
        log_success "환경 변수 파일 생성 완료 (.env)"
        log_info "필요시 .env 파일을 수정하세요"
    else
        log_info ".env 파일이 이미 존재합니다"
    fi
}

# 설치 검증
verify_installation() {
    log_info "설치 검증 중..."
    
    source venv/bin/activate
    python scripts/verify_installation.py
    
    if [ $? -eq 0 ]; then
        log_success "설치 검증 완료"
    else
        log_error "설치 검증 실패"
        exit 1
    fi
}

# Continue.dev 플러그인 준비
prepare_continue_plugin() {
    log_info "Continue.dev 플러그인 준비 중..."
    
    # VS Code 설정 파일 생성
    mkdir -p ~/.continue
    
    cat > ~/.continue/config.json << EOF
{
  "models": [
    {
      "title": "Open CodeAI",
      "provider": "openai",
      "model": "open-codeai",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "open-codeai-local-key"
    }
  ],
  "allowAnonymousTelemetry": false,
  "embeddingsProvider": {
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "apiBase": "http://localhost:8000/v1",
    "apiKey": "open-codeai-local-key"
  }
}
EOF
    
    log_success "Continue.dev 설정 파일 생성 완료"
    log_info "VS Code에서 Continue 확장을 설치하세요"
}

# 성능 튜닝
optimize_system() {
    log_info "시스템 성능 최적화 중..."
    
    source venv/bin/activate
    python -c "
from src.utils.hardware import get_hardware_info, recommend_settings
import yaml

hardware_info = get_hardware_info()
recommended = recommend_settings(hardware_info)

print('=== 하드웨어 정보 ===')
print(f'CPU: {hardware_info[\"system\"][\"cpu_count\"]}코어')
print(f'메모리: {hardware_info[\"memory\"][\"total_gb\"]}GB')
print(f'GPU: {\"사용 가능\" if hardware_info[\"gpu\"][\"available\"] else \"없음\"}')

print('\n=== 권장 설정 ===')
for key, value in recommended.items():
    print(f'{key}: {value}')
"
    
    log_success "시스템 분석 완료"
}

# 메인 설치 함수
main() {
    echo "=================================="
    echo "🚀 Open CodeAI 설치 시작"
    echo "=================================="
    
    # 설치 옵션 파싱
    INSTALL_DEV=false
    DOWNLOAD_MODELS=false
    ENABLE_GRAPH_DB=true
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev)
                INSTALL_DEV=true
                shift
                ;;
            --download-models)
                DOWNLOAD_MODELS=true
                shift
                ;;
            --no-graph-db)
                ENABLE_GRAPH_DB=false
                shift
                ;;
            --help)
                echo "사용법: $0 [옵션]"
                echo "옵션:"
                echo "  --dev              개발 패키지도 설치"
                echo "  --download-models  AI 모델 자동 다운로드"
                echo "  --no-graph-db      Graph DB 설치 건너뛰기"
                echo "  --help             도움말 표시"
                exit 0
                ;;
            *)
                log_error "알 수 없는 옵션: $1"
                exit 1
                ;;
        esac
    done
    
    # 설치 단계별 실행
    detect_system
    install_system_dependencies
    setup_python_environment
    install_python_packages
    setup_tree_sitter
    setup_databases
    setup_environment_file
    download_models
    verify_installation
    prepare_continue_plugin
    optimize_system
    
    echo "=================================="
    echo "✅ Open CodeAI 설치 완료!"
    echo "=================================="
    echo ""
    echo "다음 단계:"
    echo "1. 서버 시작: ./scripts/start.sh"
    echo "2. VS Code에서 Continue 확장 설치"
    echo "3. 프로젝트 인덱싱: python scripts/index_project.py --project-path /your/project"
    echo ""
    echo "문서: https://github.com/ChangooLee/open-codeai/wiki"
    echo "문제 해결: https://github.com/ChangooLee/open-codeai/issues"
}

# 스크립트 실행
main "$@"