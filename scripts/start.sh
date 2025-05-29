#!/bin/bash

# Open CodeAI 서버 시작 스크립트

set -e

# 색깔 출력
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 서버 시작 함수
start_server() {
    log_info "🚀 Open CodeAI 서버 시작 중..."
    
    # 가상환경 활성화
    if [ -d "venv" ]; then
        source venv/bin/activate
        log_info "Python 가상환경 활성화됨"
    else
        log_error "Python 가상환경을 찾을 수 없습니다. ./scripts/install.sh를 먼저 실행하세요."
        exit 1
    fi
    
    # 환경변수 확인
    if [ ! -f ".env" ]; then
        log_warning ".env 파일이 없습니다. .env.example을 복사합니다."
        cp .env.example .env
    fi
    
    # Docker 서비스 확인 및 시작
    if command -v docker &> /dev/null; then
        if ! docker info >/dev/null 2>&1; then
            log_info "Docker 서비스 시작 중..."
            if command -v systemctl &> /dev/null; then
                sudo systemctl start docker
            elif command -v service &> /dev/null; then
                sudo service docker start
            fi
        fi
        
        # Neo4j 컨테이너 시작
        if [ -f "docker-compose.yml" ]; then
            log_info "Neo4j 데이터베이스 시작 중..."
            docker-compose up -d neo4j
            sleep 5  # Neo4j 초기화 대기
        fi
    fi
    
    # 시스템 상태 확인
    log_info "시스템 상태 확인 중..."
    python -c "
from src.utils.hardware import get_hardware_info, check_requirements
import json

hardware_info = get_hardware_info()
requirements = check_requirements()

print('=== 시스템 정보 ===')
print(f'🖥️  CPU: {hardware_info[\"system\"][\"cpu_count\"]}코어')
print(f'💾 메모리: {hardware_info[\"memory\"][\"total_gb\"]}GB')
print(f'🎮 GPU: {\"✅\" if hardware_info[\"gpu\"][\"available\"] else \"❌\"} ({hardware_info[\"gpu\"][\"count\"]}개)')
print(f'💿 디스크: {hardware_info[\"disk\"][\"free_gb\"]}GB 사용 가능')

print('\n=== 요구사항 확인 ===')
for req, status in requirements.items():
    status_icon = '✅' if status else '❌'
    print(f'{status_icon} {req}')
"
    
    # 모델 파일 확인
    log_info "모델 파일 확인 중..."
    if [ -d "data/models" ] && [ "$(ls -A data/models 2>/dev/null)" ]; then
        log_success "모델 파일 발견됨"
    else
        log_warning "모델 파일이 없습니다. 다음 명령으로 다운로드하세요:"
        log_warning "python scripts/download_models.py --config configs/models.yaml"
    fi
    
    # FastAPI 서버 시작
    log_info "FastAPI 서버 시작 중..."
    echo "=================================="
    echo "🚀 Open CodeAI 서버가 시작됩니다"
    echo "=================================="
    echo "📡 API 서버: http://localhost:8000"
    echo "📚 API 문서: http://localhost:8000/docs"
    echo "🔌 Continue.dev 연결: ws://localhost:8001"
    echo "=================================="
    echo "종료하려면 Ctrl+C를 누르세요"
    echo ""
    
    # 서버 실행
    python src/main.py
}

# 상태 확인 함수
check_status() {
    log_info "Open CodeAI 서버 상태 확인 중..."
    
    # API 서버 상태 확인
    if curl -s http://localhost:8000/ >/dev/null 2>&1; then
        log_success "✅ API 서버: 실행 중 (http://localhost:8000)"
    else
        log_warning "❌ API 서버: 중지됨"
    fi
    
    # Neo4j 상태 확인
    if docker ps | grep -q "open-codeai-neo4j"; then
        log_success "✅ Neo4j: 실행 중"
    else
        log_warning "❌ Neo4j: 중지됨"
    fi
    
    # Continue.dev 연결 확인
    if nc -z localhost 8001 2>/dev/null; then
        log_success "✅ Continue.dev 포트: 열림 (8001)"
    else
        log_warning "❌ Continue.dev 포트: 닫힘"
    fi
}

# 서버 중지 함수
stop_server() {
    log_info "Open CodeAI 서버 중지 중..."
    
    # FastAPI 서버 중지
    pkill -f "python src/main.py" 2>/dev/null || true
    
    # Docker 컨테이너 중지
    if [ -f "docker-compose.yml" ]; then
        docker-compose down
    fi
    
    log_success "서버가 중지되었습니다"
}

# 로그 확인 함수
show_logs() {
    log_info "최근 로그 표시..."
    
    if [ -f "logs/opencodeai.log" ]; then
        tail -n 50 logs/opencodeai.log
    else
        log_warning "로그 파일을 찾을 수 없습니다"
    fi
}

# 메인 함수
main() {
    case "${1:-start}" in
        "start")
            start_server
            ;;
        "stop")
            stop_server
            ;;
        "restart")
            stop_server
            sleep 2
            start_server
            ;;
        "status")
            check_status
            ;;
        "logs")
            show_logs
            ;;
        "help"|"--help"|"-h")
            echo "사용법: $0 [명령]"
            echo ""
            echo "명령:"
            echo "  start    서버 시작 (기본값)"
            echo "  stop     서버 중지"
            echo "  restart  서버 재시작"
            echo "  status   서버 상태 확인"
            echo "  logs     최근 로그 표시"
            echo "  help     도움말 표시"
            ;;
        *)
            log_error "알 수 없는 명령: $1"
            echo "도움말: $0 help"
            exit 1
            ;;
    esac
}

# 스크립트 실행 권한 확인
if [ ! -x "$0" ]; then
    chmod +x "$0"
fi

# 메인 실행
main "$@"