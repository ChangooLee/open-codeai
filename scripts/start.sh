#!/bin/bash

# Open CodeAI 서버 시작 스크립트

set -e

# 색깔 출력
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

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

# 서버 시작 함수
start_services() {
    log_info "🚀 Open CodeAI 서비스 시작 중..."
    cd "$(dirname "$0")/.."
    
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
    echo "📡 API 서버: http://localhost:8800"
    echo "📚 API 문서: http://localhost:8800/docs"
    echo "🔌 Continue.dev: ws://localhost:8801"
    echo "🗃️  Neo4j 브라우저: http://localhost:8747"
    echo "=================================="
    echo "이 서비스는 완전 오프라인 환경에서도 동작합니다."
    echo "만약 서비스가 정상적으로 뜨지 않으면 logs/opencodeai.log 또는 'docker logs api'로 원인을 확인하세요."
    echo "종료하려면 Ctrl+C를 누르세요"
    echo ""
    # Docker 컨테이너에서 이미 API 서버가 실행 중이므로, 아래 라인은 주석 처리합니다.
    # PYTHONPATH=. python src/main.py
}

# 상태 확인 함수
check_status() {
    log_info "Open CodeAI 서버 상태 확인 중..."
    
    # API 서버 상태 확인
    if curl -s http://localhost:8800/ >/dev/null 2>&1; then
        log_success "✅ API 서버: 실행 중 (http://localhost:8800)"
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
    if nc -z localhost 8801 2>/dev/null; then
        log_success "✅ Continue.dev 포트: 열림 (8801)"
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
            start_services
            ;;
        "stop")
            stop_server
            ;;
        "restart")
            stop_server
            sleep 2
            start_services
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