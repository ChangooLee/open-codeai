#!/usr/bin/env python3
"""
Open CodeAI 간단 시작 스크립트
모델 없이도 바로 테스트할 수 있는 더미 모드 지원
"""
import os
import sys
import asyncio
import subprocess
from pathlib import Path

def check_dependencies():
    """필수 의존성 확인"""
    print("🔍 의존성 확인 중...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "pydantic-settings",
        "pyyaml",
        "loguru"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n📦 다음 패키지를 설치해야 합니다:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_minimal_config():
    """최소한의 설정 파일 생성"""
    print("⚙️ 기본 설정 파일 생성 중...")
    
    # configs 디렉토리 생성
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # 기본 config.yaml 생성
    config_content = """# Open CodeAI 기본 설정
project:
  name: "open-codeai"
  version: "1.0.0"
  max_files: 10000
  supported_extensions:
    - ".py"
    - ".js"
    - ".ts"
    - ".java"
    - ".cpp"
    - ".c"
    - ".go"
    - ".rs"

server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  reload: false
  log_level: "INFO"

# 더미 모드용 기본 LLM 설정
llm:
  main_model:
    name: "dummy-model"
    path: "./data/models/dummy"
    context_window: 4096
    max_tokens: 1024
    temperature: 0.1
    use_vllm: false
    
  embedding_model:
    name: "dummy-embedding"
    path: "./data/models/dummy-embedding"
    dimension: 1024
    batch_size: 32
    device: "cpu"
    
  graph_analysis_model:
    name: "dummy-graph"
    path: "./data/models/dummy-graph"
    enable: false

database:
  vector:
    type: "faiss"
    path: "./data/vector_index"
    index_type: "HNSW"
    memory_limit_gb: 4
    
  metadata:
    type: "sqlite"
    path: "./data/metadata/opencodeai.db"
    connection_pool_size: 10

indexing:
  strategy: "hybrid"
  chunk_size: 1000
  parallel_workers: 4
  batch_size: 50

continue_integration:
  enabled: true
  port: 8001
  auth_token: "open-codeai-secure-token"
  features:
    chat: true
    autocomplete: true
    code_review: true
"""
    
    config_path = configs_dir / "config.yaml"
    if not config_path.exists():
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        print(f"✅ 설정 파일 생성: {config_path}")
    else:
        print(f"ℹ️ 설정 파일 이미 존재: {config_path}")

def setup_env_file():
    """환경 변수 파일 생성"""
    print("🌍 환경 변수 파일 생성 중...")
    
    env_content = """# Open CodeAI 환경 변수
PROJECT_NAME=open-codeai
VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true

HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# 더미 모드 경로
MODEL_PATH=./data/models/dummy
VECTOR_INDEX_PATH=./data/vector_index
METADATA_DB_PATH=./data/metadata/opencodeai.db

API_KEY=open-codeai-local-key
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(env_content)
        print(f"✅ 환경 변수 파일 생성: {env_path}")
    else:
        print(f"ℹ️ 환경 변수 파일 이미 존재: {env_path}")

def create_directory_structure():
    """필요한 디렉토리 구조 생성"""
    print("📁 디렉토리 구조 생성 중...")
    
    directories = [
        "data/models",
        "data/vector_index",
        "data/metadata",
        "data/cache",
        "data/logs",
        "logs",
        "static",
        "src/api",
        "src/core",
        "src/utils",
        "scripts",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")

def create_minimal_files():
    """최소한의 파일들 생성"""
    print("📄 기본 파일 생성 중...")
    
    # __init__.py 파일들 생성
    init_files = [
        "src/__init__.py",
        "src/api/__init__.py", 
        "src/core/__init__.py",
        "src/utils/__init__.py"
    ]
    
    for init_file in init_files:
        init_path = Path(init_file)
        if not init_path.exists():
            init_path.touch()
            print(f"✅ {init_file}")

def start_server():
    """서버 시작"""
    print("\n🚀 Open CodeAI 서버 시작 중...")
    print("=" * 50)
    print("📡 서버 URL: http://localhost:8000")
    print("📚 API 문서: http://localhost:8000/docs")
    print("🔌 Continue.dev: http://localhost:8000/v1")
    print("=" * 50)
    print("종료하려면 Ctrl+C를 누르세요\n")
    
    try:
        # Python 경로에 현재 디렉토리 추가
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # FastAPI 앱 실행
        import uvicorn
        
        # 서버 설정
        config = uvicorn.Config(
            "src.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # 개발 모드
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        server.run()
        
    except KeyboardInterrupt:
        print("\n👋 서버를 종료합니다...")
    except ImportError as e:
        print(f"❌ 모듈 import 오류: {e}")
        print("현재 디렉토리에서 실행하고 있는지 확인하세요.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        sys.exit(1)

def show_continue_setup():
    """Continue.dev 설정 방법 표시"""
    print("\n🔌 Continue.dev 설정 방법:")
    print("=" * 40)
    print("1. VS Code에서 Continue 확장 설치")
    print("2. Continue 설정 파일에 다음 추가:")
    print()
    print('''{
  "models": [
    {
      "title": "Open CodeAI",
      "provider": "openai",
      "model": "open-codeai",
      "apiKey": "open-codeai-local-key",
      "apiBase": "http://localhost:8000/v1"
    }
  ]
}''')
    print()
    print("3. Ctrl+Shift+L로 채팅 시작!")
    print("=" * 40)

def main():
    """메인 함수"""
    print("🤖 Open CodeAI 빠른 시작")
    print("=" * 30)
    
    # 의존성 확인
    if not check_dependencies():
        print("\n❌ 필수 패키지가 설치되지 않았습니다.")
        print("다음 명령으로 설치하세요:")
        print("pip install fastapi uvicorn pydantic pydantic-settings pyyaml loguru")
        sys.exit(1)
    
    # 기본 설정
    setup_minimal_config()
    setup_env_file()
    create_directory_structure()
    create_minimal_files()
    
    print("\n✅ 초기 설정 완료!")
    
    # Continue.dev 설정 안내
    show_continue_setup()
    
    # 서버 시작 여부 확인
    response = input("\n서버를 시작하시겠습니까? (y/N): ").lower().strip()
    
    if response in ['y', 'yes']:
        start_server()
    else:
        print("\n서버를 시작하려면 다음 명령을 실행하세요:")
        print("python quick_start.py")
        print("또는")
        print("python src/main.py")

if __name__ == "__main__":
    main()