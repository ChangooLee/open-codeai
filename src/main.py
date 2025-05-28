"""
Open CodeAI - 메인 FastAPI 애플리케이션
Continue.dev 호환 AI 코드 어시스턴트 서버
"""
import os
import sys
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import time

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import settings
from src.api.openai_api import router as openai_router
from src.core.llm_manager import get_llm_manager
from src.utils.logger import get_logger, setup_logging
from src.utils.hardware import get_hardware_info

# 로거 설정
setup_logging()
logger = get_logger(__name__)

# 애플리케이션 라이프사이클 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행되는 코드"""
    
    # 시작 시
    logger.info("🚀 Open CodeAI 서버 시작 중...")
    
    # 하드웨어 정보 로깅
    hardware_info = get_hardware_info()
    logger.info(f"💻 하드웨어 정보: CPU={hardware_info['cpu']['name']}, "
               f"메모리={hardware_info['memory']['total_gb']:.1f}GB")
    
    if hardware_info['gpu']['available']:
        for i, gpu in enumerate(hardware_info['gpu']['devices']):
            logger.info(f"🎮 GPU {i}: {gpu['name']}, 메모리={gpu['memory_gb']:.1f}GB")
    else:
        logger.info("🎮 GPU: 사용 불가")
    
    # LLM 관리자 초기화 (백그라운드에서)
    try:
        llm_manager = get_llm_manager()
        logger.success("✅ LLM 관리자 초기화 완료")
    except Exception as e:
        logger.warning(f"⚠️ LLM 관리자 초기화 실패 (더미 모드로 실행): {e}")
    
    logger.success("🎉 Open CodeAI 서버 시작 완료!")
    
    yield
    
    # 종료 시
    logger.info("🛑 Open CodeAI 서버 종료 중...")
    logger.info("👋 안녕히 가세요!")

# FastAPI 앱 생성
app = FastAPI(
    title="Open CodeAI",
    description="Continue.dev 호환 AI 코드 어시스턴트",
    version=settings.VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청 로깅 미들웨어"""
    start_time = time.time()
    
    # 헬스체크는 로깅하지 않음
    if request.url.path in ["/health", "/v1/health"]:
        response = await call_next(request)
        return response
    
    # 요청 정보 로깅
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"📨 {request.method} {request.url.path} - IP: {client_ip}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # 응답 정보 로깅
        logger.info(f"📤 {response.status_code} {request.url.path} - {process_time:.3f}s")
        
        # 성능 헤더 추가
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"💥 ERROR {request.url.path} - {process_time:.3f}s: {e}")
        
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

# API 라우터 등록
app.include_router(openai_router, prefix="/v1", tags=["OpenAI Compatible API"])

# 루트 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def root():
    """루트 페이지 - 서버 정보 표시"""
    
    hardware_info = get_hardware_info()
    llm_manager = get_llm_manager()
    model_info = llm_manager.get_model_info()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Open CodeAI</title>
        <meta charset="utf-8">
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px; 
                margin: 50px auto; 
                padding: 20px;
                background: #f5f5f5;
            }}
            .header {{ 
                text-align: center; 
                margin-bottom: 40px;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .section {{ 
                background: white; 
                margin: 20px 0; 
                padding: 20px; 
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .status {{ 
                display: inline-block; 
                padding: 5px 10px; 
                border-radius: 20px; 
                font-size: 12px; 
                font-weight: bold;
            }}
            .status.running {{ background: #d4edda; color: #155724; }}
            .status.dummy {{ background: #fff3cd; color: #856404; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            pre {{ background: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            .emoji {{ font-size: 1.2em; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 Open CodeAI</h1>
            <p>Continue.dev 호환 AI 코드 어시스턴트</p>
            <span class="status {'running' if model_info['main_model']['loaded'] else 'dummy'}">
                {'🟢 모델 로드됨' if model_info['main_model']['loaded'] else '🟡 더미 모드'}
            </span>
        </div>
        
        <div class="section">
            <h2>📊 시스템 정보</h2>
            <div class="grid">
                <div>
                    <strong>🖥️ CPU:</strong> {hardware_info['cpu']['name']}<br>
                    <strong>💾 메모리:</strong> {hardware_info['memory']['total_gb']:.1f} GB<br>
                    <strong>🎮 GPU:</strong> {'사용 가능' if hardware_info['gpu']['available'] else '사용 불가'}
                </div>
                <div>
                    <strong>🐍 Python:</strong> {sys.version.split()[0]}<br>
                    <strong>🚀 버전:</strong> {settings.VERSION}<br>
                    <strong>🔧 디버그:</strong> {'On' if settings.DEBUG else 'Off'}
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🤖 모델 정보</h2>
            <div class="grid">
                <div>
                    <strong>메인 모델:</strong><br>
                    {'✅ ' + model_info['main_model']['name'] if model_info['main_model']['loaded'] else '❌ 미로드'}<br>
                    <strong>임베딩 모델:</strong><br>
                    {'✅ ' + model_info['embedding_model']['name'] if model_info['embedding_model']['loaded'] else '❌ 미로드'}
                </div>
                <div>
                    <strong>디바이스:</strong> {model_info['hardware']['device']}<br>
                    <strong>GPU 메모리:</strong> {hardware_info['gpu']['devices'][0]['memory_gb']:.1f}GB if hardware_info['gpu']['devices'] else 'N/A'}<br>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🔌 API 엔드포인트</h2>
            <ul>
                <li><strong>채팅 완성:</strong> <code>POST /v1/chat/completions</code></li>
                <li><strong>텍스트 완성:</strong> <code>POST /v1/completions</code></li>
                <li><strong>임베딩:</strong> <code>POST /v1/embeddings</code></li>
                <li><strong>모델 목록:</strong> <code>GET /v1/models</code></li>
                <li><strong>헬스체크:</strong> <code>GET /v1/health</code></li>
            </ul>
        </div>
        
        <div class="section">
            <h2>⚙️ Continue.dev 설정</h2>
            <p>VS Code 또는 JetBrains IDE에서 Continue 확장을 설치하고 다음 설정을 사용하세요:</p>
            <pre><code>{{
  "models": [
    {{
      "title": "Open CodeAI",
      "provider": "openai",
      "model": "open-codeai",
      "apiKey": "open-codeai-local-key",
      "apiBase": "http://localhost:8000/v1"
    }}
  ]
}}</code></pre>
        </div>
        
        <div class="section">
            <h2>📚 유용한 링크</h2>
            <ul>
                <li><a href="/docs" target="_blank">📖 API 문서</a></li>
                <li><a href="/v1/health" target="_blank">🏥 헬스체크</a></li>
                <li><a href="https://docs.continue.dev" target="_blank">📘 Continue.dev 문서</a></li>
                <li><a href="https://github.com/ChangooLee/open-codeai" target="_blank">💻 GitHub 저장소</a></li>
            </ul>
        </div>
        
        <div class="section">
            <h2>🚀 빠른 시작</h2>
            <ol>
                <li>VS Code에서 Continue 확장 설치</li>
                <li>Continue 설정에서 위의 API 설정 추가</li>
                <li>Ctrl+Shift+L을 눌러 채팅 시작</li>
                <li>코드를 선택하고 질문하거나 수정 요청</li>
            </ol>
        </div>
    </body>
    </html>
    """
    
    return html_content

@app.get("/status")
async def get_status():
    """서버 상태 정보"""
    
    hardware_info = get_hardware_info()
    llm_manager = get_llm_manager()
    model_info = llm_manager.get_model_info()
    
    return {
        "status": "running",
        "version": settings.VERSION,
        "timestamp": time.time(),
        "hardware": {
            "cpu_cores": hardware_info['cpu']['cores'],
            "memory_gb": hardware_info['memory']['total_gb'],
            "gpu_available": hardware_info['gpu']['available'],
            "gpu_count": len(hardware_info['gpu']['devices'])
        },
        "models": {
            "main_loaded": model_info['main_model']['loaded'],
            "embedding_loaded": model_info['embedding_model']['loaded'],
            "device": model_info['hardware']['device']
        },
        "config": {
            "debug": settings.DEBUG,
            "max_files": settings.project.max_files if hasattr(settings, 'project') else 10000,
            "supported_languages": settings.project.supported_extensions if hasattr(settings, 'project') else []
        }
    }

# 에러 핸들러
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 예외 핸들러"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 핸들러"""
    logger.error(f"Unexpected error: {exc} - {request.url}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

# 정적 파일 서빙 (선택적)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

def main():
    """메인 함수 - 서버 실행"""
    
    # 환경 변수 및 설정 확인
    logger.info("🔧 Open CodeAI 서버 설정 확인 중...")
    logger.info(f"프로젝트: {settings.PROJECT_NAME}")
    logger.info(f"버전: {settings.VERSION}")
    logger.info(f"환경: {settings.ENVIRONMENT}")
    logger.info(f"디버그: {settings.DEBUG}")
    logger.info(f"호스트: {settings.HOST}:{settings.PORT}")
    
    # 필수 디렉토리 생성
    settings.ensure_directories()
    
    # 하드웨어 정보 출력
    hardware_info = get_hardware_info()
    logger.info(f"🖥️ CPU: {hardware_info['cpu']['cores']}코어")
    logger.info(f"💾 메모리: {hardware_info['memory']['total_gb']:.1f}GB")
    logger.info(f"🎮 GPU: {'사용 가능' if hardware_info['gpu']['available'] else '사용 불가'}")
    
    # 모델 경로 확인
    if hasattr(settings, 'llm') and settings.llm:
        model_path = settings.llm.main_model.path
        if os.path.exists(model_path):
            logger.success(f"✅ 메인 모델 경로 확인: {model_path}")
        else:
            logger.warning(f"⚠️ 메인 모델 경로 없음: {model_path}")
            logger.warning("더미 모드로 실행됩니다. 모델을 다운로드하려면:")
            logger.warning("python scripts/download_models.py --config configs/models.yaml")
    
    # 서버 시작 메시지
    logger.info("=" * 60)
    logger.info("🚀 Open CodeAI 서버 시작")
    logger.info("=" * 60)
    logger.info(f"📡 서버 URL: http://{settings.HOST}:{settings.PORT}")
    logger.info(f"📚 API 문서: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"🔌 Continue.dev 연결: http://{settings.HOST}:{settings.PORT}/v1")
    logger.info("=" * 60)
    logger.info("종료하려면 Ctrl+C를 누르세요")
    logger.info("")
    
    # 서버 실행
    try:
        uvicorn.run(
            "src.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG,
            log_level="info" if settings.DEBUG else "warning",
            access_log=settings.DEBUG,
            workers=1,  # 개발 환경에서는 1개 워커만 사용
            loop="asyncio"
        )
    except KeyboardInterrupt:
        logger.info("👋 서버 종료 중...")
    except Exception as e:
        logger.error(f"💥 서버 실행 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()