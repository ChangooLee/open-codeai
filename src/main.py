"""
Open CodeAI - 메인 FastAPI 애플리케이션
RAG, Function Calling, Continue.dev 완전 통합 버전
"""
import os
import sys
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import time

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings
from api.openai_compatible import router as openai_router
from core.llm_manager import get_llm_manager
from core.rag_system import get_rag_system
from core.function_calling import get_function_registry
from utils.logger import get_logger, setup_logging
from utils.hardware import get_hardware_info

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
    logger.info(f"💻 시스템: {hardware_info['system']['platform']}")
    logger.info(f"💾 메모리: {hardware_info['memory']['total_gb']:.1f}GB")
    logger.info(f"💿 디스크: {hardware_info['disk']['free_gb']:.1f}GB 사용 가능")
    
    if hardware_info['gpu']['available']:
        for i, gpu in enumerate(hardware_info['gpu']['devices']):
            logger.info(f"🎮 GPU {i}: {gpu['name']}, 메모리={gpu['memory_gb']:.1f}GB")
    else:
        logger.info("🎮 GPU: 사용 불가")
    
    # 시스템 구성 요소 초기화
    try:
        # LLM 관리자 초기화
        llm_manager = get_llm_manager()
        model_info = llm_manager.get_model_info()
        
        if model_info['main_model']['loaded']:
            logger.success("✅ LLM 관리자 초기화 완료")
        else:
            logger.warning("⚠️ LLM 더미 모드로 실행 (모델 미로드)")
        
        # RAG 시스템 초기화
        rag_system = get_rag_system()
        stats = rag_system.indexer.get_indexing_stats()
        logger.success(f"✅ RAG 시스템 초기화 완료 (파일: {stats.get('total_files', 0)}, 청크: {stats.get('total_chunks', 0)})")
        
        # Function Calling 레지스트리 초기화
        function_registry = get_function_registry()
        available_functions = function_registry.get_available_functions()
        logger.success(f"✅ Function Calling 초기화 완료 ({len(available_functions)}개 함수)")
        
        # 백그라운드 작업 시작
        app.state.background_tasks = []
        
        # 파일 와처 시작 (변경 감지용)
        if getattr(settings.monitoring, 'file_watcher', {}).get('enable', True):
            watcher_task = asyncio.create_task(start_file_watcher())
            app.state.background_tasks.append(watcher_task)
            logger.info("📁 파일 와처 시작됨")
        
    except Exception as e:
        logger.error(f"💥 시스템 초기화 중 오류: {e}")
        logger.warning("⚠️ 일부 기능이 제한될 수 있습니다")
    
    logger.success("🎉 Open CodeAI 서버 시작 완료!")
    
    yield
    
    # 종료 시
    logger.info("🛑 Open CodeAI 서버 종료 중...")
    
    # 백그라운드 작업 정리
    if hasattr(app.state, 'background_tasks'):
        for task in app.state.background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("🧹 백그라운드 작업 정리 완료")
    
    # RAG 시스템 저장
    try:
        rag_system = get_rag_system()
        rag_system.vector_db._save_index()
        rag_system.graph_db.save_graph()
        logger.info("💾 데이터베이스 저장 완료")
    except Exception as e:
        logger.error(f"💥 데이터베이스 저장 실패: {e}")
    
    logger.info("👋 안녕히 가세요!")

# 파일 와처 (실시간 인덱싱용)
async def start_file_watcher():
    """파일 변경 감지 및 자동 인덱싱"""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        import threading
        
        class CodeFileHandler(FileSystemEventHandler):
            def __init__(self):
                self.rag_system = get_rag_system()
                self.pending_files = set()
                self.lock = threading.Lock()
                
            def on_modified(self, event):
                if event.is_directory:
                    return
                
                # 지원하는 파일 확장자만 처리
                supported_extensions = getattr(settings.project, 'supported_extensions', ['.py', '.js', '.ts'])
                if any(event.src_path.endswith(ext) for ext in supported_extensions):
                    with self.lock:
                        self.pending_files.add(event.src_path)
            
            async def process_pending_files(self):
                """대기 중인 파일들 처리"""
                if not self.pending_files:
                    return
                
                with self.lock:
                    files_to_process = list(self.pending_files)
                    self.pending_files.clear()
                
                for file_path in files_to_process:
                    try:
                        await self.rag_system.indexer.index_file(file_path)
                        logger.info(f"📝 파일 자동 인덱싱: {file_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ 파일 인덱싱 실패 {file_path}: {e}")
        
        # 와처 설정
        event_handler = CodeFileHandler()
        observer = Observer()
        
        # 현재 프로젝트 디렉토리 감시
        watch_dirs = ["."]
        if hasattr(settings, 'PROJECT_PATH') and settings.PROJECT_PATH:
            watch_dirs.append(settings.PROJECT_PATH)
        
        for watch_dir in watch_dirs:
            if os.path.exists(watch_dir):
                observer.schedule(event_handler, watch_dir, recursive=True)
        
        observer.start()
        logger.info("👁️ 파일 와처 시작됨")
        
        # 주기적으로 대기 중인 파일들 처리
        while True:
            await asyncio.sleep(5)  # 5초마다 확인
            await event_handler.process_pending_files()
            
    except ImportError:
        logger.warning("⚠️ watchdog가 설치되지 않아 파일 와처를 사용할 수 없습니다")
    except Exception as e:
        logger.error(f"💥 파일 와처 오류: {e}")

# FastAPI 앱 생성
app = FastAPI(
    title="Open CodeAI",
    description="Continue.dev 호환 AI 코드 어시스턴트 (RAG + Function Calling)",
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
    """요청 로깅 및 성능 모니터링 미들웨어"""
    start_time = time.time()
    
    # 헬스체크는 로깅하지 않음
    if request.url.path in ["/health", "/v1/health", "/favicon.ico"]:
        response = await call_next(request)
        return response
    
    # 요청 정보 로깅
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")[:50]
    
    logger.info(f"📨 {request.method} {request.url.path} - IP: {client_ip}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # 응답 정보 로깅
        status_emoji = "✅" if response.status_code < 400 else "❌"
        logger.info(f"{status_emoji} {response.status_code} {request.url.path} - {process_time:.3f}s")
        
        # 성능 헤더 추가
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Open-CodeAI-Version"] = settings.VERSION
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"💥 ERROR {request.url.path} - {process_time:.3f}s: {e}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(e) if settings.DEBUG else "서버 내부 오류가 발생했습니다"
            }
        )

# API 라우터 등록
app.include_router(openai_router, prefix="/v1", tags=["OpenAI Compatible API"])

# 루트 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def root():
    """루트 페이지 - 서버 정보 및 대시보드"""
    
    try:
        hardware_info = get_hardware_info()
        llm_manager = get_llm_manager()
        model_info = llm_manager.get_model_info()
        rag_system = get_rag_system()
        index_stats = rag_system.indexer.get_indexing_stats()
        function_registry = get_function_registry()
        available_functions = function_registry.get_available_functions()
        
        # 시스템 상태 결정
        system_status = "🟢 정상"
        if not model_info['main_model']['loaded']:
            system_status = "🟡 더미 모드"
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <title>Open CodeAI Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    color: #333;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    padding: 20px;
                }}
                .header {{ 
                    text-align: center; 
                    margin-bottom: 40px;
                    background: rgba(255,255,255,0.95);
                    padding: 40px;
                    border-radius: 20px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                    backdrop-filter: blur(10px);
                }}
                .header h1 {{ 
                    font-size: 3em; 
                    margin-bottom: 10px;
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }}
                .status-badge {{ 
                    display: inline-block; 
                    padding: 10px 20px; 
                    border-radius: 25px; 
                    font-weight: bold;
                    background: rgba(76, 175, 80, 0.1);
                    color: #4CAF50;
                    border: 2px solid #4CAF50;
                }}
                .grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); 
                    gap: 25px; 
                    margin-bottom: 30px;
                }}
                .card {{ 
                    background: rgba(255,255,255,0.95); 
                    padding: 25px; 
                    border-radius: 20px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                    backdrop-filter: blur(10px);
                    transition: transform 0.3s ease;
                }}
                .card:hover {{ transform: translateY(-5px); }}
                .card h2 {{ 
                    margin-bottom: 15px; 
                    color: #667eea;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                }}
                .metric {{ 
                    display: flex; 
                    justify-content: space-between; 
                    margin: 10px 0;
                    padding: 8px 0;
                    border-bottom: 1px solid #eee;
                }}
                .metric:last-child {{ border-bottom: none; }}
                .metric-label {{ font-weight: 500; }}
                .metric-value {{ 
                    font-weight: bold; 
                    color: #667eea;
                }}
                .code {{ 
                    background: #f8f9fa; 
                    padding: 15px; 
                    border-radius: 10px; 
                    font-family: 'Monaco', 'Menlo', monospace;
                    font-size: 0.9em;
                    border-left: 4px solid #667eea;
                    overflow-x: auto;
                }}
                .function-list {{ 
                    max-height: 200px; 
                    overflow-y: auto;
                    background: #f8f9fa;
                    padding: 10px;
                    border-radius: 10px;
                }}
                .function-item {{ 
                    padding: 5px 10px;
                    margin: 5px 0;
                    background: white;
                    border-radius: 5px;
                    border-left: 3px solid #667eea;
                }}
                .links {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 15px;
                }}
                .link-card {{ 
                    background: rgba(255,255,255,0.95);
                    padding: 20px;
                    border-radius: 15px;
                    text-decoration: none;
                    color: #333;
                    text-align: center;
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
                }}
                .link-card:hover {{ 
                    transform: translateY(-3px);
                    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
                }}
                .footer {{ 
                    text-align: center; 
                    margin-top: 40px;
                    color: rgba(255,255,255,0.8);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 Open CodeAI</h1>
                    <p>Continue.dev 호환 AI 코드 어시스턴트</p>
                    <div class="status-badge">{system_status}</div>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <h2>📊 시스템 정보</h2>
                        <div class="metric">
                            <span class="metric-label">🖥️ CPU</span>
                            <span class="metric-value">{hardware_info['system']['cpu_count']}코어</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">💾 메모리</span>
                            <span class="metric-value">{hardware_info['memory']['total_gb']:.1f}GB</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">🎮 GPU</span>
                            <span class="metric-value">{'✅ 사용 가능' if hardware_info['gpu']['available'] else '❌ 사용 불가'}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">💿 디스크</span>
                            <span class="metric-value">{hardware_info['disk']['free_gb']:.1f}GB 여유</span>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>🤖 AI 모델</h2>
                        <div class="metric">
                            <span class="metric-label">메인 LLM</span>
                            <span class="metric-value">{'✅ ' + model_info['main_model']['name'] if model_info['main_model']['loaded'] else '❌ 미로드'}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">임베딩</span>
                            <span class="metric-value">{'✅ ' + model_info['embedding_model']['name'] if model_info['embedding_model']['loaded'] else '❌ 미로드'}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">디바이스</span>
                            <span class="metric-value">{model_info['hardware']['device'].upper()}</span>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>🔍 RAG 시스템</h2>
                        <div class="metric">
                            <span class="metric-label">인덱싱된 파일</span>
                            <span class="metric-value">{index_stats.get('total_files', 0):,}개</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">총 청크</span>
                            <span class="metric-value">{index_stats.get('total_chunks', 0):,}개</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">벡터 DB 크기</span>
                            <span class="metric-value">{index_stats.get('vector_db_size', 0):,}개</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">마지막 인덱싱</span>
                            <span class="metric-value">{index_stats.get('last_full_index', 'N/A')[:10] if index_stats.get('last_full_index') else 'N/A'}</span>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>⚡ Function Calling</h2>
                        <div class="metric">
                            <span class="metric-label">사용 가능한 함수</span>
                            <span class="metric-value">{len(available_functions)}개</span>
                        </div>
                        <div class="function-list">
                            {' '.join([f'<div class="function-item">🔧 {func["name"]}</div>' for func in available_functions[:10]])}
                            {f'<div class="function-item">... 그리고 {len(available_functions) - 10}개 더</div>' if len(available_functions) > 10 else ''}
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>⚙️ Continue.dev 설정</h2>
                    <p>VS Code 또는 JetBrains IDE에서 Continue 확장을 설치하고 다음 설정을 사용하세요:</p>
                    <div class="code">{{
  "models": [
    {{
      "title": "Open CodeAI",
      "provider": "openai",
      "model": "open-codeai",
      "apiKey": "open-codeai-local-key",
      "apiBase": "http://localhost:8000/v1"
    }}
  ]
}}</div>
                </div>
                
                <div class="card">
                    <h2>🔗 빠른 링크</h2>
                    <div class="links">
                        <a href="/docs" class="link-card" target="_blank">
                            📖 API 문서
                        </a>
                        <a href="/v1/health" class="link-card" target="_blank">
                            🏥 헬스체크
                        </a>
                        <a href="http://localhost:7474" class="link-card" target="_blank">
                            🗃️ Neo4j 브라우저
                        </a>
                        <a href="https://docs.continue.dev" class="link-card" target="_blank">
                            📘 Continue.dev 문서
                        </a>
                        <a href="https://github.com/ChangooLee/open-codeai" class="link-card" target="_blank">
                            💻 GitHub 저장소
                        </a>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Open CodeAI v{settings.VERSION} | 마지막 업데이트: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Continue.dev와 완벽 호환 | RAG + Function Calling 지원</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
        
    except Exception as e:
        logger.error(f"대시보드 생성 실패: {e}")
        return "<h1>Open CodeAI</h1><p>대시보드 로딩 중 오류가 발생했습니다.</p>"

@app.get("/status")
async def get_detailed_status():
    """상세 서버 상태 정보 API"""
    
    try:
        hardware_info = get_hardware_info()
        llm_manager = get_llm_manager()
        model_info = llm_manager.get_model_info()
        rag_system = get_rag_system()
        index_stats = rag_system.indexer.get_indexing_stats()
        function_registry = get_function_registry()
        available_functions = function_registry.get_available_functions()
        
        return {
            "status": "running",
            "version": settings.VERSION,
            "timestamp": time.time(),
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            "hardware": {
                "cpu_cores": hardware_info['system']['cpu_count'],
                "memory_total_gb": hardware_info['memory']['total_gb'],
                "memory_available_gb": hardware_info['memory']['available_gb'],
                "disk_free_gb": hardware_info['disk']['free_gb'],
                "gpu_available": hardware_info['gpu']['available'],
                "gpu_count": len(hardware_info['gpu']['devices'])
            },
            "models": {
                "main_loaded": model_info['main_model']['loaded'],
                "main_name": model_info['main_model']['name'],
                "embedding_loaded": model_info['embedding_model']['loaded'],
                "embedding_name": model_info['embedding_model']['name'],
                "device": model_info['hardware']['device']
            },
            "rag_system": {
                "indexed_files": index_stats.get('total_files', 0),
                "total_chunks": index_stats.get('total_chunks', 0),
                "vector_db_size": index_stats.get('vector_db_size', 0),
                "graph_db_nodes": index_stats.get('graph_db_nodes', 0),
                "last_index": index_stats.get('last_full_index'),
                "languages": index_stats.get('languages', [])
            },
            "functions": {
                "available_count": len(available_functions),
                "function_names": [f["name"] for f in available_functions]
            },
            "config": {
                "debug": settings.DEBUG,
                "environment": settings.ENVIRONMENT,
                "max_files": getattr(settings.project, 'max_files', 10000),
                "supported_languages": getattr(settings.project, 'supported_extensions', [])
            }
        }
        
    except Exception as e:
        logger.error(f"상태 조회 실패: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

# 백그라운드 작업 엔드포인트들
@app.post("/admin/index-project")
async def admin_index_project(
    project_path: str,
    max_files: int = 1000,
    background_tasks: BackgroundTasks = None
):
    """관리자용 프로젝트 인덱싱 API"""
    
    if not settings.DEBUG:
        raise HTTPException(status_code=403, detail="관리자 기능은 디버그 모드에서만 사용 가능합니다")
    
    try:
        rag_system = get_rag_system()
        
        # 백그라운드에서 인덱싱 실행
        async def index_task():
            try:
                result = await rag_system.indexer.index_directory(project_path, max_files)
                logger.info(f"백그라운드 인덱싱 완료: {project_path} - {result.get('success_count', 0)} 파일")
            except Exception as e:
                logger.error(f"백그라운드 인덱싱 실패: {e}")
        
        if background_tasks:
            background_tasks.add_task(index_task)
            
        return {
            "status": "started",
            "message": f"프로젝트 인덱싱이 백그라운드에서 시작되었습니다: {project_path}",
            "project_path": project_path,
            "max_files": max_files
        }
        
    except Exception as e:
        logger.error(f"인덱싱 시작 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/clear-index")
async def admin_clear_index():
    """관리자용 인덱스 초기화 API"""
    
    if not settings.DEBUG:
        raise HTTPException(status_code=403, detail="관리자 기능은 디버그 모드에서만 사용 가능합니다")
    
    try:
        rag_system = get_rag_system()
        
        # 벡터 DB 초기화
        rag_system.vector_db.chunk_map.clear()
        rag_system.vector_db.file_index.clear()
        
        # 그래프 DB 초기화
        if hasattr(rag_system.graph_db, 'graph'):
            rag_system.graph_db.graph.clear()
        
        logger.warning("관리자에 의해 인덱스가 초기화되었습니다")
        
        return {
            "status": "success",
            "message": "모든 인덱스가 초기화되었습니다"
        }
        
    except Exception as e:
        logger.error(f"인덱스 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket 지원 (실시간 로그, 진행상황 등)
@app.websocket("/ws/logs")
async def websocket_logs(websocket):
    """실시간 로그 스트리밍 WebSocket"""
    await websocket.accept()
    
    try:
        # 실제 구현에서는 로그 큐나 파일 tail을 사용
        import asyncio
        while True:
            # 예시: 주기적으로 상태 정보 전송
            status_data = {
                "timestamp": time.time(),
                "type": "status",
                "data": {
                    "memory_usage": "12.3GB/32GB",
                    "active_requests": 0,
                    "last_activity": "idle"
                }
            }
            await websocket.send_json(status_data)
            await asyncio.sleep(10)
            
    except Exception as e:
        logger.error(f"WebSocket 연결 오류: {e}")
    finally:
        await websocket.close()

# 에러 핸들러들
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 예외 핸들러"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 핸들러"""
    logger.error(f"Unexpected error: {exc} - {request.url}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "서버 내부 오류가 발생했습니다",
            "status_code": 500,
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )

# 정적 파일 서빙
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

def main():
    """메인 함수 - 서버 실행"""
    
    # 시작 시간 기록
    app.state.start_time = time.time()
    
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
    logger.info(f"🖥️ CPU: {hardware_info['system']['cpu_count']}코어")
    logger.info(f"💾 메모리: {hardware_info['memory']['total_gb']:.1f}GB")
    logger.info(f"🎮 GPU: {'사용 가능' if hardware_info['gpu']['available'] else '사용 불가'}")
    
    # 성능 프로필 확인
    from .utils.hardware import get_performance_profile
    profile = get_performance_profile()
    logger.info(f"📊 성능 프로필: {profile}")
    
    # 모델 경로 확인
    if hasattr(settings, 'llm') and settings.llm:
        model_path = settings.llm.main_model.path
        if os.path.exists(model_path):
            logger.success(f"✅ 메인 모델 경로 확인: {model_path}")
        else:
            logger.warning(f"⚠️ 메인 모델 경로 없음: {model_path}")
            logger.warning("더미 모드로 실행됩니다. 모델을 다운로드하려면:")
            logger.warning("python scripts/download_models.py")
    
    # Continue.dev 설정 확인
    continue_config_path = os.path.expanduser("~/.continue/config.json")
    if os.path.exists(continue_config_path):
        logger.success("✅ Continue.dev 설정 파일 확인됨")
    else:
        logger.warning("⚠️ Continue.dev 설정 파일이 없습니다")
        logger.warning("다음 명령으로 설정하세요: cp examples/continue_config.json ~/.continue/config.json")
    
    # 서버 시작 메시지
    logger.info("=" * 70)
    logger.info("🚀 Open CodeAI 서버 시작")
    logger.info("=" * 70)
    logger.info(f"📡 API 서버: http://{settings.HOST}:{settings.PORT}")
    logger.info(f"📚 API 문서: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"🔌 Continue.dev: http://{settings.HOST}:{settings.PORT}/v1")
    logger.info(f"🗃️  Neo4j 브라우저: http://localhost:7474")
    logger.info(f"🎛️  대시보드: http://{settings.HOST}:{settings.PORT}")
    logger.info("=" * 70)
    logger.info("✨ 주요 기능:")
    logger.info("   🔍 RAG (Retrieval-Augmented Generation)")
    logger.info("   ⚡ Function Calling")
    logger.info("   🔌 Continue.dev 완전 호환")
    logger.info("   📊 실시간 파일 와처")
    logger.info("   🗃️  벡터 + 그래프 DB")
    logger.info("=" * 70)
    logger.info("종료하려면 Ctrl+C를 누르세요")
    logger.info("")
    
    # 서버 실행
    try:
        # 서버 설정 최적화
        workers = 1  # 단일 워커 (메모리 효율성)
        if hasattr(settings, 'server') and settings.server:
            workers = getattr(settings.server, 'workers', 1)
        
        uvicorn.run(
            "src.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG and workers == 1,  # 리로드는 단일 워커에서만
            log_level="info" if settings.DEBUG else "warning",
            access_log=settings.DEBUG,
            workers=workers,
            loop="asyncio",
            # 성능 최적화 설정
            backlog=2048,
            timeout_keep_alive=65,
            limit_concurrency=1000,
            limit_max_requests=10000
        )
    except KeyboardInterrupt:
        logger.info("👋 서버 종료 중...")
    except Exception as e:
        logger.error(f"💥 서버 실행 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()