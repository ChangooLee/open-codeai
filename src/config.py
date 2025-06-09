"""
Open CodeAI 설정 관리 모듈 (업데이트됨)

YAML 설정 파일과 환경변수를 통합하여 관리
하드웨어 정보와 통합된 설정 시스템
"""
import os
import sys
from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ProjectConfig(BaseModel):
    """프로젝트 기본 설정"""
    name: str = "open-codeai"
    version: str = "1.0.0"
    max_files: int = 10000
    supported_extensions: List[str] = [
        ".py", ".js", ".ts", ".java", ".cpp", ".c", 
        ".go", ".rs", ".php", ".rb", ".scala", ".kt"
    ]


class ServerConfig(BaseModel):
    """서버 설정"""
    host: str = "0.0.0.0"
    port: int = 8800
    workers: int = 1
    reload: bool = False
    log_level: str = "INFO"


class LLMModelConfig(BaseModel):
    """LLM 모델 설정"""
    name: str
    path: str
    context_window: int = 32768
    max_tokens: int = 4096
    temperature: float = 0.1
    gpu_memory_fraction: float = 0.7
    use_vllm: bool = False


class EmbeddingModelConfig(BaseModel):
    """임베딩 모델 설정"""
    name: str
    path: str
    dimension: int = 1024
    batch_size: int = 64
    device: str = "cuda"


class GraphModelConfig(BaseModel):
    """그래프 분석 모델 설정"""
    name: str
    path: str
    enable: bool = True


class LLMConfig(BaseModel):
    """LLM 통합 설정"""
    main_model: LLMModelConfig
    embedding_model: EmbeddingModelConfig
    graph_analysis_model: GraphModelConfig


class DatabaseConfig(BaseModel):
    """데이터베이스 설정"""
    vector: Dict[str, Any]
    graph: Dict[str, Any]
    metadata: Dict[str, Any]


class IndexingConfig(BaseModel):
    """인덱싱 설정"""
    strategy: str = "hybrid"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    parallel_workers: int = 16
    batch_size: int = 100
    memory_per_worker_gb: int = 4


class PerformanceConfig(BaseModel):
    """성능 최적화 설정"""
    cache: Dict[str, Any]
    memory: Dict[str, Any]
    gpu: Dict[str, Any]


class MonitoringConfig(BaseModel):
    """모니터링 설정"""
    file_watcher: Dict[str, Any]
    metrics: Dict[str, Any]


class ContinueIntegrationConfig(BaseModel):
    """Continue.dev 통합 설정"""
    enabled: bool = True
    connection_type: str = "websocket"
    port: int = 8001
    auth_token: str = "open-codeai-secure-token"
    features: Dict[str, bool]


class Settings:
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    VECTOR_DB_PORT: int = int(os.getenv("VECTOR_DB_PORT", 9000))
    GRAPH_DB_PORT: int = int(os.getenv("GRAPH_DB_PORT", 7687))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "open-codeai")
    VERSION: str = os.getenv("VERSION", "1.0.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1")
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
    GPU_MEMORY_FRACTION: float = float(os.getenv("GPU_MEMORY_FRACTION", 0.7))
    USE_MIXED_PRECISION: bool = os.getenv("USE_MIXED_PRECISION", "True").lower() in ("true", "1")
    VECTOR_INDEX_PATH: str = os.getenv("VECTOR_INDEX_PATH", "./data/vector_index")
    GRAPH_DB_PATH: str = os.getenv("GRAPH_DB_PATH", "./data/graph_db")
    METADATA_DB_PATH: str = os.getenv("METADATA_DB_PATH", "./data/metadata/metadata.db")    
    PROJECT_ROOT: str = os.getenv("PROJECT_ROOT", "/workspace")
    SUPPORTED_EXTENSIONS: str = os.getenv("SUPPORTED_EXTENSIONS", "")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))

    def ensure_directories(self) -> None:
        """필요한 디렉토리들 생성"""
        directories = [
            "data/models",
            "data/vector_index", 
            "data/graph_db",
            "data/metadata",
            "data/cache",
            "data/logs",
            "logs"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 필요한 디렉토리들이 생성되었습니다: {', '.join(directories)}")

# 전역 설정 인스턴스 생성
settings = Settings()
settings.ensure_directories()

# 하위 호환성을 위한 별칭들
PROJECT_NAME = settings.PROJECT_NAME

def setup_logging() -> None:
    """로깅 시스템 설정"""
    try:
        from .utils.logger import setup_logger
        setup_logger(
            log_level=settings.LOG_LEVEL,
            enable_file_logging=True
        )
    except ImportError:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

# === 인덱싱 확장자 처리 안내 ===
# ProjectConfig.supported_extensions는 기본값이며,
# 실제 인덱싱에서는 Settings.SUPPORTED_EXTENSIONS(환경변수 우선, 비어 있으면 전체 허용)를 사용해야 함