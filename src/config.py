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
    port: int = 8000
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


class Settings(BaseSettings):
    """통합 설정 클래스 (.env만 사용, 모든 주요 환경변수 명시)"""
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    VECTOR_DB_PORT: int = Field(default=9000, env="VECTOR_DB_PORT")
    GRAPH_DB_PORT: int = Field(default=7687, env="GRAPH_DB_PORT")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    PROJECT_NAME: str = Field(default="open-codeai", env="PROJECT_NAME")
    VERSION: str = Field(default="1.0.0", env="VERSION")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    CORS_ORIGINS: str = Field(default="http://localhost:3000,http://localhost:8080", env="CORS_ORIGINS")
    GPU_MEMORY_FRACTION: float = Field(default=0.7, env="GPU_MEMORY_FRACTION")
    USE_MIXED_PRECISION: bool = Field(default=True, env="USE_MIXED_PRECISION")
    API_KEY: str = Field(default="open-codeai-local-key", env="API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"  # 미정의 변수도 허용

    def ensure_directories(self) -> None:
        """필요한 디렉토리들 생성"""
        directories = [
            "data/models",
            "data/vector_index", 
            "data/graph_db",
            "data/metadata",
            "data/cache",
            "data/logs",
            "logs",
            "static"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 필요한 디렉토리들이 생성되었습니다: {', '.join(directories)}")

# 전역 설정 인스턴스 생성
settings = Settings()
settings.ensure_directories()

# 하위 호환성을 위한 별칭들
PROJECT_NAME = settings.PROJECT_NAME
API_KEY = settings.API_KEY

# 추가된 유용한 별칭들
HOST = settings.HOST
PORT = settings.PORT
DEBUG = settings.DEBUG
VERSION = settings.VERSION

def setup_logging():
    """로깅 시스템 설정"""
    try:
        from .utils.logger import setup_logger
        setup_logger(
            log_level=settings.log_level,
            enable_file_logging=True
        )
    except ImportError:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )