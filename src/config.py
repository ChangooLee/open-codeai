"""
Open CodeAI 설정 관리 모듈 (업데이트됨)

YAML 설정 파일과 환경변수를 통합하여 관리
하드웨어 정보와 통합된 설정 시스템
"""
import os
import sys
import yaml
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
    use_vllm: bool = True


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
    """통합 설정 클래스"""
    
    # 기본 설정들
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    server: ServerConfig = Field(default_factory=ServerConfig) 
    llm: Optional[LLMConfig] = None
    database: Optional[DatabaseConfig] = None
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    performance: Optional[PerformanceConfig] = None
    monitoring: Optional[MonitoringConfig] = None
    continue_integration: ContinueIntegrationConfig = Field(default_factory=ContinueIntegrationConfig)
    
    # 환경변수 우선 적용
    PROJECT_NAME: str = Field(default="open-codeai", env="PROJECT_NAME")
    VERSION: str = Field(default="1.0.0", env="VERSION")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    
    # 경로 설정
    MODEL_PATH: str = Field(default="./data/models/qwen2.5-coder-32b", env="MODEL_PATH")
    VECTOR_INDEX_PATH: str = Field(default="./data/vector_index", env="VECTOR_INDEX_PATH")
    GRAPH_DB_PATH: str = Field(default="./data/graph_db", env="GRAPH_DB_PATH")
    METADATA_DB_PATH: str = Field(default="./data/metadata/opencodeai.db", env="METADATA_DB_PATH")
    
    # 서버 설정
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API 설정
    API_KEY: str = Field(default="open-codeai-local-key", env="API_KEY")
    
    # CORS 설정
    CORS_ORIGINS: List[str] = Field(default=[
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "vscode-webview://*",
        "vscode://*"
    ])
    
    # GPU 설정
    GPU_MEMORY_FRACTION: float = Field(default=0.7, env="GPU_MEMORY_FRACTION")
    USE_MIXED_PRECISION: bool = Field(default=True, env="USE_MIXED_PRECISION")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    @classmethod
    def from_yaml(cls, config_path: str = "configs/config.yaml") -> "Settings":
        """YAML 파일에서 설정 로드"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return cls()
        
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)
            
            # 환경변수를 먼저 로드한 기본 설정 생성
            base_settings = cls()
            
            # YAML 데이터로 업데이트
            if yaml_data:
                # 각 섹션별로 파싱
                if "project" in yaml_data:
                    base_settings.project = ProjectConfig(**yaml_data["project"])
                
                if "server" in yaml_data:
                    base_settings.server = ServerConfig(**yaml_data["server"])
                    # 서버 설정을 환경변수로도 적용
                    base_settings.HOST = yaml_data["server"].get("host", base_settings.HOST)
                    base_settings.PORT = yaml_data["server"].get("port", base_settings.PORT)
                    base_settings.LOG_LEVEL = yaml_data["server"].get("log_level", base_settings.LOG_LEVEL)
                
                if "llm" in yaml_data:
                    llm_data = yaml_data["llm"]
                    base_settings.llm = LLMConfig(
                        main_model=LLMModelConfig(**llm_data["main_model"]),
                        embedding_model=EmbeddingModelConfig(**llm_data["embedding_model"]),
                        graph_analysis_model=GraphModelConfig(**llm_data["graph_analysis_model"])
                    )
                    # 모델 경로 환경변수로도 적용
                    base_settings.MODEL_PATH = llm_data["main_model"]["path"]
                
                if "database" in yaml_data:
                    base_settings.database = DatabaseConfig(**yaml_data["database"])
                    # 데이터베이스 경로들도 환경변수로 적용
                    if "vector" in yaml_data["database"]:
                        base_settings.VECTOR_INDEX_PATH = yaml_data["database"]["vector"].get("path", base_settings.VECTOR_INDEX_PATH)
                    if "graph" in yaml_data["database"]:
                        base_settings.GRAPH_DB_PATH = yaml_data["database"]["graph"].get("path", base_settings.GRAPH_DB_PATH)
                    if "metadata" in yaml_data["database"]:
                        base_settings.METADATA_DB_PATH = yaml_data["database"]["metadata"].get("path", base_settings.METADATA_DB_PATH)
                
                if "indexing" in yaml_data:
                    base_settings.indexing = IndexingConfig(**yaml_data["indexing"])
                
                if "performance" in yaml_data:
                    base_settings.performance = PerformanceConfig(**yaml_data["performance"])
                    # GPU 설정도 환경변수로 적용
                    if "gpu" in yaml_data["performance"]:
                        gpu_config = yaml_data["performance"]["gpu"]
                        base_settings.GPU_MEMORY_FRACTION = gpu_config.get("memory_fraction", base_settings.GPU_MEMORY_FRACTION)
                        base_settings.USE_MIXED_PRECISION = gpu_config.get("mixed_precision", base_settings.USE_MIXED_PRECISION)
                
                if "monitoring" in yaml_data:
                    base_settings.monitoring = MonitoringConfig(**yaml_data["monitoring"])
                
                if "continue_integration" in yaml_data:
                    base_settings.continue_integration = ContinueIntegrationConfig(**yaml_data["continue_integration"])
            
            return base_settings
            
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            print("Using default settings...")
            return cls()
    
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

    def get_model_path(self, model_type: str) -> str:
        """모델 타입별 경로 반환"""
        if not self.llm:
            return self.MODEL_PATH
            
        model_paths = {
            "main": self.llm.main_model.path,
            "embedding": self.llm.embedding_model.path, 
            "graph": self.llm.graph_analysis_model.path
        }
        
        return model_paths.get(model_type, self.MODEL_PATH)

    def is_gpu_available(self) -> bool:
        """GPU 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_hardware_optimized_settings(self) -> Dict[str, Any]:
        """하드웨어에 최적화된 설정 반환"""
        try:
            from .utils.hardware import get_hardware_info, recommend_settings
            hardware_info = get_hardware_info()
            return recommend_settings(hardware_info)
        except ImportError:
            return {
                "parallel_workers": 4,
                "cache_size_gb": 2,
                "memory_limit_gb": 8,
                "gpu": {"enable": False},
                "batch_size": 32
            }

def setup_logging():
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

# 전역 설정 인스턴스 생성
def create_settings() -> Settings:
    """설정 인스턴스 생성 및 초기화"""
    
    # 개발 환경에서는 현재 디렉토리 기준으로 설정
    if os.path.exists("configs/config.yaml"):
        config_path = "configs/config.yaml"
    elif os.path.exists("../configs/config.yaml"):
        config_path = "../configs/config.yaml"
    else:
        config_path = "configs/config.yaml"  # 기본값
    
    settings_instance = Settings.from_yaml(config_path)
    settings_instance.ensure_directories()
    return settings_instance


# 전역 설정 객체
settings = create_settings()

# 하위 호환성을 위한 별칭들
PROJECT_NAME = settings.PROJECT_NAME
MODEL_PATH = settings.MODEL_PATH
VECTOR_INDEX_PATH = settings.VECTOR_INDEX_PATH
DB_PATH = settings.METADATA_DB_PATH
API_KEY = settings.API_KEY

# 추가된 유용한 별칭들
HOST = settings.HOST
PORT = settings.PORT
DEBUG = settings.DEBUG
VERSION = settings.VERSION