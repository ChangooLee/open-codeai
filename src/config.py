"""
Open CodeAI 설정 관리 모듈

YAML 설정 파일과 환경변수를 통합하여 관리
"""
import os
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
    
    # API 설정
    API_KEY: str = Field(default="open-codeai-local-key", env="API_KEY")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
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
                
                if "llm" in yaml_data:
                    llm_data = yaml_data["llm"]
                    base_settings.llm = LLMConfig(
                        main_model=LLMModelConfig(**llm_data["main_model"]),
                        embedding_model=EmbeddingModelConfig(**llm_data["embedding_model"]),
                        graph_analysis_model=GraphModelConfig(**llm_data["graph_analysis_model"])
                    )
                
                if "database" in yaml_data:
                    base_settings.database = DatabaseConfig(**yaml_data["database"])
                
                if "indexing" in yaml_data:
                    base_settings.indexing = IndexingConfig(**yaml_data["indexing"])
                
                if "performance" in yaml_data:
                    base_settings.performance = PerformanceConfig(**yaml_data["performance"])
                
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
            "logs"
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


# 전역 설정 인스턴스 생성
def create_settings() -> Settings:
    """설정 인스턴스 생성 및 초기화"""
    settings_instance = Settings.from_yaml()
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