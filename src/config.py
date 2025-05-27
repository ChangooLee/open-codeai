import os
import yaml
from pydantic import BaseSettings

# data 디렉터리 자동 생성
os.makedirs("data/models", exist_ok=True)
os.makedirs("data/index", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)

class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    # 예시 필드
    project_name: str = "open-codeai"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True

    @classmethod
    def from_yaml(cls, path: str = "configs/config.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data.get("project", {}))

settings = Settings.from_yaml()

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Open CodeAI")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "data/models/qwen2.5-coder")
    VECTOR_INDEX_PATH: str = os.getenv("VECTOR_INDEX_PATH", "data/index/")
    DB_PATH: str = os.getenv("DB_PATH", "data/codeai.db")
    API_KEY: str = os.getenv("API_KEY", "open-codeai-local-key")
    # 기타 환경 변수 추가 가능

settings = Settings() 