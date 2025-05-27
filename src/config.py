import os
from dotenv import load_dotenv

# data 디렉터리 자동 생성
os.makedirs("data/models", exist_ok=True)
os.makedirs("data/index", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)

load_dotenv()

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Open CodeAI")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "data/models/qwen2.5-coder")
    VECTOR_INDEX_PATH: str = os.getenv("VECTOR_INDEX_PATH", "data/index/")
    DB_PATH: str = os.getenv("DB_PATH", "data/codeai.db")
    API_KEY: str = os.getenv("API_KEY", "open-codeai-local-key")
    # 기타 환경 변수 추가 가능

settings = Settings() 