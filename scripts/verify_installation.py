import os
import sys
from src.config import settings

def check_packages():
    try:
        import fastapi, uvicorn, sqlalchemy, faiss, llama_index, watchdog, tree_sitter, requests, pydantic
        print("[OK] 필수 패키지 설치됨.")
    except ImportError as e:
        print(f"[ERROR] 패키지 설치 필요: {e}")
        sys.exit(1)

def check_paths():
    for path in [settings.MODEL_PATH, settings.VECTOR_INDEX_PATH, settings.DB_PATH]:
        print(f"[INFO] 경로 확인: {path}")
        if not os.path.exists(os.path.dirname(path)):
            print(f"[WARN] 디렉터리 없음: {os.path.dirname(path)}")

def main():
    print("[Open CodeAI 설치 검증]")
    check_packages()
    check_paths()
    print(f"[INFO] 환경 변수: PROJECT_NAME={settings.PROJECT_NAME}, API_KEY={settings.API_KEY}")
    print("[SUCCESS] 설치 검증 완료.")

if __name__ == "__main__":
    main() 