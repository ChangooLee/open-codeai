"""
Open CodeAI 설치/환경 검증 스크립트
-------------------------------
- 필수 패키지/경로/환경변수 등 설치 상태 자동 점검
- 오프라인/컨테이너리스/샤딩 등 다양한 모드 지원

[사용법/Usage]
$ python scripts/verify_installation.py

[검증 항목/Check Items]
- 필수 Python 패키지 설치 여부
- 모델/DB/인덱스 경로 존재 여부
- 주요 환경변수 값 확인

[실행 예시/Example]
- 오프라인 설치 후 검증: python scripts/verify_installation.py
- 컨테이너리스/샤딩 등 모든 모드에서 사용 가능
"""
import os
import sys

def check_packages() -> None:
    try:
        import fastapi, uvicorn, requests, pydantic  # type: ignore
        # 선택적 패키지: sqlalchemy, faiss, llama_index, watchdog, tree_sitter
        try:
            import sqlalchemy  # type: ignore
        except ImportError:
            pass
        try:
            import faiss  # type: ignore
        except ImportError:
            pass
        try:
            import llama_index  # type: ignore
        except ImportError:
            pass
        try:
            import watchdog  # type: ignore
        except ImportError:
            pass
        try:
            import tree_sitter  # type: ignore
        except ImportError:
            pass
        print("[OK] 필수 패키지 설치됨.")
    except ImportError as e:
        print(f"[ERROR] 패키지 설치 필요: {e}")
        sys.exit(1)

def check_paths() -> None:
    # 환경변수 기반 경로 확인
    model_path = os.environ.get("MODEL_PATH", "./data/models/qwen2.5-coder-32b")
    vector_index_path = os.environ.get("VECTOR_INDEX_PATH", "./data/vector_index")
    db_path = os.environ.get("METADATA_DB_PATH", "./data/metadata/opencodeai.db")
    for path in [model_path, vector_index_path, db_path]:
        print(f"[INFO] 경로 확인: {path}")
        if not os.path.exists(os.path.dirname(path)):
            print(f"[WARN] 디렉터리 없음: {os.path.dirname(path)}")

def main() -> None:
    print("[Open CodeAI 설치 검증]")
    check_packages()
    check_paths()
    print(f"[INFO] 환경 변수: PROJECT_NAME={os.environ.get('PROJECT_NAME', '')}, API_KEY={os.environ.get('API_KEY', '')}")
    print("[SUCCESS] 설치 검증 완료.")

if __name__ == "__main__":
    main() 