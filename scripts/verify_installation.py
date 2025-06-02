import asyncio
import sys
import os
import requests
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.core.llm_manager import get_llm_manager
from src.core.rag_system import get_rag_system
from src.core.function_calling import get_function_registry

def check_packages():
    """필수 패키지 확인"""
    try:
        import fastapi, uvicorn, torch, faiss, neo4j
        print("✅ 필수 패키지 설치됨")
        return True
    except ImportError as e:
        print(f"❌ 패키지 누락: {e}")
        return False

def check_gpu():
    """GPU 사용 가능성 확인"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU 사용 가능: {gpu_name} ({gpu_count}개)")
            return True
        else:
            print("⚠️ GPU 사용 불가, CPU 모드로 실행")
            return True
    except Exception as e:
        print(f"❌ GPU 확인 실패: {e}")
        return False

def check_neo4j():
    """Neo4j 연결 확인"""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "opencodeai"))
        with driver.session() as session:
            result = session.run("RETURN 1")
            if result.single():
                print("✅ Neo4j 연결 성공")
                driver.close()
                return True
    except Exception as e:
        print(f"❌ Neo4j 연결 실패: {e}")
        return False

def check_api_server():
    """API 서버 확인"""
    try:
        # 서버가 실행 중인지 확인
        response = requests.get("http://localhost:8800/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ API 서버 실행 중")
            return True
        else:
            print("❌ API 서버 응답 오류")
            return False
    except Exception:
        print("⚠️ API 서버 미실행 (정상 - 수동 시작 필요)")
        return True

async def check_rag_system():
    """RAG 시스템 확인"""
    try:
        rag_system = get_rag_system()
        stats = rag_system.indexer.get_indexing_stats()
        print(f"✅ RAG 시스템 정상 (파일: {stats.get('total_files', 0)}, 청크: {stats.get('total_chunks', 0)})")
        return True
    except Exception as e:
        print(f"❌ RAG 시스템 오류: {e}")
        return False

def check_function_calling():
    """Function Calling 확인"""
    try:
        registry = get_function_registry()
        functions = registry.get_available_functions()
        print(f"✅ Function Calling 정상 (함수: {len(functions)}개)")
        return True
    except Exception as e:
        print(f"❌ Function Calling 오류: {e}")
        return False

async def main():
    print("=== Open CodeAI 설치 검증 ===\n")
    
    checks = [
        ("패키지", check_packages),
        ("GPU", check_gpu),
        ("Neo4j", check_neo4j),
        ("API 서버", check_api_server),
        ("RAG 시스템", check_rag_system),
        ("Function Calling", check_function_calling)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"[{name}] 확인 중...")
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {name} 확인 실패: {e}")
    
    print(f"\n=== 검증 완료: {passed}/{total} 항목 통과 ===")
    
    if passed >= total - 1:  # API 서버는 선택적
        print("🎉 설치가 성공적으로 완료되었습니다!")
        return True
    else:
        print("⚠️ 일부 구성 요소에 문제가 있습니다.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
