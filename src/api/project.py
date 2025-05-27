from fastapi import APIRouter
from typing import Dict

router = APIRouter()

@router.get("/stats/{project_path}")
def get_project_stats(project_path: str) -> Dict:
    # 실제론 core.rag_system.get_project_stats 호출
    return {
        "project_path": project_path,
        "files": 123,
        "lines": 45678,
        "functions": 789,
        "last_indexed": "2024-06-01T12:00:00"
    }

@router.post("/reindex/{project_path}")
def reindex_project(project_path: str) -> Dict:
    # 실제론 core.rag_system.reindex_project 호출
    return {
        "project_path": project_path,
        "status": "reindex started"
    }

# 프로젝트 관리 API 엔드포인트 스텁 