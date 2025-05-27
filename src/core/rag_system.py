from typing import Dict

def get_project_stats(project_path: str) -> Dict:
    # 실제론 벡터 인덱스/DB에서 통계 조회
    return {
        "project_path": project_path,
        "files": 123,
        "lines": 45678,
        "functions": 789,
        "last_indexed": "2024-06-01T12:00:00"
    }

def reindex_project(project_path: str) -> Dict:
    # 실제론 코드 임베딩 및 인덱스 재구축
    return {
        "project_path": project_path,
        "status": "reindex started"
    } 