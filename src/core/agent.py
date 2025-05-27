import os
from typing import List, Dict, Any

# 파일 읽기
def read_file(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"[ERROR] {e}"

# 파일 쓰기
def write_file(path: str, content: str) -> str:
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return "success"
    except Exception as e:
        return f"[ERROR] {e}"

# 파일 목록
def list_files(dir_path: str) -> List[str]:
    try:
        return os.listdir(dir_path)
    except Exception as e:
        return [f"[ERROR] {e}"]

# 코드 구조 분석 (mock)
def analyze_code_structure(project_path: str) -> Dict:
    return {"summary": "(모의) 코드 구조 분석 결과", "project_path": project_path}

# 코드 검색 (mock)
def search_code(keyword: str, project_path: str) -> List[str]:
    return [f"(모의) {keyword}가 포함된 파일1.py", f"(모의) {keyword}가 포함된 파일2.py"]

# 테스트 실행 (mock)
def run_tests(project_path: str) -> Dict:
    return {"result": "(모의) 모든 테스트 통과", "project_path": project_path}

# 코드 린트 (mock)
def lint_code(project_path: str) -> Dict:
    return {"result": "(모의) 린트 결과: 0 error", "project_path": project_path}

# Git diff (mock)
def get_git_diff(project_path: str) -> str:
    return "(모의) 변경사항 없음" 