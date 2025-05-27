from typing import Dict

def parse_code_structure(file_path: str) -> Dict:
    # 실제론 tree-sitter로 파싱
    return {
        "file": file_path,
        "structure": "(모의) 함수 3개, 클래스 1개"
    } 