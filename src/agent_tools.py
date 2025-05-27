from typing import List, Dict, Any, Optional
import os
import ast
import subprocess
from pathlib import Path
from dataclasses import dataclass
import json

@dataclass
class ToolResult:
    success: bool
    data: Any
    error: Optional[str] = None

class CodebaseTools:
    """AI가 사용할 수 있는 코드베이스 도구들"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.rag_system = None  # RAG 시스템 연결
    
    def read_file(self, file_path: str) -> ToolResult:
        try:
            full_path = self.project_path / file_path
            if not full_path.exists():
                return ToolResult(False, None, f"파일을 찾을 수 없습니다: {file_path}")
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            stats = full_path.stat()
            result = {
                "content": content,
                "path": str(file_path),
                "size": stats.st_size,
                "modified": stats.st_mtime,
                "lines": len(content.splitlines())
            }
            return ToolResult(True, result)
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    def write_file(self, file_path: str, content: str) -> ToolResult:
        try:
            full_path = self.project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            if full_path.exists():
                backup_path = full_path.with_suffix(f"{full_path.suffix}.backup")
                full_path.rename(backup_path)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            if self.rag_system:
                self.rag_system.index_file(str(full_path))
            return ToolResult(True, {"path": str(file_path), "size": len(content)})
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    def list_files(self, directory: str = ".", pattern: str = "*") -> ToolResult:
        try:
            search_path = self.project_path / directory
            files = []
            for file_path in search_path.rglob(pattern):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.project_path)
                    stats = file_path.stat()
                    files.append({
                        "path": str(relative_path),
                        "size": stats.st_size,
                        "modified": stats.st_mtime,
                        "extension": file_path.suffix
                    })
            return ToolResult(True, {"files": files, "count": len(files)})
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    def analyze_code_structure(self, file_path: str) -> ToolResult:
        try:
            file_result = self.read_file(file_path)
            if not file_result.success:
                return file_result
            content = file_result.data["content"]
            tree = ast.parse(content)
            structure: Dict[str, list] = {
                "imports": [],
                "classes": [],
                "functions": [],
                "variables": []
            }
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        structure["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        structure["imports"].append(f"{module}.{alias.name}")
                elif isinstance(node, ast.FunctionDef):
                    structure["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node)
                    })
                elif isinstance(node, ast.ClassDef):
                    structure["classes"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "bases": [ast.unparse(base) for base in node.bases],
                        "docstring": ast.get_docstring(node)
                    })
            return ToolResult(True, structure)
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    def search_code(self, query: str, file_types: Optional[List[str]] = None) -> ToolResult:
        try:
            if not self.rag_system:
                return ToolResult(False, None, "RAG 시스템이 초기화되지 않았습니다")
            results = self.rag_system.search_code(query)
            if file_types:
                filtered_results = []
                for result in results:
                    file_path = result.get("metadata", {}).get("file_path", "")
                    if any(file_path.endswith(ext) for ext in file_types):
                        filtered_results.append(result)
                results = filtered_results
            return ToolResult(True, {"results": results, "count": len(results)})
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    def run_tests(self, test_path: Optional[str] = None) -> ToolResult:
        try:
            cmd = ["python", "-m", "pytest"]
            if test_path:
                cmd.append(str(self.project_path / test_path))
            else:
                cmd.append(str(self.project_path))
            cmd.extend(["-v", "--tb=short", "--json-report", "--json-report-file=/tmp/test_results.json"])
            result = subprocess.run(
                cmd,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            try:
                with open("/tmp/test_results.json", "r") as f:
                    test_data = json.load(f)
            except:
                test_data = {}
            return ToolResult(True, {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "test_data": test_data
            })
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    def lint_code(self, file_path: str) -> ToolResult:
        try:
            full_path = self.project_path / file_path
            result = subprocess.run(
                ["flake8", str(full_path), "--format=json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            try:
                lint_results = json.loads(result.stdout) if result.stdout else []
            except:
                lint_results = []
            return ToolResult(True, {
                "file": str(file_path),
                "issues": lint_results,
                "issue_count": len(lint_results)
            })
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    def get_git_diff(self, file_path: Optional[str] = None) -> ToolResult:
        try:
            cmd = ["git", "diff"]
            if file_path:
                cmd.append(str(file_path))
            result = subprocess.run(
                cmd,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            return ToolResult(True, {
                "diff": result.stdout,
                "has_changes": bool(result.stdout.strip())
            })
        except Exception as e:
            return ToolResult(False, None, str(e))

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "파일의 내용을 읽어옵니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "읽을 파일의 경로"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "파일에 내용을 씁니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "쓸 파일의 경로"
                    },
                    "content": {
                        "type": "string",
                        "description": "파일에 쓸 내용"
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "디렉토리의 파일 목록을 가져옵니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "검색할 디렉토리 (기본값: 현재 디렉토리)",
                        "default": "."
                    },
                    "pattern": {
                        "type": "string", 
                        "description": "파일 패턴 (예: '*.py')",
                        "default": "*"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_code_structure",
            "description": "Python 파일의 코드 구조를 분석합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "분석할 Python 파일 경로"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "코드베이스에서 검색합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색 쿼리"
                    },
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "검색할 파일 타입 (예: ['.py', '.js'])"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "테스트를 실행합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_path": {
                        "type": "string",
                        "description": "특정 테스트 파일/디렉토리 경로"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lint_code",
            "description": "코드를 린트하여 문제점을 찾습니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "린트할 파일 경로"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_git_diff",
            "description": "Git diff를 가져옵니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "특정 파일의 diff를 가져올 경로"
                    }
                },
                "required": []
            }
        }
    }
] 