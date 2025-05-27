import os
from pathlib import Path
from typing import List, Dict, Any, Optional

SUPPORTED_EXTS = {'.py': 'python', '.js': 'javascript', '.java': 'java', '.cpp': 'cpp', '.go': 'go'}

class CodeBlock:
    def __init__(self, type_: str, name: str, code: str, file_path: str, start_line: int, end_line: int):
        self.type = type_
        self.name = name
        self.code = code
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line

    def to_dict(self):
        return {
            'type': self.type,
            'name': self.name,
            'code': self.code,
            'file_path': self.file_path,
            'start_line': self.start_line,
            'end_line': self.end_line
        }

class CodeParser:
    def __init__(self):
        # tree-sitter 언어 so파일은 ./tree-sitter-libs/{lang}.so에 위치해야 함
        from tree_sitter import Language, Parser
        self.langs = {}
        self.parsers = {}
        for ext, lang in SUPPORTED_EXTS.items():
            so_path = f'./tree-sitter-libs/{lang}.so'
            if os.path.exists(so_path):
                self.langs[lang] = Language(so_path, lang)
                parser = Parser()
                parser.set_language(self.langs[lang])
                self.parsers[lang] = parser

    def parse_file(self, file_path: str) -> List[CodeBlock]:
        ext = Path(file_path).suffix
        lang = SUPPORTED_EXTS.get(ext)
        if not lang or lang not in self.parsers:
            return []
        parser = self.parsers[lang]
        with open(file_path, 'rb') as f:
            code = f.read()
        tree = parser.parse(code)
        # 언어별로 함수/클래스 추출 (여기선 python만 예시, 나머지는 파일 전체 블록)
        if lang == 'python':
            return self._parse_python(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_str = f.read()
            return [CodeBlock('file', Path(file_path).name, code_str, file_path, 1, code_str.count('\n')+1)]

    def _parse_python(self, file_path: str) -> List[CodeBlock]:
        import ast
        blocks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start = node.lineno
                    end = getattr(node, 'end_lineno', start)
                    src = self._get_src_segment(code, node)
                    blocks.append(CodeBlock('function', node.name, src, file_path, start, end))
                elif isinstance(node, ast.ClassDef):
                    start = node.lineno
                    end = getattr(node, 'end_lineno', start)
                    src = self._get_src_segment(code, node)
                    blocks.append(CodeBlock('class', node.name, src, file_path, start, end))
        except Exception as e:
            # fallback: 파일 전체
            blocks.append(CodeBlock('file', Path(file_path).name, code, file_path, 1, code.count('\n')+1))
        return blocks

    def _get_src_segment(self, code: str, node) -> str:
        import ast
        try:
            return ast.get_source_segment(code, node) or ''
        except Exception:
            return ''

    def walk_project(self, root_dir: str) -> List[str]:
        code_files = []
        for ext in SUPPORTED_EXTS:
            code_files += list(Path(root_dir).rglob(f'*{ext}'))
        return [str(f) for f in code_files] 