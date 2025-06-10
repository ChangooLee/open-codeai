"""
Open CodeAI - Tree-sitter 대안 코드 분석기
Python AST + 정규식 기반의 강력한 코드 분석 시스템
"""
import ast
import re
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
from src.utils.mapper_xml_parser import parse_mapper_xml

class Language(Enum):
    """지원하는 언어"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    CSS = "css"
    VUE = "vue"
    HTML = "html"
    ANGULAR = "angular"

@dataclass
class CodeElement:
    """코드 요소 기본 클래스"""
    name: str
    start_line: int
    end_line: int
    element_type: str
    content: str
    language: str
    metadata: Optional[Dict[str, Any]]

@dataclass 
class FunctionElement(CodeElement):
    """함수 요소"""
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    complexity: int = 1
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)  # 호출하는 함수들

@dataclass
class ClassElement(CodeElement):
    """클래스 요소"""
    methods: List[FunctionElement] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    inheritance: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_abstract: bool = False

@dataclass
class ImportElement(CodeElement):
    """임포트 요소"""
    module: str
    items: List[str] = field(default_factory=list)
    alias: Optional[str] = None
    is_from_import: bool = False

class AlternativeCodeAnalyzer:
    """Tree-sitter 대안 코드 분석기"""
    
    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.language_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, Dict[str, Any]]:
        """언어별 패턴 초기화"""
        return {
            Language.PYTHON.value: {
                'function': r'^\s*(async\s+)?def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*([^:]+))?\s*:',
                'class': r'^\s*class\s+(\w+)(?:\([^)]*\))?\s*:',
                'import': r'^\s*(?:from\s+([\w.]+)\s+)?import\s+(.+)',
                'variable': r'^\s*(\w+)\s*=\s*(.+)',
                'decorator': r'^\s*@(\w+(?:\.\w+)*)',
                'comment': r'^\s*#.*',
                'docstring': r'^\s*"""(.*?)"""',
                'call': r'(\w+)\s*\(',
            },
            Language.JAVASCRIPT.value: {
                'function': r'^\s*(?:async\s+)?(?:function\s+(\w+)|(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))',
                'class': r'^\s*class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{',
                'import': r'^\s*(?:import\s+(?:{([^}]+)}|(\w+))\s+from\s+[\'"]([^\'"]+)[\'"]|import\s+[\'"]([^\'"]+)[\'"])',
                'variable': r'^\s*(?:const|let|var)\s+(\w+)\s*=',
                'comment': r'^\s*//.*',
                'call': r'(\w+)\s*\(',
            },
            Language.TYPESCRIPT.value: {
                'function': r'^\s*(?:async\s+)?(?:function\s+(\w+)|(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))',
                'class': r'^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*{',
                'interface': r'^\s*(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+([^{]+))?\s*{',
                'import': r'^\s*import\s+(?:{([^}]+)}|(\w+))\s+from\s+[\'"]([^\'"]+)[\'"]',
                'type': r'^\s*(?:export\s+)?type\s+(\w+)\s*=',
                'variable': r'^\s*(?:const|let|var)\s+(\w+)(?:\s*:\s*([^=]+))?\s*=',
                'comment': r'^\s*//.*',
                'call': r'(\w+)\s*\(',
            },
            Language.JAVA.value: {
                'function': r'^\s*(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*(?:abstract)?\s*(\w+|\w+<[^>]+>)\s+(\w+)\s*\([^)]*\)',
                'class': r'^\s*(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*{',
                'interface': r'^\s*(?:public|private|protected)?\s*interface\s+(\w+)(?:\s+extends\s+([^{]+))?\s*{',
                'import': r'^\s*import\s+(static\s+)?([^;]+);',
                'variable': r'^\s*(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*(\w+(?:<[^>]+>)?)\s+(\w+)\s*[=;]',
                'comment': r'^\s*//.*',
                'call': r'(\w+)\s*\(',
            },
            Language.GO.value: {
                'function': r'^\s*func\s+(?:\([^)]*\)\s+)?(\w+)\s*\([^)]*\)(?:\s*[^{]*)?{',
                'struct': r'^\s*type\s+(\w+)\s+struct\s*{',
                'interface': r'^\s*type\s+(\w+)\s+interface\s*{',
                'import': r'^\s*import\s+(?:\(\s*([^)]+)\s*\)|"([^"]+)")',
                'variable': r'^\s*(?:var\s+(\w+)|(\w+)\s*:=)',
                'comment': r'^\s*//.*',
                'call': r'(\w+)\s*\(',
            },
            Language.RUST.value: {
                'function': r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)(?:<[^>]*>)?\s*\([^)]*\)(?:\s*->\s*[^{]+)?\s*{',
                'struct': r'^\s*(?:pub\s+)?struct\s+(\w+)(?:<[^>]*>)?\s*{',
                'enum': r'^\s*(?:pub\s+)?enum\s+(\w+)(?:<[^>]*>)?\s*{',
                'trait': r'^\s*(?:pub\s+)?trait\s+(\w+)(?:<[^>]*>)?\s*{',
                'impl': r'^\s*impl(?:<[^>]*>)?\s+(?:(\w+)(?:<[^>]*>)?\s+for\s+)?(\w+)(?:<[^>]*>)?\s*{',
                'use': r'^\s*use\s+([^;]+);',
                'variable': r'^\s*let\s+(?:mut\s+)?(\w+)(?:\s*:\s*[^=]+)?\s*=',
                'comment': r'^\s*//.*',
                'call': r'(\w+)\s*\(',
            }
        }
    
    def detect_language(self, file_path: str) -> Union[Language, str]:
        """파일 확장자로 언어 감지 (확장)"""
        extension = Path(file_path).suffix.lower()
        extension_map = {
            '.py': Language.PYTHON,
            '.js': Language.JAVASCRIPT,
            '.jsx': Language.JAVASCRIPT,
            '.ts': Language.TYPESCRIPT,
            '.tsx': Language.TYPESCRIPT,
            '.mjs': Language.JAVASCRIPT,
            '.java': Language.JAVA,
            '.cpp': Language.CPP,
            '.cc': Language.CPP,
            '.cxx': Language.CPP,
            '.c': Language.C,
            '.go': Language.GO,
            '.rs': Language.RUST,
            '.php': Language.PHP,
            '.rb': Language.RUBY,
            '.css': Language.CSS,
            '.vue': Language.VUE,
            '.html': Language.HTML,
            '.htm': Language.HTML,
            '.json': 'json',
            '.md': 'markdown',
            '.markdown': 'markdown',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.sh': 'shell',
            '.ini': 'ini',
            '.cfg': 'ini',
            '.conf': 'ini',
            '.txt': 'txt',
            '.lock': 'lock',
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.env': 'env',
            '.bat': 'shell',
            '.ps1': 'shell',
            '.dockerfile': 'dockerfile',
            '.properties': 'ini',
            '.makefile': 'makefile',
            '.gradle': 'gradle',
            '.gitignore': 'gitignore',
            '.npmrc': 'ini',
            '.eslintrc': 'json',
            '.prettierrc': 'json',
            '.babelrc': 'json',
            '.editorconfig': 'ini',
        }
        # Angular: .ts + @Component decorator or .html with Angular template syntax
        if extension == '.ts':
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if '@Component' in content or '@NgModule' in content:
                    return Language.ANGULAR
            except Exception:
                pass
        if extension in {'.html', '.htm'}:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if 'ng-' in content or '*ngIf' in content or '*ngFor' in content:
                    return Language.ANGULAR
            except Exception:
                pass
        # React: .jsx/.tsx or .js/.ts/.mjs with React import
        if extension in {'.jsx', '.tsx'}:
            return Language.JAVASCRIPT
        if extension in {'.js', '.ts', '.mjs'}:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if 'from "react"' in content or "from 'react'" in content or 'React.' in content:
                    return Language.JAVASCRIPT
            except Exception:
                pass
        return extension_map.get(extension, Language.PYTHON)
    
    async def analyze_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """파일 분석 (확장자별 분기 및 fallback)"""
        try:
            if not os.path.exists(file_path):
                return None
            language = self.detect_language(file_path)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # Vue SFC
            if language == Language.VUE:
                return await self._analyze_vue_file(file_path, content)
            # Angular
            if language == Language.ANGULAR:
                return await self._analyze_angular_file(file_path, content)
            # JS/TS/JSX/TSX/MJS: tree-sitter 기반 분석
            if language in [Language.JAVASCRIPT, Language.TYPESCRIPT] and file_path.endswith(('.js', '.ts', '.jsx', '.tsx', '.mjs')):
                return await self._analyze_javascript_file_treesitter(file_path, content)
            # React
            if language == Language.JAVASCRIPT and (file_path.endswith('.jsx') or file_path.endswith('.tsx')):
                return await self._analyze_react_file(file_path, content)
            # Python: AST + tree-sitter 병합
            if language == Language.PYTHON:
                ast_result = await self._analyze_python_file(file_path, content)
                ts_result = await self._analyze_python_file_treesitter(file_path, content)
                def merge_lists(a, b):
                    return list({*a, *b})
                for key in ['functions', 'classes', 'imports', 'decorators', 'calls']:
                    if key in ast_result and key in ts_result:
                        ast_result[key] = merge_lists(ast_result[key], ts_result[key])
                    elif key in ts_result:
                        ast_result[key] = ts_result[key]
                return ast_result
            elif language == Language.JAVA:
                result = await self._analyze_generic_file(file_path, content, language)
                dir_path = os.path.dirname(file_path)
                mappers = []
                for fname in os.listdir(dir_path):
                    if fname.endswith('Mapper.xml'):
                        xml_path = os.path.join(dir_path, fname)
                        mappers.extend(parse_mapper_xml(xml_path))
                result['mappers'] = mappers
                return result
            elif language == Language.CSS:
                return await self._analyze_css_file(file_path, content)
            elif language == 'json':
                return await self._analyze_json_file(file_path, content)
            elif language == 'markdown':
                return await self._analyze_markdown_file(file_path, content)
            elif language == 'yaml':
                return await self._analyze_yaml_file(file_path, content)
            elif language == 'shell':
                return await self._analyze_shell_file(file_path, content)
            elif language == 'ini':
                return await self._analyze_ini_file(file_path, content)
            elif language == 'txt':
                return await self._analyze_txt_file(file_path, content)
            elif language == 'dockerfile':
                return await self._analyze_dockerfile(file_path, content)
            elif language == 'csv':
                return await self._analyze_csv_file(file_path, content)
            elif language == 'tsv':
                return await self._analyze_tsv_file(file_path, content)
            elif language == 'lock':
                return await self._analyze_lock_file(file_path, content)
            elif language == 'env':
                return await self._analyze_env_file(file_path, content)
            elif language == 'makefile':
                return await self._analyze_makefile(file_path, content)
            elif language == 'gradle':
                return await self._analyze_gradle_file(file_path, content)
            elif language == 'gitignore':
                return await self._analyze_gitignore_file(file_path, content)
            else:
                # 완전히 미지원 확장자: generic 분석 (최소한의 구조/통계)
                return await self._analyze_generic_file(file_path, content, language if isinstance(language, Language) else Language.PYTHON)
        except Exception as e:
            import traceback
            print(f"파일 분석 오류 {file_path}: {e}\n{traceback.format_exc()}")
            # fallback: generic 분석 시도
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return await self._analyze_generic_file(file_path, content, language if isinstance(language, Language) else Language.PYTHON)
            except Exception as e2:
                print(f"[FALLBACK] generic 분석도 실패: {file_path}: {e2}")
                return None
    
    async def _analyze_python_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Python 파일 AST 기반 분석 + Django/Flask/FastAPI 프레임워크 패턴 추출"""
        result = {
            'file_path': file_path,
            'language': Language.PYTHON.value,
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'lines_of_code': 0,
            'complexity': 0,
            'dependencies': set(),
            'exports': [],
            'views': [],
            'routers': [],
            'models': [],
            'serializers': [],
            'templates': [],
            'blueprints': [],
            'dependencies_injected': []
        }
        
        # 기본 메트릭
        lines = content.split('\n')
        result['lines_of_code'] = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        try:
            # AST 파싱
            tree = ast.parse(content)
            
            # AST 방문자 사용
            visitor = PythonASTAnalyzer(result)
            visitor.visit(tree)
            
        except SyntaxError as e:
            # AST 파싱 실패 시 정규식 폴백
            print(f"AST 파싱 실패, 정규식 사용: {e}")
            await self._analyze_python_with_regex(content, result)
        
        # 의존성을 리스트로 변환
        if isinstance(result['dependencies'], set):
            result['dependencies'] = list(result['dependencies'])
        
        # Django/Flask/FastAPI 패턴
        import re
        # Django View
        if re.search(r'from\s+django\.views', content) or re.search(r'class\s+\w+\(View\)', content):
            result['views'] = re.findall(r'class\s+(\w+)\(View\)', content)
        # Flask Blueprint
        if re.search(r'from\s+flask', content) and 'Blueprint' in content:
            result['blueprints'] = re.findall(r'Blueprint\(["\"][^"\"]+["\"]', content)
        # FastAPI router
        if re.search(r'from\s+fastapi', content) and 'APIRouter' in content:
            result['routers'] = re.findall(r'APIRouter\([\w=,\s]*\)', content)
        # Django Model
        if re.search(r'class\s+\w+\(models\.Model\)', content):
            result['models'] = re.findall(r'class\s+(\w+)\(models\.Model\)', content)
        # Django Serializer
        if re.search(r'class\s+\w+\(serializers\.Serializer\)', content):
            result['serializers'] = re.findall(r'class\s+(\w+)\(serializers\.Serializer\)', content)
        # Jinja2/템플릿
        result['templates'] = re.findall(r'render_template\(["\"][^"\"]+["\"]', content)
        # 의존성 주입(FastAPI Depends)
        result['dependencies_injected'] = re.findall(r'Depends\([\w.]+\)', content)
        
        return result
    
    async def _analyze_python_with_regex(self, content: str, result: Dict[str, Any]) -> None:
        """Python 정규식 기반 분석 (폴백)"""
        
        lines = content.split('\n')
        patterns = self.language_patterns[Language.PYTHON.value]
        
        for i, line in enumerate(lines, 1):
            # 함수 분석
            func_match = re.match(patterns['function'], line)
            if func_match:
                is_async = bool(func_match.group(1))
                func_name = func_match.group(2)
                params_str = func_match.group(3) or ""
                return_type = func_match.group(4)
                
                # 함수 끝 찾기
                end_line = self._find_python_block_end(lines, i-1)
                
                func_element = FunctionElement(
                    name=func_name,
                    start_line=i,
                    end_line=end_line,
                    element_type='function',
                    content='\n'.join(lines[i-1:end_line]),
                    language=Language.PYTHON.value,
                    parameters=self._parse_python_params(params_str),
                    return_type=return_type,
                    is_async=is_async,
                    complexity=1,
                    metadata=None
                )
                
                result['functions'].append(asdict(func_element))
            
            # 클래스 분석
            class_match = re.match(patterns['class'], line)
            if class_match:
                class_name = class_match.group(1)
                end_line = self._find_python_block_end(lines, i-1)
                
                class_element = ClassElement(
                    name=class_name,
                    start_line=i,
                    end_line=end_line,
                    element_type='class',
                    content='\n'.join(lines[i-1:end_line]),
                    language=Language.PYTHON.value,
                    methods=[],
                    attributes=[],
                    metadata=None
                )
                
                result['classes'].append(asdict(class_element))
            
            # 임포트 분석
            import_match = re.match(patterns['import'], line)
            if import_match:
                from_module = import_match.group(1)
                import_items = import_match.group(2)
                
                import_element = ImportElement(
                    name=import_items.strip(),
                    start_line=i,
                    end_line=i,
                    element_type='import',
                    content=line.strip(),
                    language=Language.PYTHON.value,
                    module=from_module or import_items.split('.')[0],
                    is_from_import=bool(from_module),
                    metadata=None
                )
                
                result['imports'].append(asdict(import_element))
                result['dependencies'].add(from_module or import_items.split('.')[0])
    
    def _find_python_block_end(self, lines: List[str], start_idx: int) -> int:
        """Python 블록의 끝 찾기"""
        if start_idx >= len(lines):
            return start_idx + 1
            
        start_line = lines[start_idx]
        base_indent = len(start_line) - len(start_line.lstrip())
        
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip():  # 빈 줄은 건너뛰기
                continue
                
            current_indent = len(line) - len(line.lstrip())
            
            # 들여쓰기가 기본 레벨보다 작거나 같으면 블록 끝
            if current_indent <= base_indent and line.strip():
                return i
        
        return len(lines)
    
    def _parse_python_params(self, params_str: str) -> List[str]:
        """Python 함수 파라미터 파싱"""
        if not params_str.strip():
            return []
        
        params = []
        for param in params_str.split(','):
            param = param.strip()
            if param and param != 'self':
                # 타입 힌트 제거
                if ':' in param:
                    param = param.split(':')[0].strip()
                # 기본값 제거
                if '=' in param:
                    param = param.split('=')[0].strip()
                params.append(param)
        
        return params
    
    async def _analyze_generic_file(self, file_path: str, content: str, language: Language) -> Dict[str, Any]:
        """일반적인 파일 정규식 기반 분석"""
        
        result: Dict[str, Any] = {
            'file_path': file_path,
            'language': language.value,
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'lines_of_code': 0,
            'complexity': 0,
            'dependencies': set(),
            'exports': []
        }
        
        lines = content.split('\n')
        result['lines_of_code'] = len([line for line in lines if line.strip() and not line.strip().startswith(('//','#','/*'))])
        
        patterns = self.language_patterns.get(language.value, {})
        
        for i, line in enumerate(lines, 1):
            # 함수 분석
            if 'function' in patterns:
                func_match = re.match(patterns['function'], line)
                if func_match:
                    func_name = self._extract_function_name(func_match, language)
                    if func_name:
                        end_line = self._find_block_end(lines, i-1, language)
                        func_element = FunctionElement(
                            name=func_name,
                            start_line=i,
                            end_line=end_line,
                            element_type='function',
                            content='\n'.join(lines[i-1:end_line]),
                            language=language.value,
                            parameters=self._extract_parameters(line, language),
                            complexity=1,
                            metadata=None
                        )
                        if isinstance(result['functions'], list):
                            result['functions'].append(asdict(func_element))
            
            # 클래스/구조체 분석
            class_patterns = ['class', 'struct', 'interface']
            for pattern_name in class_patterns:
                if pattern_name in patterns:
                    class_match = re.match(patterns[pattern_name], line)
                    if class_match:
                        class_name = class_match.group(1)
                        end_line = self._find_block_end(lines, i-1, language)
                        class_element = ClassElement(
                            name=class_name,
                            start_line=i,
                            end_line=end_line,
                            element_type=pattern_name,
                            content='\n'.join(lines[i-1:end_line]),
                            language=language.value,
                            methods=[],
                            attributes=[],
                            metadata=None
                        )
                        if isinstance(result['classes'], list):
                            result['classes'].append(asdict(class_element))
            
            # 임포트 분석
            import_patterns = ['import', 'use', '#include']
            for pattern_name in import_patterns:
                if pattern_name in patterns:
                    import_match = re.match(patterns[pattern_name], line)
                    if import_match:
                        import_info = self._extract_import_info(import_match, language)
                        if import_info:
                            import_element = ImportElement(
                                name=import_info['name'],
                                start_line=i,
                                end_line=i,
                                element_type='import',
                                content=line.strip(),
                                language=language.value,
                                module=import_info['module'],
                                is_from_import=import_info.get('is_from', False),
                                metadata=None
                            )
                            if isinstance(result['imports'], list):
                                result['imports'].append(asdict(import_element))
                            if isinstance(result['dependencies'], set):
                                result['dependencies'].add(import_info['module'])
        
        if isinstance(result['dependencies'], set):
            result['dependencies'] = list(result['dependencies'])
        return result
    
    def _extract_function_name(self, match: re.Match, language: Language) -> Optional[str]:
        """정규식 매치에서 함수명 추출"""
        if language == Language.JAVASCRIPT or language == Language.TYPESCRIPT:
            # function name() 또는 name = function() 패턴
            val = match.group(1) or match.group(2)
            return str(val) if val else None
        elif language == Language.JAVA:
            val = match.group(2)
            return str(val) if val else None
        elif language == Language.GO or language == Language.RUST:
            val = match.group(1)
            return str(val) if val else None
        else:
            return str(match.group(1)) if match.groups() and match.group(1) else None
    
    def _extract_parameters(self, line: str, language: Language) -> List[str]:
        """함수 파라미터 추출"""
        
        # 괄호 안의 내용 추출
        paren_match = re.search(r'\(([^)]*)\)', line)
        if not paren_match:
            return []
        
        params_str = paren_match.group(1).strip()
        if not params_str:
            return []
        
        params = []
        for param in params_str.split(','):
            param = param.strip()
            if param:
                # 언어별 파라미터 정리
                if language == Language.PYTHON:
                    if ':' in param:
                        param = param.split(':')[0].strip()
                elif language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
                    if ':' in param:
                        param = param.split(':')[0].strip()
                elif language == Language.JAVA:
                    # 타입 제거 (타입 이름 순서)
                    parts = param.split()
                    if len(parts) >= 2:
                        param = parts[-1]  # 마지막이 변수명
                
                if param and param not in ['self', 'this']:
                    params.append(param)
        
        return params
    
    def _extract_import_info(self, match: re.Match, language: Language) -> Optional[Dict[str, Any]]:
        """임포트 정보 추출"""
        
        if language == Language.PYTHON:
            # 이미 Python용으로 처리됨
            return None
        elif language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
            # import { items } from 'module' 또는 import module from 'module'
            items = match.group(1)
            module_name = match.group(2) or match.group(3)
            
            return {
                'name': items or module_name,
                'module': module_name,
                'is_from': bool(items)
            }
        elif language == Language.JAVA:
            # import package.Class;
            import_path = match.group(2)
            return {
                'name': import_path.split('.')[-1],
                'module': import_path,
                'is_from': False
            }
        elif language == Language.GO:
            # import "package" 또는 import ( ... )
            package = match.group(1) or match.group(2)
            return {
                'name': package.split('/')[-1],
                'module': package,
                'is_from': False
            }
        elif language == Language.RUST:
            # use std::collections::HashMap;
            use_path = match.group(1)
            return {
                'name': use_path.split('::')[-1],
                'module': use_path,
                'is_from': False
            }
        
        return None
    
    def _find_block_end(self, lines: List[str], start_idx: int, language: Language) -> int:
        """블록의 끝 찾기 (언어별)"""
        
        if language == Language.PYTHON:
            return self._find_python_block_end(lines, start_idx)
        
        # 중괄호 기반 언어들
        if language in [Language.JAVASCRIPT, Language.TYPESCRIPT, Language.JAVA, 
                       Language.CPP, Language.C, Language.GO, Language.RUST]:
            return self._find_brace_block_end(lines, start_idx)
        
        # 기본: 빈 줄까지
        for i in range(start_idx + 1, len(lines)):
            if not lines[i].strip():
                return i
        
        return len(lines)
    
    def _find_brace_block_end(self, lines: List[str], start_idx: int) -> int:
        """중괄호 기반 블록 끝 찾기"""
        
        brace_count = 0
        found_opening = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            
            for char in line:
                if char == '{':
                    brace_count += 1
                    found_opening = True
                elif char == '}':
                    brace_count -= 1
                    
                    if found_opening and brace_count == 0:
                        return i + 1
        
        return len(lines)
    
    async def analyze_directory(self, directory_path: str, max_files: int = 1000) -> Dict[str, Any]:
        """디렉토리 전체 분석"""
        
        try:
            directory = Path(directory_path)
            if not directory.exists():
                return {'error': f'Directory not found: {directory_path}'}
            
            # 지원하는 파일 확장자
            extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.go', '.rs', '.php', '.rb']
            
            # 파일 수집
            files = []
            for ext in extensions:
                pattern = f"**/*{ext}"
                found_files = list(directory.glob(pattern))
                files.extend(found_files)
            
            # 제외할 디렉토리
            exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build'}
            files = [f for f in files if not any(part in exclude_dirs for part in f.parts)]
            
            if max_files:
                files = files[:max_files]
            
            print(f"분석할 파일: {len(files)}개")
            
            # 병렬 분석
            tasks = []
            semaphore = asyncio.Semaphore(8)  # 동시 처리 제한
            
            async def analyze_with_semaphore(file_path: Path) -> Optional[Dict[str, Any]]:
                async with semaphore:
                    return await self.analyze_file(str(file_path))
            
            for file_path in files:
                task = asyncio.create_task(analyze_with_semaphore(file_path))
                tasks.append(task)
            
            # 결과 수집
            results = []
            for i, task in enumerate(asyncio.as_completed(tasks)):
                try:
                    result = await task
                    if result:
                        results.append(result)
                    
                    if (i + 1) % 50 == 0:
                        print(f"진행상황: {i + 1}/{len(files)} 파일 분석 완료")
                        
                except Exception as e:
                    print(f"파일 분석 오류: {e}")
                    continue
            
            # 통계 생성
            stats = self._generate_analysis_stats(results)
            
            return {
                'directory': str(directory),
                'total_files': len(files),
                'analyzed_files': len(results),
                'statistics': stats,
                'files': results
            }
            
        except Exception as e:
            return {'error': f'Directory analysis failed: {str(e)}'}
    
    def _generate_analysis_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """분석 통계 생성"""
        stats: Dict[str, Any] = {
            'languages': {},
            'total_functions': 0,
            'total_classes': 0,
            'total_imports': 0,
            'total_lines': 0,
            'complexity_distribution': {'low': 0, 'medium': 0, 'high': 0},
            'top_dependencies': {},
            'file_types': {}
        }
        for result in results:
            if not isinstance(result, dict):
                continue
            language = result['language']
            # 언어별 통계
            if not isinstance(stats['languages'], dict):
                stats['languages'] = {}
            if language not in stats['languages']:
                stats['languages'][language] = {
                    'files': 0,
                    'functions': 0,
                    'classes': 0,
                    'lines': 0
                }
            if not isinstance(stats['languages'][language], dict):
                stats['languages'][language] = {'files': 0, 'functions': 0, 'classes': 0, 'lines': 0}
            stats['languages'][language]['files'] += 1
            stats['languages'][language]['functions'] += len(result.get('functions', []))
            stats['languages'][language]['classes'] += len(result.get('classes', []))
            stats['languages'][language]['lines'] += result.get('lines_of_code', 0)
            # 전체 통계
            stats['total_functions'] += len(result.get('functions', []))
            stats['total_classes'] += len(result.get('classes', []))
            stats['total_imports'] += len(result.get('imports', []))
            stats['total_lines'] += result.get('lines_of_code', 0)
            # 복잡도 분포
            if not isinstance(stats['complexity_distribution'], dict):
                stats['complexity_distribution'] = {'low': 0, 'medium': 0, 'high': 0}
            complexity = result.get('complexity', 0)
            if complexity < 5:
                stats['complexity_distribution']['low'] += 1
            elif complexity < 15:
                stats['complexity_distribution']['medium'] += 1
            else:
                stats['complexity_distribution']['high'] += 1
            # 의존성 통계
            if not isinstance(stats['top_dependencies'], dict):
                stats['top_dependencies'] = {}
            for dep in result.get('dependencies', []):
                stats['top_dependencies'][dep] = stats['top_dependencies'].get(dep, 0) + 1
            # 파일 타입 통계
            if not isinstance(stats['file_types'], dict):
                stats['file_types'] = {}
            file_ext = Path(result['file_path']).suffix
            stats['file_types'][file_ext] = stats['file_types'].get(file_ext, 0) + 1
        # 상위 의존성 정렬
        if not isinstance(stats['top_dependencies'], dict):
            stats['top_dependencies'] = {}
        stats['top_dependencies'] = dict(
            sorted(stats['top_dependencies'].items(), key=lambda x: x[1], reverse=True)[:10]
        )
        return stats

    async def _analyze_css_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """CSS 파일 분석 (tree-sitter-css → fallback: 정규식)"""
        result = {
            'file_path': file_path,
            'language': Language.CSS.value,
            'selectors': [],
            'at_rules': [],
            'lines_of_code': 0,
            'complexity': 0,
            'dependencies': [],
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'exports': []
        }
        lines = content.split('\n')
        result['lines_of_code'] = len([line for line in lines if line.strip() and not line.strip().startswith('/*')])
        try:
            from tree_sitter import Parser
            from tree_sitter_css import CSS
            parser = Parser()
            parser.language = CSS
            tree = parser.parse(bytes(content, 'utf8'))
            root = tree.root_node
            selectors = set()
            at_rules = set()
            def walk(node):
                if node.type == 'qualified_rule':
                    sel_text = content[node.start_byte:node.end_byte].split('{')[0].strip()
                    selectors.add(sel_text)
                elif node.type == 'at_rule':
                    at_text = content[node.start_byte:node.end_byte].split('{')[0].strip()
                    at_rules.add(at_text)
                for child in node.children:
                    walk(child)
            walk(root)
            result['selectors'] = list(selectors)
            result['at_rules'] = list(at_rules)
        except Exception as e:
            # Fallback: 정규식 기반 분석
            import re
            result['selectors'] = re.findall(r'([.#]?[a-zA-Z0-9_-]+)\s*{', content)
            result['at_rules'] = re.findall(r'@([a-zA-Z\-]+)\s*[^\{]*{', content)
            result['fallback_reason'] = f"tree-sitter-css import/분석 실패: {e}"
        return result

    async def _analyze_vue_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Vue SFC(.vue) 파일 분석: template/script/style 분리 및 컴포넌트/props/emit/slot/라우트/스토어/스타일 추출"""
        import re
        from collections import defaultdict
        result = {
            'file_path': file_path,
            'language': Language.VUE.value,
            'components': [],
            'props': [],
            'emits': [],
            'slots': [],
            'used_components': [],
            'router_links': [],
            'store_usage': [],
            'style_classes': [],
            'template_classes': [],
            'methods': [],
            'computed': [],
            'watch': [],
            'imports': [],
            'lines_of_code': 0,
            'complexity': 0,
            'dependencies': set(),
            'exports': []
        }
        # SFC 분리
        template = re.search(r'<template[\s\S]*?>([\s\S]*?)<\/template>', content)
        script = re.search(r'<script[\s\S]*?>([\s\S]*?)<\/script>', content)
        style = re.search(r'<style[\s\S]*?>([\s\S]*?)<\/style>', content)
        template_content = template.group(1) if template else ''
        script_content = script.group(1) if script else ''
        style_content = style.group(1) if style else ''
        # template: 사용 컴포넌트, slot, props, emits, router-link, v-*, class 등
        result['slots'] = re.findall(r'<slot(?:\s+name=["\"][^"\"]+["\"])?>', template_content)
        result['router_links'] = re.findall(r'<router-link[^>]*to=["\"][^"\"]+["\"]', template_content)
        result['template_classes'] = re.findall(r'class=["\"][^"\"]+["\"]', template_content)
        # script: export default, props, emits, methods, computed, watch, import
        result['props'] = re.findall(r'props\s*:\s*\[([^\]]+)\]', script_content)
        result['emits'] = re.findall(r'emits\s*:\s*\[([^\]]+)\]', script_content)
        result['methods'] = re.findall(r'methods\s*:\s*{([\s\S]*?)}', script_content)
        result['computed'] = re.findall(r'computed\s*:\s*{([\s\S]*?)}', script_content)
        result['watch'] = re.findall(r'watch\s*:\s*{([\s\S]*?)}', script_content)
        result['imports'] = re.findall(r'import\s+([\w{}*,\s]+)\s+from\s+["\"][^"\"]+["\"]', script_content)
        # style: 클래스/ID 추출
        result['style_classes'] = re.findall(r'\.([\w-]+)\s*{', style_content)
        # used components (template 태그명 중 대문자/케밥케이스)
        result['used_components'] = re.findall(r'<([A-Z][\w-]*)', template_content)
        # lines of code
        result['lines_of_code'] = len(content.split('\n'))
        return result

    async def _analyze_angular_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Angular 컴포넌트/서비스/모듈/라우트 분석: @Component, @NgModule, @Injectable, @Input, @Output, 라우트, DI 등 추출"""
        import re
        result = {
            'file_path': file_path,
            'language': Language.ANGULAR.value,
            'components': [],
            'services': [],
            'modules': [],
            'routes': [],
            'inputs': [],
            'outputs': [],
            'injectables': [],
            'imports': [],
            'lines_of_code': 0,
            'complexity': 0,
            'dependencies': set(),
            'exports': []
        }
        # 컴포넌트
        result['components'] = re.findall(r'@Component\s*\(([^)]*)\)', content)
        result['services'] = re.findall(r'@Injectable\s*\(([^)]*)\)', content)
        result['modules'] = re.findall(r'@NgModule\s*\(([^)]*)\)', content)
        result['routes'] = re.findall(r'RouterModule\.forRoot\(([^)]*)\)', content)
        result['inputs'] = re.findall(r'@Input\(\)\s*(\w+)', content)
        result['outputs'] = re.findall(r'@Output\(\)\s*(\w+)', content)
        result['injectables'] = re.findall(r'constructor\s*\(([^)]*)\)', content)
        result['imports'] = re.findall(r'import\s+([\w{}*,\s]+)\s+from\s+["\"][^"\"]+["\"]', content)
        result['lines_of_code'] = len(content.split('\n'))
        return result

    async def _analyze_react_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """React(JSX/TSX) 파일 분석: 컴포넌트, props, state, hooks, 이벤트, 라우트 등 추출"""
        import re
        result = {
            'file_path': file_path,
            'language': 'react',
            'components': [],
            'props': [],
            'state': [],
            'hooks': [],
            'events': [],
            'routes': [],
            'imports': [],
            'lines_of_code': len(content.split('\n')),
            'complexity': 0,
            'dependencies': set(),
            'exports': []
        }
        # 컴포넌트 정의 (function/class)
        result['components'] = re.findall(r'(?:function|const|let|var|class)\s+(\w+)\s*[=\(]', content)
        # props
        result['props'] = re.findall(r'props\.([\w]+)', content)
        # state (useState)
        result['state'] = re.findall(r'const\s*\[([\w]+),', content)
        # hooks
        result['hooks'] = re.findall(r'use(\w+)\s*\(', content)
        # 이벤트 핸들러
        result['events'] = re.findall(r'on\w+\s*=\s*\{(\w+)', content)
        # 라우트 (react-router)
        result['routes'] = re.findall(r'<Route[^>]+path=["\"][^"\"]+["\"]', content)
        # import
        result['imports'] = re.findall(r'import\s+([\w{}*,\s]+)\s+from\s+["\'][^"\']+["\']', content)
        return result

    async def _analyze_javascript_file_treesitter(self, file_path: str, content: str) -> Dict[str, Any]:
        """JS/TS/JSX/TSX 파일 tree-sitter 기반 구조 분석 (함수/클래스/임포트/호출/컴포넌트 등)"""
        from tree_sitter import Parser, Node
        from tree_sitter_languages import get_language as get_ts_language
        import re
        parser = Parser()
        ext = file_path.split('.')[-1]
        lang = 'tsx' if file_path.endswith('.tsx') else 'jsx' if file_path.endswith('.jsx') else 'typescript' if file_path.endswith('.ts') else 'javascript'
        parser.language = get_ts_language(lang)
        tree = parser.parse(bytes(content, 'utf8'))
        root = tree.root_node
        result: Dict[str, Any] = {
            'file_path': file_path,
            'language': lang,
            'functions': [],
            'classes': [],
            'imports': [],
            'exports': [],
            'components': [],
            'props': [],
            'state': [],
            'hooks': [],
            'events': [],
            'calls': [],
            'lines_of_code': len(content.split('\n')),
            'complexity': 0,
            'dependencies': set(),
        }
        # tree-sitter 기반 구조 탐색
        def walk(node: Node, parent: Optional[Node] = None) -> None:
            # Ensure all result fields are lists before appending
            for key in ['functions', 'classes', 'imports', 'exports', 'components', 'props', 'state', 'hooks', 'events', 'calls']:
                if not isinstance(result[key], list):
                    if isinstance(result[key], (set, tuple)):
                        result[key] = [str(x) for x in result[key]]
                    else:
                        result[key] = []
            if node.type == 'function_declaration':
                name = node.child_by_field_name('name')
                if name is not None and hasattr(name, 'text') and name.text is not None:
                    if isinstance(result['functions'], list):
                        result['functions'].append(name.text.decode())
            elif node.type == 'class_declaration':
                name = node.child_by_field_name('name')
                if name is not None and hasattr(name, 'text') and name.text is not None:
                    if isinstance(result['classes'], list):
                        result['classes'].append(name.text.decode())
            elif node.type == 'import_statement':
                if isinstance(result['imports'], list):
                    result['imports'].append(content[node.start_byte:node.end_byte])
            elif node.type == 'export_statement':
                if isinstance(result['exports'], list):
                    result['exports'].append(content[node.start_byte:node.end_byte])
            elif node.type == 'call_expression':
                callee = node.child_by_field_name('function')
                if callee is not None and hasattr(callee, 'start_byte') and hasattr(callee, 'end_byte'):
                    if isinstance(result['calls'], list):
                        result['calls'].append(content[callee.start_byte:callee.end_byte])
            elif node.type == 'variable_declaration':
                # useState, useEffect, etc.
                text = content[node.start_byte:node.end_byte]
                if 'useState' in text:
                    if isinstance(result['state'], list):
                        result['state'].append(text)
                if 'useEffect' in text:
                    if isinstance(result['hooks'], list):
                        result['hooks'].append('useEffect')
                if 'useContext' in text:
                    if isinstance(result['hooks'], list):
                        result['hooks'].append('useContext')
            elif node.type == 'jsx_opening_element':
                tag = node.child_by_field_name('name')
                if tag is not None and hasattr(tag, 'text') and tag.text is not None and re.match(b'[A-Z]', tag.text):
                    if isinstance(result['components'], list):
                        result['components'].append(tag.text.decode())
            for child in node.children:
                walk(child, node)
        walk(root)
        # props, events (JSX/TSX)
        if file_path.endswith(('.jsx', '.tsx')):
            result['props'] = re.findall(r'props\.([\w]+)', content)
            result['events'] = re.findall(r'on\w+\s*=\s*\{(\w+)', content)
        if isinstance(result['dependencies'], set):
            result['dependencies'] = [str(x) for x in result['dependencies']]
        # 상세 로깅
        print(f"[tree-sitter JS 분석] {file_path}: functions={result['functions']}, classes={result['classes']}, components={result['components']}, imports={len(list(result['imports']))}, calls={len(list(result['calls']))}")
        return result

    async def _analyze_python_file_treesitter(self, file_path: str, content: str) -> Dict[str, Any]:
        """Python 파일 tree-sitter 기반 구조 분석 (함수/클래스/임포트/데코레이터/호출 등)"""
        from tree_sitter import Parser, Node
        from tree_sitter_languages import get_language as get_ts_language
        import re
        parser = Parser()
        parser.language = get_ts_language('python')
        tree = parser.parse(bytes(content, 'utf8'))
        root = tree.root_node
        result = {
            'functions': [],
            'classes': [],
            'imports': [],
            'decorators': [],
            'calls': [],
        }
        def walk(node: Node, parent: Node = None) -> None:
            if node.type == 'function_definition':
                name = node.child_by_field_name('name')
                if name is not None and hasattr(name, 'text') and name.text is not None:
                    result['functions'].append(name.text.decode())
                # decorators
                decorators = [c for c in node.children if c.type == 'decorator']
                for deco in decorators:
                    deco_name = deco.child_by_field_name('name')
                    if deco_name is not None and hasattr(deco_name, 'text') and deco_name.text is not None:
                        result['decorators'].append(deco_name.text.decode())
            elif node.type == 'class_definition':
                name = node.child_by_field_name('name')
                if name is not None and hasattr(name, 'text') and name.text is not None:
                    result['classes'].append(name.text.decode())
            elif node.type == 'import_statement':
                result['imports'].append(content[node.start_byte:node.end_byte])
            elif node.type == 'call':
                call_name = node.child_by_field_name('function')
                if call_name is not None and hasattr(call_name, 'text') and call_name.text is not None:
                    result['calls'].append(call_name.text.decode())
            for child in node.children:
                walk(child, node)
        walk(root)
        return result

    async def _analyze_json_file(self, file_path: str, content: str) -> Dict[str, Any]:
        import json
        result = {'file_path': file_path, 'language': 'json', 'keys': [], 'num_keys': 0, 'lines_of_code': len(content.splitlines())}
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                result['keys'] = list(data.keys())
                result['num_keys'] = len(data)
            elif isinstance(data, list):
                result['keys'] = [f'list[{len(data)}]']
                result['num_keys'] = len(data)
        except Exception as e:
            result['error'] = str(e)
        return result

    async def _analyze_markdown_file(self, file_path: str, content: str) -> Dict[str, Any]:
        import re
        result = {'file_path': file_path, 'language': 'markdown', 'headers': [], 'links': [], 'code_blocks': [], 'lines_of_code': len(content.splitlines())}
        result['headers'] = re.findall(r'^(#+)\s+(.*)', content, re.MULTILINE)
        result['links'] = re.findall(r'\[(.*?)\]\((.*?)\)', content)
        result['code_blocks'] = re.findall(r'```([\w]*)\n([\s\S]*?)```', content)
        return result

    async def _analyze_yaml_file(self, file_path: str, content: str) -> Dict[str, Any]:
        try:
            import ruamel.yaml
            yaml = ruamel.yaml.YAML(typ='safe')
            data = yaml.load(content)
            keys = list(data.keys()) if isinstance(data, dict) else []
            return {'file_path': file_path, 'language': 'yaml', 'keys': keys, 'lines_of_code': len(content.splitlines())}
        except Exception as e:
            return {'file_path': file_path, 'language': 'yaml', 'error': str(e), 'lines_of_code': len(content.splitlines())}

    async def _analyze_shell_file(self, file_path: str, content: str) -> Dict[str, Any]:
        import re
        result = {'file_path': file_path, 'language': 'shell', 'functions': [], 'variables': [], 'lines_of_code': len(content.splitlines())}
        result['functions'] = re.findall(r'^([\w_]+)\s*\(\)\s*\{', content, re.MULTILINE)
        result['variables'] = re.findall(r'^([\w_]+)=', content, re.MULTILINE)
        return result

    async def _analyze_ini_file(self, file_path: str, content: str) -> Dict[str, Any]:
        import configparser
        from io import StringIO
        parser = configparser.ConfigParser()
        result = {'file_path': file_path, 'language': 'ini', 'sections': [], 'lines_of_code': len(content.splitlines())}
        try:
            parser.read_file(StringIO(content))
            result['sections'] = parser.sections()
        except Exception as e:
            result['error'] = str(e)
        return result

    async def _analyze_txt_file(self, file_path: str, content: str) -> Dict[str, Any]:
        lines = content.splitlines()
        return {'file_path': file_path, 'language': 'txt', 'num_lines': len(lines), 'num_words': sum(len(line.split()) for line in lines)}

    async def _analyze_dockerfile(self, file_path: str, content: str) -> Dict[str, Any]:
        import re
        result = {'file_path': file_path, 'language': 'dockerfile', 'instructions': [], 'lines_of_code': len(content.splitlines())}
        result['instructions'] = re.findall(r'^(FROM|RUN|CMD|LABEL|EXPOSE|ENV|ADD|COPY|ENTRYPOINT|VOLUME|USER|WORKDIR|ARG|ONBUILD|STOPSIGNAL|HEALTHCHECK|SHELL)\b', content, re.MULTILINE)
        return result

    async def _analyze_csv_file(self, file_path: str, content: str) -> Dict[str, Any]:
        import csv
        from io import StringIO
        reader = csv.reader(StringIO(content))
        rows = list(reader)
        return {'file_path': file_path, 'language': 'csv', 'num_rows': len(rows), 'num_columns': len(rows[0]) if rows else 0}

    async def _analyze_tsv_file(self, file_path: str, content: str) -> Dict[str, Any]:
        import csv
        from io import StringIO
        reader = csv.reader(StringIO(content), delimiter='\t')
        rows = list(reader)
        return {'file_path': file_path, 'language': 'tsv', 'num_rows': len(rows), 'num_columns': len(rows[0]) if rows else 0}

    async def _analyze_lock_file(self, file_path: str, content: str) -> Dict[str, Any]:
        lines = content.splitlines()
        return {'file_path': file_path, 'language': 'lock', 'num_lines': len(lines)}

    async def _analyze_env_file(self, file_path: str, content: str) -> Dict[str, Any]:
        import re
        result = {'file_path': file_path, 'language': 'env', 'variables': [], 'lines_of_code': len(content.splitlines())}
        result['variables'] = re.findall(r'^([\w_]+)=', content, re.MULTILINE)
        return result

    async def _analyze_makefile(self, file_path: str, content: str) -> Dict[str, Any]:
        import re
        result = {'file_path': file_path, 'language': 'makefile', 'targets': [], 'lines_of_code': len(content.splitlines())}
        result['targets'] = re.findall(r'^([\w\-]+):', content, re.MULTILINE)
        return result

    async def _analyze_gradle_file(self, file_path: str, content: str) -> Dict[str, Any]:
        lines = content.splitlines()
        return {'file_path': file_path, 'language': 'gradle', 'num_lines': len(lines)}

    async def _analyze_gitignore_file(self, file_path: str, content: str) -> Dict[str, Any]:
        lines = content.splitlines()
        patterns = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        return {'file_path': file_path, 'language': 'gitignore', 'patterns': patterns, 'num_patterns': len(patterns)}

class PythonASTAnalyzer(ast.NodeVisitor):
    """Python AST 전용 분석기"""
    
    def __init__(self, result: Dict[str, Any]) -> None:
        self.result = result
        self.current_class = None
        self.function_calls: list = []
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """함수 정의 방문"""
        self._process_function(node, is_async=False)
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """비동기 함수 정의 방문"""
        self._process_function(node, is_async=True)
        self.generic_visit(node)
    
    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool) -> None:
        """함수 처리"""
        
        # 파라미터 추출
        parameters = []
        for arg in node.args.args:
            if arg.arg != 'self':
                parameters.append(arg.arg)
        
        # 반환 타입 추출
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        
        # 독스트링 추출
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        # 데코레이터 추출
        decorators = []
        for decorator in node.decorator_list:
            if hasattr(ast, 'unparse'):
                decorators.append(ast.unparse(decorator))
            else:
                # Python 3.8 이하 호환성
                if isinstance(decorator, ast.Name):
                    decorators.append(getattr(decorator, 'id', ''))
                elif isinstance(decorator, ast.Attribute):
                    decorators.append(f"{getattr(decorator.value, 'id', '')}.{decorator.attr}")
        
        # 함수 호출 추출
        call_visitor = FunctionCallVisitor()
        call_visitor.visit(node)
        
        # 복잡도 계산
        if isinstance(node, ast.FunctionDef):
            complexity = self._calculate_complexity(node)
        else:
            complexity = 1
        
        func_element = FunctionElement(
            name=node.name,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            element_type='function',
            content='',  # 필요시 나중에 채움
            language=Language.PYTHON.value,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            complexity=complexity,
            is_async=is_async,
            decorators=decorators,
            calls=call_visitor.calls,
            metadata=None
        )
        
        if self.current_class:
            self.current_class['methods'].append(asdict(func_element))
        else:
            self.result['functions'].append(asdict(func_element))
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """클래스 정의 방문"""
        
        # 상속 정보 추출
        inheritance = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                inheritance.append(base.id)
            elif isinstance(base, ast.Attribute):
                # module.Class 형태
                inheritance.append(f"{getattr(base.value, 'id', '')}.{base.attr}" if hasattr(base.value, 'id') else base.attr)
        
        # 독스트링 추출
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        class_element = ClassElement(
            name=node.name,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            element_type='class',
            content='',
            language=Language.PYTHON.value,
            methods=[],
            attributes=[],
            inheritance=inheritance,
            docstring=docstring,
            metadata=None
        )
        
        # 현재 클래스 설정
        old_class = self.current_class
        self.current_class = asdict(class_element) if class_element else None
        
        # 클래스 내부 방문
        self.generic_visit(node)
        
        # 속성 추출
        attr_visitor = AttributeVisitor()
        attr_visitor.visit(node)
        if self.current_class is not None:
            self.current_class['attributes'] = attr_visitor.attributes
        
        if self.current_class is not None:
            self.result['classes'].append(self.current_class)
        self.current_class = old_class
    
    def visit_Import(self, node: ast.Import) -> None:
        """import 방문"""
        for alias in node.names:
            import_element = ImportElement(
                name=alias.name,
                start_line=node.lineno,
                end_line=node.lineno,
                element_type='import',
                content=f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                language=Language.PYTHON.value,
                module=alias.name,
                alias=alias.asname,
                is_from_import=False,
                metadata=None
            )
            
            self.result['imports'].append(asdict(import_element))
            self.result['dependencies'].add(alias.name.split('.')[0])
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """from import 방문"""
        if node.module:
            items = [alias.name for alias in node.names]
            
            import_element = ImportElement(
                name=', '.join(items),
                start_line=node.lineno,
                end_line=node.lineno,
                element_type='import',
                content=f"from {node.module} import {', '.join(items)}",
                language=Language.PYTHON.value,
                module=node.module,
                items=items,
                is_from_import=True,
                metadata=None
            )
            
            self.result['imports'].append(asdict(import_element))
            self.result['dependencies'].add(node.module.split('.')[0])
        
        self.generic_visit(node)
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """함수 복잡도 계산 (순환 복잡도)"""
        complexity = 1  # 기본 복잡도
        
        for child in ast.walk(node):
            # 분기문
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            # 예외 처리
            elif isinstance(child, (ast.Try, ast.ExceptHandler)):
                complexity += 1
            # 논리 연산자
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            # 조건 표현식
            elif isinstance(child, ast.IfExp):
                complexity += 1
        
        return complexity

class FunctionCallVisitor(ast.NodeVisitor):
    """함수 호출 추출 방문자"""
    
    def __init__(self) -> None:
        self.calls: list = []
    
    def visit_Call(self, node: ast.Call) -> None:
        """함수 호출 방문"""
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)
        
        self.generic_visit(node)

class AttributeVisitor(ast.NodeVisitor):
    """클래스 속성 추출 방문자"""
    
    def __init__(self) -> None:
        self.attributes: list = []
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """할당문 방문"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.attributes.append(target.id)
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if target.value.id == 'self':
                    self.attributes.append(target.attr)
        
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """타입 어노테이션 할당 방문"""
        if isinstance(node.target, ast.Name):
            self.attributes.append(node.target.id)
        elif isinstance(node.target, ast.Attribute) and isinstance(node.target.value, ast.Name):
            if node.target.value.id == 'self':
                self.attributes.append(node.target.attr)
        
        self.generic_visit(node)

def get_code_analyzer() -> AlternativeCodeAnalyzer:
    """공식 코드 분석기 인스턴스 반환"""
    return AlternativeCodeAnalyzer()