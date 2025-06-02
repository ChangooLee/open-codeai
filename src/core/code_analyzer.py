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
    
    def __init__(self):
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
    
    def detect_language(self, file_path: str) -> Language:
        """파일 확장자로 언어 감지"""
        extension = Path(file_path).suffix.lower()
        
        extension_map = {
            '.py': Language.PYTHON,
            '.js': Language.JAVASCRIPT,
            '.jsx': Language.JAVASCRIPT,
            '.ts': Language.TYPESCRIPT,
            '.tsx': Language.TYPESCRIPT,
            '.java': Language.JAVA,
            '.cpp': Language.CPP,
            '.cc': Language.CPP,
            '.cxx': Language.CPP,
            '.c': Language.C,
            '.go': Language.GO,
            '.rs': Language.RUST,
            '.php': Language.PHP,
            '.rb': Language.RUBY,
        }
        
        return extension_map.get(extension, Language.PYTHON)
    
    async def analyze_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """파일 분석"""
        try:
            if not os.path.exists(file_path):
                return None
            
            language = self.detect_language(file_path)
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 언어별 분석
            if language == Language.PYTHON:
                return await self._analyze_python_file(file_path, content)
            elif language == Language.JAVA:
                # Java 분석 + 매퍼 XML 연동
                result = await self._analyze_generic_file(file_path, content, language)
                # 같은 디렉토리 내 매퍼 XML 자동 탐색
                dir_path = os.path.dirname(file_path)
                mappers = []
                for fname in os.listdir(dir_path):
                    if fname.endswith('Mapper.xml'):
                        xml_path = os.path.join(dir_path, fname)
                        mappers.extend(parse_mapper_xml(xml_path))
                result['mappers'] = mappers
                return result
            else:
                return await self._analyze_generic_file(file_path, content, language)
                
        except Exception as e:
            print(f"파일 분석 오류 {file_path}: {e}")
            return None
    
    async def _analyze_python_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Python 파일 AST 기반 분석"""
        
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
            'exports': []
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
        result['dependencies'] = list(result['dependencies'])
        
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
        stats = {
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

class PythonASTAnalyzer(ast.NodeVisitor):
    """Python AST 전용 분석기"""
    
    def __init__(self, result: Dict[str, Any]) -> None:
        self.result = result
        self.current_class = None
        self.function_calls = []
    
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
                    decorators.append(decorator.id)
                elif isinstance(decorator, ast.Attribute):
                    decorators.append(f"{decorator.value.id}.{decorator.attr}")
        
        # 함수 호출 추출
        call_visitor = FunctionCallVisitor()
        call_visitor.visit(node)
        
        # 복잡도 계산
        complexity = self._calculate_complexity(node)
        
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
                inheritance.append(f"{base.value.id}.{base.attr}" if hasattr(base.value, 'id') else base.attr)
        
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
        self.current_class = asdict(class_element)
        
        # 클래스 내부 방문
        self.generic_visit(node)
        
        # 속성 추출
        attr_visitor = AttributeVisitor()
        attr_visitor.visit(node)
        self.current_class['attributes'] = attr_visitor.attributes
        
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
        self.calls: List[str] = []
    
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
        self.attributes: List[str] = []
    
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